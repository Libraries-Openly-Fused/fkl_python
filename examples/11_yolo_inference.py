"""Example 11 — REAL end-to-end pipeline: YOLOv8 object detection with
fkl-python GPU preprocessing.

    JPEG -> [GPU: ONE fused kernel] -> YOLOv8n (onnxruntime) -> boxes

The preprocessing that ultralytics does in several numpy/torch steps
(letterbox resize + RGB normalize + HWC->CHW) is ONE fused FKL kernel:

    TensorRead          uchar3 frame (any size)
    Resize(640, 640,    bilinear letterbox, gray(114) padding --
      preserve AR)        the exact YOLO convention
    Div(255)            normalize to [0,1]
    TensorSplit         packed HWC -> planar CHW float, NCHW-ready

Requirements (see examples/assets/):
    pip install numpy pillow onnxruntime
    assets/bus.jpg       (ultralytics sample image)
    assets/yolov8n.onnx  (COCO-pretrained YOLOv8n, 1x3x640x640 input)

Run:  python3 11_yolo_inference.py [image.jpg]
"""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

import fkl

ASSETS = Path(__file__).parent / "assets"
NET = 640
CONF_THRES, IOU_THRES = 0.25, 0.45

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic-light "
        "fire-hydrant stop-sign parking-meter bench bird cat dog horse sheep cow "
        "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports-ball kite baseball-bat baseball-glove skateboard "
        "surfboard tennis-racket bottle wine-glass cup fork knife spoon bowl banana "
        "apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch "
        "potted-plant bed dining-table toilet tv laptop mouse remote keyboard "
        "cell-phone microwave oven toaster sink refrigerator book clock vase "
        "scissors teddy-bear hair-drier toothbrush").split()


# --------------------------------------------------------------------------
# GPU preprocessing: ONE fused kernel, compiled once, reused every frame
# --------------------------------------------------------------------------

def build_preprocessor():
    return fkl.compose(
        fkl.TensorRead(),                                  # uchar3 HxW
        fkl.Resize(NET, NET, interp="linear",
                   aspect_ratio="preserve",                # letterbox
                   background=(114.0, 114.0, 114.0)),      # YOLO gray pad
        fkl.Div((255.0, 255.0, 255.0)),                    # [0,1]
        fkl.TensorSplit(),                                 # HWC -> CHW float
    )


def upload_image(img_np):
    """HxWx3 uint8 numpy -> fkl.DeviceBuffer (zero further copies)."""
    h, w, _ = img_np.shape
    buf = fkl.DeviceBuffer(w, h, "uint8", channels=3)
    buf.copy_from_host(img_np.tobytes())
    return buf


def gpu_preprocess(pipe, frame_dev):
    """Run the fused kernel; return a (1,3,640,640) float32 numpy view."""
    out = pipe(frame_dev)                                  # ONE kernel launch
    chw = np.frombuffer(out.copy_to_host(), dtype=np.float32)
    return chw.reshape(1, 3, NET, NET)


# --------------------------------------------------------------------------
# YOLO postprocessing (standard: decode + NMS), CPU numpy
# --------------------------------------------------------------------------

def letterbox_params(w, h):
    """Replicate FKL's PRESERVE_AR placement: scale + centered offsets."""
    scale = min(NET / w, NET / h)
    nw, nh = round(w * scale), round(h * scale)
    return scale, (NET - nw) // 2, (NET - nh) // 2


def postprocess(pred, orig_w, orig_h):
    """pred: (1, 84, 8400) -> [(x1,y1,x2,y2,conf,cls), ...] in image coords."""
    p = pred[0].T                                          # (8400, 84)
    boxes, scores = p[:, :4], p[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= CONF_THRES
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

    # xywh (letterbox space) -> xyxy (original image space)
    scale, ox, oy = letterbox_params(orig_w, orig_h)
    x, y, w, h = boxes.T
    x1 = (x - w / 2 - ox) / scale
    y1 = (y - h / 2 - oy) / scale
    x2 = (x + w / 2 - ox) / scale
    y2 = (y + h / 2 - oy) / scale
    boxes = np.stack([x1.clip(0, orig_w), y1.clip(0, orig_h),
                      x2.clip(0, orig_w), y2.clip(0, orig_h)], 1)

    # class-aware NMS
    order = conf.argsort()[::-1]
    result = []
    while order.size:
        i = order[0]
        result.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        same = cls[rest] == cls[i]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[~(same & (iou > IOU_THRES))]
    return [(boxes[i], conf[i], cls[i]) for i in result]


# --------------------------------------------------------------------------

def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ASSETS / "bus.jpg"
    model_path = ASSETS / "yolov8n.onnx"
    for p in (img_path, model_path):
        if not p.exists():
            sys.exit(f"missing {p} — see the docstring for setup")

    img = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    print(f"image: {img_path.name} {W}x{H}")

    session = ort.InferenceSession(str(model_path),
                                   providers=["CPUExecutionProvider"])
    pipe = build_preprocessor()
    frame = upload_image(img)

    # warmup (compiles the fused kernel once; cached on disk afterwards)
    t0 = time.perf_counter()
    blob = gpu_preprocess(pipe, frame)
    t_first = (time.perf_counter() - t0) * 1e3

    # steady-state preprocessing timing
    t0 = time.perf_counter()
    N = 50
    for _ in range(N):
        blob = gpu_preprocess(pipe, frame)
    t_pre = (time.perf_counter() - t0) * 1e3 / N

    t0 = time.perf_counter()
    pred = session.run(None, {"images": blob})[0]
    t_inf = (time.perf_counter() - t0) * 1e3

    dets = postprocess(pred, W, H)
    print(f"\npreprocess (fused GPU kernel + D2H): first {t_first:.1f} ms, "
          f"steady {t_pre:.2f} ms")
    print(f"inference (onnxruntime CPU): {t_inf:.1f} ms")
    print(f"\n{len(dets)} detections:")
    for box, conf, c in dets:
        x1, y1, x2, y2 = (int(v) for v in box)
        print(f"  {COCO[c]:<14} {conf:.2f}  [{x1},{y1} -> {x2},{y2}]")

    persons = sum(1 for _, _, c in dets if COCO[c] == "person")
    buses = sum(1 for _, _, c in dets if COCO[c] == "bus")
    print(f"\nsanity: {persons} persons + {buses} bus "
          f"({'OK - expected for bus.jpg' if persons >= 3 and buses >= 1 else 'check image'})")


if __name__ == "__main__":
    main()
