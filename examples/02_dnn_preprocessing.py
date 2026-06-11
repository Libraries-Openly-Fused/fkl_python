"""Example 02 — DNN preprocessing in ONE kernel (the paper's flagship case).

Classic inference ingest for an RGB uint8 image:

    crop ROI -> resize to model input -> normalize -> NCHW planar float

In OpenCV/torch this is 4+ kernels and 3 intermediate buffers. With FKL it
is ONE kernel and ZERO intermediates:

  * Crop and Resize are ReadBack ops -> Backwards Vertical Fusion folds
    them INTO the read (each output thread samples the source directly).
  * The arithmetic is Vertical Fusion (registers).
  * TensorSplit writes planar CHW, ready for a DNN.
"""
import fkl
from _util import gpu_image_u8, to_floats, synthetic_rgb

SRC_W, SRC_H = 320, 240          # camera frame
NET_W, NET_H = 32, 32            # model input

frame = gpu_image_u8(synthetic_rgb(SRC_W, SRC_H), SRC_W, SRC_H, channels=3)

MEAN = (123.675, 116.28, 103.53)     # ImageNet mean (RGB)
STD = (58.395, 57.12, 57.375)

preprocess = fkl.compose(
    fkl.TensorRead(),                          # uchar3 frame
    fkl.Crop(40, 30, 240, 180),                # ROI (BVF: fused into read)
    fkl.Resize(NET_W, NET_H),                  # bilinear (BVF, stacked)
    fkl.Sub(MEAN),                             # float3 per-channel
    fkl.Div(STD),
    fkl.TensorSplit(),                         # packed -> planar CHW float
)

out = preprocess(frame)                        # ONE kernel launch
chw = to_floats(out, NET_W * NET_H * 3)

print(f"OK  {SRC_W}x{SRC_H} uchar3 -> crop -> {NET_W}x{NET_H} -> norm -> CHW")
print(f"    output: {len(chw)} floats = 3 planes x {NET_W}x{NET_H}")
print(f"    R-plane[:4] = {[round(v, 3) for v in chw[:4]]}")
print(f"    G-plane[:4] = {[round(v, 3) for v in chw[NET_W*NET_H:NET_W*NET_H+4]]}")

# Values (rect, mean, std, sizes) live in params[]: a tracking ROI that
# moves every frame NEVER recompiles.
for i, rect in enumerate([(50, 35, 240, 180), (60, 40, 240, 180)]):
    moving = fkl.compose(fkl.TensorRead(), fkl.Crop(*rect),
                         fkl.Resize(NET_W, NET_H), fkl.Sub(MEAN),
                         fkl.Div(STD), fkl.TensorSplit())
    moving(frame)
print("OK  moving ROI across frames reuses the same cached kernel")
