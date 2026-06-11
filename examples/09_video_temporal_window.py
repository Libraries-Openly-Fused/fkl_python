"""Example 09 — CircularTensor: a rolling temporal window for video models.

Temporal DNNs (action recognition, video super-res, tracking) consume the
last N frames as a single tensor. fkl.CircularTensor keeps that window ON
the GPU: each update() preprocesses the incoming frame AND rotates the
window in ONE fused kernel (Divergent HF under the hood — exactly the
CircularTensor mechanism described in the FKL paper).

    no per-frame reallocation - no host round-trips - no N-copy rotation
"""
import fkl
from _util import gpu_image_u8, to_floats, synthetic_rgb

W, H, WINDOW = 64, 48, 4

# the window stores normalized float planes in planar CHW (DNN-ready)
ct = fkl.CircularTensor(W, H, batch=WINDOW, dtype="uint8", channels=3,
                        order="newest_first", layout="planar",
                        out_dtype="float32")

# simulate a camera: push 7 frames through the same fused preproc
for frame_idx in range(7):
    rgb = synthetic_rgb(W, H)
    # encode the frame number in the red channel so we can verify rotation
    rgb = [min(255, v + frame_idx * 10) if i % 3 == 0 else v
           for i, v in enumerate(rgb)]
    frame = gpu_image_u8(rgb, W, H, channels=3)

    ct.update(frame, ops=[fkl.Cast("float32"), fkl.Div(255.0)])

    window = ct.snapshot()        # DeviceBuffer: (WINDOW*3 planes, H, W)
    vals = to_floats(window, W * H * 3 * WINDOW)
    r_means = [sum(vals[k * W * H * 3: k * W * H * 3 + W * H]) / (W * H)
               for k in range(WINDOW)]
    print(f"frame {frame_idx}: window R-plane means "
          f"{[round(m, 3) for m in r_means]}  (newest first)")

print(f"OK  pushed {ct.frames_pushed} frames through a {WINDOW}-frame window")
print("OK  each update = ONE kernel (preproc + insert + rotate, Divergent HF)")
print("    snapshot shape: (WINDOW*C planes, H, W) planar float -> torch.from_dlpack ready")
