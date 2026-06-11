"""Example 04 — Horizontal Fusion over a batch of SEPARATE images.

Multi-camera / multi-stream ingest: N same-size frames from different
buffers, all processed by ONE kernel (FKL BatchRead). From Python you just
pass a LIST of images to the call instead of a single image.

Also shows Divergent HF (the paper's 4th technique): different planes run
DIFFERENT op sequences inside the same kernel.
"""
import fkl
from _util import gpu_image_f32, to_floats

W, H = 128, 96

# ---- 4 "cameras" -----------------------------------------------------------
frames = []
for cam in range(4):
    frames.append(gpu_image_f32(
        [float((cam * 1000 + i) % 256) for i in range(W * H)], W, H))

# ---- same processing for every camera: plain batch HF ----------------------
pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(0.5), fkl.Add(1.0),
                   fkl.TensorWrite())

out = pipe(frames)                       # LIST of images => batch HF
vals = to_floats(out, W * H * 4)
print(f"OK  4 cameras x {W}x{H} -> ONE kernel -> Tensor(4, {H}, {W})")
for cam in range(4):
    plane = vals[cam * W * H:(cam + 1) * W * H]
    print(f"    cam{cam}: out[0]={plane[0]:.1f}")

# ---- different processing per camera: DIVERGENT HF --------------------------
# cam0 is a thermal sensor (needs gain), cams 1-3 are RGB (need offset).
divergent = fkl.compose_divergent(
    [1, 2, 2, 2],                                          # plane -> sequence
    [fkl.TensorRead(), fkl.Mul(4.0), fkl.TensorWrite()],   # seq 1: thermal
    [fkl.TensorRead(), fkl.Add(50.0), fkl.TensorWrite()],  # seq 2: rgb
)
out2 = divergent(frames)                 # still ONE kernel
vals2 = to_floats(out2, W * H * 4)
print("OK  Divergent HF: plane0 ran Mul(4), planes 1-3 ran Add(50)")
print(f"    cam0 out[0] = {vals2[0]:.1f}  (in*4)")
print(f"    cam1 out[0] = {vals2[W*H]:.1f}  (in+50)")

# Batch size B is part of the kernel TYPE: first call per B compiles once.
two = pipe(frames[:2])                   # B=2 -> new signature, cached
print("OK  same pipe with B=2 compiled its own variant (then cached)")
