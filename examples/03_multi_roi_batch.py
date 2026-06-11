"""Example 03 — Horizontal Fusion: many ROIs from one image, ONE kernel.

The detection->classification pattern: a detector yields N boxes and every
box must be cropped, resized and normalized for the classifier. Naive code
launches N x (crop + resize + normalize) kernels; FKL launches ONE kernel
where each thread-plane (blockIdx.z) processes one ROI.

From Python you just pass a LIST of rects instead of a single rect
(Oscar's rule: lists of parameters = horizontal fusion).
"""
import fkl
from _util import gpu_image_u8, to_floats, synthetic_rgb

SRC_W, SRC_H = 640, 480
frame = gpu_image_u8(synthetic_rgb(SRC_W, SRC_H), SRC_W, SRC_H, channels=3)

# pretend these came from a detector (x, y, w, h) — different positions OK
boxes = [
    (12,  40, 100, 100),
    (300, 80, 100, 100),
    (500, 200, 100, 100),
    (90, 300, 100, 100),
    (420, 350, 100, 100),
]

CLS_W, CLS_H = 64, 64

batch_pipe = fkl.compose(
    fkl.TensorRead(),
    fkl.Crop(boxes),               # LIST => B=5 thread-planes (HF)
    fkl.Resize(CLS_W, CLS_H),      # per-plane resize
    fkl.Mul((1/255.0,) * 3),       # normalize to [0,1]
    fkl.TensorWrite(),             # Tensor with 5 planes
)

out = batch_pipe(frame)            # ONE kernel for all 5 ROIs
n_per_roi = CLS_W * CLS_H * 3
vals = to_floats(out, n_per_roi * len(boxes))

print(f"OK  {len(boxes)} ROIs -> resize {CLS_W}x{CLS_H} -> [0,1] in 1 kernel")
for i in range(len(boxes)):
    plane = vals[i * n_per_roi:(i + 1) * n_per_roi]
    print(f"    roi[{i}] {boxes[i]}: mean={sum(plane)/len(plane):.4f}")

# Box positions are VALUES -> tracking boxes that move every frame reuse
# the kernel. Changing the NUMBER of boxes changes std::array<Rect,B>
# (a C++ type) -> compiles once per distinct B, then cached.
moved = [(x + 4, y + 2, w, h) for (x, y, w, h) in boxes]
batch_pipe2 = fkl.compose(fkl.TensorRead(), fkl.Crop(moved),
                          fkl.Resize(CLS_W, CLS_H), fkl.Mul((1/255.0,) * 3),
                          fkl.TensorWrite())
batch_pipe2(frame)
print("OK  moved boxes (same count) reused the cached kernel")
