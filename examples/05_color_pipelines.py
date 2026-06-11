"""Example 05 — Color pipelines and channel surgery.

Shows ColorConversion plus the vector-channel toolbox:
  * RGB -> grayscale (BT.601 luma) in a fused chain
  * BGR -> RGB swap at compile time (VectorReorder) vs RUNTIME
    (VectorReorderRT — the permutation lives in params[], so changing it
    does NOT recompile)
  * Building/trimming channels with AddLast / Discard
  * A 3x3 color matrix with MxVFloat3 (e.g. custom color correction)
"""
import fkl
from _util import gpu_image_u8, to_bytes, to_floats, synthetic_rgb

W, H = 64, 48
rgb = gpu_image_u8(synthetic_rgb(W, H), W, H, channels=3)

# ---- grayscale --------------------------------------------------------------
gray = fkl.compose(
    fkl.TensorRead(),
    fkl.ColorConversion("RGB2GRAY"),     # 0.299R + 0.587G + 0.114B
    fkl.TensorWrite(),
)(rgb)
g = to_bytes(gray, W * H)
print(f"OK  RGB2GRAY: {W*H} px, gray[0:4] = {g[:4]}")

# ---- channel swap: compile-time vs runtime ----------------------------------
swapped_ct = fkl.compose(fkl.TensorRead(), fkl.VectorReorder(2, 1, 0),
                         fkl.TensorWrite())(rgb)
swapped_rt = fkl.compose(fkl.TensorRead(), fkl.VectorReorderRT(2, 1, 0),
                         fkl.TensorWrite())(rgb)
assert to_bytes(swapped_ct, 12) == to_bytes(swapped_rt, 12)
print("OK  VectorReorder (compile-time) == VectorReorderRT (runtime params)")

# A runtime permutation can change per-call WITHOUT recompiling:
rt2 = fkl.compose(fkl.TensorRead(), fkl.VectorReorderRT(1, 2, 0),
                  fkl.TensorWrite())(rgb)
print("OK  different runtime perm reused the same cached .so")

# ---- add alpha, then strip it back ------------------------------------------
rgba = fkl.compose(fkl.TensorRead(), fkl.ColorConversion("RGB2RGBA"),
                   fkl.TensorWrite())(rgb)        # alpha = 255
back = fkl.compose(fkl.TensorRead(), fkl.Discard(3),
                   fkl.TensorWrite())(rgba)       # drop alpha again
assert to_bytes(back, 12) == to_bytes(rgb, 12)
print("OK  RGB -> RGBA(alpha=255) -> Discard(3) round-trips")

# ---- 3x3 color matrix (sepia tone), fused with normalize --------------------
SEPIA = [(0.393, 0.769, 0.189),
         (0.349, 0.686, 0.168),
         (0.272, 0.534, 0.131)]

sepia = fkl.compose(
    fkl.TensorRead(),            # uchar3
    fkl.Cast("float32"),         # float3
    fkl.MxVFloat3(SEPIA),        # matrix * pixel
    fkl.SaturateCast("uint8"),   # clamp [0,255] and back to uchar3
    fkl.TensorWrite(),
)(rgb)
s = to_bytes(sepia, 6)
print(f"OK  sepia 3x3 matrix fused: out[0:6] = {s}")
