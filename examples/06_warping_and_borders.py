"""Example 06 — Geometry: affine/perspective warps and border policies.

  * Warping (BVF): the output grid samples the source through a 2x3 affine
    or 3x3 perspective matrix. The matrix lives in params[] -> animating a
    rotation per frame never recompiles.
  * BorderReader: what to read when a Crop (or warp) reaches outside the
    image: constant fill, replicate edges, reflect, wrap, reflect101.
"""
import math
import fkl
from _util import gpu_image_f32, to_floats

W, H = 64, 64
img = gpu_image_f32([float((x + y) % 50) for y in range(H) for x in range(W)],
                    W, H)

# ---- affine: rotate around the image center ---------------------------------
def rotation_2x3(deg, cx, cy):
    """Maps OUTPUT pixel -> SOURCE pixel (inverse transform)."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    # x_src = c*(x-cx) - s*(y-cy) + cx ; y_src = s*(x-cx) + c*(y-cy) + cy
    return [(c, -s, cx - c * cx + s * cy),
            (s,  c, cy - s * cx - c * cy)]

rotated = fkl.compose(
    fkl.TensorRead(),
    fkl.Warping(rotation_2x3(15.0, W / 2, H / 2), W, H),
    fkl.TensorWrite(),
)(img)
print(f"OK  15deg rotation: out[0:4] = {[round(v,1) for v in to_floats(rotated, 4)]}")

# animate WITHOUT recompiling: a new matrix is just new params
for deg in (30.0, 45.0, 60.0):
    fkl.compose(fkl.TensorRead(),
                fkl.Warping(rotation_2x3(deg, W / 2, H / 2), W, H),
                fkl.TensorWrite())(img)
print("OK  3 more angles reused the same cached kernel")

# ---- perspective (3x3) -------------------------------------------------------
PERSP = [(1.0, 0.1, 0.0),
         (0.05, 1.0, 0.0),
         (0.0005, 0.0, 1.0)]
persp = fkl.compose(fkl.TensorRead(), fkl.Warping(PERSP, W, H),
                    fkl.TensorWrite())(img)
print(f"OK  perspective warp: out[0:4] = {[round(v,1) for v in to_floats(persp, 4)]}")

# ---- border policies on an out-of-bounds crop --------------------------------
# crop hangs 8 px past the right/bottom edge of the image
for mode, extra in (("constant", {"value": -1.0}), ("replicate", {}),
                    ("reflect101", {})):
    out = fkl.compose(
        fkl.TensorRead(),
        fkl.BorderReader(mode, **extra),
        fkl.Crop(W - 8, H - 8, 16, 16),
        fkl.TensorWrite(),
    )(img)
    vals = to_floats(out, 16 * 16)
    corner = vals[-1]                       # bottom-right = fully OOB
    print(f"OK  BorderReader({mode:10s}): OOB corner reads {corner:.1f}")
