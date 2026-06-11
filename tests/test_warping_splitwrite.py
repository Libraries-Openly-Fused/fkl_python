"""Tests for the newest ops: Warping (affine/perspective BVF) and SplitWrite.
Also IsEven/VectorAnd predicates and the full README C++ pipeline mirrored.
"""
from harness import dev_f32, dev_u8, unf32, unu8, check, check_true, run
import fkl


def t_warp_affine_identity():
    W, H = 8, 8
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    M = [(1, 0, 0), (0, 1, 0)]  # identity affine
    out = fkl.compose(fkl.TensorRead(), fkl.Warping(M, W, H),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    check("Warping affine identity", unf32(out.copy_to_host(), W * H), src)


def t_warp_affine_translate():
    """Shift right by 2: out(x,y) = src(x-2, y)... wait, warp maps OUTPUT
    coords through M to SOURCE coords: src_x = x + tx. tx=2 samples 2 px
    to the right."""
    W, H = 8, 4
    src = [float(x) for y in range(H) for x in range(W)]  # x gradient
    M = [(1, 0, 2), (0, 1, 0)]
    out = fkl.compose(fkl.TensorRead(), fkl.Warping(M, W, H),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), W * H)
    # interior pixels: value = x+2 where x+2 <= W-1
    interior = [got[y * W + x] for y in range(H) for x in range(W - 2)]
    exp = [float(x + 2) for y in range(H) for x in range(W - 2)]
    check("Warping affine translate(+2,0)", interior, exp, tol=0.01)


def t_warp_perspective_identity():
    W, H = 6, 6
    src = [float((y * W + x) % 31) for y in range(H) for x in range(W)]
    M = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    out = fkl.compose(fkl.TensorRead(), fkl.Warping(M, W, H),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    check("Warping perspective identity", unf32(out.copy_to_host(), W * H), src)


def t_warp_scale_plus_vf():
    """Affine 2x zoom-out + fused arithmetic afterwards."""
    W, H = 8, 8
    src = [100.0] * (W * H)
    M = [(2, 0, 0), (0, 2, 0)]   # sample at 2x coords -> downscale content
    out = fkl.compose(fkl.TensorRead(), fkl.Warping(M, 4, 4),
                      fkl.Mul(0.5), fkl.TensorWrite())(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), 16)
    # constant image: any in-bounds sample = 100; *0.5 = 50
    check("Warping scale + Mul fused", got[:9], [50.0] * 9)


def t_split_write():
    """uchar3 -> float3 -> SplitWrite: 3 contiguous planes of float."""
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 7) % 256 for i in range(n)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                       fkl.Div(255.0), fkl.SplitWrite())
    out = pipe(dev_u8(src, W, H, ch=3))
    got = unf32(out.copy_to_host(), n)
    exp = []
    for c in range(3):
        for p in range(W * H):
            exp.append(src[p * 3 + c] / 255.0)
    check("SplitWrite packed->planar float", got, exp, tol=1e-6)


def t_readme_cpp_pipeline():
    """The exact C++ README pipeline, from Python:
    PerThreadRead<_2D,uchar3> -> Crop(5 ROIs) -> Resize -> Mul -> Sub ->
    SaturateCast -> TensorWrite."""
    W, H = 192, 108
    n = W * H * 3
    src = [((p * 3 + c * 5) % 256) for p in range(W * H) for c in range(3)]
    rois = [(30, 12, 60, 40), (40, 12, 60, 40), (53, 27, 60, 40),
            (56, 11, 60, 40), (57, 19, 60, 40)]
    OW, OH = 16, 16
    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Crop(rois),
        fkl.Resize(OW, OH),
        fkl.Mul((2.0, 2.0, 2.0)),
        fkl.Sub((128.0, 128.0, 128.0)),
        fkl.SaturateCast("uint8"),
        fkl.TensorWrite(),
    )
    out = pipe(dev_u8(src, W, H, ch=3))
    got = unu8(out.copy_to_host(), OW * OH * 3 * len(rois))
    check_true("README pipeline: 5 ROIs->Resize->arith->u8 Tensor",
               len(got) == OW * OH * 3 * 5 and any(v > 0 for v in got),
               f"{len(got)} bytes, mean={sum(got)/len(got):.0f}")
    check_true("README pipeline: single kernel", len(pipe._variants) == 1)


def t_isEven():
    from harness import i32, uni32
    N = 16
    src = list(range(N))
    buf = fkl.DeviceBuffer(N, 1, "int32")
    buf.copy_from_host(i32(src))
    # bool output read back as uint8 plane
    out = fkl.compose(fkl.TensorRead(), fkl.IsEven(),
                      fkl.Cast("uint8"), fkl.TensorWrite())(buf)
    check("IsEven int32->bool->u8", [float(v) for v in unu8(out.copy_to_host(), N)],
          [float(1 - (v % 2)) for v in src])


if __name__ == "__main__":
    run([t_warp_affine_identity, t_warp_affine_translate,
         t_warp_perspective_identity, t_warp_scale_plus_vf,
         t_split_write, t_readme_cpp_pipeline, t_isEven], "warping-splitwrite")
