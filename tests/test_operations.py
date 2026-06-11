"""Per-operation coverage tests: every op exposed in fkl/__init__ exercised
at least once against a CPU reference. Complements the fusion-mode tests.
"""
from harness import (PASS, FAIL, dev_f32, dev_u8, unf32, unu8, check,
                     check_true, run)
import fkl


# ---- arithmetic ----------------------------------------------------------

def t_add_sub_mul_div_scalar():
    N = 64
    src = [float(i) + 1 for i in range(N)]
    for op, fn in [(fkl.Add(5.0), lambda v: v + 5),
                   (fkl.Sub(3.0), lambda v: v - 3),
                   (fkl.Mul(7.0), lambda v: v * 7),
                   (fkl.Div(4.0), lambda v: v / 4)]:
        out = fkl.compose(fkl.TensorRead(), op, fkl.TensorWrite())(dev_f32(src, N))
        check(f"op {op.name}", unf32(out.copy_to_host(), N), [fn(v) for v in src])


def t_vector_value_broadcast():
    """Scalar value against vector pixels broadcasts to all channels."""
    W, H = 4, 2
    n = W * H * 3
    src = [float(i) for i in range(n)]
    out = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0),
                      fkl.TensorWrite())(dev_f32(src, W, H, ch=3))
    check("Mul scalar->float3 broadcast", unf32(out.copy_to_host(), n),
          [v * 2 for v in src])


def t_max_min():
    N = 64
    src = [float(i - 32) for i in range(N)]
    out = fkl.compose(fkl.TensorRead(), fkl.Max(0.0), fkl.TensorWrite())(dev_f32(src, N))
    check("Max(0) relu", unf32(out.copy_to_host(), N), [max(0.0, v) for v in src])
    out = fkl.compose(fkl.TensorRead(), fkl.Min(0.0), fkl.TensorWrite())(dev_f32(src, N))
    check("Min(0)", unf32(out.copy_to_host(), N), [min(0.0, v) for v in src])


# ---- casts / saturation --------------------------------------------------

def t_cast_f2i():
    N = 32
    from harness import uni32
    src = [float(i) * 1.0 for i in range(N)]
    out = fkl.compose(fkl.TensorRead(), fkl.Cast("int32"), fkl.TensorWrite())(dev_f32(src, N))
    check("Cast float->int32", [float(v) for v in uni32(out.copy_to_host(), N)],
          src)


def t_saturate_cast_negative():
    N = 16
    src = [float(i * 40 - 200) for i in range(N)]  # -200..400
    out = fkl.compose(fkl.TensorRead(), fkl.SaturateCast("uint8"),
                      fkl.TensorWrite())(dev_f32(src, N))
    check("SaturateCast clamps [-200,400]->[0,255]",
          [float(v) for v in unu8(out.copy_to_host(), N)],
          [float(max(0, min(255, round(v)))) for v in src])


def t_saturate_float():
    N = 16
    src = [float(i) / 8.0 - 0.5 for i in range(N)]  # -0.5 .. 1.375
    out = fkl.compose(fkl.TensorRead(), fkl.SaturateFloat(),
                      fkl.TensorWrite())(dev_f32(src, N))
    check("SaturateFloat [0,1]", unf32(out.copy_to_host(), N),
          [max(0.0, min(1.0, v)) for v in src])


def t_saturate_range():
    N = 16
    src = [float(i - 8) for i in range(N)]
    out = fkl.compose(fkl.TensorRead(), fkl.Saturate(-2.5, 2.5),
                      fkl.TensorWrite())(dev_f32(src, N))
    check("Saturate(-2.5,2.5)", unf32(out.copy_to_host(), N),
          [max(-2.5, min(2.5, v)) for v in src])


# ---- vector ops -----------------------------------------------------------

def t_vector_reduce_add():
    W, H = 4, 2
    n = W * H * 4
    src = [float(i % 13) for i in range(n)]
    out = fkl.compose(fkl.TensorRead(), fkl.VectorReduce("Add"),
                      fkl.TensorWrite())(dev_f32(src, W, H, ch=4))
    check("VectorReduce(Add) float4->float", unf32(out.copy_to_host(), W * H),
          [sum(src[p*4:(p+1)*4]) for p in range(W * H)])


def t_discard():
    W, H = 4, 2
    n4 = W * H * 4
    src = [float(i) for i in range(n4)]
    out = fkl.compose(fkl.TensorRead(), fkl.Discard(2),
                      fkl.TensorWrite())(dev_f32(src, W, H, ch=4))
    exp = []
    for p in range(W * H):
        exp.extend(src[p*4:p*4+2])
    check("Discard float4->float2", unf32(out.copy_to_host(), W * H * 2), exp)


def t_addlast():
    W, H = 4, 2
    n3 = W * H * 3
    src = [float(i) for i in range(n3)]
    out = fkl.compose(fkl.TensorRead(), fkl.AddLast(1.0),
                      fkl.TensorWrite())(dev_f32(src, W, H, ch=3))
    exp = []
    for p in range(W * H):
        exp.extend(src[p*3:(p+1)*3] + [1.0])
    check("AddLast float3->float4", unf32(out.copy_to_host(), W * H * 4), exp)


def t_vector_reorder_rgba():
    W, H = 2, 2
    n = W * H * 4
    src = [(i * 31) % 256 for i in range(n)]
    out = fkl.compose(fkl.TensorRead(), fkl.VectorReorder(3, 2, 1, 0),
                      fkl.TensorWrite())(dev_u8(src, W, H, ch=4))
    exp = []
    for p in range(W * H):
        q = src[p*4:(p+1)*4]
        exp.extend([float(q[3]), float(q[2]), float(q[1]), float(q[0])])
    check("VectorReorder(3,2,1,0) uchar4", 
          [float(v) for v in unu8(out.copy_to_host(), n)], exp)


# ---- algebraic -------------------------------------------------------------

def t_mxv_identity():
    W, H = 4, 2
    n = W * H * 3
    src = [float(i % 29) for i in range(n)]
    I3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    out = fkl.compose(fkl.TensorRead(), fkl.MxVFloat3(I3),
                      fkl.TensorWrite())(dev_f32(src, W, H, ch=3))
    check("MxVFloat3 identity matrix", unf32(out.copy_to_host(), n), src)


def t_mxv_bt601():
    """Luma via matrix (every output channel = BT.601 luma)."""
    W, H = 4, 2
    n = W * H * 3
    src = [float((i * 11) % 200) for i in range(n)]
    M = [(0.299, 0.587, 0.114)] * 3
    out = fkl.compose(fkl.TensorRead(), fkl.MxVFloat3(M),
                      fkl.TensorWrite())(dev_f32(src, W, H, ch=3))
    got = unf32(out.copy_to_host(), n)
    exp = []
    for p in range(W * H):
        r, g, b = src[p*3:(p+1)*3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        exp.extend([y, y, y])
    check("MxVFloat3 BT.601 luma", got, exp, tol=1e-2)


# ---- color -----------------------------------------------------------------

def t_color_rgb2bgr():
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 13) % 256 for i in range(n)]
    out = fkl.compose(fkl.TensorRead(), fkl.ColorConversion("RGB2BGR"),
                      fkl.TensorWrite())(dev_u8(src, W, H, ch=3))
    exp = []
    for p in range(W * H):
        r, g, b = src[p*3:(p+1)*3]
        exp.extend([float(b), float(g), float(r)])
    check("ColorConversion RGB2BGR", 
          [float(v) for v in unu8(out.copy_to_host(), n)], exp)


def t_color_rgb2rgba():
    W, H = 4, 2
    src = [(i * 9) % 256 for i in range(W * H * 3)]
    out = fkl.compose(fkl.TensorRead(), fkl.ColorConversion("RGB2RGBA"),
                      fkl.TensorWrite())(dev_u8(src, W, H, ch=3))
    got = [float(v) for v in unu8(out.copy_to_host(), W * H * 4)]
    exp = []
    for p in range(W * H):
        exp.extend([float(v) for v in src[p*3:(p+1)*3]] + [255.0])
    check("ColorConversion RGB2RGBA (alpha=255)", got, exp)


def t_color_bgr2gray():
    W, H = 8, 2
    n = W * H * 3
    src = [(i * 17) % 256 for i in range(n)]
    out = fkl.compose(fkl.TensorRead(), fkl.ColorConversion("BGR2GRAY"),
                      fkl.TensorWrite())(dev_u8(src, W, H, ch=3))
    got = [float(v) for v in unu8(out.copy_to_host(), W * H)]
    exp = []
    for p in range(W * H):
        b, g, r = src[p*3:(p+1)*3]
        exp.append(float(int(0.299*r + 0.587*g + 0.114*b + 0.5)))
    check("ColorConversion BGR2GRAY", got, exp, tol=1.0)


# ---- memory layout ---------------------------------------------------------

def t_tensor_split():
    """Packed uchar3 HxW -> planar: 3 planes of HxW (DNN layout)."""
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 7) % 256 for i in range(n)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.TensorSplit())
    out = pipe(dev_u8(src, W, H, ch=3))
    got = [float(v) for v in unu8(out.copy_to_host(), n)]
    exp = []
    for c in range(3):                       # plane-major
        for p in range(W * H):
            exp.append(float(src[p * 3 + c]))
    check("TensorSplit packed->planar", got, exp)


if __name__ == "__main__":
    run([t_add_sub_mul_div_scalar, t_vector_value_broadcast, t_max_min,
         t_cast_f2i, t_saturate_cast_negative, t_saturate_float,
         t_saturate_range, t_vector_reduce_add, t_discard, t_addlast,
         t_vector_reorder_rgba, t_mxv_identity, t_mxv_bt601,
         t_color_rgb2bgr, t_color_rgb2rgba, t_color_bgr2gray,
         t_tensor_split], "operations")
