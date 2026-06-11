"""VERTICAL FUSION tests: chains of compute ops -> ONE kernel, intermediates
in registers. Mirrors the paper's section VI-B experiments (Mul/Add chains).
"""
from harness import (PASS, FAIL, dev_f32, dev_u8, unf32, unu8, f32,
                     check, check_true, run)
import fkl


def t_two_ops():
    N = 256
    src = [float(i % 97) for i in range(N)]
    out = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0),
                      fkl.TensorWrite())(dev_f32(src, N))
    check("VF: Mul+Add", unf32(out.copy_to_host(), N),
          [v * 2 + 1 for v in src])


def t_six_op_chain():
    N = 128
    src = [float(i % 53) + 1.0 for i in range(N)]
    pipe = fkl.compose(fkl.TensorRead(),
                       fkl.Mul(3.0), fkl.Add(7.0), fkl.Sub(2.0),
                       fkl.Div(2.0), fkl.Max(1.0), fkl.Min(80.0),
                       fkl.TensorWrite())
    out = pipe(dev_f32(src, N))
    exp = [min(80.0, max(1.0, ((v * 3 + 7 - 2) / 2))) for v in src]
    check("VF: 6-op chain Mul+Add+Sub+Div+Max+Min", unf32(out.copy_to_host(), N), exp)


def t_static_loop():
    """StaticLoop: 64 fused Adds without 64 params -- paper's VF-scaling trick."""
    N = 64
    src = [float(i) for i in range(N)]
    pipe = fkl.compose(fkl.TensorRead(),
                       fkl.StaticLoop(fkl.Add(1.0), 64),
                       fkl.TensorWrite())
    out = pipe(dev_f32(src, N))
    check("VF: StaticLoop(Add(1) x64)", unf32(out.copy_to_host(), N),
          [v + 64.0 for v in src])


def t_2d_layout():
    W, H = 33, 17  # non-pow2 to catch pitch bugs
    n = W * H
    src = [float((x * 7 + y * 13) % 101) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite())(
        dev_f32(src, W, H))
    check("VF: 2D non-pow2 (33x17)", unf32(out.copy_to_host(), n),
          [v * 2 for v in src])


def t_dtype_int32():
    N = 64
    src = list(range(-32, 32))
    from harness import i32, uni32
    buf = fkl.DeviceBuffer(N, 1, "int32")
    buf.copy_from_host(i32(src))
    out = fkl.compose(fkl.TensorRead(), fkl.Mul(3), fkl.Add(2), fkl.TensorWrite())(buf)
    check("VF: int32 chain", [float(v) for v in uni32(out.copy_to_host(), N)],
          [float(v * 3 + 2) for v in src])


def t_dtype_double():
    N = 32
    import struct
    src = [float(i) * 1e-7 for i in range(N)]
    buf = fkl.DeviceBuffer(N, 1, "float64")
    buf.copy_from_host(struct.pack(f"{N}d", *src))
    out = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite())(buf)
    got = list(struct.unpack(f"{N}d", out.copy_to_host()))
    check("VF: float64 precision", got, [v * 2 for v in src], tol=1e-15)


def t_values_no_recompile():
    """Same chain TYPES with different VALUES must reuse the same .so."""
    N = 16
    src = [float(i) for i in range(N)]
    p1 = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite())
    p2 = fkl.compose(fkl.TensorRead(), fkl.Mul(5.0), fkl.TensorWrite())
    o1 = p1(dev_f32(src, N))
    o2 = p2(dev_f32(src, N))
    so1 = list(p1._variants.values())[0][3]
    so2 = list(p2._variants.values())[0][3]
    check_true("VF: value change reuses cached .so", so1 == so2, str(so1.name))
    check("VF: ...and computes correctly (x2)", unf32(o1.copy_to_host(), N),
          [v * 2 for v in src])
    check("VF: ...and computes correctly (x5)", unf32(o2.copy_to_host(), N),
          [v * 5 for v in src])


def t_vector_chain_float3():
    W, H = 8, 4
    n = W * H * 3
    src = [float((i * 3) % 64) for i in range(n)]
    pipe = fkl.compose(fkl.TensorRead(),
                       fkl.Mul((1.0, 2.0, 3.0)),
                       fkl.Add((10.0, 20.0, 30.0)),
                       fkl.TensorWrite())
    out = pipe(dev_f32(src, W, H, ch=3))
    fac, off = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    check("VF: float3 per-channel Mul+Add", unf32(out.copy_to_host(), n),
          [src[i] * fac[i % 3] + off[i % 3] for i in range(n)])


def t_uchar4_pipeline():
    W, H = 4, 4
    n = W * H * 4
    src = [(i * 17) % 256 for i in range(n)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                       fkl.Div(2.0), fkl.SaturateCast("uint8"),
                       fkl.TensorWrite())
    out = pipe(dev_u8(src, W, H, ch=4))
    check("VF: uchar4 normalize roundtrip", 
          [float(v) for v in unu8(out.copy_to_host(), n)],
          [float(int(v / 2.0 + 0.5)) for v in src], tol=1.0)


if __name__ == "__main__":
    run([t_two_ops, t_six_op_chain, t_static_loop, t_2d_layout,
         t_dtype_int32, t_dtype_double, t_values_no_recompile,
         t_vector_chain_float3, t_uchar4_pipeline], "vertical-fusion")
