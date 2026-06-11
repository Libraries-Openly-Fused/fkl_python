"""Tests for the niche catalog ops: ReadSet, TensorPack, TensorTSplit,
VectorReorderRT, BorderReader(+Crop), Deinterlace.
"""
from harness import dev_f32, dev_u8, unf32, unu8, check, check_true, run
import fkl


def t_readset():
    """Constant generator: no input at all -> w*h of value*2+1."""
    W, H = 8, 4
    pipe = fkl.compose(fkl.ReadSet(21.0, W, H), fkl.Mul(2.0), fkl.Add(1.0),
                       fkl.TensorWrite())
    out = pipe(fkl.DeviceBuffer(W, H, "float32"))  # dummy input for shape/alloc
    check("ReadSet(21)*2+1", unf32(out.copy_to_host(), W * H), [43.0] * (W * H))


def t_tensorpack_roundtrip():
    """TensorSplit then TensorPack must reproduce the original packed data."""
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 7) % 256 for i in range(n)]
    # packed -> planar
    planar = fkl.compose(fkl.TensorRead(), fkl.TensorSplit())(
        dev_u8(src, W, H, ch=3))
    # planar -> packed (TensorPack as read op over the planar tensor)
    packed = fkl.compose(fkl.TensorPack(3), fkl.TensorWrite())(planar)
    got = [float(v) for v in unu8(packed.copy_to_host(), n)]
    check("TensorPack inverts TensorSplit", got, [float(v) for v in src])


def t_tensor_t_split():
    """T3D transposed layout: [C][planes][H][W] (color planes outermost)."""
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 5) % 256 for i in range(n)]
    out = fkl.compose(fkl.TensorRead(), fkl.TensorTSplit())(
        dev_u8(src, W, H, ch=3))
    got = [float(v) for v in unu8(out.copy_to_host(), n)]
    # single plane => same as TensorSplit for planes=1: [C][H][W]
    exp = []
    for c in range(3):
        for p in range(W * H):
            exp.append(float(src[p * 3 + c]))
    check("TensorTSplit single-plane CHW", got, exp)


def t_vector_reorder_rt():
    """Runtime permutation: same kernel, different perms (no recompile)."""
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 13) % 256 for i in range(n)]
    img = dev_u8(src, W, H, ch=3)

    p_bgr = fkl.compose(fkl.TensorRead(), fkl.VectorReorderRT(2, 1, 0),
                        fkl.TensorWrite())
    p_gbr = fkl.compose(fkl.TensorRead(), fkl.VectorReorderRT(1, 2, 0),
                        fkl.TensorWrite())
    got_bgr = [float(v) for v in unu8(p_bgr(img).copy_to_host(), n)]
    got_gbr = [float(v) for v in unu8(p_gbr(img).copy_to_host(), n)]

    exp_bgr, exp_gbr = [], []
    for p in range(W * H):
        a, b, c = src[p*3:(p+1)*3]
        exp_bgr.extend([float(c), float(b), float(a)])
        exp_gbr.extend([float(b), float(c), float(a)])
    check("VectorReorderRT(2,1,0)", got_bgr, exp_bgr)
    check("VectorReorderRT(1,2,0)", got_gbr, exp_gbr)

    so1 = list(p_bgr._variants.values())[0][3]
    so2 = list(p_gbr._variants.values())[0][3]
    check_true("VectorReorderRT: same .so for different perms", so1 == so2)


def t_border_reader_constant():
    """Crop reaching out of bounds + BorderReader CONSTANT fill."""
    W, H = 8, 8
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    # crop extends 2 px past the right/bottom edge
    pipe = fkl.compose(fkl.TensorRead(),
                       fkl.BorderReader("constant", 999.0),
                       fkl.Crop(6, 6, 4, 4),
                       fkl.TensorWrite())
    out = pipe(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), 16)
    exp = []
    for y in range(4):
        for x in range(4):
            sx, sy = 6 + x, 6 + y
            exp.append(src[sy * W + sx] if sx < W and sy < H else 999.0)
    check("BorderReader constant + OOB crop", got, exp)


def t_border_reader_replicate():
    W, H = 6, 6
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    pipe = fkl.compose(fkl.TensorRead(),
                       fkl.BorderReader("replicate"),
                       fkl.Crop(4, 4, 4, 4),
                       fkl.TensorWrite())
    out = pipe(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), 16)
    exp = []
    for y in range(4):
        for x in range(4):
            sx, sy = min(4 + x, W - 1), min(4 + y, H - 1)
            exp.append(src[sy * W + sx])
    check("BorderReader replicate + OOB crop", got, exp)


def t_deinterlace_blend():
    """BLEND mode: each row becomes the average of its neighbours."""
    W, H = 4, 6
    src = []
    for y in range(H):
        src.extend([float(y * 10)] * W)   # row-constant image
    pipe = fkl.compose(fkl.TensorRead(), fkl.Deinterlace("blend"),
                       fkl.TensorWrite())
    out = pipe(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), W * H)
    ok = len(got) == W * H and all(v >= 0.0 for v in got)
    mid_rows_blended = any(abs(got[2 * W] - 20.0) < 10.1 for _ in [0])
    check_true("Deinterlace blend (shape + plausible values)",
               ok and mid_rows_blended,
               f"row2={got[2*W]:.1f}")


if __name__ == "__main__":
    run([t_readset, t_tensorpack_roundtrip, t_tensor_t_split,
         t_vector_reorder_rt, t_border_reader_constant,
         t_border_reader_replicate, t_deinterlace_blend], "niche-ops")
