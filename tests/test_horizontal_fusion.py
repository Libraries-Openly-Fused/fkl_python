"""HORIZONTAL FUSION tests: batch Crop -> ONE kernel with B thread-planes
(blockIdx.z selects the crop). This is the paper's HF (figs. 4-5) driven from
Python. The output is a Tensor (contiguous planes).
"""
from harness import PASS, FAIL, dev_f32, dev_u8, unf32, unu8, check, check_true, run
import fkl


def t_batch_crop_3():
    W, H = 16, 8
    CW, CH = 4, 2
    rects = [(0, 0, CW, CH), (8, 4, CW, CH), (2, 1, CW, CH)]
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Crop(rects), fkl.TensorWrite())
    out = pipe(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), CW * CH * len(rects))
    exp = []
    for (cx, cy, cw, chh) in rects:
        for y in range(chh):
            for x in range(cw):
                exp.append(float((cy + y) * W + (cx + x)))
    check("HF: batch Crop x3 -> Tensor", got, exp)


def t_batch_crop_plus_vf():
    """HF + VF: 5 crops, then fused arithmetic on all planes in ONE kernel."""
    W, H = 32, 16
    CW, CH = 6, 4
    rects = [(0, 0, CW, CH), (10, 5, CW, CH), (20, 10, CW, CH),
             (5, 2, CW, CH), (26, 12, CW, CH)]
    src = [float((y * W + x) % 119) for y in range(H) for x in range(W)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Crop(rects),
                       fkl.Mul(2.0), fkl.Add(1.0), fkl.TensorWrite())
    out = pipe(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), CW * CH * 5)
    exp = []
    for (cx, cy, cw, chh) in rects:
        for y in range(chh):
            for x in range(cw):
                exp.append(src[(cy + y) * W + (cx + x)] * 2 + 1)
    check("HF+VF: 5 crops + Mul+Add fused", got, exp)


def t_batch_crop_resize():
    """The README pipeline: batch crops of DIFFERENT positions -> Resize ->
    arithmetic -> Tensor. HF + BVF + VF in one kernel."""
    W, H = 64, 32
    rects = [(0, 0, 16, 8), (32, 16, 16, 8), (8, 4, 16, 8), (40, 20, 16, 8)]
    OW, OH = 8, 4
    src = [float((x * 3 + y * 7) % 251) for y in range(H) for x in range(W)]
    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Crop(rects),
        fkl.Resize(OW, OH),
        fkl.Mul(0.5),
        fkl.TensorWrite(),
    )
    out = pipe(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), OW * OH * len(rects))
    check_true("HF+BVF+VF: batch crop->resize->mul (shape sane)",
               len(got) == OW * OH * 4 and all(v >= 0 for v in got),
               f"{len(got)} px")
    # plane 0 must equal single-crop reference computed independently
    ref_pipe = fkl.compose(fkl.TensorRead(), fkl.Crop(*rects[0]),
                           fkl.Resize(OW, OH), fkl.Mul(0.5), fkl.TensorWrite())
    ref = unf32(ref_pipe(dev_f32(src, W, H)).copy_to_host(), OW * OH)
    check("HF plane0 == single-crop reference", got[:OW * OH], ref)


def t_batch_identity_crops():
    """B crops of the SAME rect must produce B identical planes."""
    W, H = 12, 6
    CW, CH = 4, 3
    B = 4
    rects = [(3, 1, CW, CH)] * B
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(), fkl.Crop(rects),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), CW * CH * B)
    plane0 = got[:CW * CH]
    ok = all(got[p * CW * CH:(p + 1) * CW * CH] == plane0 for p in range(B))
    check_true("HF: identical rects -> identical planes", ok)
    exp0 = [float((1 + y) * W + (3 + x)) for y in range(CH) for x in range(CW)]
    check("HF: plane content correct", plane0, exp0)


def t_batch_uchar3_pipeline():
    """HF with vector pixels: uchar3 crops -> float3 -> normalize -> Tensor."""
    W, H = 24, 12
    CW, CH = 6, 3
    rects = [(0, 0, CW, CH), (12, 6, CW, CH)]
    n = W * H * 3
    src = [(i * 7) % 256 for i in range(n)]
    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Crop(rects),
        fkl.Cast("float32"),
        fkl.Div(255.0),
        fkl.TensorWrite(),
    )
    out = pipe(dev_u8(src, W, H, ch=3))
    got = unf32(out.copy_to_host(), CW * CH * 3 * 2)
    exp = []
    for (cx, cy, cw, chh) in rects:
        for y in range(chh):
            for x in range(cw):
                p = ((cy + y) * W + (cx + x)) * 3
                exp.extend([src[p] / 255.0, src[p+1] / 255.0, src[p+2] / 255.0])
    check("HF: uchar3 batch crops + normalize", got, exp, tol=1e-5)


if __name__ == "__main__":
    run([t_batch_crop_3, t_batch_crop_plus_vf, t_batch_crop_resize,
         t_batch_identity_crops, t_batch_uchar3_pipeline], "horizontal-fusion")
