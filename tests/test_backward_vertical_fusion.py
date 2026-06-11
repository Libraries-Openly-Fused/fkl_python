"""BACKWARDS VERTICAL FUSION tests: Crop/Resize are ReadBack ops fused
backwards onto the Read by FKL's BackFuser (paper's BVF; OpenCV-filter-like
but generic). Python emits them as plain IOps -- fusion happens in C++.
"""
from harness import PASS, FAIL, dev_f32, dev_u8, unf32, unu8, check, check_true, run
import fkl


def t_crop_exact():
    W, H = 24, 12
    CX, CY, CW, CH = 6, 3, 8, 5
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(), fkl.Crop(CX, CY, CW, CH),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    exp = [float((CY + y) * W + (CX + x)) for y in range(CH) for x in range(CW)]
    check("BVF: Crop exact pixels", unf32(out.copy_to_host(), CW * CH), exp)


def t_crop_plus_compute():
    """BVF + VF together: crop fused backwards, arithmetics fused forwards."""
    W, H = 16, 16
    src = [float((y * W + x) % 71) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(), fkl.Crop(2, 2, 8, 8),
                      fkl.Mul(2.0), fkl.Add(0.5),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    exp = [src[(2 + y) * W + (2 + x)] * 2 + 0.5 for y in range(8) for x in range(8)]
    check("BVF+VF: Crop->Mul->Add", unf32(out.copy_to_host(), 64), exp)


def t_resize_identity():
    """Resize to the same size with linear interp == identity."""
    W, H = 8, 8
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(), fkl.Resize(W, H),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    check("BVF: Resize identity (8x8->8x8)", unf32(out.copy_to_host(), W * H), src)


def t_resize_upscale():
    """2x upscale of a constant image stays constant everywhere."""
    W, H = 4, 4
    src = [42.0] * (W * H)
    out = fkl.compose(fkl.TensorRead(), fkl.Resize(8, 8),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    check("BVF: Resize 2x upscale of constant", unf32(out.copy_to_host(), 64),
          [42.0] * 64)


def t_resize_gradient():
    W, H, OW, OH = 16, 4, 8, 4
    src = [float(x) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(), fkl.Resize(OW, OH),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    exp = []
    for y in range(OH):
        for x in range(OW):
            exp.append(min(max((x + 0.5) * 2 - 0.5, 0.0), W - 1.0))
    check("BVF: Resize x-gradient halved", unf32(out.copy_to_host(), OW * OH),
          exp, tol=0.51)


def t_crop_resize_stack():
    """Two stacked ReadBacks: Read <- Crop <- Resize (paper's README example)."""
    W, H = 32, 32
    src = [float((x + y) % 50) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(),
                      fkl.Crop(8, 8, 16, 16),
                      fkl.Resize(16, 16),   # identity-size resize of the crop
                      fkl.TensorWrite())(dev_f32(src, W, H))
    exp = [src[(8 + y) * W + (8 + x)] for y in range(16) for x in range(16)]
    check("BVF: Crop->Resize identity stack", unf32(out.copy_to_host(), 256), exp)


def t_full_preproc_pipeline():
    """The cvGS-style DNN preprocessing pipeline from the paper, single image:
    Crop -> Resize -> Mul -> Sub -> Div -> SaturateCast."""
    W, H = 64, 48
    src = [(x * 3 + y * 5) % 256 for y in range(H) for x in range(W)]
    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Crop(10, 10, 32, 24),
        fkl.Resize(16, 12),
        fkl.Mul(0.5),
        fkl.Add(8.0),
        fkl.SaturateCast("uint8"),
        fkl.TensorWrite(),
    )
    out = pipe(dev_u8(src, W, H))
    got = unu8(out.copy_to_host(), 16 * 12)
    ok = len(got) == 192 and all(0 <= v <= 255 for v in got) and \
        sum(got) > 0  # non-degenerate
    check_true("BVF: full preproc Crop->Resize->arith->cast", ok,
               f"192 px, mean={sum(got)/len(got):.1f}")


def t_resize_preserve_ar():
    """PRESERVE_AR: 16x8 -> 8x8 letterboxed with background bands."""
    W, H = 16, 8
    src = [100.0] * (W * H)
    out = fkl.compose(fkl.TensorRead(),
                      fkl.Resize(8, 8, aspect_ratio="preserve", background=7.0),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    got = unf32(out.copy_to_host(), 64)
    # aspect 2:1 into square -> content rows centered, bands top+bottom = 7.0
    top_band = got[:8]
    middle = got[3 * 8: 4 * 8]
    check("BVF: Resize PRESERVE_AR bands", top_band, [7.0] * 8)
    check("BVF: Resize PRESERVE_AR content", middle, [100.0] * 8, tol=0.5)


if __name__ == "__main__":
    run([t_crop_exact, t_crop_plus_compute, t_resize_identity,
         t_resize_upscale, t_resize_gradient, t_crop_resize_stack,
         t_full_preproc_pipeline, t_resize_preserve_ar],
        "backwards-vertical-fusion")
