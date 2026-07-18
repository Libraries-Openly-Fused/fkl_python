"""fkl.F — eager, torch-style convenience functions.

Each function is a ONE-op fused pipeline (fkl.pipe(x).<op>().run()): first
call per signature JIT-compiles (disk-cached), then it costs a single kernel
launch. For multi-step preprocessing prefer fkl.pipe(...) so the WHOLE chain
fuses into one kernel instead of one kernel per call.

    import fkl
    y = fkl.F.resize(img, (64, 64))
    g = fkl.F.cvt_color(img, "RGB2GRAY")
    n = fkl.F.normalize(img, MEAN, STD)
    p = fkl.F.to_planar(img)                 # HWC -> CHW planar
"""
from .highlevel import pipe


def resize(x, size, interp="linear", aspect_ratio="ignore", background=None,
           out=None, stream=None):
    """Bilinear/nearest resize to size=(w, h) (int = square)."""
    return pipe(x).resize(size, interp=interp, aspect_ratio=aspect_ratio,
                          background=background).run(out=out, stream=stream)


def crop(x, rect, out=None, stream=None):
    """rect=(x, y, w, h), or a list of rects for multi-ROI (batch planes)."""
    return pipe(x).crop(rect).run(out=out, stream=stream)


def cvt_color(x, code, out=None, stream=None):
    """OpenCV-style color conversion, e.g. 'RGB2BGR', 'BGR2GRAY'."""
    return pipe(x).cvt_color(code).run(out=out, stream=stream)


def normalize(x, mean, std, out=None, stream=None):
    """(x - mean) / std per channel (auto-casts non-float chains to float32)."""
    return pipe(x).normalize(mean, std).run(out=out, stream=stream)


def cast(x, dtype, out=None, stream=None):
    return pipe(x).cast(dtype).run(out=out, stream=stream)


def saturate_cast(x, dtype, out=None, stream=None):
    return pipe(x).saturate_cast(dtype).run(out=out, stream=stream)


def to_planar(x, layout="CHW", out=None, stream=None):
    """Packed HWC pixels -> planar 'CHW' ('NCHW' for batches)."""
    return pipe(x).to_planar(layout).run(out=out, stream=stream)


def mul(x, value, out=None, stream=None):
    return pipe(x).mul(value).run(out=out, stream=stream)


def add(x, value, out=None, stream=None):
    return pipe(x).add(value).run(out=out, stream=stream)


def sub(x, value, out=None, stream=None):
    return pipe(x).sub(value).run(out=out, stream=stream)


def div(x, value, out=None, stream=None):
    return pipe(x).div(value).run(out=out, stream=stream)


__all__ = ["resize", "crop", "cvt_color", "normalize", "cast",
           "saturate_cast", "to_planar", "mul", "add", "sub", "div"]
