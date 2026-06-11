"""Full FKL operation catalog for Python composition.

Each Op is a lazy descriptor (the Python mirror of an IOp). It declares:
  - cpp(state, pbase)     -> the C++ build() expression (or None if implicit)
  - out_dtype(in_dtype)   -> dtype transform through the chain
  - out_shape(in_shape)   -> (w, h, planes) transform through the chain
  - values                -> runtime scalars appended to params[] (chain order)
  - token(state)          -> contribution to the compile-cache signature

Covered library surface:
  arithmetic : Add, Sub, Mul, Div
  logical    : Max, Min, IsEven
  cast       : Cast, SaturateCast, SaturateFloat, Saturate
  vector_ops : VectorReduce(via Sum), Discard, AddLast, VectorAnd, VectorReorder
  color      : ColorConversion (all OpenCV-style codes FKL implements)
  image      : Crop, Resize (INTER_LINEAR; IGNORE_AR / PRESERVE_AR / *_LEFT / *_RN_EVEN)
  memory     : PerThreadRead/Write (2D), TensorRead/Write (3D), TensorSplit/Pack
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .types import DType, dtype as _dt, pack_value, make_expr

READ, COMPUTE, WRITE = "read", "compute", "write"


@dataclass
class ChainState:
    """Type/shape state threaded through codegen, like OutputType through
    FKL's template chain."""
    dtype: DType
    width: int = 0
    height: int = 0
    planes: int = 1


class Op:
    role = COMPUTE
    name = "Op"

    # runtime scalar values consumed from params[]
    @property
    def values(self) -> List[float]:
        return []

    def out_dtype(self, dt: DType) -> DType:
        return dt

    def out_shape(self, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return shape

    def cpp(self, state: ChainState, pbase: int) -> Optional[str]:
        raise NotImplementedError

    def token(self, state: ChainState) -> str:
        return f"{self.name}<{state.dtype}>"


# ===================== arithmetic / logical (Binary COps) =================

class _BinaryValueOp(Op):
    """Mul/Add/Sub/Div/Max/Min: Name<T>::build(value). Value lives in
    params[] so changing it does NOT recompile."""
    def __init__(self, value):
        self._raw = value

    @property
    def values(self):
        # resolved at codegen time against the chain dtype; stored after bind
        return self._vals

    def bind(self, dt: DType):
        self._vals = pack_value(dt, self._raw)

    def cpp(self, state, pbase):
        self.bind(state.dtype)
        return f"{self.name}<{state.dtype}>::build({make_expr(state.dtype, pbase)})"


class Mul(_BinaryValueOp):  name = "Mul"
class Add(_BinaryValueOp):  name = "Add"
class Sub(_BinaryValueOp):  name = "Sub"
class Div(_BinaryValueOp):  name = "Div"
class Max(_BinaryValueOp):  name = "Max"
class Min(_BinaryValueOp):  name = "Min"


class Saturate(Op):
    """Clamp to [lo, hi]. ParamsType is VectorType_t<VBase<T>,2>."""
    name = "Saturate"

    def __init__(self, lo, hi):
        self._lo, self._hi = float(lo), float(hi)

    @property
    def values(self):
        return [self._lo, self._hi]

    def cpp(self, state, pbase):
        b = state.dtype.base_ctype
        return (f"Saturate<{state.dtype}>::build("
                f"make_<VectorType_t<{b}, 2>>(({b})params[{pbase}], ({b})params[{pbase + 1}]))")


# ===================== casts (Unary COps) =================================

class Cast(Op):
    name = "Cast"

    def __init__(self, to):
        self._to = _dt(to)

    def out_dtype(self, dt):
        # channel count follows the chain; base changes
        return self._to if self._to.channels == dt.channels else \
            self._to.with_channels(dt.channels)

    def cpp(self, state, pbase):
        out = self.out_dtype(state.dtype)
        return f"Cast<{state.dtype}, {out}>::build()"

    def token(self, state):
        return f"Cast<{state.dtype},{self.out_dtype(state.dtype)}>"


class SaturateCast(Cast):
    name = "SaturateCast"

    def cpp(self, state, pbase):
        out = self.out_dtype(state.dtype)
        return f"SaturateCast<{state.dtype}, {out}>::build()"

    def token(self, state):
        return f"SaturateCast<{state.dtype},{self.out_dtype(state.dtype)}>"


class SaturateFloat(Op):
    """Clamp float data to [0,1]. Unary."""
    name = "SaturateFloat"

    def cpp(self, state, pbase):
        return f"SaturateFloat<{state.dtype}>::build()"


# ===================== vector ops (Unary COps) ============================

class VectorReduce(Op):
    """Reduce vector channels with Sum -> scalar. e.g. float3 -> float."""
    name = "VectorReduce"

    def __init__(self, op: str = "Add"):
        self._op = op

    def out_dtype(self, dt):
        return dt.with_channels(1)

    def cpp(self, state, pbase):
        t = state.dtype
        s = t.with_channels(1)
        return (f"VectorReduce<{t}, {self._op}<{s}, {s}, {s}, UnaryType>>::build()")

    def token(self, state):
        return f"VectorReduce<{state.dtype},{self._op}>"


class Discard(Op):
    """Drop trailing channels: float4 -> float2, etc."""
    name = "Discard"

    def __init__(self, keep: int):
        self._keep = keep

    def out_dtype(self, dt):
        return dt.with_channels(self._keep)

    def cpp(self, state, pbase):
        return f"Discard<{state.dtype}, {self.out_dtype(state.dtype)}>::build()"

    def token(self, state):
        return f"Discard<{state.dtype},{self._keep}>"


class AddLast(Op):
    """Append one channel with a constant value: float3 -> float4."""
    name = "AddLast"

    def __init__(self, value: float):
        self._v = float(value)

    @property
    def values(self):
        return [self._v]

    def out_dtype(self, dt):
        return dt.with_channels(dt.channels + 1)

    def cpp(self, state, pbase):
        out = self.out_dtype(state.dtype)
        b = state.dtype.base_ctype
        return f"AddLast<{state.dtype}, {out}>::build(({b})params[{pbase}])"


class VectorReorder(Op):
    """Compile-time channel shuffle, e.g. VectorReorder(2,1,0) = RGB<->BGR."""
    name = "VectorReorder"

    def __init__(self, *idx: int):
        self._idx = idx

    def cpp(self, state, pbase):
        seq = ", ".join(str(i) for i in self._idx)
        return f"VectorReorder<{state.dtype}, {seq}>::build()"

    def token(self, state):
        return f"VectorReorder<{state.dtype},{self._idx}>"


# ===================== color conversion ===================================

_CC_OUT_CHANNELS = {  # code suffix -> output channel count
    "GRAY": 1, "BGRA": 4, "RGBA": 4, "BGR": 3, "RGB": 3,
}


class ColorConversion(Op):
    """ColorConversion<COLOR_x2y, I, O>::build(). code e.g. 'RGB2BGR',
    'BGR2GRAY', 'RGB2RGBA'..."""
    name = "ColorConversion"

    def __init__(self, code: str):
        self._code = code.upper().replace("COLOR_", "")

    def _out_ch(self):
        dst = self._code.split("2")[-1]
        for suffix, n in _CC_OUT_CHANNELS.items():
            if dst.startswith(suffix):
                return n
        raise ValueError(f"cannot infer channels for color code {self._code}")

    def out_dtype(self, dt):
        return dt.with_channels(self._out_ch())

    def cpp(self, state, pbase):
        out = self.out_dtype(state.dtype)
        code = self._code
        # FKL main-branch bug: BGR*2GRAY aliases expand to
        # FusedOperation<raw Operations...> which lacks ::build. Decompose to
        # the equivalent reorder + RGB2GRAY (identical semantics).
        if code == "BGR2GRAY":
            return (f"VectorReorder<{state.dtype}, 2, 1, 0>::build(), "
                    f"ColorConversion<ColorConversionCodes::COLOR_RGB2GRAY, "
                    f"{state.dtype}, {out}>::build()")
        if code == "BGRA2GRAY":
            return (f"VectorReorder<{state.dtype}, 2, 1, 0, 3>::build(), "
                    f"ColorConversion<ColorConversionCodes::COLOR_RGBA2GRAY, "
                    f"{state.dtype}, {out}>::build()")
        return (f"ColorConversion<ColorConversionCodes::COLOR_{code}, "
                f"{state.dtype}, {out}>::build()")

    def token(self, state):
        return f"CC<{self._code},{state.dtype}>"


# ===================== image ops (ReadBack / BVF) =========================

class Crop(Op):
    """Backwards-Vertical-Fused crop.

    Single : Crop(x, y, w, h)            -> Crop<>::build(Rect)
    Batch  : Crop([(x,y,w,h), ...])      -> Crop<>::build(std::array<Rect,B>)
             This is HORIZONTAL FUSION: one kernel, B thread-planes, each
             reading a different crop (BatchRead under the hood). All rects
             must share w,h unless a Resize follows in the chain.
    Fused backwards onto the Read by BackFuser inside executeOperations."""
    role = COMPUTE  # emitted in the IOp list; BackFuser handles the rest
    name = "Crop"

    def __init__(self, *args):
        if len(args) == 4 and all(isinstance(a, (int, float)) for a in args):
            self._rects = [tuple(int(v) for v in args)]
            self._batch = False
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            self._rects = [tuple(int(v) for v in r) for r in args[0]]
            self._batch = True
            if not self._rects:
                raise ValueError("batch Crop needs at least one rect")
        else:
            raise ValueError("Crop(x,y,w,h) or Crop([(x,y,w,h), ...])")

    @property
    def values(self):
        return [float(v) for r in self._rects for v in r]

    def out_shape(self, shape):
        w, h = self._rects[0][2], self._rects[0][3]
        planes = len(self._rects) if self._batch else shape[2]
        return (w, h, planes)

    def cpp(self, state, pbase):
        if not self._batch:
            return (f"Crop<>::build(Rect((int)params[{pbase}], (int)params[{pbase+1}], "
                    f"(int)params[{pbase+2}], (int)params[{pbase+3}]))")
        B = len(self._rects)
        rects = ", ".join(
            f"Rect((int)params[{pbase + i*4}], (int)params[{pbase + i*4 + 1}], "
            f"(int)params[{pbase + i*4 + 2}], (int)params[{pbase + i*4 + 3}])"
            for i in range(B))
        return f"Crop<>::build(std::array<Rect, {B}>{{ {rects} }})"

    def token(self, state):
        b = f",B{len(self._rects)}" if self._batch else ""
        return f"Crop<{state.dtype}{b}>"


_AR = {
    "ignore": "AspectRatio::IGNORE_AR",
    "preserve": "AspectRatio::PRESERVE_AR",
    "preserve_left": "AspectRatio::PRESERVE_AR_LEFT",
    "preserve_even": "AspectRatio::PRESERVE_AR_RN_EVEN",
}
_INTERP = {
    "linear": "InterpolationType::INTER_LINEAR",
    "nearest": "InterpolationType::INTER_NEAREST",
}


class Resize(Op):
    """Resize<INTER, AR>::build(Size(w,h)[, background]). Output dtype becomes
    float-based (interpolation result), channels preserved."""
    name = "Resize"

    def __init__(self, w: int, h: int, interp: str = "linear",
                 aspect_ratio: str = "ignore", background=None):
        self._w, self._h = int(w), int(h)
        self._interp = interp
        self._ar = aspect_ratio
        self._bg = background
        if aspect_ratio != "ignore" and background is None:
            raise ValueError("preserve aspect-ratio modes need a background value")

    @property
    def values(self):
        out = [float(self._w), float(self._h)]
        if self._bg is not None:
            out.extend(float(v) for v in (
                self._bg if isinstance(self._bg, (tuple, list))
                else [self._bg]))
        return out

    def out_dtype(self, dt):
        return dt.with_base("float32")  # interpolation yields float

    def out_shape(self, shape):
        return (self._w, self._h, shape[2])

    def cpp(self, state, pbase):
        it, ar = _INTERP[self._interp], _AR[self._ar]
        size = f"Size((int)params[{pbase}], (int)params[{pbase+1}])"
        if self._bg is None:
            return f"Resize<{it}, {ar}>::build({size})"
        ch = state.dtype.channels
        bg_dt = state.dtype.with_base("float32")
        nbg = ch if isinstance(self._bg, (tuple, list)) else 1
        if nbg == 1 and ch > 1:
            comps = ", ".join(f"(float)params[{pbase+2}]" for _ in range(ch))
        else:
            comps = ", ".join(f"(float)params[{pbase+2+i}]" for i in range(nbg))
        bg = f"make_<{bg_dt.ctype}>({comps})" if ch > 1 else f"(float)params[{pbase+2}]"
        return f"Resize<{it}, {ar}>::build({size}, {bg})"

    def token(self, state):
        return f"Resize<{self._interp},{self._ar},{state.dtype}>"


# ===================== memory ops =========================================

class _ReadOp(Op):
    role = READ


class _WriteOp(Op):
    role = WRITE


class TensorRead(_ReadOp):
    """Read input. 2D Ptr2D -> PerThreadRead<_2D>; 3D Tensor -> TensorRead."""
    name = "TensorRead"

    def cpp(self, state, pbase):
        if state.planes > 1:
            return f"TensorRead<{state.dtype}>::build(in_tensor)"
        return f"PerThreadRead<ND::_2D, {state.dtype}>::build(input.ptr())"

    def token(self, state):
        return f"Read<{state.dtype},p{state.planes}>"


class TensorWrite(_WriteOp):
    name = "TensorWrite"

    def cpp(self, state, pbase):
        if state.planes > 1:
            return f"TensorWrite<{state.dtype}>::build(out_tensor)"
        return f"PerThreadWrite<ND::_2D, {state.dtype}>::build(output.ptr())"

    def token(self, state):
        return f"Write<{state.dtype},p{state.planes}>"


class TensorSplit(_WriteOp):
    """Write vector pixels as separate planes (planar layout for DNNs).
    uchar3 HxW -> 3 planes of HxW uchar."""
    name = "TensorSplit"

    def out_shape(self, shape):
        return shape  # planes encoded by channels at alloc time

    def cpp(self, state, pbase):
        return f"TensorSplit<{state.dtype}>::build(out_tensor)"

    def token(self, state):
        return f"TensorSplit<{state.dtype}>"


# aliases mirroring the C++ spelling
PerThreadRead = TensorRead
PerThreadWrite = TensorWrite


# ===================== algebraic ==========================================

class MxVFloat3(Op):
    """3x3 matrix x float3 vector: color-space transforms, etc.
    MxVFloat3<>::build(M3x3Float{...}). Matrix rows in params[] (9 floats)."""
    name = "MxVFloat3"

    def __init__(self, matrix):
        rows = [list(r) for r in matrix]
        if len(rows) != 3 or any(len(r) != 3 for r in rows):
            raise ValueError("MxVFloat3 needs a 3x3 matrix")
        self._m = [float(v) for r in rows for v in r]

    @property
    def values(self):
        return self._m

    def cpp(self, state, pbase):
        if state.dtype.ctype != "float3":
            raise TypeError(f"MxVFloat3 needs float3 input, chain has {state.dtype}")
        mk = lambda i: (f"make_<float3>((float)params[{pbase+i}], "
                        f"(float)params[{pbase+i+1}], (float)params[{pbase+i+2}])")
        return (f"MxVFloat3<>::build(M3x3Float{{ {mk(0)}, {mk(3)}, {mk(6)} }})")

    def token(self, state):
        return f"MxVFloat3<{state.dtype}>"


# ===================== logical (unary predicates) =========================

class IsEven(Op):
    """Integral -> bool."""
    name = "IsEven"

    def out_dtype(self, dt):
        return dt.with_base("bool")

    def cpp(self, state, pbase):
        return f"IsEven<{state.dtype}>::build()"


class VectorAnd(Op):
    """boolN -> bool (all channels true)."""
    name = "VectorAnd"

    def out_dtype(self, dt):
        return dt.with_channels(1).with_base("bool")

    def cpp(self, state, pbase):
        return f"VectorAnd<{state.dtype}>::build()"


# ===================== control flow =======================================

class StaticLoop(Op):
    """Apply a Binary op N times in registers: StaticLoop<Op<T>, N>::build(v).
    The paper's trick for huge VF chains without param-space blowup."""
    name = "StaticLoop"

    def __init__(self, op, iterations: int):
        if not isinstance(op, _BinaryValueOp):
            raise TypeError("StaticLoop wraps a binary value op (Mul/Add/Sub/Div/Max/Min)")
        self._inner = op
        self._n = int(iterations)

    @property
    def values(self):
        return self._inner.values

    def bind(self, dt):
        self._inner.bind(dt)

    def cpp(self, state, pbase):
        self._inner.bind(state.dtype)
        t = state.dtype
        return (f"StaticLoop<{self._inner.name}<{t}>, {self._n}>::build("
                f"{make_expr(t, pbase)})")

    def token(self, state):
        return f"StaticLoop<{self._inner.name},{self._n},{state.dtype}>"


# ===================== warping (ReadBack / BVF) ===========================

class Warping(Op):
    """Affine (2x3) or Perspective (3x3) warp, fused backwards onto the read.
    Warping<WarpType::X>::build(WarpingParameters{matrix, dstSize}).
    Matrix + dst size travel in params[] (values-only changes don't recompile).
    """
    name = "Warping"

    def __init__(self, matrix, dst_w: int, dst_h: int):
        rows = [list(r) for r in matrix]
        if len(rows) == 2 and all(len(r) == 3 for r in rows):
            self._kind = "Affine"
            rows = rows + [[0.0, 0.0, 1.0]]   # stored 3x3; affine emits 3x2 raw
        elif len(rows) == 3 and all(len(r) == 3 for r in rows):
            self._kind = "Perspective"
        else:
            raise ValueError("Warping needs a 2x3 (affine) or 3x3 (perspective) matrix")
        self._m = [float(v) for r in rows for v in r]
        self._w, self._h = int(dst_w), int(dst_h)

    @property
    def values(self):
        return self._m + [float(self._w), float(self._h)]

    def out_dtype(self, dt):
        return dt.with_base("float32")   # interpolated output

    def out_shape(self, shape):
        return (self._w, self._h, shape[2])

    def cpp(self, state, pbase):
        wt = f"WarpType::{self._kind}"
        nrows = 2 if self._kind == "Affine" else 3
        rows = ", ".join(
            "{ " + ", ".join(f"(float)params[{pbase + r*3 + c}]" for c in range(3)) + " }"
            for r in range(nrows))
        sz = f"Size((int)params[{pbase + 9}], (int)params[{pbase + 10}])"
        return (f"Warping<{wt}>::build(WarpingParameters<{wt}>{{ "
                f"{{{{ {rows} }}}}, {sz} }})")

    def token(self, state):
        return f"Warping<{self._kind},{state.dtype}>"


# ===================== split write (planar outputs) =======================

class SplitWrite(_WriteOp):
    """Write a vector-pixel stream into per-channel 2D planes laid out
    contiguously (like TensorSplit but via SplitWrite params)."""
    name = "SplitWrite"

    def cpp(self, state, pbase):
        return None  # constructed by codegen from output pointer

    def token(self, state):
        return f"SplitWrite<{state.dtype}>"


# ===================== niche ops (catalog completion) =====================
# NOTE on fk::Equal: its InputType is Tuple<I1, I2> (two data streams).
# Linear chains carry ONE value between ops, so Equal is not representable
# in compose() until FKL exposes tuple-building ops in chains. Excluded.


class VectorReorderRT(Op):
    """RUNTIME channel shuffle: indices live in params[] (no recompile when
    the permutation changes -- unlike VectorReorder which is compile-time).
    VectorReorderRT<T>::build(intN{idx...})."""
    name = "VectorReorderRT"

    def __init__(self, *idx: int):
        if len(idx) not in (2, 3, 4):
            raise ValueError("VectorReorderRT needs 2..4 indices")
        self._idx = [int(i) for i in idx]

    @property
    def values(self):
        return [float(i) for i in self._idx]

    def cpp(self, state, pbase):
        n = state.dtype.channels
        if n != len(self._idx):
            raise TypeError(f"VectorReorderRT({len(self._idx)} idx) on {n}-channel chain")
        comps = ", ".join(f"(int)params[{pbase + i}]" for i in range(n))
        return (f"VectorReorderRT<{state.dtype}>::build(make_<int{n}>({comps}))")

    def token(self, state):
        return f"VectorReorderRT<{state.dtype}>"


class ReadSet(_ReadOp):
    """Generate a constant value for every thread (no DRAM read at all).
    ReadSet<T>::build(value, ActiveThreads(w, h, 1)).
    Use as the chain's READ op: fkl.compose(fkl.ReadSet(0.5, w, h), ...)"""
    name = "ReadSet"

    def __init__(self, value, width: int, height: int = 1, dtype="float32"):
        from .types import dtype as _d
        self._value = value
        self._w, self._h = int(width), int(height)
        self._dt = _d(dtype) if isinstance(dtype, str) else dtype

    @property
    def values(self):
        return pack_value(self._dt, self._value) + [float(self._w), float(self._h)]

    def out_shape(self, shape):
        return (self._w, self._h, 1)

    def out_dtype(self, dt):
        return self._dt

    def cpp(self, state, pbase):
        n = self._dt.channels
        val = make_expr(self._dt, pbase)
        at = (f"ActiveThreads((uint)params[{pbase + n}], "
              f"(uint)params[{pbase + n + 1}], 1u)")
        return f"ReadSet<{self._dt}>::build({val}, {at})"

    def token(self, state):
        return f"ReadSet<{self._dt}>"


class TensorPack(_ReadOp):
    """Read PLANAR (CHW) data as packed vector pixels: the inverse of
    TensorSplit. Input buffer must be planar with `channels` planes."""
    name = "TensorPack"

    def __init__(self, channels: int):
        if channels not in (2, 3, 4):
            raise ValueError("TensorPack needs 2..4 channels")
        self._ch = channels

    def out_dtype(self, dt):
        return dt.with_channels(self._ch)

    def cpp(self, state, pbase):
        return None  # built by codegen from the input pointer

    def token(self, state):
        return f"TensorPack<{state.dtype}c{self._ch}>"


class TensorTSplit(_WriteOp):
    """Write to T3D transposed-planes layout (color_planes outermost):
    layout [C][planes][H][W] instead of TensorSplit's [planes][C][H][W]."""
    name = "TensorTSplit"

    def cpp(self, state, pbase):
        return None  # built by codegen from the output pointer

    def token(self, state):
        return f"TensorTSplit<{state.dtype}>"


_BORDER_TYPES = ("constant", "replicate", "reflect", "wrap", "reflect101")


class BorderReader(Op):
    """Out-of-bounds policy fused onto the read (BVF). Per Oscar: the input
    is an Image/Ptr2D read, and BorderReader takes that complete Read IOp as
    its backIOp directly: BorderReader<BT>::build(readIOp[, defaultValue]).
    Codegen consumes the read expression (marker: _fuse_with_read)."""
    name = "BorderReader"
    _fuse_with_read = True

    def __init__(self, border: str = "replicate", value=0.0):
        b = border.lower()
        if b not in _BORDER_TYPES:
            raise ValueError(f"border must be one of {_BORDER_TYPES}")
        self._b = b
        self._value = value

    @property
    def values(self):
        if self._b == "constant":
            return self._vals
        return []

    def bind(self, dt):
        if self._b == "constant":
            self._vals = pack_value(dt, self._value)

    def cpp_with_read(self, state, pbase, read_expr: str) -> str:
        bt = {"constant": "CONSTANT", "replicate": "REPLICATE",
              "reflect": "REFLECT", "wrap": "WRAP",
              "reflect101": "REFLECT_101"}[self._b]
        if self._b == "constant":
            self.bind(state.dtype)
            return (f"BorderReader<BorderType::{bt}>::build("
                    f"{read_expr}, {make_expr(state.dtype, pbase)})")
        return f"BorderReader<BorderType::{bt}>::build({read_expr})"

    def cpp(self, state, pbase):
        raise RuntimeError("BorderReader must be emitted via cpp_with_read")

    def token(self, state):
        return f"BorderReader<{self._b},{state.dtype}>"


_DEINTERLACE = {"blend": "BLEND", "linear": "INTER_LINEAR"}


class Deinterlace(Op):
    """Deinterlace a video field (BVF ReadBack). modes: 'blend', 'linear'."""
    name = "Deinterlace"

    def __init__(self, mode: str = "blend"):
        m = mode.lower()
        if m not in _DEINTERLACE:
            raise ValueError(f"mode must be one of {tuple(_DEINTERLACE)}")
        self._m = m

    def out_dtype(self, dt):
        return dt.with_base("float32") if self._m == "linear" else dt

    def cpp(self, state, pbase):
        return f"Deinterlace<DeinterlaceType::{_DEINTERLACE[self._m]}>::build()"

    def token(self, state):
        return f"Deinterlace<{self._m},{state.dtype}>"
