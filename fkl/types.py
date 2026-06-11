"""dtype registry: scalar AND vector (CUDA pixel) types.

FKL works with CUDA vector types (uchar3, float4...). Frameworks expose an
HxWxC array instead. We map (base dtype, channels) <-> FKL C++ type, and
parse __cuda_array_interface__ shapes accordingly:

    shape (H, W)    typestr <f4  -> float,  width=W, height=H
    shape (H, W, 3) typestr |u1  -> uchar3, width=W, height=H
    shape (P, H, W, C)           -> 3D tensor of vector type (P planes)
"""
from __future__ import annotations
from dataclasses import dataclass

# base scalar name -> (C++ base, CAI typestr, itemsize)
_BASES = {
    "uint8":   ("uchar",  "|u1", 1),
    "int8":    ("char",   "|i1", 1),
    "uint16":  ("ushort", "<u2", 2),
    "int16":   ("short",  "<i2", 2),
    "int32":   ("int",    "<i4", 4),
    "uint32":  ("uint",   "<u4", 4),
    "float32": ("float",  "<f4", 4),
    "float64": ("double", "<f8", 8),
    "bool":    ("bool",   "|b1", 1),
}
_TYPESTR_TO_BASE = {v[1]: k for k, v in _BASES.items()}
_TYPESTR_TO_BASE["<u1"] = "uint8"  # alt encoding


@dataclass(frozen=True)
class DType:
    base: str       # 'float32'
    channels: int   # 1..4

    @property
    def ctype(self) -> str:
        """The FKL/C++ element type, e.g. 'float', 'uchar3'."""
        cbase = _BASES[self.base][0]
        return cbase if self.channels == 1 else f"{cbase}{self.channels}"

    @property
    def base_ctype(self) -> str:
        return _BASES[self.base][0]

    @property
    def itemsize(self) -> int:
        return _BASES[self.base][2] * self.channels

    @property
    def typestr(self) -> str:
        return _BASES[self.base][1]

    def with_channels(self, n: int) -> "DType":
        return DType(self.base, n)

    def with_base(self, base: str) -> "DType":
        return DType(base, self.channels)

    def __str__(self):
        return self.ctype


def dtype(spec) -> DType:
    """Parse 'float32', 'uint8x3', ('float32', 3), or DType."""
    if isinstance(spec, DType):
        return spec
    if isinstance(spec, tuple):
        return DType(spec[0], spec[1])
    if "x" in spec and spec.rsplit("x", 1)[-1].isdigit():
        b, c = spec.rsplit("x", 1)
        return DType(b, int(c))
    if spec not in _BASES:
        raise TypeError(f"unknown dtype {spec!r}")
    return DType(spec, 1)


def from_cai(shape, typestr) -> tuple["DType", int, int, int]:
    """(shape, typestr) -> (DType, width, height, planes).

    Trailing dim of size 2..4 is folded into a vector pixel type.
    """
    base = _TYPESTR_TO_BASE.get(typestr)
    if base is None:
        raise TypeError(f"unsupported typestr {typestr!r}")
    ch = 1
    dims = list(shape)
    if len(dims) >= 2 and 2 <= dims[-1] <= 4:
        ch = dims[-1]
        dims = dims[:-1]
    if len(dims) == 1:
        w, h, p = dims[0], 1, 1
    elif len(dims) == 2:
        h, w = dims
        p = 1
    elif len(dims) == 3:
        p, h, w = dims
    else:
        raise TypeError(f"unsupported shape {shape}")
    return DType(base, ch), int(w), int(h), int(p)


# constants for value packing in codegen
def pack_value(dt: DType, value) -> list[float]:
    """Normalize a python scalar/tuple to dt.channels floats for params[]."""
    if isinstance(value, (int, float)):
        return [float(value)] * dt.channels
    vals = [float(v) for v in value]
    if len(vals) != dt.channels:
        raise ValueError(f"value {value} has {len(vals)} channels, dtype {dt} needs {dt.channels}")
    return vals


def make_expr(dt: DType, param_base: int) -> str:
    """C++ expression reconstructing a (possibly vector) value of type dt
    from the runtime params[] array starting at param_base."""
    b = dt.base_ctype
    if dt.channels == 1:
        return f"({b})params[{param_base}]"
    comps = ", ".join(f"({b})params[{param_base + i}]" for i in range(dt.channels))
    return f"make_<{dt.ctype}>({comps})"
