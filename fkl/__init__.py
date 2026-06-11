"""FKL-Python: zero-overhead Python front-end for the Fused Kernel Library.

Lazy composition (the paper's IOp model) -> single-step host+device JIT
(.so with extern "C" fkl_entry; clang preferred, nvcc fallback) -> cached
forever per chain signature -> hot path is one ctypes call, zero copies.
"""
from .operations import (
    # memory
    TensorRead, TensorWrite, TensorSplit, SplitWrite,
    TensorPack, TensorTSplit, ReadSet,
    PerThreadRead, PerThreadWrite,
    # arithmetic / logical
    Mul, Add, Sub, Div, Max, Min, IsEven, VectorAnd,
    # casts / saturation
    Cast, SaturateCast, SaturateFloat, Saturate,
    # vector ops
    VectorReduce, Discard, AddLast, VectorReorder, VectorReorderRT,
    # algebraic / control flow
    MxVFloat3, StaticLoop,
    # color / image (BVF + batch HF)
    ColorConversion, Crop, Resize, Warping, BorderReader, Deinterlace,
)
from .jit import compose, compose_divergent, FusedKernel, DivergentKernel
from .circular import CircularTensor
from .attention import flash_attention, FlashAttention, compress_kv, CompressedKV
from .flex import ScoreMod, ALiBi, SoftCap, SlidingWindow, BlockMask
from .backend import CompilerBackend, set_backend, get_backend, clear_cache
from .tensor import DeviceBuffer
from .types import DType, dtype

__all__ = [
    "TensorRead", "TensorWrite", "TensorSplit", "SplitWrite",
    "TensorPack", "TensorTSplit", "ReadSet",
    "PerThreadRead", "PerThreadWrite",
    "Mul", "Add", "Sub", "Div", "Max", "Min", "IsEven", "VectorAnd",
    "Cast", "SaturateCast", "SaturateFloat", "Saturate",
    "VectorReduce", "Discard", "AddLast", "VectorReorder", "VectorReorderRT",
    "MxVFloat3", "StaticLoop",
    "ColorConversion", "Crop", "Resize", "Warping", "BorderReader", "Deinterlace",
    "compose", "compose_divergent", "FusedKernel", "DivergentKernel",
    "CircularTensor",
    "flash_attention", "FlashAttention", "compress_kv", "CompressedKV",
    "ScoreMod", "ALiBi", "SoftCap", "SlidingWindow", "BlockMask",
    "CompilerBackend", "set_backend", "get_backend", "clear_cache",
    "DeviceBuffer", "DType", "dtype",
]

__version__ = "0.6.0"
