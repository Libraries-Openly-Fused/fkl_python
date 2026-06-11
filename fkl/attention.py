"""fkl.attention — FlashAttention-2 forward on FKL's cooperative DPPs.

What makes this different from calling a handmade FA kernel:

  1. EPILOGUE FUSION: pass ops=[...] (any fkl compute ops) and they run
     IN-REGISTER on the attention output inside the same kernel — no
     second kernel, no DRAM round-trip:
         fkl.flash_attention(q, k, v, epilogue=[fkl.Mul(2.0), fkl.Add(1.0)])
  2. COMPRESSED KV CACHE: kv_layout="int8" stores K/V as int8 with one
     fp32 scale per token (4x smaller than fp32, 2x than fp16) and
     dequantizes in-register inside the fused kernel:
         kc = fkl.compress_kv(k); vc = fkl.compress_kv(v)
         out = fkl.flash_attention(q, kc, vc, causal=True)

Layout: (batch*heads, seq, head_dim) C-contiguous; head_dim in {32,64,128,...}.
fp32 accumulation always. Targets SM 12x (SIMT mapping per fa-5090; no
TMEM). Compiled once per (dtype, head_dim, layout, epilogue-shape, arch);
runtime values (scales, sizes, causal flag, epilogue params) never
recompile.
"""
from __future__ import annotations
import ctypes
import math

from .backend import get_backend
from .codegen import CODEGEN_VERSION
from .operations import READ, WRITE, ChainState
from .tensor import DeviceBuffer, as_device_view, stream_handle
from .types import dtype as _dtype


class CompressedKV:
    """int8-per-token compressed KV tensor + per-token scales (on GPU)."""

    def __init__(self, data: DeviceBuffer, scales: DeviceBuffer,
                 batch_heads: int, seq: int, head_dim: int):
        self.data, self.scales = data, scales
        self.batch_heads, self.seq, self.head_dim = batch_heads, seq, head_dim

    @property
    def nbytes(self) -> int:
        return self.data._nbytes + self.scales._nbytes


_quant_lib = None


def _quant_so():
    """Tiny standalone kernel for GPU-side per-token int8 quantization."""
    global _quant_lib
    if _quant_lib is not None:
        return _quant_lib
    src = r"""
#include <cuda_runtime.h>
#include <math.h>
extern "C" {
__global__ void quant_kernel(const float* dense, signed char* q8, float* scales,
                             int tokens, int headDim) {
    const int t = blockIdx.x;
    if (t >= tokens) return;
    __shared__ float smax[256];
    float mx = 0.f;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x)
        mx = fmaxf(mx, fabsf(dense[(long)t * headDim + d]));
    smax[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        __syncthreads();
    }
    const float sc = smax[0] > 0.f ? smax[0] / 127.f : 1.f;
    if (threadIdx.x == 0) scales[t] = sc;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        q8[(long)t * headDim + d] = (signed char)nearbyintf(dense[(long)t * headDim + d] / sc);
    }
}
void quantize(const float* dense, signed char* q8, float* scales,
              int tokens, int headDim, void* stream) {
    quant_kernel<<<tokens, 256, 0, (cudaStream_t)stream>>>(dense, q8, scales, tokens, headDim);
    cudaStreamSynchronize((cudaStream_t)stream);
}
}
"""
    so = get_backend().compile(src, f"kvquant;cg={CODEGEN_VERSION}")
    lib = ctypes.CDLL(str(so))
    lib.quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _quant_lib = lib
    return lib


def compress_kv(kv) -> CompressedKV:
    """Compress a (batch_heads, seq, head_dim) float32 K or V tensor to the
    int8-per-token layout ON the GPU. ~4x memory reduction vs fp32."""
    v = as_device_view(kv)
    if v.dtype.base != "float32":
        raise TypeError("compress_kv expects float32 input (cast first)")
    bh, seq, hd = v.planes, v.height, v.width
    if bh == 1 and v.planes == 1:
        raise ValueError("expected 3D (batch_heads, seq, head_dim)")
    tokens = bh * seq
    q8 = DeviceBuffer(hd, seq, "int8", planes=bh)
    scales = DeviceBuffer(tokens, 1, "float32")
    _quant_so().quantize(ctypes.c_void_p(v.ptr), ctypes.c_void_p(q8.ptr),
                         ctypes.c_void_p(scales.ptr), tokens, hd,
                         ctypes.c_void_p(0))
    return CompressedKV(q8, scales, bh, seq, hd)


class FlashAttention:
    """Compiled FA-2 forward for one (head_dim, dtype, layout, epilogue) shape."""

    def __init__(self, head_dim: int, kv_layout: str = "dense",
                 epilogue=None):
        if head_dim % 32 != 0:
            raise ValueError("head_dim must be a multiple of 32")
        if kv_layout not in ("dense", "int8"):
            raise ValueError("kv_layout must be 'dense' or 'int8'")
        self.head_dim = head_dim
        self.kv_layout = kv_layout
        self.epilogue = list(epilogue or [])
        for op in self.epilogue:
            if op.role in (READ, WRITE):
                raise ValueError("epilogue ops must be compute-only")
        self._lib = None

    # ---- public ------------------------------------------------------------
    def __call__(self, q, k, v, out=None, causal=False, scale=None,
                 stream=None):
        vq = as_device_view(q)
        bh, seq_q, hd = vq.planes, vq.height, vq.width
        if hd != self.head_dim:
            raise ValueError(f"q head_dim {hd} != compiled {self.head_dim}")

        if self.kv_layout == "int8":
            if not isinstance(k, CompressedKV) or not isinstance(v, CompressedKV):
                raise TypeError("int8 layout expects CompressedKV (use fkl.compress_kv)")
            seq_k = k.seq
            kptr, vptr = k.data.ptr, v.data.ptr
            ks, vs = k.scales.ptr, v.scales.ptr
        else:
            vk, vv = as_device_view(k), as_device_view(v)
            seq_k = vk.height
            kptr, vptr, ks, vs = vk.ptr, vv.ptr, 0, 0

        self._ensure_compiled(vq.dtype.base)
        if out is None:
            out = DeviceBuffer(hd, seq_q, "float32", planes=bh)
        vout = as_device_view(out)

        eps = []
        dt = _dtype("float32")
        for op in self.epilogue:
            if hasattr(op, "bind"):
                op.bind(dt)
            eps.extend(op.values)
            dt = op.out_dtype(dt)
        pbuf = (ctypes.c_float * max(1, len(eps)))(*eps)

        sc = float(scale) if scale else 1.0 / math.sqrt(self.head_dim)
        self._lib.fa_forward(ctypes.c_void_p(vq.ptr), ctypes.c_void_p(kptr),
                             ctypes.c_void_p(vptr), ctypes.c_void_p(ks),
                             ctypes.c_void_p(vs), ctypes.c_void_p(vout.ptr),
                             bh, seq_q, seq_k, ctypes.c_float(sc),
                             1 if causal else 0,
                             ctypes.cast(pbuf, ctypes.c_void_p),
                             ctypes.c_void_p(stream_handle(stream)))
        return out

    # ---- compilation ---------------------------------------------------------
    def _ensure_compiled(self, in_dtype: str):
        if self._lib is not None:
            return
        if in_dtype != "float32":  # base name
            raise TypeError("flash_attention currently takes float32 q/k/v")
        from .jit import _ARCH

        # epilogue chain -> C++ expression chained with .then()
        st = ChainState(_dtype("float32"), self.head_dim, 1, 1)
        exprs, pbase, toks = [], 0, []
        dt = _dtype("float32")
        for op in self.epilogue:
            if hasattr(op, "bind"):
                op.bind(dt)
            exprs.append(op.cpp(st, pbase))
            toks.append(op.token(st))
            pbase += len(op.values)
            dt = op.out_dtype(dt)
        if exprs:
            ep_build = exprs[0] + "".join(f".then({e})" for e in exprs[1:])
            ep_type = f"decltype({ep_build})"
        else:
            ep_build = "AttentionIdentityEpilogue{}"
            ep_type = "AttentionIdentityEpilogue"

        kvl = ("KVLayout::INT8_PER_TOKEN" if self.kv_layout == "int8"
               else "KVLayout::DENSE")
        kvt = "signed char" if self.kv_layout == "int8" else "float"

        src = f"""// AUTO-GENERATED by fkl-python (FlashAttention). Host+device TU.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/attention/flash_attention.h>

using namespace fk;

extern "C" {{
void fa_forward(const float* q, const void* k, const void* v,
                const float* kScale, const float* vScale, float* o,
                int bh, int seqQ, int seqK, float scale, int causal,
                const float* params, void* ext_stream) {{
    const auto epilogue = {ep_build};
    using Ep = {ep_type};
    if (ext_stream != nullptr) {{
        Stream stream(reinterpret_cast<cudaStream_t>(ext_stream));
        executeFlashAttention<float, {self.head_dim}, {kvl},
                              ({self.head_dim} >= 64 ? 32 : 32), 4, Ep>(
            q, (const {kvt}*)k, (const {kvt}*)v, o, bh, seqQ, seqK,
            causal != 0, stream, kScale, vScale, scale, epilogue);
    }} else {{
        static Stream stream;
        executeFlashAttention<float, {self.head_dim}, {kvl},
                              ({self.head_dim} >= 64 ? 32 : 32), 4, Ep>(
            q, (const {kvt}*)k, (const {kvt}*)v, o, bh, seqQ, seqK,
            causal != 0, stream, kScale, vScale, scale, epilogue);
        stream.sync();
    }}
}}
}} // extern "C"
"""
        sig = (f"flashattn;arch={_ARCH};cg={CODEGEN_VERSION};d={self.head_dim};"
               f"kv={self.kv_layout};ep=" + "|".join(toks))
        so = get_backend().compile(src, sig)
        lib = ctypes.CDLL(str(so))
        lib.fa_forward.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._lib = lib


_fa_cache = {}


def flash_attention(q, k, v, out=None, causal=False, scale=None, stream=None,
                    epilogue=None, kv_layout=None):
    """One-shot API. Auto-detects head_dim and kv layout; compiled kernels
    are cached per (head_dim, layout, epilogue shape)."""
    if kv_layout is None:
        kv_layout = "int8" if isinstance(k, CompressedKV) else "dense"
    vq = as_device_view(q)
    ep_key = tuple(type(op).__name__ for op in (epilogue or []))
    key = (vq.width, kv_layout, ep_key)
    fa = _fa_cache.get(key)
    if fa is None:
        fa = FlashAttention(vq.width, kv_layout, epilogue)
        _fa_cache[key] = fa
    elif epilogue:
        fa.epilogue = list(epilogue)   # same shape, fresh values
    return fa(q, k, v, out=out, causal=causal, scale=scale, stream=stream)
