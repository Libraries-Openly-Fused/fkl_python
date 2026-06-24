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
    """Compressed KV tensor + per-token scales (on GPU). fmt: 'int8' | 'fp8'."""

    def __init__(self, data: DeviceBuffer, scales: DeviceBuffer,
                 batch_heads: int, seq: int, head_dim: int, fmt: str = "int8"):
        self.data, self.scales = data, scales
        self.batch_heads, self.seq, self.head_dim = batch_heads, seq, head_dim
        self.fmt = fmt

    @property
    def nbytes(self) -> int:
        return self.data._nbytes + self.scales._nbytes


_quant_lib = None


def _quant_so():
    """Tiny standalone kernel for GPU-side per-token int8/fp8 quantization."""
    global _quant_lib
    if _quant_lib is not None:
        return _quant_lib
    src = r"""
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <math.h>
template <bool FP8>
__global__ void quant_kernel_t(const float* dense, signed char* q8, float* scales,
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
    const float sc = smax[0] > 0.f ? smax[0] / (FP8 ? 448.f : 127.f) : 1.f;
    if (threadIdx.x == 0) scales[t] = sc;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        const float x = dense[(long)t * headDim + d] / sc;
        if (FP8) {
            const __nv_fp8_e4m3 f8(x);
            q8[(long)t * headDim + d] = (signed char)f8.__x;
        } else {
            q8[(long)t * headDim + d] = (signed char)nearbyintf(x);
        }
    }
}
extern "C" {
void quantize(const float* dense, signed char* q8, float* scales,
              int tokens, int headDim, void* stream) {
    quant_kernel_t<false><<<tokens, 256, 0, (cudaStream_t)stream>>>(dense, q8, scales, tokens, headDim);
    cudaStreamSynchronize((cudaStream_t)stream);
}
void quantize_fp8(const float* dense, signed char* q8, float* scales,
                  int tokens, int headDim, void* stream) {
    quant_kernel_t<true><<<tokens, 256, 0, (cudaStream_t)stream>>>(dense, q8, scales, tokens, headDim);
    cudaStreamSynchronize((cudaStream_t)stream);
}
}
"""
    so = get_backend().compile(src, f"kvquant;cg={CODEGEN_VERSION};v=2")
    lib = ctypes.CDLL(str(so))
    for fn in (lib.quantize, lib.quantize_fp8):
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _quant_lib = lib
    return lib


def compress_kv(kv, fmt: str = "int8") -> CompressedKV:
    """Compress a (batch_heads, seq, head_dim) float32 K or V tensor to the
    int8- or fp8(e4m3)-per-token layout ON the GPU. ~4x smaller than fp32.
    fp8 keeps more precision near zero (attention tails) at the same size."""
    if fmt not in ("int8", "fp8"):
        raise ValueError("fmt must be 'int8' or 'fp8'")
    v = as_device_view(kv)
    if v.dtype.base != "float32":
        raise TypeError("compress_kv expects float32 input (cast first)")
    bh, seq, hd = v.planes, v.height, v.width
    if bh == 1 and v.planes == 1:
        raise ValueError("expected 3D (batch_heads, seq, head_dim)")
    tokens = bh * seq
    q8 = DeviceBuffer(hd, seq, "int8", planes=bh)
    scales = DeviceBuffer(tokens, 1, "float32")
    fn = _quant_so().quantize_fp8 if fmt == "fp8" else _quant_so().quantize
    fn(ctypes.c_void_p(v.ptr), ctypes.c_void_p(q8.ptr),
       ctypes.c_void_p(scales.ptr), tokens, hd, ctypes.c_void_p(0))
    return CompressedKV(q8, scales, bh, seq, hd, fmt)


class FlashAttention:
    """Compiled FA-2 forward for one (head_dim, dtype, layout, prologue,
    epilogue) shape.

    PROLOGUE FUSION (mirrors the C++ IOp-first API): prologue_q/k/v are
    lists of fkl compute ops fused onto the Q/K/V *Read IOps* — every
    element enters the kernel through read.then(op1).then(op2)...,
    in-register, at load time. With kv_layout="int8" the K/V prologue
    chains AFTER the dequantizing Int8TokenDequantRead. Op VALUES live in
    params[] — changing them never recompiles."""

    def __init__(self, head_dim: int, kv_layout: str = "dense",
                 epilogue=None, prologue_q=None, prologue_k=None,
                 prologue_v=None, mma: bool = False, score_mod=None,
                 block_sparse: bool = False):
        if head_dim % 32 != 0:
            raise ValueError("head_dim must be a multiple of 32")
        if kv_layout not in ("dense", "int8", "fp8"):
            raise ValueError("kv_layout must be 'dense', 'int8' or 'fp8'")
        if (score_mod is not None or block_sparse) and not mma:
            raise ValueError("score_mod/block_mask require mma=True "
                             "(tensor-core path)")
        self.head_dim = head_dim
        self.kv_layout = kv_layout
        self.mma = bool(mma)   # tensor-core (bf16 mma.sync) path
        self.score_mod = score_mod          # flex-attention score mod
        self.block_sparse = bool(block_sparse)
        self.epilogue = list(epilogue or [])
        self.prologue_q = list(prologue_q or [])
        self.prologue_k = list(prologue_k or [])
        self.prologue_v = list(prologue_v or [])
        for op in (self.epilogue + self.prologue_q + self.prologue_k
                   + self.prologue_v):
            if op.role in (READ, WRITE):
                raise ValueError("prologue/epilogue ops must be compute-only")
        self._lib = None

    # ---- public ------------------------------------------------------------
    def __call__(self, q, k, v, out=None, causal=False, scale=None,
                 stream=None, block_mask=None):
        if self.block_sparse != (block_mask is not None):
            raise ValueError("kernel compiled with block_sparse="
                             f"{self.block_sparse} but block_mask is "
                             f"{'set' if block_mask else 'None'}")
        vq = as_device_view(q)
        bh, seq_q, hd = vq.planes, vq.height, vq.width
        if hd != self.head_dim:
            raise ValueError(f"q head_dim {hd} != compiled {self.head_dim}")

        if self.kv_layout in ("int8", "fp8"):
            if not isinstance(k, CompressedKV) or not isinstance(v, CompressedKV):
                raise TypeError("compressed layout expects CompressedKV (use fkl.compress_kv)")
            if k.fmt != self.kv_layout or v.fmt != self.kv_layout:
                raise TypeError(f"CompressedKV fmt mismatch: kernel={self.kv_layout}, "
                                f"k={k.fmt}, v={v.fmt}")
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
        for ops in (self.prologue_q, self.prologue_k, self.prologue_v,
                    self.epilogue):
            dt = _dtype("float32")
            for op in ops:
                if hasattr(op, "bind"):
                    op.bind(dt)
                eps.extend(op.values)
                dt = op.out_dtype(dt)
        pbuf = (ctypes.c_float * max(1, len(eps)))(*eps)

        sc = float(scale) if scale else 1.0 / math.sqrt(self.head_dim)

        # flex score-mod runtime values (slope/cap/window — never recompiles)
        mvals = list(self.score_mod.values) if self.score_mod else []
        mbuf = (ctypes.c_float * max(1, len(mvals)))(*mvals)

        # block sparsity: mask ptr + geometry (0s = dense)
        if block_mask is not None:
            mk_ptr, nqb, nkb = block_mask.ptr, block_mask.n_q_blocks, block_mask.n_kv_blocks
            mbq, mbkv = block_mask.block_q, block_mask.block_kv
        else:
            mk_ptr, nqb, nkb, mbq, mbkv = 0, 0, 0, 128, 128

        self._lib.fa_forward(ctypes.c_void_p(vq.ptr), ctypes.c_void_p(kptr),
                             ctypes.c_void_p(vptr), ctypes.c_void_p(ks),
                             ctypes.c_void_p(vs), ctypes.c_void_p(vout.ptr),
                             bh, seq_q, seq_k, ctypes.c_float(sc),
                             1 if causal else 0,
                             ctypes.cast(pbuf, ctypes.c_void_p),
                             ctypes.cast(mbuf, ctypes.c_void_p),
                             ctypes.c_void_p(mk_ptr), nqb, nkb, mbq, mbkv,
                             ctypes.c_void_p(stream_handle(stream)))
        return out

    # ---- compilation ---------------------------------------------------------
    def _chain(self, ops, pbase):
        """ops -> (list of C++ exprs, tokens, new pbase)."""
        st = ChainState(_dtype("float32"), self.head_dim, 1, 1)
        exprs, toks = [], []
        dt = _dtype("float32")
        for op in ops:
            if hasattr(op, "bind"):
                op.bind(dt)
            exprs.append(op.cpp(st, pbase))
            toks.append(op.token(st))
            pbase += len(op.values)
            dt = op.out_dtype(dt)
        return exprs, toks, pbase

    def _ensure_compiled(self, in_dtype: str):
        if self._lib is not None:
            return
        if in_dtype != "float32":  # base name
            raise TypeError("flash_attention currently takes float32 q/k/v")
        from .jit import _ARCH

        # prologue chains fuse onto the Read IOps; epilogue onto the output.
        pbase = 0
        q_ex, q_tk, pbase = self._chain(self.prologue_q, pbase)
        k_ex, k_tk, pbase = self._chain(self.prologue_k, pbase)
        v_ex, v_tk, pbase = self._chain(self.prologue_v, pbase)
        e_ex, e_tk, pbase = self._chain(self.epilogue, pbase)

        def fuse(base, exprs):
            return base + "".join(f".then({e})" for e in exprs)

        if self.kv_layout == "int8":
            k_read = "makeInt8KVRead((const signed char*)k, kScale, bh, seqK, HD)"
            v_read = "makeInt8KVRead((const signed char*)v, vScale, bh, seqK, HD)"
        elif self.kv_layout == "fp8":
            k_read = "makeFp8KVRead(k, kScale, bh, seqK, HD)"
            v_read = "makeFp8KVRead(v, vScale, bh, seqK, HD)"
        else:
            k_read = "makeAttentionRead((const float*)k, bh, seqK, HD)"
            v_read = "makeAttentionRead((const float*)v, bh, seqK, HD)"
        q_build = fuse("makeAttentionRead(q, bh, seqQ, HD)", q_ex)
        k_build = fuse(k_read, k_ex)
        v_build = fuse(v_read, v_ex)

        if e_ex:
            ep_build = fuse(e_ex[0], e_ex[1:])
        else:
            ep_build = "AttentionIdentityEpilogue{}"

        exec_fn = ("executeFlashAttentionMma" if self.mma
                   else "executeFlashAttention")

        # flex score mod + block sparsity (mma path only; validated in ctor)
        mod_ctor = (self.score_mod.cpp_ctor if self.score_mod
                    else "NoScoreMod{}")
        if self.mma:
            sparse_decl = ("    const BlockSparsity sparse{ blockMask, nQB, nKB, "
                           "maskBQ, maskBKV };\n")
            extra_args = ", epilogue, scoreMod, sparse"
        else:
            sparse_decl = ""
            extra_args = ", epilogue"

        call_tail = extra_args if self.mma else ", epilogue"

        src = f"""// AUTO-GENERATED by fkl-python (FlashAttention). Host+device TU.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/attention/flash_attention_mma.h>

using namespace fk;

constexpr int HD = {self.head_dim};

extern "C" {{
void fa_forward(const float* q, const void* k, const void* v,
                const float* kScale, const float* vScale, float* o,
                int bh, int seqQ, int seqK, float scale, int causal,
                const float* params, const float* modParams,
                const unsigned char* blockMask, int nQB, int nKB,
                int maskBQ, int maskBKV, void* ext_stream) {{
    // PROLOGUES: Q/K/V are Read IOps (possibly .then-fused); every element
    // enters the DPP through them, in-register, at load time.
    const auto qIOp = {q_build};
    const auto kIOp = {k_build};
    const auto vIOp = {v_build};
    const auto epilogue = {ep_build};
    const auto scoreMod = {mod_ctor};
    (void)scoreMod; (void)modParams;
{sparse_decl}    if (ext_stream != nullptr) {{
        Stream stream(reinterpret_cast<cudaStream_t>(ext_stream));
        {exec_fn}<HD>(qIOp, kIOp, vIOp, o, bh, seqQ, seqK,
                      causal != 0, stream, scale{call_tail});
    }} else {{
        static Stream stream;
        {exec_fn}<HD>(qIOp, kIOp, vIOp, o, bh, seqQ, seqK,
                      causal != 0, stream, scale{call_tail});
        stream.sync();
    }}
}}
}} // extern "C"
"""
        mod_tok = self.score_mod.token() if self.score_mod else "none"
        sig = (f"flashattn;arch={_ARCH};cg={CODEGEN_VERSION};d={self.head_dim};"
               f"kv={self.kv_layout};mma={int(self.mma)};mod={mod_tok};"
               f"bs={int(self.block_sparse)};pq=" + "|".join(q_tk) +
               ";pk=" + "|".join(k_tk) + ";pv=" + "|".join(v_tk) +
               ";ep=" + "|".join(e_tk))
        so = get_backend().compile(src, sig)
        lib = ctypes.CDLL(str(so))
        lib.fa_forward.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p]
        self._lib = lib


_fa_cache = {}


def flash_attention(q, k, v, out=None, causal=False, scale=None, stream=None,
                    epilogue=None, kv_layout=None,
                    prologue_q=None, prologue_k=None, prologue_v=None,
                    mma=False, score_mod=None, block_mask=None):
    """One-shot API. Auto-detects head_dim and kv layout; compiled kernels
    are cached per (head_dim, layout, mma, prologue shapes, epilogue shape).

    prologue_q/k/v: compute op chains fused onto the Q/K/V Read IOps —
    every element is preprocessed in-register at load time (Oscar's
    prologue: the input is a ReadOperation, not a pointer).
    mma=True: tensor-core path (bf16 mma.sync m16n8k16, fp32 accum) —
    ~1e-3 abs error on unit inputs (vs ~5e-7 SIMT fp32) but order-of-
    magnitude faster. Same prologue/epilogue fusion."""
    if kv_layout is None:
        kv_layout = k.fmt if isinstance(k, CompressedKV) else "dense"
    if (score_mod is not None or block_mask is not None) and not mma:
        mma = True   # flex/sparse live on the tensor-core path
    vq = as_device_view(q)

    def _key(ops):
        return tuple(type(op).__name__ for op in (ops or []))

    key = (vq.width, kv_layout, bool(mma), _key(prologue_q), _key(prologue_k),
           _key(prologue_v), _key(epilogue),
           type(score_mod).__name__ if score_mod else "none",
           block_mask is not None)
    fa = _fa_cache.get(key)
    if fa is None:
        fa = FlashAttention(vq.width, kv_layout, epilogue,
                            prologue_q, prologue_k, prologue_v, mma=mma,
                            score_mod=score_mod,
                            block_sparse=block_mask is not None)
        _fa_cache[key] = fa
    else:  # same shapes, fresh values (never recompiles)
        if epilogue:
            fa.epilogue = list(epilogue)
        if prologue_q:
            fa.prologue_q = list(prologue_q)
        if prologue_k:
            fa.prologue_k = list(prologue_k)
        if prologue_v:
            fa.prologue_v = list(prologue_v)
        if score_mod is not None:
            fa.score_mod = score_mod   # same TYPE (key) — fresh values
    return fa(q, k, v, out=out, causal=causal, scale=scale, stream=stream,
              block_mask=block_mask)
