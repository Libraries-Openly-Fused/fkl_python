"""compose() + FusedKernel v2: shape/dtype-aware, full library surface.

    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Crop(100, 50, 640, 360),
        fkl.Resize(64, 64, interp="linear"),
        fkl.Mul((2.0, 2.0, 2.0)),
        fkl.Sub(128.0),
        fkl.SaturateCast("uint8"),
        fkl.TensorWrite(),
    )
    out = pipe(x)    # cuda torch tensor HxWx3 uint8 -> 64x64x3 uint8

Because op TYPES (not values) define the kernel, the first call with a given
(chain, input dtype/layout) compiles once; every other call -- with any crop
rect, any mul factor, any size values that keep the same types -- reuses the
cached .so. Values travel through params[].
"""
from __future__ import annotations
import ctypes
from typing import List, Optional

from .operations import Op, READ, WRITE
from .codegen import generate_cu, signature, collect_params, plan
from .backend import get_backend, _ARCH
from .tensor import as_device_view, stream_handle, DeviceView
from .types import DType, from_cai


class _FklDims(ctypes.Structure):
    _fields_ = [("in_w", ctypes.c_int), ("in_h", ctypes.c_int),
                ("in_planes", ctypes.c_int),
                ("out_w", ctypes.c_int), ("out_h", ctypes.c_int),
                ("out_planes", ctypes.c_int)]


class FusedKernel:
    """A composed chain. Compiles lazily on first call (input dtype/layout
    is only known then) and caches per signature.

    target="gpu" (default) JITs a CUDA .so; target="cpu" JITs a plain C++
    .so running FKL's ParArch::CPU executor on host memory (numpy in/out,
    no CUDA required)."""

    def __init__(self, ops: List[Op], target: str = "gpu",
                 thread_fusion: bool = False):
        if not ops or ops[0].role != READ:
            raise ValueError("chain must start with TensorRead()")
        if ops[-1].role != WRITE:
            raise ValueError("chain must end with TensorWrite()/TensorSplit()")
        if target not in ("gpu", "cpu"):
            raise ValueError("target must be 'gpu' or 'cpu'")
        self.ops = ops
        self.target = target
        self.thread_fusion = bool(thread_fusion) and target == "gpu"
        self._variants = {}  # signature -> (entry fn, params buffer, out_state)

    # ---- compile path (cold, once per signature) --------------------------
    def _tf_effective(self, dt: DType, shape) -> bool:
        """ThreadFusion vectorizes row accesses (e.g. float4). With external
        tight-pitched pointers every row must stay 16-byte aligned, i.e.
        width * itemsize % 16 == 0. Otherwise fall back to the scalar DPP
        for THIS shape (correctness first; the .so is per-signature)."""
        if not self.thread_fusion:
            return False
        return (shape[0] * dt.itemsize) % 16 == 0

    def _get_variant(self, dt: DType, shape, n_inputs: int = 1):
        arch = _ARCH if self.target == "gpu" else "host"
        tf = self._tf_effective(dt, shape)
        sig = (signature(self.ops, dt, shape, arch, n_inputs)
               + f";t={self.target};tf={int(tf)}")
        hit = self._variants.get(sig)
        if hit is not None:
            return hit
        cu = generate_cu(self.ops, dt, shape, n_inputs, target=self.target,
                         thread_fusion=tf)
        if self.target == "cpu":
            from .backend import CompilerBackend
            so = CompilerBackend(CompilerBackend.CPU).compile(cu, sig)
        else:
            so = get_backend().compile(cu, sig)
        lib = ctypes.CDLL(str(so))
        entry = lib.fkl_entry
        entry.restype = None
        entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.POINTER(_FklDims),
                          ctypes.c_void_p, ctypes.c_void_p]
        params = collect_params(self.ops, dt, shape, n_inputs)
        pbuf = (ctypes.c_float * max(1, len(params)))(*params)
        _, out_st = plan(self.ops, dt, shape, n_inputs)
        variant = (entry, pbuf, out_st, so)
        self._variants[sig] = variant
        return variant

    # ---- HOT PATH ----------------------------------------------------------
    def __call__(self, x, out=None, stream=None):
        if self.target == "cpu":
            return self._call_cpu(x, out)
        if isinstance(x, (list, tuple)):
            return self._call_batch(x, out, stream)
        vin: DeviceView = as_device_view(x)
        if vin.device != 0:
            # make the input's device current for the launch + output alloc
            from .tensor import DeviceBuffer
            DeviceBuffer._load_driver()
            DeviceBuffer._activate(vin.device)
        in_shape = (vin.width, vin.height, vin.planes)
        entry, pbuf, out_st, _ = self._get_variant(vin.dtype, in_shape)

        if out is None:
            out = self._alloc_out(vin, out_st)
        vout: DeviceView = as_device_view(out)

        dims = _FklDims(vin.width, vin.height, vin.planes,
                        out_st.width, out_st.height, out_st.planes)
        entry(ctypes.c_void_p(vin.ptr), ctypes.c_void_p(vout.ptr),
              ctypes.byref(dims),
              ctypes.cast(pbuf, ctypes.c_void_p),
              ctypes.c_void_p(stream_handle(stream)))
        return out

    def _call_batch(self, xs, out=None, stream=None):
        """HORIZONTAL FUSION over a batch of same-size images.

        `xs` is a list of CUDA arrays (all same dtype + WxH). One kernel
        processes all B images as thread-planes (BatchRead). The output is
        a Tensor with B planes. The batch size B is part of the kernel TYPE
        (std::array<Ptr2D,B>) -> one compile per B, cached.
        """
        views = [as_device_view(x) for x in xs]
        v0 = views[0]
        for v in views[1:]:
            if (v.width, v.height, v.dtype) != (v0.width, v0.height, v0.dtype):
                raise ValueError("batch HF requires same size+dtype for all images")
        B = len(views)
        in_shape = (v0.width, v0.height, 1)
        entry, pbuf, out_st, _ = self._get_variant(v0.dtype, in_shape, B)

        if out is None:
            out = self._alloc_out(v0, out_st)
        vout: DeviceView = as_device_view(out)

        # device-pointer array (host-side; FKL copies Ptr2D params by value
        # into kernel arguments at launch)
        ptrs = (ctypes.c_void_p * B)(*[v.ptr for v in views])
        dims = _FklDims(v0.width, v0.height, 1,
                        out_st.width, out_st.height, out_st.planes)
        entry(ctypes.cast(ptrs, ctypes.c_void_p), ctypes.c_void_p(vout.ptr),
              ctypes.byref(dims),
              ctypes.cast(pbuf, ctypes.c_void_p),
              ctypes.c_void_p(stream_handle(stream)))
        # keep views alive until the call returns (sync path) or rely on
        # caller keeping tensors alive for async streams (documented).
        return out

    def _call_cpu(self, x, out=None):
        """CPU executor: numpy arrays in/out (host memory, synchronous)."""
        import numpy as np
        from .types import from_cai
        x = np.ascontiguousarray(x)
        ai = x.__array_interface__
        dt, w, h, p = from_cai(ai["shape"], ai["typestr"])
        entry, pbuf, out_st, _ = self._get_variant(dt, (w, h, p))

        ch = out_st.dtype.channels
        is_split = self.ops[-1].name in ("TensorSplit", "TensorTSplit")
        if is_split:
            shape = ((out_st.planes if out_st.planes > 1 else 1) * ch,
                     out_st.height, out_st.width)
        elif out_st.planes > 1:
            shape = (out_st.planes, out_st.height, out_st.width) + ((ch,) if ch > 1 else ())
        elif out_st.height > 1:
            shape = (out_st.height, out_st.width) + ((ch,) if ch > 1 else ())
        else:
            shape = (out_st.width,) + ((ch,) if ch > 1 else ())
        if out is None:
            out = np.empty(shape, dtype=out_st.dtype.base)
        out = np.ascontiguousarray(out)

        dims = _FklDims(w, h, p, out_st.width, out_st.height, out_st.planes)
        entry(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(out.ctypes.data),
              ctypes.byref(dims), ctypes.cast(pbuf, ctypes.c_void_p),
              ctypes.c_void_p(0))
        return out

    # ---- output allocation --------------------------------------------------
    def _alloc_out(self, vin: DeviceView, out_st):
        mod = type(vin.obj).__module__.split(".")[0]
        ch = out_st.dtype.channels
        is_split = self.ops[-1].name in ("TensorSplit", "TensorTSplit")
        if is_split:
            # planar output: channels become leading planes (CHW / NCHW)
            nplanes = (out_st.planes if out_st.planes > 1 else 1) * ch
            shape = (nplanes, out_st.height, out_st.width)
            alloc_dtype = out_st.dtype.base
        elif out_st.planes > 1:
            shape = (out_st.planes, out_st.height, out_st.width) + ((ch,) if ch > 1 else ())
            alloc_dtype = out_st.dtype.base
        elif out_st.height > 1:
            shape = (out_st.height, out_st.width) + ((ch,) if ch > 1 else ())
            alloc_dtype = out_st.dtype.base
        else:
            shape = (out_st.width,) + ((ch,) if ch > 1 else ())
            alloc_dtype = out_st.dtype.base
        if mod == "torch":
            import torch
            tmap = {"float32": torch.float32, "float64": torch.float64,
                    "uint8": torch.uint8, "int32": torch.int32,
                    "int16": torch.int16, "uint16": torch.uint16,
                    "int8": torch.int8}
            return torch.empty(shape, device=f"cuda:{vin.device}",
                               dtype=tmap[out_st.dtype.base])
        if mod == "cupy":
            import cupy
            with cupy.cuda.Device(vin.device):
                return cupy.empty(shape, dtype=out_st.dtype.base)
        from .tensor import DeviceBuffer
        return DeviceBuffer.from_state(out_st, device=vin.device)

    def source_for(self, dtype_spec, shape):
        """Debug: show the generated C++ for a given input."""
        from .types import dtype as _d
        return generate_cu(self.ops, _d(dtype_spec), tuple(shape))


def compose(*ops: Op, target: str = "gpu",
            thread_fusion: bool = False) -> FusedKernel:
    """thread_fusion=True opts into FKL's ThreadFusion (TF::ENABLED):
    each thread processes multiple elements with vectorized accesses.
    DEFAULT IS FALSE, matching FKL's own TransformDPP<> default
    (TF::DISABLED): per upstream guidance it only pays off in a small set
    of cases (wide images, trivial chains, bandwidth-bound) — benchmark
    your pipeline before enabling. GPU-only."""
    return FusedKernel(list(ops), target=target, thread_fusion=thread_fusion)


# ===================== Divergent Horizontal Fusion ========================

class DivergentKernel:
    """DIVERGENT HF (the paper's 4th fusion type): one kernel, B thread-
    planes, where DIFFERENT planes execute DIFFERENT op sequences.

        pipe = fkl.compose_divergent(
            [1, 2, 2],                                   # plane -> sequence (1-based)
            [fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite()],   # seq 1
            [fkl.TensorRead(), fkl.Add(5.0), fkl.TensorWrite()],   # seq 2
        )
        out = pipe([img0, img1, img2])    # B=3 same-size images -> Tensor(B)

    Maps to Executor<DivergentBatchTransformDPP<GPU_NVIDIA, Selector>>::
    executeOperations(stream, seq1, seq2) with a generated SequenceSelector
    whose at(z) returns the 1-based sequence for each plane. All sequences
    read the same input batch and write the same output Tensor; the selector
    decides which fused chain each plane runs (paper fig: divergent HF).
    """

    def __init__(self, plane_map, chains):
        if not chains or not plane_map:
            raise ValueError("need plane_map and at least one chain")
        smax = max(plane_map)
        if smax > len(chains) or min(plane_map) < 1:
            raise ValueError("plane_map entries are 1-based sequence indices")
        for ch in chains:
            if ch[0].role != READ or ch[-1].role != WRITE:
                raise ValueError("every chain needs TensorRead() ... TensorWrite()")
        self.plane_map = list(plane_map)
        self.chains = [list(c) for c in chains]
        self._variants = {}

    def _selector_cpp(self):
        # FKL convention (see SequenceSelectorType in circular_tensor.h):
        # at(z) is uint, 1-based sequence index. Generate a chain of ternaries.
        terms = []
        for z, s in enumerate(self.plane_map):
            terms.append((z, s))
        expr = f"{self.plane_map[-1]}u"
        for z, s in reversed(terms[:-1]):
            expr = f"(index == {z}u ? {s}u : {expr})"
        return (
            "struct PySequenceSelector {\n"
            "    FK_HOST_DEVICE_FUSE uint at(const uint& index) {\n"
            f"        return {expr};\n"
            "    }\n"
            "};")

    def _get_variant(self, dt, shape, B):
        from .codegen import plan as _plan
        key = (str(dt), shape, B)
        hit = self._variants.get(key)
        if hit is not None:
            return hit

        # all chains must agree on output dtype/shape (they share the Tensor)
        outs = [_plan(ch, dt, shape, B)[1] for ch in self.chains]
        o0 = outs[0]
        for o in outs[1:]:
            if (str(o.dtype), o.width, o.height) != (str(o0.dtype), o0.width, o0.height):
                raise ValueError("all divergent chains must produce the same output type/shape")
        out_st = o0
        out_st.planes = B

        in_t = dt.ctype
        out_t = out_st.dtype.ctype

        elems = ", ".join(
            f"Ptr2D<{in_t}>((({in_t}**)d_in)[{i}], (uint)dims->in_w, "
            f"(uint)dims->in_h, (uint)(dims->in_w * sizeof({in_t})), MemType::Device)"
            for i in range(B))

        seq_decls = []
        seq_names = []
        for si, ch in enumerate(self.chains):
            states, _ = _plan(ch, dt, shape, B)
            exprs = []
            pbase = sum(len(o.values) for c in self.chains[:si] for o in c)
            for op, st in zip(ch, states):
                if hasattr(op, "bind"):
                    op.bind(st.dtype)
                if op.role not in (READ, WRITE):
                    exprs.append(op.cpp(st, pbase))
                pbase += len(op.values)
            seq = ", ".join(
                [f"PerThreadRead<ND::_2D, {in_t}>::build(input)"] + exprs +
                [f"TensorWrite<{out_t}>::build(output)"])
            # IOpSequences must be lvalues: BaseExecutor forwards by ref
            seq_decls.append(f"const auto seq{si} = buildOperationSequence({seq});")
            seq_names.append(f"seq{si}")
        pall = [v for c in self.chains for o in c for v in o.values]

        seq_decl_block = "\n    ".join(seq_decls)
        seqs_joined = ", ".join(seq_names)
        src = f"""// AUTO-GENERATED by fkl-python (Divergent HF). Single-step host+device TU.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

{self._selector_cpp()}

extern "C" {{
struct FklDims {{
    int in_w, in_h, in_planes;
    int out_w, out_h, out_planes;
}};

void fkl_entry(void* d_in, void* d_out, const FklDims* dims,
               const float* params, void* ext_stream)
{{
    const std::array<Ptr2D<{in_t}>, {B}> input{{ {elems} }};
    Tensor<{out_t}> output(({out_t}*)d_out, (uint)dims->out_w,
                           (uint)dims->out_h, (uint)dims->out_planes, 1,
                           MemType::Device);
    {seq_decl_block}
    // NOTE: direct kernel launch instead of Executor<DivergentBatchTransformDPP>.
    // LTS fixed the fuse_back compile error (issue #245), BUT the Executor's
    // getActiveThreads SUMS the z extents of all sequences. Our model gives
    // EVERY sequence a full-batch read (selector picks which planes run it),
    // so Executor would launch num_sequences * B planes and write out of
    // bounds. grid.z = B with the selector is the intended semantics here.
    const dim3 block(cxp::min::f((uint)dims->in_w, 32u),
                     cxp::min::f((uint)dims->in_h, 8u));
    const dim3 grid((uint)ceil(dims->in_w / (float)block.x),
                    (uint)ceil(dims->in_h / (float)block.y),
                    (uint){B});
    const DivergentBatchTransformDPPDetails<ParArch::GPU_NVIDIA> details{{}};
    if (ext_stream != nullptr) {{
        Stream stream(reinterpret_cast<cudaStream_t>(ext_stream));
        launchDivergentBatchTransformDPP_Kernel<ParArch::GPU_NVIDIA, PySequenceSelector>
            <<<grid, block, 0, stream.getCUDAStream()>>>(details, {seqs_joined});
        gpuErrchk(cudaGetLastError());
    }} else {{
        static Stream stream;
        launchDivergentBatchTransformDPP_Kernel<ParArch::GPU_NVIDIA, PySequenceSelector>
            <<<grid, block, 0, stream.getCUDAStream()>>>(details, {seqs_joined});
        gpuErrchk(cudaGetLastError());
        stream.sync();
    }}
}}
}} // extern "C"
"""
        chain_toks = ";".join(
            "|".join(op.token(st) for op, st in
                     zip(ch, _plan(ch, dt, shape, B)[0]))
            for ch in self.chains)
        sig = (f"divergent;arch={_ARCH};in={dt}x{B};map={tuple(self.plane_map)};"
               f"chains={chain_toks}")
        so = get_backend().compile(src, sig)
        lib = ctypes.CDLL(str(so))
        entry = lib.fkl_entry
        entry.restype = None
        entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.POINTER(_FklDims),
                          ctypes.c_void_p, ctypes.c_void_p]
        pbuf = (ctypes.c_float * max(1, len(pall)))(*pall)
        variant = (entry, pbuf, out_st, so)
        self._variants[key] = variant
        return variant

    def __call__(self, xs, out=None, stream=None):
        views = [as_device_view(x) for x in xs]
        v0 = views[0]
        B = len(views)
        if B != len(self.plane_map):
            raise ValueError(f"plane_map has {len(self.plane_map)} entries, got {B} images")
        entry, pbuf, out_st, _ = self._get_variant(
            v0.dtype, (v0.width, v0.height, 1), B)
        if out is None:
            from .tensor import DeviceBuffer
            out = DeviceBuffer.from_state(out_st)
        vout = as_device_view(out)
        ptrs = (ctypes.c_void_p * B)(*[v.ptr for v in views])
        dims = _FklDims(v0.width, v0.height, 1,
                        out_st.width, out_st.height, out_st.planes)
        entry(ctypes.cast(ptrs, ctypes.c_void_p), ctypes.c_void_p(vout.ptr),
              ctypes.byref(dims), ctypes.cast(pbuf, ctypes.c_void_p),
              ctypes.c_void_p(stream_handle(stream)))
        return out


def compose_divergent(plane_map, *chains) -> DivergentKernel:
    return DivergentKernel(plane_map, chains)
