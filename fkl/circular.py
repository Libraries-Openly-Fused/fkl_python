"""CircularTensor: stateful temporal window of the last BATCH frames.

The only FKL feature that is NOT expressible as a chain op: it owns GPU
state (the rolling tensor + a temp tensor + the rotation index) that
persists across calls. The generated .so holds a heap-allocated PyCT
(data Tensor + temp Tensor + rotation idx) mirroring fk::CircularTensor's
double-buffer design; every update() is ONE fused kernel that:

  1. runs your preprocessing chain on the incoming frame, writes it into
     the temp tensor's rotation slot (CircularTensorWrite) AND into the
     newest data plane, and
  2. rotate-copies the other BATCH-1 planes from temp into data
     (CircularTensorRead with the order's direction),

as a Divergent HF kernel: SequenceSelectorType routes plane z to sequence
1 (update) or 2 (copy), grid.z = BATCH exactly.

Why not call fk::CircularTensor::update directly: the Executor's
getActiveThreads SUMS the z extents of all sequences (1 frame-read +
BATCH circular-read = BATCH+1 planes) and the extra plane writes past the
data tensor. Race-safety of our launch is auditable: the temp slot written
by seq1 is never among the slots read by seq2 in the same launch, for both
orders and every rotation index (see test_circular_tensor.py).

Usage:
    ct = fkl.CircularTensor(64, 48, batch=4, dtype="uint8", channels=3,
                            order="newest_first", layout="packed")
    for frame in camera:                  # frame: HxWx3 uint8 on GPU
        ct.update(frame, ops=[fkl.Cast("float32"), fkl.Div(255.0)])
        dnn_input = ct.snapshot()         # DeviceBuffer (BATCH, H, W, C)

Layouts: "packed"  -> TensorWrite   (BATCH, H, W, C)
         "planar"  -> TensorSplit   (BATCH, C, H, W)   [DNN NCHW ingest]
Order:   "newest_first" (plane 0 = latest frame) or "oldest_first".
"""
from __future__ import annotations
import ctypes

from .backend import get_backend
from .codegen import CODEGEN_VERSION
from .operations import READ, WRITE, ChainState
from .tensor import DeviceBuffer, as_device_view, stream_handle
from .types import dtype as _dtype

_ORDERS = {"newest_first": "CircularTensorOrder::NewestFirst",
           "oldest_first": "CircularTensorOrder::OldestFirst"}
_LAYOUTS = ("packed", "planar")


class CircularTensor:
    def __init__(self, width: int, height: int, batch: int,
                 dtype="uint8", channels: int = 1,
                 order: str = "newest_first", layout: str = "packed",
                 out_dtype=None):
        if order not in _ORDERS:
            raise ValueError(f"order must be one of {tuple(_ORDERS)}")
        if layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}")
        if batch < 2:
            raise ValueError("batch must be >= 2")
        base = _dtype(dtype) if isinstance(dtype, str) else dtype
        self.in_dtype = base.with_channels(channels) if channels > 1 else base
        # the STORED dtype: what your preproc chain outputs (default = input)
        od = out_dtype or self.in_dtype
        self.store_dtype = _dtype(od) if isinstance(od, str) else od
        if self.in_dtype.channels > 1 and self.store_dtype.channels == 1:
            self.store_dtype = self.store_dtype.with_channels(self.in_dtype.channels)
        self.width, self.height, self.batch = int(width), int(height), int(batch)
        self.order, self.layout = order, layout
        self._lib = None
        self._handle = None
        self._chain_key = None
        self._pushed = 0

    # ---- public API ---------------------------------------------------------

    def update(self, frame, ops=None, stream=None):
        """Insert `frame` (preprocessed by `ops`) into the rolling window.
        ONE fused kernel: preproc+insert / rotate-copy (Divergent HF)."""
        ops = list(ops or [])
        for op in ops:
            if op.role in (READ, WRITE):
                raise ValueError("ops must be compute-only (no read/write)")
        v = as_device_view(frame)
        if (v.width, v.height) != (self.width, self.height):
            raise ValueError(f"frame is {v.width}x{v.height}, "
                             f"CircularTensor is {self.width}x{self.height}")
        if str(v.dtype) != str(self.in_dtype):
            raise TypeError(f"frame dtype {v.dtype} != declared {self.in_dtype}")

        self._ensure_compiled(ops)
        params = []
        shape = (self.width, self.height, 1)
        dt = self.in_dtype
        for op in ops:
            if hasattr(op, "bind"):
                op.bind(dt)
            params.extend(op.values)
            dt = op.out_dtype(dt)
        pbuf = (ctypes.c_float * max(1, len(params)))(*params)
        self._lib.ct_update(self._handle, ctypes.c_void_p(v.ptr),
                            ctypes.cast(pbuf, ctypes.c_void_p),
                            ctypes.c_void_p(stream_handle(stream)))
        self._pushed += 1
        return self

    def snapshot(self, stream=None):
        """Copy the current window into a fresh DeviceBuffer.
        packed: (BATCH, H, W[, C]) -- planar: (BATCH*C planes, H, W)."""
        ch = self.store_dtype.channels
        if self.layout == "planar":
            out = DeviceBuffer(self.width, self.height,
                               self.store_dtype.base, channels=1,
                               planes=self.batch * ch)
        else:
            out = DeviceBuffer(self.width, self.height, self.store_dtype.base,
                               channels=ch, planes=self.batch)
        self._lib.ct_snapshot(self._handle, ctypes.c_void_p(out.ptr),
                              ctypes.c_void_p(stream_handle(stream)))
        return out

    @property
    def frames_pushed(self) -> int:
        return self._pushed

    def __len__(self):
        return self.batch

    def close(self):
        if self._handle and self._lib:
            self._lib.ct_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---- compilation --------------------------------------------------------

    def _plan_chain(self, ops):
        """Thread dtype/shape through the preproc chain; returns (states, key).
        Raises if the chain output doesn't match the declared store dtype."""
        states, toks = [], []
        dt = self.in_dtype
        shape = (self.width, self.height, 1)
        for op in ops:
            st = ChainState(dt, *shape)
            states.append(st)
            if hasattr(op, "bind"):
                op.bind(dt)
            toks.append(op.token(st))
            dt = op.out_dtype(dt)
            shape = op.out_shape(shape)
        if str(dt) != str(self.store_dtype):
            raise TypeError(f"preproc chain outputs {dt}, CircularTensor "
                            f"stores {self.store_dtype}")
        return states, "|".join(toks)

    def generate_source(self, ops=None):
        """Return the generated .cu for this CircularTensor + preproc chain
        WITHOUT compiling (debugging / CI compile checks)."""
        ops = list(ops or [])
        states, _ = self._plan_chain(ops)
        return self._generate_cu(ops, states)

    def _ensure_compiled(self, ops):
        from .jit import _ARCH
        states, key = self._plan_chain(ops)
        if self._lib is not None and key == self._chain_key:
            return
        if self._lib is not None:
            raise RuntimeError(
                "this CircularTensor was already compiled with a different "
                "ops chain; create a new CircularTensor per chain")

        src = self._generate_cu(ops, states)
        sig = (f"circular;arch={_ARCH};cg={CODEGEN_VERSION};"
               f"in={self.in_dtype};store={self.store_dtype};"
               f"b={self.batch};{self.order};{self.layout};chain={key}")
        so = get_backend().compile(src, sig)
        lib = ctypes.CDLL(str(so))
        lib.ct_create.restype = ctypes.c_void_p
        lib.ct_create.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.ct_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p]
        lib.ct_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p]
        lib.ct_destroy.argtypes = [ctypes.c_void_p]
        self._handle = lib.ct_create(self.width, self.height)
        if not self._handle:
            raise RuntimeError("ct_create failed")
        self._lib = lib
        self._chain_key = key

    def _generate_cu(self, ops, states) -> str:
        in_t = self.in_dtype.ctype
        store_t = self.store_dtype.ctype
        base_t = self.store_dtype.base_ctype
        ch = self.store_dtype.channels
        order = _ORDERS[self.order]
        B = self.batch

        exprs = []
        pbase = 0
        for op, st in zip(ops, states):
            exprs.append(op.cpp(st, pbase))
            pbase += len(op.values)
        chain = ("".join(e + ",\n        " for e in exprs))

        if self.layout == "planar":
            w_op = f"TensorSplit<{store_t}>"
            r_op = f"TensorPack<{store_t}>"
            cp = ch
            snap_bytes = f"(size_t)ct->w * ct->h * {B} * {ch} * sizeof({base_t})"

        else:
            w_op = f"TensorWrite<{store_t}>"
            r_op = f"TensorRead<{store_t}>"
            cp = 1
            snap_bytes = f"(size_t)ct->w * ct->h * {B} * sizeof({store_t})"

        tensor_t = base_t if self.layout == "planar" else store_t

        return f"""// AUTO-GENERATED by fkl-python (CircularTensor). Host+device TU.
// NOTE: we re-implement fk::CircularTensor::update's two-sequence divergent
// launch instead of calling it: the Executor's getActiveThreads SUMS the z
// of all sequences (1 frame-read + BATCH circular-read = BATCH+1 planes).
// The extra plane writes past the destination tensor and corrupts the temp
// tensor allocated next to it. We launch with grid.z = BATCH (plane 0 =
// update sequence, planes 1..BATCH-1 = rotate-copy), which is the intent
// encoded in SequenceSelectorType.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/core/data/circular_tensor.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

using StoreT  = {store_t};
using TensorTy = Tensor<{tensor_t}>;
using WOp = {w_op};
using ROp = {r_op};
constexpr int B = {B};
using Selector = SequenceSelectorType<{order}, B>;

struct PyCT {{
    TensorTy data;
    TensorTy temp;
    uint w, h;
    int idx;
    PyCT(uint w_, uint h_)
        : data(w_, h_, B, {cp}, MemType::Device),
          temp(w_, h_, B, {cp}, MemType::Device), w(w_), h(h_), idx(0) {{}}
}};

template <typename Seq1, typename Seq2>
static void launch_ct(const Seq1& s1, const Seq2& s2, cudaStream_t stream,
                      uint w, uint h) {{
    const dim3 block(cxp::min::f(w, 32u), cxp::min::f(h, 8u));
    const dim3 grid((uint)ceil(w / (float)block.x),
                    (uint)ceil(h / (float)block.y), (uint)B);
    const DivergentBatchTransformDPPDetails<ParArch::GPU_NVIDIA> details{{}};
    launchDivergentBatchTransformDPP_Kernel<ParArch::GPU_NVIDIA, Selector>
        <<<grid, block, 0, stream>>>(details, s1, s2);
    gpuErrchk(cudaGetLastError());
}}

extern "C" {{

void* ct_create(int w, int h) {{
    try {{ return new PyCT((uint)w, (uint)h); }} catch (...) {{ return nullptr; }}
}}

void ct_update(void* h, void* d_frame, const float* params, void* ext_stream) {{
    PyCT* ct = (PyCT*)h;
    Ptr2D<{in_t}> frame(({in_t}*)d_frame, ct->w, ct->h,
                        (uint)(ct->w * sizeof({in_t})), MemType::Device);

    MidWrite<CircularTensorWrite<CircularDirection::Ascendent, WOp, B>> toTemp;
    toTemp.params.first = ct->idx;
    toTemp.params.opData.params = ct->temp.ptr();

    Read<CircularTensorRead<CTReadDirection_v<{order}>, ROp, B>> fromTemp;
    fromTemp.params.first = ct->idx;
    fromTemp.params.opData.params = ct->temp.ptr();

    const auto seqUpdate = buildOperationSequence(
        PerThreadRead<ND::_2D, {in_t}>::build(frame),
        {chain}toTemp,
        WOp::build(ct->data));
    const auto seqCopy = buildOperationSequence(fromTemp, WOp::build(ct->data));

    cudaStream_t s;
    if (ext_stream != nullptr) {{
        s = reinterpret_cast<cudaStream_t>(ext_stream);
        launch_ct(seqUpdate, seqCopy, s, ct->w, ct->h);
    }} else {{
        static Stream stream;
        s = stream.getCUDAStream();
        launch_ct(seqUpdate, seqCopy, s, ct->w, ct->h);
        stream.sync();
    }}
    ct->idx = (ct->idx + 1) % B;
}}

void ct_snapshot(void* h, void* d_out, void* ext_stream) {{
    PyCT* ct = (PyCT*)h;
    const size_t bytes = {snap_bytes};
    cudaStream_t s = ext_stream != nullptr
        ? reinterpret_cast<cudaStream_t>(ext_stream) : (cudaStream_t)0;
    cudaMemcpyAsync(d_out, ct->data.ptr().data, bytes,
                    cudaMemcpyDeviceToDevice, s);
    if (ext_stream == nullptr) {{ cudaStreamSynchronize(s); }}
}}

void ct_snapshot_temp(void* h, void* d_out, void* ext_stream) {{
    PyCT* ct = (PyCT*)h;
    const size_t bytes = {snap_bytes};
    cudaStream_t s = ext_stream != nullptr
        ? reinterpret_cast<cudaStream_t>(ext_stream) : (cudaStream_t)0;
    cudaMemcpyAsync(d_out, ct->temp.ptr().data, bytes,
                    cudaMemcpyDeviceToDevice, s);
    if (ext_stream == nullptr) {{ cudaStreamSynchronize(s); }}
}}

void ct_destroy(void* h) {{ delete (PyCT*)h; }}

}} // extern "C"
"""
