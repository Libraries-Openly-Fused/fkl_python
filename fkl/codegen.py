"""C++ code generation v2: full library surface.

Threads a ChainState (dtype + shape) through the op chain exactly like FKL
threads OutputType through its template chain, then emits ONE host+device TU:

  extern "C" void fkl_entry(void* in, void* out, FklDims* dims,
                            const float* params, void* stream)

Why one .so per chain *signature*: the C++ types ARE the kernel. Runtime
values (mul factors, crop rects, resize sizes) travel in params[] so the same
.so serves any values without recompiling.

BVF note: Crop/Resize are ReadBack ops. We emit them as plain IOps in the
executeOperations call; FKL's BackFuser::fuse_back does the Backwards
Vertical Fusion at C++ compile time. Python NEVER reimplements fusion.
"""
from __future__ import annotations
from typing import List, Tuple

from .operations import Op, ChainState, READ, WRITE
from .types import DType

# bump when generate_cu's emitted C++ changes for the SAME signature inputs
CODEGEN_VERSION = 7


def plan(ops: List[Op], in_dtype: DType, in_shape: Tuple[int, int, int],
         n_inputs: int = 1):
    """Walk the chain, computing per-op input state + final output state.
    n_inputs > 1 = HF over a batch of same-size images: the read produces
    n_inputs thread-planes (BatchRead under the hood)."""
    if not ops or ops[0].role != READ:
        raise ValueError("chain must start with TensorRead()")
    if ops[-1].role != WRITE:
        raise ValueError("chain must end with TensorWrite()/TensorSplit()")
    states = []
    dt, shape = in_dtype, in_shape
    for i, op in enumerate(ops):
        st = ChainState(dt, *shape)
        states.append(st)
        dt = op.out_dtype(dt)
        shape = op.out_shape(shape)
        if i == 0 and n_inputs > 1:
            shape = (shape[0], shape[1], n_inputs)
    return states, ChainState(dt, *shape)


def signature(ops: List[Op], in_dtype: DType, in_shape: Tuple[int, int, int],
              arch: str, n_inputs: int = 1) -> str:
    states, out_st = plan(ops, in_dtype, in_shape, n_inputs)
    toks = [op.token(st) for op, st in zip(ops, states)]
    # planes affect Ptr kind (2D vs Tensor) -> part of the type signature
    return (f"arch={arch};cg={CODEGEN_VERSION};in={in_dtype}p{in_shape[2]}x{n_inputs};out={out_st.dtype}"
            f"p{out_st.planes};chain=" + "|".join(toks))


def collect_params(ops: List[Op], in_dtype: DType,
                   in_shape: Tuple[int, int, int],
                   n_inputs: int = 1) -> List[float]:
    states, _ = plan(ops, in_dtype, in_shape, n_inputs)
    out: List[float] = []
    for op, st in zip(ops, states):
        if hasattr(op, "bind"):
            op.bind(st.dtype)
        out.extend(op.values)
    return out


def generate_cu(ops: List[Op], in_dtype: DType,
                in_shape: Tuple[int, int, int], n_inputs: int = 1,
                target: str = "gpu", thread_fusion: bool = False) -> str:
    states, out_st = plan(ops, in_dtype, in_shape, n_inputs)
    in_st = states[0]

    # emit build() expressions for everything except read/write. The read
    # and write IOps construct their OWN host-side IO objects via
    # emit_read()/emit_write() (the IO is part of the IOp, mirroring how the
    # C++ side carries the buffer inside the read/write IOp type) instead of
    # a closed if/elif ladder here.
    iop_exprs = []
    batch_fuse_head = False  # batch ReadBack ops need explicit fuse() w/ read
    fuse_with_read = None    # op that consumes the read expr (BorderReader)
    pbase = 0
    pbase_by_op = []
    for op, st in zip(ops, states):
        if hasattr(op, "bind"):
            op.bind(st.dtype)
        pbase_by_op.append(pbase)
        if op.role not in (READ, WRITE):
            if getattr(op, "_fuse_with_read", False):
                if fuse_with_read is not None:
                    raise ValueError("only one read-fusing op per chain")
                fuse_with_read = (op, st, pbase)
                iop_exprs.append(None)  # placeholder, filled below
            else:
                iop_exprs.append(op.cpp(st, pbase))
                if getattr(op, "_batch", False):
                    batch_fuse_head = True
        pbase += len(op.values)

    read_op, write_op = ops[0], ops[-1]
    MEM = "MemType::Device" if target == "gpu" else "MemType::Host"
    DPP = ("TransformDPP<ParArch::GPU_NVIDIA, TF::ENABLED>" if thread_fusion
           else "TransformDPP<>")

    # ---- IO construction (host): delegated to the read/write IOps ----
    in_decl, read_expr = read_op.emit_read(in_st, MEM, n_inputs)
    out_decl, write_expr = write_op.emit_write(out_st, MEM, pbase_by_op[-1])

    if fuse_with_read is not None:
        # Per Oscar: the BorderReader takes the Image/Ptr2D read IOp as its
        # backIOp: BorderReader<BT>::build(readIOp[, value]). The combined
        # expression REPLACES the read at the head of the chain.
        op_f, st_f, pb_f = fuse_with_read
        read_expr = op_f.cpp_with_read(st_f, pb_f, read_expr)
        iop_exprs = [e for e in iop_exprs if e is not None]

    if batch_fuse_head and iop_exprs:
        # Batch ReadBack (e.g. Crop<>::build(std::array<Rect,B>)) returns a
        # Read<BatchRead<...>> whose InstanceType is ReadType, so BackFuser's
        # idxFirstNonBack misses it unless another ReadBack follows. Fuse it
        # with the read explicitly via fk::fuse (operator&), exactly like the
        # library does internally.
        head = f"fuse({read_expr}, {iop_exprs[0]})"
        all_iops = ",\n            ".join([head, *iop_exprs[1:], write_expr])
    else:
        all_iops = ",\n            ".join([read_expr, *iop_exprs, write_expr])

    if target == "cpu":
        return f"""// AUTO-GENERATED by fkl-python codegen v2 (CPU backend). Plain C++ TU.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

extern "C" {{
struct FklDims {{
    int in_w, in_h, in_planes;
    int out_w, out_h, out_planes;
}};

void fkl_entry(void* d_in, void* d_out, const FklDims* dims,
               const float* params, void* ext_stream)
{{
    (void)ext_stream;  // CPU executor is synchronous
    {in_decl}
    {out_decl}

    Stream_<ParArch::CPU> stream;
    executeOperations<TransformDPP<ParArch::CPU>>(stream,
        {all_iops});
    stream.sync();
}}
}} // extern "C"
"""

    return f"""// AUTO-GENERATED by fkl-python codegen v2. Single-step host+device TU.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

extern "C" {{
struct FklDims {{
    int in_w, in_h, in_planes;
    int out_w, out_h, out_planes;
}};

void fkl_entry(void* d_in, void* d_out, const FklDims* dims,
               const float* params, void* ext_stream)
{{
    {in_decl}
    {out_decl}

    if (ext_stream != nullptr) {{
        Stream stream(reinterpret_cast<cudaStream_t>(ext_stream));
        executeOperations<{DPP}>(stream,
            {all_iops});
        // caller owns the stream: stay async
    }} else {{
        static Stream stream;  // persistent: avoid create/destroy per call
        executeOperations<{DPP}>(stream,
            {all_iops});
        stream.sync();
    }}
}}
}} // extern "C"
"""
