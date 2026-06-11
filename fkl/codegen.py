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
                in_shape: Tuple[int, int, int], n_inputs: int = 1) -> str:
    states, out_st = plan(ops, in_dtype, in_shape, n_inputs)
    in_st = states[0]

    # emit build() expressions for everything except read/write (those are
    # constructed from the IO pointers below)
    iop_exprs = []
    batch_fuse_head = False  # batch ReadBack ops need explicit fuse() w/ read
    fuse_with_read = None    # op that consumes the read expr (BorderReader)
    pbase = 0
    for op, st in zip(ops, states):
        if hasattr(op, "bind"):
            op.bind(st.dtype)
        nvals = len(op.values)
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
        pbase += nvals

    read_op, write_op = ops[0], ops[-1]
    in_t, out_t = in_st.dtype.ctype, out_st.dtype.ctype

    # ---- IO construction (host) ----
    if read_op.name == "ReadSet":
        # constant generator: no input pointer at all
        in_decl = "// ReadSet: no DRAM input"
        read_expr = ops[0].cpp(in_st, 0)
    elif read_op.name == "TensorPack":
        # planar (CHW) input read back as packed vector pixels
        base_in = in_st.dtype.base_ctype
        packed_t = ops[0].out_dtype(in_st.dtype).ctype
        in_decl = (f"Tensor<{base_in}> input(({base_in}*)d_in, (uint)dims->in_w, "
                   f"(uint)dims->in_h, (uint)dims->in_planes, "
                   f"(uint){ops[0]._ch}, MemType::Device);")
        read_expr = f"TensorPack<{packed_t}>::build(input)"
    elif n_inputs > 1:
        # HF over a batch of images: d_in is void** (array of device ptrs).
        # std::array<Ptr2D<T>, B> -> PerThreadRead<BatchRead> inside FKL.
        elems = ", ".join(
            f"Ptr2D<{in_t}>((({in_t}**)d_in)[{i}], (uint)dims->in_w, "
            f"(uint)dims->in_h, (uint)(dims->in_w * sizeof({in_t})), MemType::Device)"
            for i in range(n_inputs))
        in_decl = (f"const std::array<Ptr2D<{in_t}>, {n_inputs}> "
                   f"input{{ {elems} }};")
        read_expr = f"PerThreadRead<ND::_2D, {in_t}>::build(input)"
    elif in_st.planes > 1:
        in_decl = (f"Tensor<{in_t}> input(({in_t}*)d_in, (uint)dims->in_w, "
                   f"(uint)dims->in_h, (uint)dims->in_planes, 1, MemType::Device);")
        read_expr = f"TensorRead<{in_t}>::build(input)"
    else:
        in_decl = (f"Ptr2D<{in_t}> input(({in_t}*)d_in, (uint)dims->in_w, "
                   f"(uint)dims->in_h, (uint)(dims->in_w * sizeof({in_t})), "
                   f"MemType::Device);")
        read_expr = f"PerThreadRead<ND::_2D, {in_t}>::build(input)"

    if write_op.name == "TensorSplit":
        base_t = out_st.dtype.base_ctype
        # planes = batch (thread.z); color_planes = channels (split offsets)
        out_decl = (f"Tensor<{base_t}> output(({base_t}*)d_out, (uint)dims->out_w, "
                    f"(uint)dims->out_h, (uint)dims->out_planes, "
                    f"(uint){out_st.dtype.channels}, MemType::Device);")
        write_expr = f"TensorSplit<{out_t}>::build(output)"
    elif write_op.name == "TensorTSplit":
        base_t = out_st.dtype.base_ctype
        ch = out_st.dtype.channels
        # TensorT's (data, w, h, planes, cp) ctor leaves pitches at 0 (they
        # are only filled by h_malloc_init on allocation). For an external
        # pointer, build the RawPtr<T3D> with explicit pitches and pass it
        # straight to build(params).
        pitch = f"(uint)(dims->out_w * sizeof({base_t}))"
        out_decl = (
            f"PtrDims<ND::T3D> outDims((uint)dims->out_w, (uint)dims->out_h, "
            f"(uint)dims->out_planes, (uint){ch});\n"
            f"    outDims.pitch = {pitch};\n"
            f"    outDims.plane_pitch = outDims.pitch * outDims.height;\n"
            f"    outDims.color_planes_pitch = outDims.plane_pitch * outDims.planes;\n"
            f"    RawPtr<ND::T3D, {base_t}> outRaw{{ ({base_t}*)d_out, outDims }};")
        write_expr = f"TensorTSplit<{out_t}>::build(outRaw)"
    elif write_op.name == "SplitWrite":
        base_t = out_st.dtype.base_ctype
        ch = out_st.dtype.channels
        plane = f"(dims->out_w * dims->out_h)"
        ptrs = ", ".join(
            f"Ptr2D<{base_t}>((({base_t}*)d_out) + {c} * {plane}, "
            f"(uint)dims->out_w, (uint)dims->out_h, "
            f"(uint)(dims->out_w * sizeof({base_t})), MemType::Device)"
            for c in range(ch))
        out_decl = f"const std::vector<Ptr2D<{base_t}>> output{{ {ptrs} }};"
        write_expr = f"SplitWrite<ND::_2D, {out_t}>::build(output)"
    elif out_st.planes > 1:
        out_decl = (f"Tensor<{out_t}> output(({out_t}*)d_out, (uint)dims->out_w, "
                    f"(uint)dims->out_h, (uint)dims->out_planes, 1, MemType::Device);")
        write_expr = f"TensorWrite<{out_t}>::build(output)"
    else:
        out_decl = (f"Ptr2D<{out_t}> output(({out_t}*)d_out, (uint)dims->out_w, "
                    f"(uint)dims->out_h, (uint)(dims->out_w * sizeof({out_t})), "
                    f"MemType::Device);")
        write_expr = f"PerThreadWrite<ND::_2D, {out_t}>::build(output)"

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
        executeOperations<TransformDPP<>>(stream,
            {all_iops});
        // caller owns the stream: stay async
    }} else {{
        static Stream stream;  // persistent: avoid create/destroy per call
        executeOperations<TransformDPP<>>(stream,
            {all_iops});
        stream.sync();
    }}
}}
}} // extern "C"
"""
