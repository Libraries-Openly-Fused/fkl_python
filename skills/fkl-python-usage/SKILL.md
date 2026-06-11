---
name: fkl-python-usage
description: Use the fkl-python front-end to compose and run fused GPU kernels from Python with zero overhead. Covers compose(), all operations (arithmetic, casts, vector ops, color conversion, Crop/Resize), Vertical Fusion, Backwards Vertical Fusion, Horizontal Fusion (batch crops), torch/cupy zero-copy interop, and the compile cache. Use when writing Python code that uses the fkl package or when asked how to build a fused pipeline from Python.
---

# fkl-python: composing fused GPU kernels from Python

## Mental model (read this first)

fkl-python is a LAZY front-end over the C++ Fused Kernel Library (FKL):

1. `fkl.compose(...)` builds a symbolic op chain. NO kernel exists yet.
2. First call with a concrete input compiles ONE .so for that chain's
   *type signature* (ops + dtype + layout + arch) and caches it forever
   under `~/.cache/fkl/`.
3. Every subsequent call is a single ctypes call into the cached
   `extern "C" fkl_entry`. Zero copies, no Python per-element work.

CRITICAL: runtime VALUES (mul factors, crop rects, resize sizes) travel in
a params[] array -> changing values NEVER recompiles. Only changing the
chain structure, dtypes or layouts triggers a new compile.

The fusion itself (VF + HF + BVF via BackFuser) happens in C++ at compile
time. Python never reimplements it.

## Setup

```bash
export FKL_INCLUDE=/path/to/FusedKernelLibrary/include
export FKL_ROOT=/path/to/FusedKernelLibrary
export CUDA_HOME=/usr/local/cuda
export FKL_ARCH=sm_120        # match your GPU
export PYTHONPATH=/path/to/fkl_python
```

## Basic usage (Vertical Fusion)

```python
import fkl

pipe = fkl.compose(
    fkl.TensorRead(),      # chain MUST start with a read
    fkl.Mul(2.0),          # all compute ops fuse into ONE kernel,
    fkl.Add(1.0),          # intermediates stay in registers
    fkl.TensorWrite(),     # chain MUST end with a write
)
out = pipe(x)              # x: cuda torch tensor / cupy array / DeviceBuffer
out = pipe(x, out=y)       # reuse a preallocated output (fastest)
out = pipe(x, stream=torch.cuda.current_stream())  # async on torch stream
```

With an external stream the call does NOT sync (caller owns the stream).
Without one, an internal static stream is used and synced before return.

## Operation catalog

Memory:    TensorRead(), TensorWrite(), TensorSplit() [packed->planar],
           SplitWrite() [packed->planar via per-channel Ptr2D planes],
           TensorPack(ch) [planar CHW -> packed read, inverse of TensorSplit],
           TensorTSplit() [T3D transposed layout: C outermost],
           ReadSet(value, w, h) [constant generator, zero DRAM reads]
Arith:     Mul(v) Add(v) Sub(v) Div(v) Max(v) Min(v)
           - v scalar broadcasts over channels; v tuple = per-channel
Logic:     IsEven() [int->bool], VectorAnd() [boolN->bool]
Casts:     Cast("float32"), SaturateCast("uint8"), SaturateFloat(),
           Saturate(lo, hi)
Vector:    VectorReduce("Add"), Discard(keep_n), AddLast(val),
           VectorReorder(2,1,0),   # compile-time channel shuffle
           VectorReorderRT(2,1,0)  # RUNTIME shuffle: perm in params[],
                                   # changing it reuses the same .so
Algebraic: MxVFloat3(3x3_matrix)  # needs float3 chain state
Color:     ColorConversion("RGB2BGR" | "RGB2GRAY" | "RGB2RGBA" | ...)
Image:     Crop(x,y,w,h), Crop([(x,y,w,h),...]),
           Resize(w,h, interp="linear", aspect_ratio="ignore"|"preserve"|
                  "preserve_left"|"preserve_even", background=val_or_tuple),
           Warping([(a,b,c),(d,e,f)], dst_w, dst_h)        # affine 2x3
           Warping(3x3_matrix, dst_w, dst_h)               # perspective
           BorderReader("constant"|"replicate"|"reflect"|"wrap"|
                        "reflect101", value=...)  # OOB policy on the read;
                        # place right after TensorRead(), before Crop
           Deinterlace("blend"|"linear")
Control:   StaticLoop(fkl.Add(1.0), 64)  # N fused reps, 1 param slot

NOT wrapped: fk::Equal (InputType is Tuple<I1,I2> = two data streams;
linear chains carry one value, so it is not representable in compose());
CircularBatch*/CircularTensor* (temporal video state machine — needs a
stateful Python object, not a chain op).

## DLPack export (zero-copy into torch/cupy/jax)

DeviceBuffer implements __dlpack__/__dlpack_device__:
```python
out = pipe(x)                          # fkl DeviceBuffer on GPU
t = torch.from_dlpack(out)             # SAME device memory, no copy
```
Ownership transfers to the consumer (the DLPack deleter frees the CUDA
allocation; fkl's __del__ won't double-free). Shapes map as
(planes, H, W[, C]) like __cuda_array_interface__.

## The canonical ROI use case

A vector of ROIs cropped from one image, processed, written to a Tensor —
HF + BVF + VF in ONE kernel (mirror of the C++ README example):

```python
rois = [(30,12,60,40), (40,12,60,40), (53,27,60,40)]   # (x, y, w, h)
pipe = fkl.compose(
    fkl.TensorRead(),                  # PerThreadRead<_2D, uchar3>
    fkl.Crop(rois),                    # Crop<>::build(std::array<Rect,B>)
    fkl.Resize(16, 16),                # per-plane resize of each ROI
    fkl.Mul((2.0, 2.0, 2.0)),          # Mul<float3>
    fkl.Sub((128.0,)*3),               # Sub<float3>
    fkl.SaturateCast("uint8"),
    fkl.TensorWrite(),                 # Tensor with B=3 planes
)
out = pipe(image_u8_hxwx3)             # shape (3, 16, 16, 3)
```
Changing the rect VALUES later reuses the same kernel; changing the NUMBER
of ROIs (B) is a new C++ type -> compiles once more, then cached.

## dtypes and vector pixel types

Input arrays map shape -> FKL type: (H, W) float32 -> float;
(H, W, 3) uint8 -> uchar3; (planes, H, W[, C]) -> 3D Tensor.
A trailing dim of 2..4 is ALWAYS folded into a vector pixel type.
Arrays must be C-contiguous (call .contiguous() on torch tensors).

## The four fusion modes

VERTICAL (VF): any sequence of compute ops between read and write.

BACKWARDS VERTICAL (BVF): Crop/Resize/Warping are ReadBack ops. Put them
right after TensorRead(); FKL's BackFuser fuses them INTO the read at C++
compile time. Stack them: Crop then Resize works (crop region resized).

HORIZONTAL (HF), two flavours:
1. Batch crops from ONE image: pass a LIST of rects to Crop:
```python
pipe = fkl.compose(
    fkl.TensorRead(),
    fkl.Crop([(0,0,64,64), (100,50,64,64), (10,80,64,64)]),  # B=3 planes
    fkl.Resize(32, 32),
    fkl.Mul(1/255.0),
    fkl.TensorWrite(),       # output is a 3-plane Tensor
)
out = pipe(image)
```
2. Batch of SEPARATE same-size images: pass a LIST of arrays to the call:
```python
pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite())
out = pipe([img0, img1, img2])     # ONE kernel, 3 thread-planes -> Tensor(3)
```
B (batch size) is part of the C++ type: each distinct B compiles once.

DIVERGENT HF (paper's 4th technique): different planes run DIFFERENT
fused sequences, selected by blockIdx.z, in ONE kernel:
```python
pipe = fkl.compose_divergent(
    [1, 2, 2],                                            # plane -> seq (1-based)
    [fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite()],  # sequence 1
    [fkl.TensorRead(), fkl.Add(5.0), fkl.TensorWrite()],  # sequence 2
)
out = pipe([img0, img1, img2])
```
All sequences must produce the same output dtype/shape (they share the
output Tensor). Chains can have different lengths/ops.

## Debugging

```python
print(pipe.source_for("float32", (640, 480, 1)))  # generated C++
fkl.get_backend().kind          # 'clang' or 'nvcc'
fkl.set_backend("nvcc")         # force a backend
fkl.clear_cache()               # nuke ~/.cache/fkl
```
On compile errors the offending .cu is kept in ~/.cache/fkl/ - read it.

## Pitfalls

1. Chain must be TensorRead() first, TensorWrite()/TensorSplit() last.
2. Non-contiguous tensors are rejected (zero-copy needs contiguity).
3. MxVFloat3 requires the chain state to be exactly float3.
4. ColorConversion("BGR2GRAY"/"BGRA2GRAY") is decomposed into
   VectorReorder + RGB2GRAY because the FKL main-branch alias for these
   expands to a FusedOperation without ::build (upstream bug).
5. Resize with aspect_ratio != "ignore" REQUIRES background=...
6. First call per signature takes seconds (compile); benchmark AFTER it.
7. Vector value ops bind channel count at codegen: Mul((1,2,3)) on a
   single-channel chain raises (3 channels vs 1).
