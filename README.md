# fkl-python

Zero-overhead Python front-end for the **Fused Kernel Library** (FKL).

## The idea (and why NVRTC alone does not work)

In FKL the fusion is **host code at compile time**: `build()` constructs the
IOps, `BackFuser` performs Backwards Vertical Fusion, and `executeOperations`
infers the grid and launches. Only `exec()` is device code.

NVRTC compiles **only device code**, so it can compile `exec()` but throws away
the entire methodology (build + BackFuser + grid inference + param packing).
Using NVRTC would mean reimplementing all of that in Python and maintaining two
copies forever. That is the opposite of "one standard C++ implementation".

So this front-end does what works (and what Oscar concluded in Nov 2025):

> compile host+device together in **one step** into a `.so`, expose a C-ABI
> `fkl_entry`, and drive it from Python with ctypes.

```
compose(ops...)  ->  symbolic chain (lazy IOps, no kernel yet)
                 ->  signature string (op types + dtype + arch)
                 ->  disk cache lookup  (~/.cache/fkl/<hash>.so)
                       HIT : reuse compiled .so          (~2 ms, just dlopen)
                       MISS: codegen .cu -> compile once  (~3 s, then cached forever)
kernel(x, y)     ->  HOT PATH: zero-copy device ptr + ONE ctypes call. No sync.
```

Python is only on the **cold path** (compose + first compile). The hot path is a
single C call into a cached, fully-fused kernel — no per-element Python, no copies.

## Backends

- **clang** (`-x cuda`): single step, full device std lib, Apache-2.0, no NVIDIA
  EULA inside your bundle. Preferred when clang's CUDA support matches your toolkit.
- **nvcc**: automatic fallback when clang's CUDA support lags the installed CUDA
  (e.g. clang-21 vs CUDA 13.3). You never have to choose — it falls back silently.

## Usage

```python
import torch, fkl

pipe = fkl.compose(
    fkl.TensorRead(),
    fkl.Mul(2.0),     # fused
    fkl.Add(1.0),     # fused   -> ONE kernel, one DRAM read + one write
    fkl.Mul(3.0),     # fused
    fkl.TensorWrite(),
)

x = torch.arange(16, device="cuda", dtype=torch.float32)
out = pipe(x, stream=torch.cuda.current_stream())   # zero-copy, async
```

No torch? Use the built-in dependency-free `DeviceBuffer` (CUDA driver via ctypes).

## Setup

```bash
export FKL_INCLUDE=/path/to/FusedKernelLibrary/include
export FKL_ROOT=/path/to/FusedKernelLibrary
export FKL_ARCH=sm_120          # your GPU arch
pip install -e .
python tests/test_e2e.py
```

## Status

Verified end-to-end on RTX PRO 6000 Blackwell (sm_120), CUDA 13.3, on BOTH
backends (clang with auto-shims, nvcc):

| suite                            | cases | clang | nvcc |
|----------------------------------|-------|-------|------|
| test_vertical_fusion             | 11    | PASS  | PASS |
| test_backward_vertical_fusion    | 9     | PASS  | PASS |
| test_horizontal_fusion           | 7     | PASS  | PASS |
| test_batch_divergent_hf          | 12    | PASS  | PASS |
| test_operations (per-op vs CPU)  | 21    | PASS  | PASS |
| test_matrix                      | 12    | PASS  | PASS |
| test_roi_use_case                | 3     | PASS  | PASS |
| test_warping_splitwrite          | 8     | PASS  | PASS |
| test_niche_ops                   | 9     | PASS  | PASS |
| test_dlpack                      | 9     | PASS  | PASS |
| test_circular_tensor             | 10    | PASS  | PASS |
| test_e2e (timing/cache)          | 1     | PASS  | PASS |

112 checks per backend. Steady-state hot launch ~70 µs/call; cache-hit
compose ~0 ms (lazy); cold compile ~1-3 s once per chain signature.

## Temporal video: CircularTensor

`fkl.CircularTensor(w, h, batch=N, ...)` keeps a rolling window of the last
N frames ON the GPU. Each `update(frame, ops=[...])` preprocesses the frame
AND rotates the window in ONE fused kernel (the paper's CircularTensor
mechanism, built on Divergent HF):

```python
ct = fkl.CircularTensor(640, 480, batch=4, dtype="uint8", channels=3,
                        layout="planar", out_dtype="float32")
for frame in camera:
    ct.update(frame, ops=[fkl.Cast("float32"), fkl.Div(255.0)])
    window = ct.snapshot()       # (4*C planes, H, W) -> torch.from_dlpack
```

## Interop

- Input: anything with `__cuda_array_interface__` (torch cuda tensors,
  cupy, numba, fkl.DeviceBuffer). C-contiguous required.
- Output: `DeviceBuffer` exposes both `__cuda_array_interface__` and
  DLPack (`__dlpack__`/`__dlpack_device__`), so `torch.from_dlpack(out)` /
  `cupy.from_dlpack(out)` reuse the device memory with zero copies.

## Upstream contributions

- Issues filed: [#244](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary/issues/244)
  (ColorConversion FusedOperation aliases ill-formed),
  [#245](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary/issues/245)
  (Divergent Executor fuse_back forwarding-reference bug).
- Fix PR: [#248](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary/pull/248)
  with regression utests; full in-tree suite 66/66 on CUDA 13.3 / sm_120.

## Fusion modes from Python (all four from the paper)

- Vertical Fusion: any compute chain -> one kernel (StaticLoop for huge chains).
- Backwards Vertical Fusion: Crop/Resize/Warping fused INTO the read by BackFuser.
- Horizontal Fusion: `Crop([(x,y,w,h), ...])` (batch crops of one image) or
  `pipe([img0, img1, ...])` (batch of separate same-size images via BatchRead).
- Divergent Horizontal Fusion: `compose_divergent(plane_map, chain1, chain2, ...)`
  -> one kernel where different thread-planes run different fused sequences.

## In-repo skills

`skills/fkl-python-usage`, `skills/fkl-python-extending`,
`skills/fkl-python-testing` — agent-ready guides for using, extending and
testing the package.

## Examples

`examples/` — 8 runnable, dependency-free scripts (see `examples/README.md`):
vertical fusion basics, single-kernel DNN preprocessing, multi-ROI batch
(HF), multi-camera + Divergent HF, color pipelines, warping + border
policies, torch/DLPack interop, and a fused-vs-unfused benchmark
(~5x at 1080p for a 6-op chain).

## Current scope / next steps

- TernaryType ops generic path, CircularBatch ops, Divergent HF
  (per-plane different op sequences), warping/deinterlace descriptors,
  DLPack __dlpack__ export on DeviceBuffer, wheels + CI.
