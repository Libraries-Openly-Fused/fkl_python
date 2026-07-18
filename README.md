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

## PyTorch-style API: Tensor / Image / pipe

A torch-flavoured layer over the same machinery — for users who think in
tensors and image batches rather than op chains. Everything is zero-copy and
every pipeline still lowers to **one** fused kernel:

```python
import torch, fkl

frames = torch.randint(0, 256, (8, 720, 1280, 3),           # NHWC uint8
                       device="cuda", dtype=torch.uint8)
batch = fkl.ImageBatch(frames)          # zero-copy per-plane views
                                        # (or a list of same-size images)

out = (fkl.pipe(batch)
       .resize((224, 224))              # BVF: fused into the read
       .normalize(MEAN, STD)            # Cast+Sub+Div, in registers
       .to_planar("CHW")                # planar NCHW write
       .run())                          # ONE kernel for all 8 images

net(out.torch())                        # zero-copy (8, 3, 224, 224) cuda tensor
```

- `fkl.Tensor(obj)` wraps torch CUDA tensors, `__cuda_array_interface__`
  objects (cupy/numba) and `DeviceBuffer` without copying; exposes
  `.shape/.dtype/.device`, `__cuda_array_interface__`, DLPack and
  `.torch()`. `fkl.Tensor.empty(shape, dtype)` allocates.
- `fkl.Image` = HWC image semantics (`width/height/channels`, C 1..4);
  `fkl.ImageBatch` = list of same-shape images OR one batched
  `(N, H, W[, C])` tensor. Input layout is `'HWC'`; planar `'CHW'` is an
  OUTPUT layout produced by `.to_planar()` (the existing split ops).
- `fkl.pipe(x)` methods (`crop`, `resize`, `border`, `cvt_color`,
  `normalize`, `mul/add/sub/div`, `cast`, `saturate_cast`, `apply(op)`,
  `to_planar`) each append an existing symbolic op; `.run()` goes through
  `compose()`'s JIT cache — same rules: types compile once, values travel
  in `params[]`. Geometry ops come first (they fuse into the read).
- One-op eager calls: `fkl.F.resize(img, (64, 64))`,
  `fkl.F.cvt_color(img, "BGR2GRAY")`, ... (each a one-op fused pipeline).

Out of scope by design (use torch itself): autograd, broadcasting,
>4-channel images (FKL vector pixels are 1..4 channels), multi-input graphs
(the ABI is single-input/single-output; batches are horizontal fusion).

## Setup

### Plug and play (wheel with vendored headers)

```bash
pip install fkl-python      # once published; or the wheel from GitHub Releases
python -c "import fkl; ..."  # just works: no env vars, no FKL checkout
```

The wheel ships the FKL headers inside (`fkl/_vendor/`, header-only,
Apache-2.0, upstream commit recorded in `VENDOR_INFO.txt`). At first use it
auto-detects your GPU arch (CUDA driver API) and CUDA toolkit, JIT-compiles
the chain, and caches the .so under `~/.cache/fkl`. Requirements on the
machine: an NVIDIA driver + a CUDA toolkit (nvcc) or clang++.

### Development (use your own FKL checkout)

```bash
export FKL_INCLUDE=/path/to/FusedKernelLibrary/include   # overrides vendored
export FKL_ARCH=sm_120                                   # optional override
pip install -e .
python tests/test_e2e.py
```

Resolution order: `FKL_INCLUDE` env var > vendored headers in the wheel >
sibling dev checkout. Arch: `FKL_ARCH` > driver query of GPU 0 > sm_75.
To (re)vendor headers before building a wheel:
`python scripts/vendor_fkl.py --ref LTS-C++17`.

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
| test_circular_tensor             | 21    | PASS  | PASS |
| test_thread_fusion               | 5     | PASS  | PASS |
| test_e2e (timing/cache)          | 1     | PASS  | PASS |

Plus, separately: test_torch_integration (9 checks, real torch 2.12+cu130:
zero-copy both ways, external streams, CircularTensor->torch) and
test_cpu_backend (5 checks, ParArch::CPU with numpy in/out, CPU==GPU
cross-validated).

## CPU backend (no CUDA required)

```python
pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite(),
                   target="cpu")
out = pipe(numpy_array)        # numpy in -> numpy out, ParArch::CPU
```
Same fused chains, FKL's CPU executor, plain C++ .so (clang++; g++ rejects
Stream_<ParArch::CPU>'s ctor spelling). Results cross-validated against
the GPU backend.

## ThreadFusion

```python
pipe = fkl.compose(..., thread_fusion=True)   # opt-in, GPU only
```
**Disabled by default**, matching FKL's own default (`TransformDPP<>` =
`TF::DISABLED`). Per Oscar: ThreadFusion only improves performance in a
small set of cases (wide images, trivial per-pixel chains, bandwidth-bound)
— benchmark YOUR pipeline before enabling it; it is not a general speedup.
When enabled it emits `TransformDPP<GPU_NVIDIA, TF::ENABLED>` and
auto-falls back to scalar for shapes whose row bytes aren't 16-aligned
(external tight-pitch pointers would fault on vectorized loads).

## Multi-GPU

`DeviceBuffer(..., device=N)`, `CircularTensor(..., device=N)`; `compose`
pipelines follow the input tensor's device (torch `cuda:N`, cupy device id,
DeviceBuffer.device). DLPack exports carry the right device id. NOTE: this
box has 1 GPU — device routing is implemented and exercised on device 0;
true multi-GPU runs still need a 2+ GPU machine.

## Known structural limits

- `fk::Equal` and other Tuple-input ops need a multi-source read (a read
  producing `Tuple<A,B>` from two pointers). FKL has no such Read op yet —
  upstream feature candidate, not wrappable from here.
- Divergent HF goes through a direct kernel launch (not the Executor)
  because of upstream issue
  [#250](https://github.com/Libraries-Openly-Fused/FusedKernelLibrary/issues/250)
  (grid.z = sum of sequence z-extents).

128 checks per backend (+ 9 torch + 5 cpu separately). Steady-state hot launch ~70 µs/call; cache-hit
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

`examples/` — 12 runnable scripts, most dependency-free (see
`examples/README.md`): vertical fusion basics, single-kernel DNN
preprocessing, multi-ROI batch (HF), multi-camera + Divergent HF, color
pipelines, warping + border policies, torch/DLPack interop, a
fused-vs-unfused benchmark (~5x at 1080p for a 6-op chain), temporal video
windows, YOLO end-to-end, and the torch-style `pipe()` API.

## Current scope / next steps

- TernaryType ops generic path, CircularBatch ops, Divergent HF
  (per-plane different op sequences), warping/deinterlace descriptors,
  DLPack __dlpack__ export on DeviceBuffer, wheels + CI.
