# fkl-python examples

Self-contained, dependency-free scripts (no numpy/torch needed — inputs are
built with `fkl.DeviceBuffer`). Each one prints `OK` lines and exits 0.

## Run

```bash
export FKL_INCLUDE=/path/to/FusedKernelLibrary/include
export FKL_ROOT=/path/to/FusedKernelLibrary
export CUDA_HOME=/usr/local/cuda
export FKL_ARCH=sm_120                  # match your GPU
export PYTHONPATH=/path/to/fkl_python

cd examples
python3 01_basic_vertical_fusion.py
```

First run of each example compiles its kernels (seconds); re-runs hit the
disk cache in `~/.cache/fkl/` and start instantly.

## Index

| example | what it shows |
|---|---|
| `01_basic_vertical_fusion.py` | compose() basics; 6 ops -> 1 kernel; value changes don't recompile |
| `02_dnn_preprocessing.py` | crop -> resize -> normalize -> CHW planar in ONE kernel (BVF+VF); moving ROI reuses kernel |
| `03_multi_roi_batch.py` | Horizontal Fusion: list of rects -> N ROIs in one kernel (detection->classification ingest) |
| `04_multi_camera_divergent.py` | batch of separate images (list input) + Divergent HF: per-plane different op sequences |
| `05_color_pipelines.py` | ColorConversion, compile-time vs runtime channel reorder, alpha add/strip, 3x3 sepia matrix |
| `06_warping_and_borders.py` | affine rotation / perspective warp (animatable, no recompile), border policies on OOB crops |
| `07_torch_interop_dlpack.py` | torch/cupy inputs, DLPack zero-copy outputs, external streams, preallocated outputs |
| `08_performance.py` | cold compile vs cache hit vs hot launch; fused chain vs 6 separate kernels |

## The one-page mental model

```python
pipe = fkl.compose(fkl.TensorRead(), <ops...>, fkl.TensorWrite())  # lazy
out  = pipe(x)            # 1st call: JIT one .so for (ops, dtype, layout)
out  = pipe(x2)           # later calls: single ctypes call, ~70 us
```

- TYPES (op list, dtype, channels, batch size) define the kernel -> compile.
- VALUES (factors, rects, matrices, sizes) travel in `params[]` -> never
  recompile.
- Lists = Horizontal Fusion: `Crop([r1, r2, ...])` or `pipe([img1, img2])`.
- `compose_divergent(plane_map, chain1, chain2)` = Divergent HF.
