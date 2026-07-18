---
name: fkl-python-testing
description: Run and write tests for fkl-python (fused GPU kernel front-end). Covers the test harness, per-fusion-mode suites, CPU reference patterns, backend matrix runs (nvcc + clang), and debugging compile failures from cached .cu files. Use when running fkl_python tests, adding tests for a new op, or diagnosing test failures.
---

# Testing fkl-python

## Running

```bash
cd fkl_python/tests
export PYTHONPATH=/path/to/fkl_python:. CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

python3 test_vertical_fusion.py            # VF: op chains, dtypes, cache
python3 test_backward_vertical_fusion.py   # BVF: Crop/Resize fused into read
python3 test_horizontal_fusion.py          # HF: batch crops -> planes
python3 test_operations.py                 # every op vs CPU reference
python3 test_matrix.py                     # legacy broad matrix
python3 test_e2e.py                        # timing + cache behaviour
python3 test_bound_compose.py              # compose-time binding + from_ptr
```

Each suite exits 0/1 and prints [PASS]/[FAIL] per case. First run compiles
each unique signature (~2-3 s each); later runs hit ~/.cache/fkl (ms).

## Backend matrix (always test BOTH before claiming victory)

```python
import fkl
fkl.set_backend("nvcc")   # or "clang"
```
Wipe state between backend runs: `rm -f ~/.cache/fkl/fkl_* ~/.cache/fkl/.backend`.
The .so cache key includes the backend, but .backend pins auto-selection.

## Writing a new test

Use tests/harness.py - dependency-free (no numpy/torch needed):

```python
from harness import dev_f32, dev_u8, unf32, unu8, check, check_true, run
import fkl

def t_my_case():
    W, H = 16, 8
    src = [float((y * W + x) % 97) for y in range(H) for x in range(W)]
    out = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite())(
        dev_f32(src, W, H))
    check("my case", unf32(out.copy_to_host(), W * H), [v * 2 for v in src])

if __name__ == "__main__":
    run([t_my_case], "my-suite")
```

Conventions that catch real bugs:
- Non-pow2 sizes (33x17) to catch pitch errors.
- Position-encoded pixels `y*W+x` so wrong reads show WHERE they read.
- Independent single-op reference pipelines to validate batch planes
  (see t_batch_crop_resize: plane0 vs single-crop pipeline).
- Tolerances: exact ops tol<=1e-3; interpolation 0.51; BT.601 color 1.0;
  float64 1e-15.
- uint8 round-trips: reference must round (int(v + 0.5)), not truncate.

## Debugging failures

1. [FAIL] with wrong values -> print got vs expect (check() shows 8).
   Wrong WHERE: read offset bug (pitch/rect). Wrong SCALE: params order.
2. RuntimeError: FKL JIT compile failed -> the .cu path is in the message.
   Iterate directly:
   `nvcc -std=c++20 -arch=sm_120 -shared -Xcompiler -fPIC -I$FKL_INCLUDE <file>.cu -o /tmp/t.so`
3. Param misalignment symptom: values shifted between ops (op N reads
   op N+1's value). Cause: descriptor's `values` length doesn't match the
   params[] indices consumed in `cpp()`. Audit pbase arithmetic.
4. Stale-cache suspicion: `fkl.clear_cache()` and rerun. If it passes only
   after clearing, some VALUE leaked into the C++ instead of params[].
5. Batch/HF failures with operator| errors at data_parallel_patterns.h(77):
   the batch ReadBack wasn't fused with the read - the op must set
   `self._batch = True` so codegen wraps it in fuse().
