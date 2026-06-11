"""Example 08 — Performance: cold compile vs hot path, and why fusion wins.

Measures on YOUR GPU:
  1. cold compile (once per chain signature, then cached on disk)
  2. warm re-compose (cache hit)
  3. steady-state launch overhead (single ctypes call)
  4. fused chain vs running each op as its own kernel (DRAM round-trips)
"""
import time
import fkl
from _util import gpu_image_f32, to_floats

W, H = 1920, 1080
x = gpu_image_f32([float(i % 255) for i in range(W * H)], W, H)
dst = fkl.DeviceBuffer(W, H, "float32")

# ---- 1. cold compile ----------------------------------------------------------
ops = lambda: (fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(3.0), fkl.Sub(1.0),
               fkl.Div(2.0), fkl.Max(0.0), fkl.Min(255.0), fkl.TensorWrite())
t0 = time.perf_counter()
pipe = fkl.compose(*ops())
pipe(x, out=dst)                                   # triggers compile
t_cold = time.perf_counter() - t0
print(f"cold first call (compile + dlopen + run): {t_cold*1e3:8.1f} ms")

# ---- 2. warm re-compose ---------------------------------------------------------
t0 = time.perf_counter()
pipe_b = fkl.compose(*ops())
pipe_b(x, out=dst)                                 # same signature: cache hit
t_warm = time.perf_counter() - t0
print(f"new compose, cached signature:            {t_warm*1e3:8.1f} ms")

# ---- 3. steady-state launches ----------------------------------------------------
N = 200
pipe(x, out=dst)
t0 = time.perf_counter()
for _ in range(N):
    pipe(x, out=dst)
t_hot = (time.perf_counter() - t0) / N
print(f"steady-state fused launch (6 ops):        {t_hot*1e6:8.1f} us/call")

# ---- 4. fused vs op-by-op ---------------------------------------------------------
single = [fkl.compose(fkl.TensorRead(), op, fkl.TensorWrite())
          for op in (fkl.Mul(2.0), fkl.Add(3.0), fkl.Sub(1.0),
                     fkl.Div(2.0), fkl.Max(0.0), fkl.Min(255.0))]
tmp = fkl.DeviceBuffer(W, H, "float32")
for p in single:
    p(x, out=tmp)                                  # compile all variants

t0 = time.perf_counter()
for _ in range(N):
    buf = x
    for p in single:                               # 6 kernels, 5 DRAM round-trips
        p(buf, out=tmp)
        buf = tmp
t_unfused = (time.perf_counter() - t0) / N
print(f"same math as 6 separate kernels:          {t_unfused*1e6:8.1f} us/call")
print(f"fusion speedup at {W}x{H}:                {t_unfused/t_hot:8.1f}x")

got = to_floats(dst, 4)
print(f"sanity out[:4] = {[round(v, 2) for v in got]}")
