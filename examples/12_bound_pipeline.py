"""Example 12 — Compose-time buffer binding (argument-free kernels).

Bind the input/output buffers directly on the read/write ops:

    k = fkl.compose(fkl.TensorRead(x), <ops...>, fkl.TensorWrite(out))
    k()          # no arguments: reads x, writes out

Key ideas demonstrated:
  * Binding makes compose() EAGER: the kernel compiles inside compose()
    (dtype/shape come from the bound buffers), so it is ready-to-run
    before the first call — no first-call compile hiccup in the hot loop.
  * Bound buffers are call DEFAULTS: k(y) / k(y, out=z) override them,
    validated against the compiled signature (clear errors on mismatch).
  * DeviceBuffer.from_ptr(ptr, shape, dtype) wraps a RAW device pointer
    (non-owning) so C-style integrations bind pointers into the chain.
"""
import struct
import fkl
from _util import gpu_image_f32, to_floats

W, H = 64, 32
src = [float(i % 100) for i in range(W * H)]

# ---- 1. bind input AND output: a fully wired, argument-free kernel --------
x = gpu_image_f32(src, W, H)
out = fkl.DeviceBuffer(W, H, "float32")

k = fkl.compose(
    fkl.TensorRead(x),        # <- input bound here
    fkl.Mul(2.0),
    fkl.Add(10.0),
    fkl.TensorWrite(out),     # <- output bound here
)
# compose() already compiled the kernel (eager because the read is bound):
assert len(k._variants) == 1, "expected eager compilation at compose() time"
print("OK  compose() compiled eagerly: kernel ready before the first call")

k()                           # the whole pipeline, no arguments
got = to_floats(out, W * H)
assert all(abs(g - (v * 2 + 10)) < 1e-5 for g, v in zip(got, src))
print(f"OK  k() ran argument-free: out[:4] = {got[:4]}")

# The binding is by POINTER: update the input in place and just call again.
x.copy_from_host(struct.pack(f"{W * H}f", *[v + 1 for v in src]))
k()
got = to_floats(out, W * H)
assert abs(got[0] - ((src[0] + 1) * 2 + 10)) < 1e-5
print("OK  in-place input update + k(): same kernel, new data")

# ---- 2. overrides still work (validated against the compiled signature) ---
y = gpu_image_f32([v * 3 for v in src], W, H)
k(y)                          # same dtype/shape: reuses the compiled kernel
assert abs(to_floats(out, 1)[0] - (src[0] * 3 * 2 + 10)) < 1e-4
print("OK  k(y) overrode the bound input, still wrote the bound output")

try:
    k(gpu_image_f32([0.0] * 16, 4, 4))   # wrong shape -> clear error
    raise SystemExit("expected a ValueError!")
except ValueError as e:
    print(f"OK  mismatching override rejected: {str(e)[:56]}...")

# ---- 3. raw pointers: DeviceBuffer.from_ptr (non-owning) -------------------
# Wrap an EXTERNAL device pointer (here: another buffer's own pointer, the
# way a C library would hand you cudaMalloc'd memory). The wrapper never
# frees it — the caller keeps ownership and must keep it alive.
dst = fkl.DeviceBuffer(W, H, "float32")
dst_view = fkl.DeviceBuffer.from_ptr(dst.ptr, (H, W), "float32")

k2 = fkl.compose(fkl.TensorRead(x), fkl.Div(2.0), fkl.TensorWrite(dst_view))
k2()
assert abs(to_floats(dst, 1)[0] - (src[0] + 1) / 2) < 1e-5
del dst_view                  # non-owning: dst's memory is untouched
assert abs(to_floats(dst, 1)[0] - (src[0] + 1) / 2) < 1e-5
print("OK  from_ptr: raw device pointer bound as the kernel's destination")

print("\nAll bound-pipeline demos passed.")
