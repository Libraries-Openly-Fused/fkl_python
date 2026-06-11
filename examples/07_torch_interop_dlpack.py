"""Example 07 — Interop: torch/cupy in, DLPack out, external CUDA streams.

fkl accepts ANY object with __cuda_array_interface__ as input (cuda torch
tensors, cupy arrays, numba device arrays, fkl.DeviceBuffer). Outputs are
DeviceBuffers exposing BOTH __cuda_array_interface__ and DLPack, so they
flow back into frameworks with zero copies.

    x  = torch.rand(480, 640, 3, device="cuda")        # torch owns input
    y  = pipe(x, stream=torch.cuda.current_stream())   # async on torch stream
    t  = torch.from_dlpack(y)                          # zero-copy back

This example runs without torch installed: it validates the same contract
at the DLPack ABI level (pointer identity = zero-copy proof).
"""
import ctypes
import fkl
from fkl.tensor import _DLManagedTensor
from _util import gpu_image_f32, to_floats

W, H = 320, 240
x = gpu_image_f32([float(i % 251) for i in range(W * H)], W, H)

pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.Sub(1.0),
                   fkl.TensorWrite())
y = pipe(x)

# ---- consume via __cuda_array_interface__ (cupy/numba do this) ---------------
cai = y.__cuda_array_interface__
print(f"OK  __cuda_array_interface__: shape={cai['shape']} typestr={cai['typestr']}")

# ---- consume via DLPack (torch.from_dlpack does this) ------------------------
device = y.__dlpack_device__()
capsule = y.__dlpack__()

get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
get_ptr.restype = ctypes.c_void_p
get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
mt = ctypes.cast(get_ptr(capsule, b"dltensor"),
                 ctypes.POINTER(_DLManagedTensor)).contents

assert device == (2, 0), "kDLCUDA device"
assert mt.dl_tensor.data == cai["data"][0], "DLPack ptr == CAI ptr (zero-copy)"
shape = [mt.dl_tensor.shape[i] for i in range(mt.dl_tensor.ndim)]
print(f"OK  DLPack: device=kDLCUDA shape={shape} "
      f"ptr=0x{mt.dl_tensor.data:x} (same memory, no copy)")
mt.deleter(ctypes.pointer(mt))             # consumer frees when done
print("OK  DLPack deleter released the buffer (consumer-owned lifetime)")

# ---- external streams ---------------------------------------------------------
# pipe(x, stream=<torch stream | cupy stream | raw handle int>) makes the
# launch ASYNC on that stream and skips the internal sync: you own ordering.
y2 = pipe(gpu_image_f32([1.0] * (W * H), W, H), stream=0)  # 0 = default stream
print("OK  ran async on an externally-provided stream (no internal sync)")

# ---- preallocated outputs ------------------------------------------------------
# In tight loops, reuse one output buffer instead of allocating per call:
src = gpu_image_f32([3.0] * (W * H), W, H)
dst = fkl.DeviceBuffer(W, H, "float32")
for _ in range(10):
    pipe(src, out=dst)                     # zero allocations per iteration
print(f"OK  10 calls into a preallocated output: out[0]={to_floats(dst, 1)[0]}")
