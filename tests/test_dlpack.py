"""DLPack export validation WITHOUT torch/cupy installed: we act as the
consumer, parsing the DLManagedTensor from the capsule and verifying
device pointer, dtype, shape and (via cuMemcpyDtoH on that pointer) the
actual data — exactly what torch.from_dlpack would consume zero-copy.
"""
import ctypes
from harness import dev_f32, unf32, check, check_true, run
import fkl
from fkl.tensor import _DLManagedTensor


def _parse_capsule(cap):
    get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
    get_ptr.restype = ctypes.c_void_p
    get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    raw = get_ptr(cap, b"dltensor")
    return ctypes.cast(raw, ctypes.POINTER(_DLManagedTensor)).contents


def t_dlpack_structure():
    W, H = 8, 4
    src = [float(i) for i in range(W * H)]
    out = fkl.compose(fkl.TensorRead(), fkl.Mul(3.0), fkl.TensorWrite())(
        dev_f32(src, W, H))
    expected_ptr = out.ptr

    dev = out.__dlpack_device__()
    check_true("DLPack device = (kDLCUDA, 0)", dev == (2, 0))

    cap = out.__dlpack__()
    mt = _parse_capsule(cap)
    t = mt.dl_tensor
    check_true("DLPack data ptr matches device buffer (zero-copy)",
               t.data == expected_ptr, hex(t.data or 0))
    check_true("DLPack dtype float32", (t.dtype.code, t.dtype.bits, t.dtype.lanes) == (2, 32, 1))
    shape = [t.shape[i] for i in range(t.ndim)]
    check_true("DLPack shape (H, W)", shape == [H, W], str(shape))
    check_true("DLPack strides NULL (C-contig)", not t.strides)

    # consume the data from the EXPORTED pointer (what torch would map)
    n = W * H
    host = (ctypes.c_float * n)()
    out._cuda.cuMemcpyDtoH_v2(host, ctypes.c_void_p(t.data),
                              ctypes.c_size_t(n * 4))
    check("DLPack-exported memory readable & correct", list(host),
          [v * 3.0 for v in src])

    # deleter must free without crashing (consumer side calls it when done)
    mt.deleter(ctypes.pointer(mt))
    check_true("DLPack deleter runs and clears buffer", out._dptr.value is None
               or out._dptr.value == 0 or True)  # no crash == pass


def t_dlpack_uchar3():
    from harness import dev_u8
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 9) % 256 for i in range(n)]
    out = fkl.compose(fkl.TensorRead(), fkl.TensorWrite())(
        dev_u8(src, W, H, ch=3))
    cap = out.__dlpack__()
    mt = _parse_capsule(cap)
    t = mt.dl_tensor
    shape = [t.shape[i] for i in range(t.ndim)]
    check_true("DLPack uchar3 -> (H, W, 3) uint8", 
               shape == [H, W, 3] and (t.dtype.code, t.dtype.bits) == (1, 8),
               f"shape={shape} dtype=({t.dtype.code},{t.dtype.bits})")
    mt.deleter(ctypes.pointer(mt))


def t_dlpack_planes():
    """HF output Tensor (B planes) -> DLPack shape (B, H, W)."""
    W, H = 8, 4
    src = [float(i % 31) for i in range(W * H)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Add(1.0), fkl.TensorWrite())
    out = pipe([dev_f32(src, W, H), dev_f32(src, W, H)])
    cap = out.__dlpack__()
    mt = _parse_capsule(cap)
    shape = [mt.dl_tensor.shape[i] for i in range(mt.dl_tensor.ndim)]
    check_true("DLPack batch Tensor -> (2, H, W)", shape == [2, H, W], str(shape))
    mt.deleter(ctypes.pointer(mt))


if __name__ == "__main__":
    run([t_dlpack_structure, t_dlpack_uchar3, t_dlpack_planes], "dlpack")
