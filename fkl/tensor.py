"""Zero-copy device-pointer extraction (v2: vector dtypes + planes).

Supports torch.Tensor(cuda), cupy.ndarray, numba, and the built-in
DeviceBuffer via __cuda_array_interface__. Trailing dims of size 2..4 map to
CUDA vector pixel types (uchar3, float4, ...) per fkl.types.from_cai.
"""
from __future__ import annotations
from dataclasses import dataclass
import ctypes

from .types import DType, from_cai, _BASES


@dataclass
class DeviceView:
    ptr: int
    width: int
    height: int
    planes: int
    dtype: DType
    obj: object  # keeps the underlying buffer alive
    device: int = 0


def as_device_view(x) -> "DeviceView":
    cai = getattr(x, "__cuda_array_interface__", None)
    if cai is None:
        raise TypeError(
            f"{type(x).__name__} is not a CUDA array (needs "
            "__cuda_array_interface__: cuda torch.Tensor, cupy, numba, DeviceBuffer)")
    if cai.get("strides") is not None:
        # require C-contiguous for now (zero-copy without repacking)
        raise ValueError("non-contiguous arrays not supported; call .contiguous() first")
    dt, w, h, p = from_cai(cai["shape"], cai["typestr"])
    return DeviceView(int(cai["data"][0]), w, h, p, dt, x, _device_of(x))


def _device_of(x) -> int:
    """Best-effort device index: DeviceBuffer.device, torch .device.index,
    cupy .device.id; default 0 (CAI itself carries no device id)."""
    d = getattr(x, "device", None)
    if d is None:
        return 0
    if isinstance(d, int):
        return d
    idx = getattr(d, "index", None)       # torch.device
    if idx is not None:
        return int(idx)
    did = getattr(d, "id", None)          # cupy.cuda.Device
    return int(did) if did is not None else 0


def stream_handle(stream) -> int:
    if stream is None:
        return 0
    h = getattr(stream, "cuda_stream", None)  # torch
    if h is not None:
        return int(h)
    h = getattr(stream, "ptr", None)          # cupy
    if h is not None:
        return int(h)
    return int(stream) if isinstance(stream, int) else 0


class DeviceBuffer:
    """Dependency-free device buffer via the CUDA driver API.

    device=N allocates on GPU N (contexts are retained per device and the
    allocation happens with that device's primary context current)."""
    _cuda = None
    _ctxs = {}

    def __init__(self, width: int, height: int = 1, dtype="float32",
                 channels: int = 1, planes: int = 1, device: int = 0):
        self._load_driver()
        self.device = int(device)
        self._activate(self.device)
        from .types import dtype as _d
        base = _d(dtype) if isinstance(dtype, str) else dtype
        self.dtype = DType(base.base, channels if channels > 1 else base.channels)
        self.width, self.height, self.planes = int(width), int(height), int(planes)
        self._nbytes = self.width * self.height * self.planes * self.dtype.itemsize
        self._dptr = ctypes.c_void_p()
        self._check(self._cuda.cuMemAlloc_v2(ctypes.byref(self._dptr),
                                             ctypes.c_size_t(self._nbytes)))

    @classmethod
    def from_state(cls, st, device: int = 0):
        return cls(st.width, st.height, st.dtype.base,
                   channels=st.dtype.channels, planes=st.planes, device=device)

    @classmethod
    def _activate(cls, device: int):
        """Make `device`'s primary context current for this thread."""
        if device not in cls._ctxs:
            dev = ctypes.c_int(0)
            if cls._cuda.cuDeviceGet(ctypes.byref(dev), device) != 0:
                raise RuntimeError(f"cuDeviceGet({device}) failed — "
                                   f"is GPU {device} present?")
            ctx = ctypes.c_void_p()
            if cls._cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != 0:
                raise RuntimeError(f"cuDevicePrimaryCtxRetain({device}) failed")
            cls._ctxs[device] = ctx
        cls._cuda.cuCtxSetCurrent(cls._ctxs[device])

    @property
    def ptr(self) -> int:
        return int(self._dptr.value)

    @classmethod
    def _load_driver(cls):
        if cls._cuda is None:
            lib = ctypes.CDLL("libcuda.so.1")
            lib.cuInit.argtypes = [ctypes.c_uint]
            lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            lib.cuDevicePrimaryCtxRetain.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
            lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            lib.cuMemFree_v2.argtypes = [ctypes.c_void_p]
            lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            if lib.cuInit(0) != 0:
                raise RuntimeError("cuInit failed")
            cls._cuda = lib
            cls._activate(0)

    def _check(self, code):
        if code != 0:
            raise RuntimeError(f"CUDA driver error {code}")

    def copy_from_host(self, data_bytes: bytes):
        buf = (ctypes.c_char * len(data_bytes)).from_buffer_copy(data_bytes)
        self._check(self._cuda.cuMemcpyHtoD_v2(self._dptr, buf,
                                               ctypes.c_size_t(len(data_bytes))))

    def copy_to_host(self) -> bytes:
        out = (ctypes.c_char * self._nbytes)()
        self._check(self._cuda.cuMemcpyDtoH_v2(out, self._dptr,
                                               ctypes.c_size_t(self._nbytes)))
        return bytes(out)

    @property
    def __cuda_array_interface__(self):
        ts = self.dtype.typestr
        dims = []
        if self.planes > 1:
            dims.append(self.planes)
        if self.height > 1 or self.planes > 1:
            dims.append(self.height)
        dims.append(self.width)
        if self.dtype.channels > 1:
            dims.append(self.dtype.channels)
        return {"shape": tuple(dims), "typestr": ts,
                "data": (self.ptr, False), "version": 3}

    # ---- DLPack export (zero-copy into torch/cupy/jax) ----------------------
    # torch.from_dlpack(buf) / cupy.from_dlpack(buf) reuse OUR device memory:
    # no copy, no sync beyond what the consumer framework inserts.

    def __dlpack_device__(self):
        return (2, self.device)  # (kDLCUDA, device_id)

    def __dlpack__(self, stream=None):
        return _make_dlpack_capsule(self)

    def __del__(self):
        try:
            if getattr(self, "_dptr", None) and self._dptr.value:
                if getattr(self, "_exported_dlpack", False):
                    return  # ownership moved to the DLPack consumer
                self._cuda.cuMemFree_v2(self._dptr)
        except Exception:
            pass


# ===================== DLPack capsule construction =========================
# Implements the DLPack ABI (dlpack.h v0.8) with pure ctypes: DLManagedTensor
# wrapping the DeviceBuffer's pointer. The deleter frees the device memory,
# which is handed over to the consumer (PyCapsule protocol).

class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice),
                ("ndim", ctypes.c_int), ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER_T = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [("dl_tensor", _DLTensor),
                             ("manager_ctx", ctypes.c_void_p),
                             ("deleter", _DELETER_T)]

# dtype.base -> (DLDataTypeCode, bits). codes: 0=int 1=uint 2=float 6=bool
_DL_CODES = {
    "float32": (2, 32), "float64": (2, 64),
    "int32": (0, 32), "int16": (0, 16), "int64": (0, 64),
    "uint8": (1, 8), "uint16": (1, 16), "uint32": (1, 32),
    "bool": (6, 8),
}

_dlpack_keepalive = {}  # capsule id -> (managed, shape_arr, buffer)


def _make_dlpack_capsule(buf: "DeviceBuffer"):
    code_bits = _DL_CODES.get(buf.dtype.base)
    if code_bits is None:
        raise TypeError(f"DLPack: unsupported base dtype {buf.dtype.base}")
    code, bits = code_bits

    dims = []
    if buf.planes > 1:
        dims.append(buf.planes)
    if buf.height > 1 or buf.planes > 1:
        dims.append(buf.height)
    dims.append(buf.width)
    if buf.dtype.channels > 1:
        dims.append(buf.dtype.channels)
    ndim = len(dims)
    shape_arr = (ctypes.c_int64 * ndim)(*dims)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(buf.ptr)
    managed.dl_tensor.device = _DLDevice(2, getattr(buf, "device", 0))  # kDLCUDA
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = _DLDataType(code, bits, 1)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = None                    # C-contiguous
    managed.dl_tensor.byte_offset = 0

    key = id(managed)

    @_DELETER_T
    def _deleter(mt_ptr):
        entry = _dlpack_keepalive.pop(key, None)
        if entry is not None:
            b = entry[2]
            try:
                if getattr(b, "_dptr", None) and b._dptr.value:
                    b._cuda.cuMemFree_v2(b._dptr)
                    b._dptr = ctypes.c_void_p()
            except Exception:
                pass

    managed.deleter = _deleter
    # keep the struct + shape array + deleter + buffer alive until consumed
    _dlpack_keepalive[key] = (managed, shape_arr, buf, _deleter)
    buf._exported_dlpack = True   # our __del__ must not double-free

    pycapsule_new = ctypes.pythonapi.PyCapsule_New
    pycapsule_new.restype = ctypes.py_object
    pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return pycapsule_new(ctypes.byref(managed), b"dltensor", None)
