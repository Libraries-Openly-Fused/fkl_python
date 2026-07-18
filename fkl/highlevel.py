"""PyTorch-flavoured high-level API: Tensor / Image / ImageBatch + pipe().

An API LAYER over the existing machinery, not a framework:

  fkl.Tensor       thin zero-copy wrapper over torch CUDA tensors,
                   __cuda_array_interface__ objects and DeviceBuffer, with
                   .torch() / DLPack export and .reshape() (metadata only).
  fkl.Image        image semantics on a Tensor: width/height/channels,
                   'HWC' input layout (planar 'CHW' OUTPUT comes from
                   .to_planar(), which lowers to the existing split ops).
  fkl.ImageBatch   batch of same-shape images: a list of Images OR a
                   batched (N, H, W[, C]) tensor (split into zero-copy
                   per-plane views; runs as horizontal fusion).
  fkl.pipe(x)      fluent builder. Every method appends an existing
                   symbolic op; .run() lowers the WHOLE chain to ONE fused
                   FKL kernel through compose()'s machinery (JIT-compiled
                   once per signature, disk-cached, then a single launch).

        out = (fkl.pipe(batch)
                  .resize((64, 64))            # BVF: fused into the read
                  .cvt_color("RGB2BGR")        # VF: registers
                  .normalize(MEAN, STD)        # Cast+Sub+Div composition
                  .to_planar("CHW")            # planar NCHW write
                  .run())                      # ONE kernel launch
        out.torch()                            # zero-copy cuda tensor

Explicitly OUT of scope (use torch itself for these):
  - autograd (fkl kernels are preprocessing, not differentiable graph nodes)
  - broadcasting (values broadcast per-channel only, like the raw ops)
  - >4-channel images (FKL vector pixels are 1..4 channels)
  - multi-input graphs (the ABI is single-input/single-output; batches are
    horizontal fusion over one logical input)
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from .jit import FusedKernel
from .codegen import plan, generate_cu
from .operations import (Op, COMPUTE, TensorRead, TensorWrite, TensorSplit,
                         Mul, Add, Sub, Div, Cast, SaturateCast, Crop, Resize,
                         ColorConversion, Warping, BorderReader, Deinterlace)
from .tensor import as_device_view, DeviceBuffer, _make_dlpack_capsule
from .types import DType, dtype as _dt, from_shape, _BASES


def _prod(dims) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def _natural_shape(obj) -> Tuple[int, ...]:
    s = getattr(obj, "shape", None)          # torch / cupy / numba
    if s is not None:
        return tuple(int(d) for d in s)
    cai = getattr(obj, "__cuda_array_interface__", None)
    if cai is not None:
        return tuple(int(d) for d in cai["shape"])
    raise TypeError(f"cannot infer shape of {type(obj).__name__}")


class Tensor:
    """Zero-copy wrapper over any CUDA array (torch tensor, cupy/numba array,
    DeviceBuffer, ... anything with __cuda_array_interface__).

    Never copies, never owns new memory (except Tensor.empty, which allocates
    a DeviceBuffer). Exposes .shape/.dtype/.device, __cuda_array_interface__,
    __dlpack__/__dlpack_device__ and .torch() (zero-copy view when torch is
    installed). C-contiguous inputs only, like the rest of fkl."""
    __slots__ = ("_base", "_view", "_base_shape", "_shape", "_torch_view")

    def __init__(self, data):
        if isinstance(data, Tensor):
            self._base, self._view = data._base, data._view
            self._base_shape, self._shape = data._base_shape, data._shape
            self._torch_view = data._torch_view
            return
        self._view = as_device_view(data)   # validates CUDA + C-contiguity
        if self._view.dtype.base not in _BASES:
            # e.g. float16/bfloat16/int64 torch tensors: as_device_view maps
            # them, but there is no FKL kernel/interop surface for them yet.
            raise TypeError(
                f"fkl.Tensor does not support {self._view.dtype.base} yet; "
                "cast to a supported dtype first (e.g. .to(torch.float32))")
        self._base = data
        self._base_shape = _natural_shape(data)
        self._shape = self._base_shape
        self._torch_view = None

    @classmethod
    def empty(cls, shape, dtype="float32", device: int = 0) -> "Tensor":
        """Allocate an uninitialized DeviceBuffer-backed Tensor.

        dtype is a scalar spec ('float32', 'uint8', ...; channels live in
        shape, torch-style) or a vector spec like 'uint8x3', which APPENDS
        the channel dim to shape: Tensor.empty((4, 6), 'uint8x3') has shape
        (4, 6, 3) and scalar dtype uint8 — exactly like
        Tensor.empty((4, 6, 3), 'uint8')."""
        shape = tuple(int(s) for s in ((shape,) if isinstance(shape, int)
                                       else shape))
        dt = _dt(dtype)
        if dt.channels > 1:
            if len(shape) == 1:
                w, h, p = shape[0], 1, 1
            elif len(shape) == 2:
                (h, w), p = shape, 1
            elif len(shape) == 3:
                p, h, w = shape
            else:
                raise ValueError(f"unsupported shape {shape} for vector dtype")
            ch = dt.channels
            shape = shape + (ch,)   # channels live in .shape (torch-style)
        else:
            d2, w, h, p = from_shape(shape, dt.base)
            ch = d2.channels
        buf = DeviceBuffer(w, h, dt.base, channels=ch, planes=p, device=device)
        t = cls(buf)
        return t if t._shape == shape else t.reshape(shape)

    # ---- torch-style metadata ------------------------------------------
    @property
    def base(self):
        """The wrapped object (torch tensor / DeviceBuffer / ...)."""
        return self._base

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> DType:
        """Scalar element type (channels live in .shape, torch-style)."""
        return DType(self._view.dtype.base, 1)

    @property
    def device(self) -> int:
        return self._view.device

    @property
    def ptr(self) -> int:
        return self._view.ptr

    @property
    def nbytes(self) -> int:
        return _prod(self._shape) * _BASES[self._view.dtype.base][2]

    def reshape(self, *shape) -> "Tensor":
        """Metadata-only reshape (zero-copy; memory is C-contiguous)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(s) for s in shape)
        if _prod(shape) != _prod(self._shape):
            raise ValueError(f"cannot reshape {self._shape} -> {shape} "
                             "(element counts differ)")
        t = Tensor.__new__(Tensor)
        t._base, t._view = self._base, self._view
        t._base_shape, t._shape = self._base_shape, shape
        t._torch_view = None
        return t

    # ---- interop --------------------------------------------------------
    @property
    def __cuda_array_interface__(self):
        return {"shape": self._shape, "typestr": self._view.dtype.typestr,
                "data": (self._view.ptr, False), "version": 3}

    def __dlpack_device__(self):
        return (2, self._view.device)   # (kDLCUDA, device_id)

    def __dlpack__(self, stream=None):
        if isinstance(self._base, DeviceBuffer):
            # our own allocation: export under THIS Tensor's (possibly
            # reshaped) dims. Ownership moves to the consumers: the device
            # memory is freed when the LAST exported capsule is released
            # (several Tensor views may export the same buffer).
            return _make_dlpack_capsule(self._base, dims=self._shape)
        if self._shape == self._base_shape and hasattr(self._base, "__dlpack__"):
            if stream is None:
                return self._base.__dlpack__()
            return self._base.__dlpack__(stream=stream)
        raise BufferError(
            "cannot export DLPack: the base object has no __dlpack__ or the "
            "Tensor was reshaped; use .torch() or export .base directly")

    def torch(self):
        """Zero-copy torch view of this Tensor (torch must be installed).

        DeviceBuffer-backed Tensors hand memory ownership to torch (DLPack
        deleter): the device memory is freed when the last exported view is
        released, so several Tensor views over one buffer are safe. The
        view is cached so repeated calls return the same tensor."""
        if self._torch_view is not None:
            return self._torch_view
        try:
            import torch as _torch
        except ImportError as e:  # pragma: no cover - torch-less machines
            raise RuntimeError("fkl.Tensor.torch() requires torch") from e
        if isinstance(self._base, _torch.Tensor):
            t = self._base
            if tuple(t.shape) != self._shape:
                t = t.view(self._shape)
        else:
            try:
                t = _torch.from_dlpack(self)
            except (BufferError, TypeError, RuntimeError):
                # CAI-only objects without DLPack: torch.as_tensor consumes
                # __cuda_array_interface__ zero-copy on matching device.
                t = _torch.as_tensor(self._base,
                                     device=f"cuda:{self.device}")
            if tuple(t.shape) != self._shape:
                t = t.reshape(self._shape)
        self._torch_view = t
        return t

    # kernels take the raw base object (so auto-allocated outputs match the
    # input's framework); a reshaped Tensor must pass itself, since only its
    # own __cuda_array_interface__ carries the override shape.
    @property
    def _kernel_arg(self):
        return self._base if self._shape == self._base_shape else self

    def __repr__(self):
        return (f"fkl.Tensor(shape={self._shape}, dtype={self.dtype.base}, "
                f"device=cuda:{self.device})")


class Image(Tensor):
    """A single 'HWC' image: shape (H, W) or (H, W, C) with C in 2..4.

    Input layout is 'HWC' (packed vector pixels — what cameras, decoders and
    OpenCV produce, and what FKL's ops consume). Planar 'CHW' is an OUTPUT
    layout: produce it with pipe(...).to_planar('CHW') / fkl.F.to_planar,
    which lower to the existing TensorSplit write."""
    __slots__ = ()

    def __init__(self, data, layout: str = "HWC"):
        if str(layout).upper() != "HWC":
            raise ValueError(
                f"unsupported input layout {layout!r}: Image inputs are "
                "'HWC'; planar 'CHW' output comes from .to_planar()")
        super().__init__(data)
        s = self._shape
        if len(s) == 3 and not (2 <= s[-1] <= 4):
            raise ValueError(
                f"Image got shape {s}: trailing dim {s[-1]} is not a channel "
                "count (2..4). >4-channel images are out of scope; squeeze a "
                "trailing 1; for a batch of 2-D images use fkl.ImageBatch")
        if len(s) not in (2, 3):
            raise ValueError(
                f"Image needs (H, W) or (H, W, C<=4), got {s}; for a batch "
                "use fkl.ImageBatch")

    @classmethod
    def from_tensor(cls, t, layout: str = "HWC") -> "Image":
        """From an fkl.Tensor or any CUDA array object (zero-copy)."""
        return cls(t, layout=layout)

    @property
    def layout(self) -> str:
        return "HWC"

    @property
    def height(self) -> int:
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]

    @property
    def channels(self) -> int:
        return self._shape[2] if len(self._shape) == 3 else 1

    def __repr__(self):
        return (f"fkl.Image({self.width}x{self.height}x{self.channels} HWC, "
                f"dtype={self.dtype.base}, device=cuda:{self.device})")


class _PlaneView:
    """Zero-copy view of one plane of a batched (N, ...) CUDA array. Exposes
    just enough for as_device_view (__cuda_array_interface__ + .device) and
    keeps the owner alive."""
    __slots__ = ("_cai", "device", "_owner")

    def __init__(self, ptr, shape, typestr, device, owner):
        self._cai = {"shape": tuple(int(d) for d in shape), "typestr": typestr,
                     "data": (int(ptr), False), "version": 3}
        self.device = int(device)
        self._owner = owner

    @property
    def __cuda_array_interface__(self):
        return self._cai


class ImageBatch:
    """A batch of same-shape 'HWC' images.

    Build it from a list of Images/arrays, or from ONE batched tensor of
    shape (N, H, W) or (N, H, W, C<=4) — the tensor is split into zero-copy
    per-plane views. Pipelines over an ImageBatch run as HORIZONTAL FUSION:
    one kernel, N thread-planes (batch reads), N part of the kernel type."""

    def __init__(self, images, layout: str = "HWC"):
        if str(layout).upper() != "HWC":
            raise ValueError("ImageBatch input layout must be 'HWC'")
        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("ImageBatch needs at least one image")
            imgs = [im if isinstance(im, Image) else Image(im)
                    for im in images]
            key0 = (imgs[0].width, imgs[0].height, imgs[0].channels,
                    imgs[0].dtype.base, imgs[0].device)
            for im in imgs[1:]:
                if (im.width, im.height, im.channels, im.dtype.base,
                        im.device) != key0:
                    raise ValueError("all images in a batch must share "
                                     "(W, H, C, dtype, device)")
            self._images = imgs
            self._tensor = None
            return
        t = images if isinstance(images, Tensor) else Tensor(images)
        s = t.shape
        if len(s) == 4 and not (2 <= s[-1] <= 4):
            raise ValueError(f"batched tensor got shape {s}: trailing dim "
                             "must be a channel count (2..4)")
        if len(s) not in (3, 4):
            raise ValueError(
                f"batched tensor must be (N, H, W) or (N, H, W, C<=4), got {s}")
        stride = _prod(s[1:]) * _BASES[t.dtype.base][2]
        ts = t.__cuda_array_interface__["typestr"]
        self._images = [Image(_PlaneView(t.ptr + i * stride, s[1:], ts,
                                         t.device, t))
                        for i in range(s[0])]
        self._tensor = t

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, i) -> Image:
        return self._images[i]

    def __iter__(self):
        return iter(self._images)

    @property
    def images(self) -> Tuple[Image, ...]:
        return tuple(self._images)

    @property
    def tensor(self) -> Optional[Tensor]:
        """The batched source tensor, if this batch was built from one."""
        return self._tensor

    @property
    def width(self) -> int:
        return self._images[0].width

    @property
    def height(self) -> int:
        return self._images[0].height

    @property
    def channels(self) -> int:
        return self._images[0].channels

    @property
    def dtype(self) -> DType:
        return self._images[0].dtype

    @property
    def device(self) -> int:
        return self._images[0].device

    @property
    def layout(self) -> str:
        return "HWC"

    def __repr__(self):
        return (f"fkl.ImageBatch({len(self)} x {self.width}x{self.height}"
                f"x{self.channels} HWC, dtype={self.dtype.base})")


# ===================== fluent pipeline builder =============================

# ReadBack-family ops: fused backwards into the read (BVF) -> must be the
# LEADING stages of a chain, exactly like in compose().
_GEOMETRY_OPS = (Crop, Resize, Warping, BorderReader, Deinterlace)


class Pipeline:
    """Fluent, source-bound pipeline that lowers to ONE fused kernel.

    Every method appends an existing symbolic op (the same descriptors
    compose() takes) and returns self. .run() materializes: the whole chain
    becomes a single fused FKL kernel — geometry ops fuse backwards into the
    read (BVF), compute ops fuse vertically in registers (VF), an ImageBatch
    source adds horizontal fusion (HF) — JIT-compiled once per signature and
    disk-cached, exactly like any compose() chain. Values (rects, sizes,
    means) travel in params[]: an identical chain built with different
    values reuses the cached .so.

    Ordering rule (mirrors compose()): crop/resize/warp/border are ReadBack
    ops and must come before compute ops (cvt_color/normalize/...).
    Reuse the Pipeline object for hot loops: the kernel handle is cached on
    it after the first .run().

    Note: value arithmetic (mul/add/sub/div) on integer VECTOR-pixel chains
    (e.g. uchar3) does not compile in FKL (make_ promotes small-int vectors)
    — .cast('float32') first; .normalize() and .resize() already move the
    chain to float.
    """

    def __init__(self, src, thread_fusion: bool = False):
        if isinstance(src, (list, tuple)):
            src = ImageBatch(list(src))
        if isinstance(src, ImageBatch):
            self._batch, self._single = src, None
            im0 = src[0]
            self._in_dt = DType(im0.dtype.base, im0.channels)
            self._in_shape = (im0.width, im0.height, 1)
            self._n_inputs = len(src)
            self._keep_h = True     # images are 2-D by definition
        else:
            t = src if isinstance(src, Tensor) else Tensor(src)
            self._batch, self._single = None, t
            dt, w, h, p = from_shape(t.shape, t.dtype.base)
            self._in_dt = dt
            self._in_shape = (w, h, p)
            self._n_inputs = 1
            # 2-D-natured sources keep their height dim in the output even
            # when it is 1 ((1, W) round-trips); 1-D vectors stay 1-D. Same
            # channel folding rule as from_shape.
            dims = t.shape
            if len(dims) >= 2 and 2 <= dims[-1] <= 4:
                dims = dims[:-1]
            self._keep_h = len(dims) >= 2
        self._ops: List[Op] = []
        self._write: Optional[Op] = None
        self._closed = False
        self._geometry_open = True
        self._thread_fusion = bool(thread_fusion)
        self._kernel: Optional[FusedKernel] = None

    # ---- builder plumbing -------------------------------------------------
    def _append(self, op: Op) -> "Pipeline":
        if self._closed:
            raise RuntimeError(
                ".to_planar() is the pipeline's write stage; no ops after it")
        if isinstance(op, _GEOMETRY_OPS):
            if not self._geometry_open:
                raise ValueError(
                    f"{type(op).__name__} must come before compute ops: "
                    "geometry ops fuse backwards into the read (BVF), so "
                    "put crop/resize/warp/border first, like in compose()")
        else:
            self._geometry_open = False
        self._ops.append(op)
        self._kernel = None
        return self

    def _chain_dtype(self) -> DType:
        dt = self._in_dt
        for op in self._ops:
            dt = op.out_dtype(dt)
        return dt

    @property
    def ops(self) -> Tuple[Op, ...]:
        """The symbolic compute ops appended so far (read/write excluded)."""
        return tuple(self._ops)

    # ---- geometry (ReadBack / BVF) -----------------------------------------
    def crop(self, *args) -> "Pipeline":
        """crop(x, y, w, h) / crop((x, y, w, h)) — single ROI.
        crop([(x, y, w, h), ...]) — multi-ROI from ONE image (each rect
        becomes a batch plane: horizontal fusion)."""
        if len(args) == 4:
            return self._append(Crop(*args))
        if len(args) == 1:
            a = args[0]
            if len(a) == 4 and all(isinstance(v, (int, float)) for v in a):
                return self._append(Crop(*a))
            if self._n_inputs > 1:
                raise ValueError(
                    "multi-ROI crop needs a single source image (the rect "
                    "list itself produces the batch planes)")
            return self._append(Crop(list(a)))
        raise ValueError(
            "crop(x, y, w, h), crop((x, y, w, h)) or crop([(x, y, w, h), ...])")

    def resize(self, size, interp: str = "linear",
               aspect_ratio: str = "ignore", background=None) -> "Pipeline":
        """size = (w, h) target size, or an int for a square. Output becomes
        float32 (interpolation), like fkl.Resize."""
        if isinstance(size, int):
            w = h = size
        else:
            w, h = size
        return self._append(Resize(int(w), int(h), interp=interp,
                                   aspect_ratio=aspect_ratio,
                                   background=background))

    def border(self, mode: str = "replicate", value=0.0) -> "Pipeline":
        """Out-of-bounds read policy (place before crop/resize)."""
        return self._append(BorderReader(mode, value=value))

    # ---- compute (VF) --------------------------------------------------------
    def cvt_color(self, code: str) -> "Pipeline":
        """OpenCV-style color conversion code, e.g. 'RGB2BGR', 'BGR2GRAY'."""
        return self._append(ColorConversion(code))

    def normalize(self, mean, std) -> "Pipeline":
        """(x - mean) / std, per channel. Composed from the existing ops:
        Cast('float32') (only when the chain is not float yet) + Sub + Div —
        all fused in registers. mean/std are given in the chain's value
        range at this point (e.g. 0..255 for uint8 sources)."""
        dt = self._chain_dtype()
        for name, v in (("mean", mean), ("std", std)):
            if isinstance(v, (list, tuple)) and len(v) != dt.channels:
                raise ValueError(
                    f"normalize {name} has {len(v)} values but the chain "
                    f"carries {dt.channels} channel(s) here")
        if dt.base not in ("float32", "float64"):
            self._append(Cast("float32"))
        self._append(Sub(mean))
        return self._append(Div(std))

    def mul(self, value) -> "Pipeline":
        return self._append(Mul(value))

    def add(self, value) -> "Pipeline":
        return self._append(Add(value))

    def sub(self, value) -> "Pipeline":
        return self._append(Sub(value))

    def div(self, value) -> "Pipeline":
        return self._append(Div(value))

    def cast(self, to) -> "Pipeline":
        return self._append(Cast(to))

    def saturate_cast(self, to) -> "Pipeline":
        return self._append(SaturateCast(to))

    def apply(self, op: Op) -> "Pipeline":
        """Escape hatch: append any fkl symbolic op (e.g. fkl.VectorReorder,
        fkl.MxVFloat3) to the fused chain."""
        if getattr(op, "role", None) != COMPUTE:
            raise ValueError("apply() takes compute ops; the read/write "
                             "stages are managed by the pipeline")
        return self._append(op)

    # ---- write stage -----------------------------------------------------
    def to_planar(self, layout: str = "CHW") -> "Pipeline":
        """Terminal layout change: packed HWC pixels -> planar 'CHW' (or
        'NCHW' for batches) via the existing TensorSplit write. Must be the
        last stage; single-channel chains are already planar."""
        if str(layout).upper() not in ("CHW", "NCHW"):
            raise ValueError("to_planar supports 'CHW' / 'NCHW' output only")
        if self._closed:
            raise RuntimeError("to_planar() already applied")
        dt = self._chain_dtype()
        if dt.channels < 2:
            raise ValueError("to_planar needs a multi-channel chain "
                             "(single-channel data is already planar)")
        self._write = TensorSplit()
        self._closed = True
        self._kernel = None
        return self

    # ---- materialization ---------------------------------------------------
    def _full_ops(self) -> List[Op]:
        write = self._write if self._write is not None else TensorWrite()
        return [TensorRead(), *self._ops, write]

    def _ensure_kernel(self) -> FusedKernel:
        if self._kernel is None:
            self._kernel = FusedKernel(self._full_ops(), target="gpu",
                                       thread_fusion=self._thread_fusion)
        return self._kernel

    def _check_out(self, out_arg, out_st, sem: Tuple[int, ...]):
        """Eager validation of a preallocated out= BEFORE the launch: a
        too-small buffer would be an out-of-bounds device write, a wrong
        dtype a silent reinterpretation."""
        vout = as_device_view(out_arg)
        want = (out_st.width * out_st.height * out_st.planes
                * out_st.dtype.channels)
        got = vout.width * vout.height * vout.planes * vout.dtype.channels
        if vout.dtype.base != out_st.dtype.base or got != want:
            raise ValueError(
                f"out= mismatch: the pipeline produces {sem} "
                f"{out_st.dtype.base} ({want} elements), out= provides "
                f"{_natural_shape(out_arg)} {vout.dtype.base} "
                f"({got} elements)")
        in_dev = (self._batch.device if self._batch is not None
                  else self._single.device)
        if vout.device != in_dev:
            raise ValueError(f"out= lives on cuda:{vout.device} but the "
                             f"pipeline input is on cuda:{in_dev}")

    def run(self, out=None, stream=None) -> Tensor:
        """Execute the fused chain: ONE kernel launch. First call per
        signature JIT-compiles (disk-cached forever); afterwards it is a
        single ctypes call. stream: torch stream / cupy stream / raw handle
        (async, caller syncs); None = internal stream, synced before return.
        out: preallocated output (fkl.Tensor / torch tensor / DeviceBuffer),
        validated against the planned output before the launch.
        Returns an fkl.Tensor (use .torch() for a zero-copy torch view)."""
        k = self._ensure_kernel()
        _, out_st = plan(k.ops, self._in_dt, self._in_shape, self._n_inputs)
        sem = _semantic_shape(out_st, split=isinstance(k.ops[-1], TensorSplit),
                              keep_h=self._keep_h)
        src = self._batch.tensor if self._batch is not None else self._single
        if out is not None:
            out_arg = out._kernel_arg if isinstance(out, Tensor) else out
            self._check_out(out_arg, out_st, sem)
        else:
            # allocate in the SOURCE's framework (torch in -> torch out),
            # even when the kernel args are fkl-internal views (reshaped
            # Tensors, batched-tensor plane views). A batch built from a
            # LIST allocates from its first image inside the kernel call.
            base = None if src is None else src.base
            if isinstance(base, _PlaneView):
                base = base._owner.base   # plane of a batched source tensor
            out_arg = (None if base is None
                       else k._alloc_out(as_device_view(base), out_st))
        if self._batch is not None:
            args = [im._kernel_arg for im in self._batch.images]
            raw = k(args, out=out_arg, stream=stream)
        else:
            raw = k(self._single._kernel_arg, out=out_arg, stream=stream)
        return Tensor(raw).reshape(sem)

    def source(self) -> str:
        """Debug: the generated C++ for the exact variant .run() would
        launch (including the effective ThreadFusion choice for this
        source's dtype/width) — one TU, one executeOperations call."""
        k = self._ensure_kernel()
        return generate_cu(k.ops, self._in_dt, self._in_shape, self._n_inputs,
                           thread_fusion=k._tf_effective(self._in_dt,
                                                         self._in_shape))

    def __repr__(self):
        names = [type(op).__name__ for op in self._ops]
        if self._write is not None:
            names.append("to_planar")
        src = (f"batch[{self._n_inputs}]" if self._batch is not None
               else "image")
        return f"fkl.Pipeline({src}: {' -> '.join(names) or '<empty>'})"


def _semantic_shape(out_st, split: bool, keep_h: bool = False) -> Tuple[int, ...]:
    """Logical output shape for a chain output state: planar (C, H, W) /
    (N, C, H, W) for split writes, (H, W[, C]) / (N, H, W[, C]) otherwise.
    keep_h preserves a height-1 dim (2-D sources round-trip (1, W));
    without it height-1 collapses to (W,), matching 1-D vector sources."""
    ch, p = out_st.dtype.channels, out_st.planes
    H, W = out_st.height, out_st.width
    if split:
        core = (ch, H, W)
    else:
        core = (H, W) if (H > 1 or keep_h) else (W,)
        if ch > 1:
            core = core + (ch,)
    return (p,) + core if p > 1 else core


def pipe(src, thread_fusion: bool = False) -> Pipeline:
    """Start a fluent pipeline bound to `src`: an Image / ImageBatch /
    Tensor, a raw CUDA array, or a list of same-shape images (treated as an
    ImageBatch -> horizontal fusion). The chain fuses into ONE kernel at
    .run(). GPU only (for the CPU executor use fkl.compose(target='cpu'))."""
    return Pipeline(src, thread_fusion=thread_fusion)
