"""High-level torch-style API (Tensor / Image / ImageBatch / pipe / F).

Dependency-free (DeviceBuffer-backed). Pipelines are validated against
pure-Python per-op references, or against the equivalent hand-built
compose() chain (whose ops are themselves reference-tested in
test_operations.py). Torch interop lives in test_highlevel_torch.py.
"""
from harness import (dev_f32, dev_u8, f32, unf32, unu8, u8, check,
                     check_true, run)
import fkl


# ---- Tensor ---------------------------------------------------------------

def t_tensor_wrap():
    W, H = 8, 4
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    buf = dev_f32(src, W, H)
    t = fkl.Tensor(buf)
    check_true("Tensor wrap: shape/dtype/device",
               t.shape == (H, W) and t.dtype.base == "float32"
               and t.device == 0 and t.ndim == 2)
    check_true("Tensor wrap: zero-copy",
               t.base is buf and t.ptr == buf.ptr
               and t.__cuda_array_interface__["data"][0] == buf.ptr)


def t_tensor_empty():
    t = fkl.Tensor.empty((4, 6, 3), "uint8")
    check_true("Tensor.empty: shape/dtype/nbytes",
               t.shape == (4, 6, 3) and t.dtype.base == "uint8"
               and t.nbytes == 72)
    vals = [(i * 5) % 256 for i in range(72)]
    t.base.copy_from_host(u8(vals))
    check("Tensor.empty: DeviceBuffer roundtrip",
          unu8(t.base.copy_to_host(), 72), vals)


def t_tensor_empty_vector():
    t = fkl.Tensor.empty((4, 6), "uint8x3")   # channel dim appended to shape
    check_true("Tensor.empty vector spec: (4, 6) x uint8x3 -> (4, 6, 3)",
               t.shape == (4, 6, 3) and t.dtype.base == "uint8"
               and t.nbytes == 72)
    t1 = fkl.Tensor.empty((5,), "float32x2")        # 1-D: (W,) -> (W, C)
    t3 = fkl.Tensor.empty((2, 3, 4), "uint8x4")     # 3-D: (P, H, W) -> +C
    check_true("Tensor.empty vector spec: 1-D and 3-D shapes",
               t1.shape == (5, 2) and t3.shape == (2, 3, 4, 4))
    eq = fkl.Tensor.empty((4, 6, 3), "uint8")       # scalar-spec equivalent
    check_true("Tensor.empty: vector spec == scalar spec with channel dim",
               eq.shape == t.shape and eq.dtype == t.dtype
               and eq.nbytes == t.nbytes)


def t_tensor_reshape():
    t = fkl.Tensor.empty((2, 8), "float32")
    r = t.reshape(4, 4)
    check_true("reshape: zero-copy metadata",
               r.shape == (4, 4) and r.ptr == t.ptr and r.base is t.base)
    try:
        t.reshape(5, 5)
        ok = False
    except ValueError:
        ok = True
    check_true("reshape: validates element count", ok)


def t_tensor_dlpack():
    src = [float(i) for i in range(16)]
    t = fkl.Tensor(dev_f32(src, 8, 2))
    check_true("__dlpack_device__ = (kDLCUDA, 0)",
               t.__dlpack_device__() == (2, 0))
    cap = t.reshape(16).__dlpack__()   # reshaped export over DeviceBuffer
    check_true("__dlpack__ capsule (reshaped DeviceBuffer base)",
               cap is not None)


def t_dlpack_shared_buffer_ownership():
    """Two Tensor views exporting ONE DeviceBuffer: the memory must survive
    until the LAST consumer releases its capsule (no use-after-free)."""
    import ctypes
    from fkl.tensor import _DLManagedTensor
    t = fkl.Tensor.empty((4, 4), "float32")
    buf = t.base
    caps = [t.__dlpack__(), t.reshape(16).__dlpack__()]
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]

    def release(cap):   # what a DLPack consumer does when it is dropped
        mt = ctypes.cast(get(cap, b"dltensor"),
                         ctypes.POINTER(_DLManagedTensor))
        mt.contents.deleter(mt)

    release(caps[0])
    alive_after_first = bool(buf._dptr.value)
    release(caps[1])
    check_true("DLPack: shared buffer freed by the LAST exported view",
               alive_after_first and not buf._dptr.value)


# ---- Image / ImageBatch ----------------------------------------------------

def t_image_semantics():
    W, H = 6, 4
    img = fkl.Image(dev_u8([(i * 3) % 256 for i in range(W * H * 3)], W, H, ch=3))
    check_true("Image: w/h/c/layout",
               img.width == W and img.height == H and img.channels == 3
               and img.layout == "HWC" and img.shape == (H, W, 3))
    img2 = fkl.Image.from_tensor(fkl.Tensor(dev_u8([0] * (W * H), W, H)))
    check_true("Image.from_tensor (single channel)",
               isinstance(img2, fkl.Image) and img2.channels == 1)


def t_image_rejects():
    bad = 0
    try:  # 5 "channels" is neither a vector pixel nor a valid image
        fkl.Image(fkl.DeviceBuffer(5, 4, "uint8", channels=5))
    except ValueError:
        bad += 1
    try:  # batch-shaped (N, H, W) input
        fkl.Image(fkl.DeviceBuffer(16, 8, "float32", planes=3))
    except ValueError:
        bad += 1
    try:  # CHW input layout is not supported (output-only, via to_planar)
        fkl.Image(fkl.DeviceBuffer(8, 8, "uint8", channels=3), layout="CHW")
    except ValueError:
        bad += 1
    check_true("Image rejects >4ch / batch-shaped / CHW-input", bad == 3)


def t_image_batch_list():
    W, H = 8, 6
    imgs = [dev_u8([(i + k) % 256 for i in range(W * H * 3)], W, H, ch=3)
            for k in range(3)]
    b = fkl.ImageBatch(imgs)
    check_true("ImageBatch(list): len/w/h/c",
               len(b) == 3 and b.width == W and b.height == H
               and b.channels == 3 and isinstance(b[0], fkl.Image))


def t_image_batch_tensor():
    W, H, N = 4, 3, 2
    base = fkl.DeviceBuffer(W, H, "uint8", channels=3, planes=N)
    base.copy_from_host(u8([(i * 7) % 256 for i in range(N * H * W * 3)]))
    b = fkl.ImageBatch(base)
    per_plane = H * W * 3
    check_true("ImageBatch(tensor): zero-copy plane views",
               len(b) == N and b[0].ptr == base.ptr
               and b[1].ptr == base.ptr + per_plane
               and b[0].shape == (H, W, 3) and b.tensor is not None)


def t_image_batch_device_mismatch():
    from fkl.highlevel import _PlaneView
    W, H = 4, 3
    a = dev_u8([0] * (W * H * 3), W, H, ch=3)
    b = dev_u8([0] * (W * H * 3), W, H, ch=3)
    ghost = _PlaneView(b.ptr, (H, W, 3), "|u1", 1, b)   # claims cuda:1
    try:
        fkl.ImageBatch([a, ghost])
        ok = False
    except ValueError:
        ok = True
    check_true("ImageBatch(list): mixed devices rejected eagerly", ok)


# ---- pipelines vs references ------------------------------------------------

def t_pipe_elementwise():
    W, H = 16, 8
    src = [float((y * W + x) % 97) for y in range(H) for x in range(W)]
    out = fkl.pipe(fkl.Image(dev_f32(src, W, H))).mul(2.0).add(1.0).run()
    check_true("pipe: returns fkl.Tensor with image shape",
               isinstance(out, fkl.Tensor) and out.shape == (H, W))
    check("pipe: mul+add vs python ref",
          unf32(out.base.copy_to_host(), W * H), [v * 2 + 1 for v in src])


def t_pipe_crop():
    W, H = 33, 17                       # non-pow2 to catch pitch errors
    src = [(y * W + x) % 251 for y in range(H) for x in range(W)]
    out = fkl.pipe(fkl.Image(dev_u8(src, W, H))).crop(5, 3, 8, 6).run()
    exp = [float(((y + 3) * W + (x + 5)) % 251)
           for y in range(6) for x in range(8)]
    check_true("pipe: crop output shape", out.shape == (6, 8))
    check("pipe: crop vs position-encoded ref",
          [float(v) for v in unu8(out.base.copy_to_host(), 48)], exp)


def t_pipe_resize():
    W, H = 16, 8
    src = [float((y * W + x) % 53) for y in range(H) for x in range(W)]
    out = fkl.pipe(fkl.Image(dev_f32(src, W, H))).resize((7, 5)).run()
    ref = fkl.compose(fkl.TensorRead(), fkl.Resize(7, 5),
                      fkl.TensorWrite())(dev_f32(src, W, H))
    check_true("pipe: resize output shape+dtype",
               out.shape == (5, 7) and out.dtype.base == "float32")
    check("pipe: resize == compose(Resize) chain",
          unf32(out.base.copy_to_host(), 35),
          unf32(ref.copy_to_host(), 35), tol=1e-5)


def t_pipe_cvt_color():
    W, H = 8, 2
    src = [(i * 17) % 256 for i in range(W * H * 3)]
    out = fkl.pipe(fkl.Image(dev_u8(src, W, H, ch=3))).cvt_color("BGR2GRAY").run()
    exp = []
    for p in range(W * H):
        b, g, r = src[p * 3:(p + 1) * 3]
        exp.append(float(int(0.299 * r + 0.587 * g + 0.114 * b + 0.5)))
    check("pipe: cvt_color BGR2GRAY vs BT.601 ref",
          [float(v) for v in unu8(out.base.copy_to_host(), W * H)], exp,
          tol=1.0)


def t_pipe_normalize_autocast():
    W, H = 6, 4
    MEAN, STD = (110.0, 120.0, 130.0), (50.0, 55.0, 60.0)
    src = [(i * 13) % 256 for i in range(W * H * 3)]
    out = fkl.pipe(fkl.Image(dev_u8(src, W, H, ch=3))).normalize(MEAN, STD).run()
    exp = [(float(v) - MEAN[i % 3]) / STD[i % 3] for i, v in enumerate(src)]
    check_true("pipe: normalize auto-casts uint8 -> float32",
               out.dtype.base == "float32" and out.shape == (H, W, 3))
    check("pipe: normalize vs (x-mean)/std ref",
          unf32(out.base.copy_to_host(), W * H * 3), exp)


def t_pipe_to_planar():
    W, H = 4, 3
    src = [(i * 7) % 256 for i in range(W * H * 3)]
    p = fkl.pipe(fkl.Image(dev_u8(src, W, H, ch=3))).to_planar("CHW")
    out = p.run()
    exp = [float(src[px * 3 + c]) for c in range(3) for px in range(W * H)]
    check_true("pipe: to_planar semantic shape (C, H, W)",
               out.shape == (3, H, W))
    check("pipe: to_planar HWC->CHW vs ref",
          [float(v) for v in unu8(out.base.copy_to_host(), W * H * 3)], exp)
    try:
        p.mul(2.0)
        ok = False
    except RuntimeError:
        ok = True
    check_true("pipe: no ops after to_planar", ok)


def t_pipe_dnn_single_kernel():
    """Flagship chain: crop -> resize -> normalize -> CHW, ONE fused kernel."""
    W, H = 32, 24
    MEAN = (123.675, 116.28, 103.53)
    STD = (58.395, 57.12, 57.375)
    src = [(i * 3 + (i % 3) * 7) % 256 for i in range(W * H * 3)]
    p = (fkl.pipe(fkl.Image(dev_u8(src, W, H, ch=3)))
         .crop(4, 2, 24, 20)
         .resize((8, 8))
         .normalize(MEAN, STD)
         .to_planar("CHW"))
    out = p.run()
    ref = fkl.compose(fkl.TensorRead(), fkl.Crop(4, 2, 24, 20),
                      fkl.Resize(8, 8), fkl.Sub(MEAN), fkl.Div(STD),
                      fkl.TensorSplit())(dev_u8(src, W, H, ch=3))
    check_true("pipe: DNN chain output (3, 8, 8)", out.shape == (3, 8, 8))
    check("pipe: DNN chain == compose reference chain",
          unf32(out.base.copy_to_host(), 3 * 64),
          unf32(ref.copy_to_host(), 3 * 64))
    # ONE fused kernel: a single TU with a single entry; every stage appears
    # inside the one fused launch (executeOperations shows up twice only
    # because the template has an async and a sync stream branch).
    src = p.source()
    check_true("pipe: lowers to ONE fused kernel",
               src.count("void fkl_entry") == 1
               and src.count("executeOperations") == 2
               and all(tok in src for tok in
                       ("Crop", "Resize", "Sub", "Div", "TensorSplit")))


def t_pipe_batch_list():
    """HF: batch pipeline plane b == single-image pipeline on image b."""
    W, H = 12, 10
    MEAN, STD = (100.0, 110.0, 120.0), (50.0, 51.0, 52.0)
    srcs = [[(i * (k + 3)) % 256 for i in range(W * H * 3)] for k in range(3)]
    imgs = [dev_u8(s, W, H, ch=3) for s in srcs]
    out = (fkl.pipe(fkl.ImageBatch(imgs))
           .resize((6, 5)).normalize(MEAN, STD).to_planar("CHW").run())
    check_true("pipe(batch): semantic shape (N, C, H, W)",
               out.shape == (3, 3, 5, 6))
    got = unf32(out.base.copy_to_host(), 3 * 3 * 5 * 6)
    per = 3 * 5 * 6
    for b in range(3):
        single = (fkl.pipe(fkl.Image(imgs[b]))
                  .resize((6, 5)).normalize(MEAN, STD).to_planar("CHW").run())
        check(f"pipe(batch): plane {b} == single-image pipeline",
              got[b * per:(b + 1) * per],
              unf32(single.base.copy_to_host(), per))


def t_pipe_batch_tensor_equals_list():
    W, H, N = 8, 6, 3
    flat = [(i * 11) % 256 for i in range(N * H * W * 3)]
    per = H * W * 3
    batched = fkl.DeviceBuffer(W, H, "uint8", channels=3, planes=N)
    batched.copy_from_host(u8(flat))
    imgs = [dev_u8(flat[k * per:(k + 1) * per], W, H, ch=3) for k in range(N)]
    ops = lambda p: p.resize((4, 3)).mul(2.0).to_planar("CHW").run()
    a = ops(fkl.pipe(fkl.ImageBatch(batched)))
    b = ops(fkl.pipe(fkl.ImageBatch(imgs)))
    n = N * 3 * 3 * 4
    check_true("pipe: batched tensor shape == list shape", a.shape == b.shape)
    check("pipe: ImageBatch(tensor) == ImageBatch(list)",
          unf32(a.base.copy_to_host(), n), unf32(b.base.copy_to_host(), n))


def t_functional_eager():
    W, H = 10, 6
    src = [(i * 19) % 256 for i in range(W * H * 3)]
    img = fkl.Image(dev_u8(src, W, H, ch=3))
    r = fkl.F.resize(img, (5, 3))
    ref = fkl.compose(fkl.TensorRead(), fkl.Resize(5, 3),
                      fkl.TensorWrite())(dev_u8(src, W, H, ch=3))
    check("F.resize == compose(Resize)",
          unf32(r.base.copy_to_host(), 45), unf32(ref.copy_to_host(), 45),
          tol=1e-5)
    m = fkl.F.mul(fkl.Image(dev_f32([float(i) for i in range(24)], 6, 4)), 3.0)
    check("F.mul vs python ref", unf32(m.base.copy_to_host(), 24),
          [i * 3.0 for i in range(24)])
    pl = fkl.F.to_planar(img)
    exp = [float(src[px * 3 + c]) for c in range(3) for px in range(W * H)]
    check("F.to_planar vs ref",
          [float(v) for v in unu8(pl.base.copy_to_host(), W * H * 3)], exp)


def t_pipe_out_and_reuse():
    W, H = 8, 4
    src = [float(i % 31) for i in range(W * H)]
    p = fkl.pipe(fkl.Image(dev_f32(src, W, H))).mul(3.0)
    p.run()
    pre = fkl.Tensor.empty((H, W), "float32")
    out = p.run(out=pre)
    check_true("pipe: out= writes into the provided Tensor",
               out.ptr == pre.ptr)
    check("pipe: out= values", unf32(pre.base.copy_to_host(), W * H),
          [v * 3 for v in src])
    check_true("pipe: kernel cached across runs (one variant)",
               p._kernel is not None and len(p._kernel._variants) == 1)


def t_pipe_out_validation():
    W, H = 8, 4
    src = [float(i % 29) for i in range(W * H)]
    p = fkl.pipe(fkl.Image(dev_f32(src, W, H))).mul(2.0)
    small = fkl.Tensor.empty((2, 2), "float32")
    small.base.copy_from_host(f32([-7.0] * 4))
    bad = 0
    try:    # too small: would be an out-of-bounds device write
        p.run(out=small)
    except ValueError:
        bad += 1
    try:    # same element count, wrong dtype: silent reinterpretation
        p.run(out=fkl.Tensor.empty((H, W), "int32"))
    except ValueError:
        bad += 1
    check_true("pipe: out= size/dtype validated BEFORE the launch", bad == 2)
    check("pipe: rejected out= buffer left untouched",
          unf32(small.base.copy_to_host(), 4), [-7.0] * 4)


def t_pipe_source_thread_fusion():
    src = [float(i % 19) for i in range(8 * 4)]
    p = fkl.pipe(fkl.Image(dev_f32(src, 8, 4)), thread_fusion=True).mul(2.0)
    check_true("pipe: source() shows the TF variant that will run",
               "TF::ENABLED" in p.source())
    out = p.run()
    check("pipe: thread_fusion run vs python ref",
          unf32(out.base.copy_to_host(), 32), [v * 2 for v in src])
    q = fkl.pipe(fkl.Image(dev_f32([0.0] * 24, 6, 4)),
                 thread_fusion=True).mul(2.0)
    check_true("pipe: source() shows scalar fallback (row not 16B-aligned)",
               "TF::ENABLED" not in q.source())


def t_pipe_height1_roundtrip():
    W = 8
    src = [float(i) for i in range(W)]
    img = fkl.Image(fkl.Tensor(dev_f32(src, W)).reshape(1, W))
    out = fkl.pipe(img).mul(2.0).run()
    check_true("pipe: (1, W) image keeps its rank", out.shape == (1, W))
    check("pipe: (1, W) values", unf32(out.base.copy_to_host(), W),
          [v * 2 for v in src])
    vec = fkl.pipe(fkl.Tensor(dev_f32(src, W))).mul(2.0).run()
    check_true("pipe: 1-D source stays 1-D", vec.shape == (W,))


def t_pipe_errors():
    W, H = 8, 8
    img = fkl.Image(dev_u8([0] * (W * H * 3), W, H, ch=3))
    bad = 0
    try:   # geometry after compute: BVF ordering violated
        fkl.pipe(img).mul(2.0).resize((4, 4))
    except ValueError:
        bad += 1
    try:   # multi-ROI crop over a batch source
        imgs = [dev_u8([0] * (W * H * 3), W, H, ch=3) for _ in range(2)]
        fkl.pipe(fkl.ImageBatch(imgs)).crop([(0, 0, 4, 4), (1, 1, 4, 4)])
    except ValueError:
        bad += 1
    try:   # normalize channel mismatch
        fkl.pipe(img).normalize((1.0, 2.0), (1.0, 2.0))
    except ValueError:
        bad += 1
    try:   # to_planar on single-channel chain
        fkl.pipe(fkl.Image(dev_u8([0] * (W * H), W, H))).to_planar()
    except ValueError:
        bad += 1
    check_true("pipe: ordering/batch/channel/planar errors are eager",
               bad == 4)


if __name__ == "__main__":
    run([t_tensor_wrap, t_tensor_empty, t_tensor_empty_vector,
         t_tensor_reshape, t_tensor_dlpack, t_dlpack_shared_buffer_ownership,
         t_image_semantics, t_image_rejects, t_image_batch_list,
         t_image_batch_tensor, t_image_batch_device_mismatch,
         t_pipe_elementwise, t_pipe_crop,
         t_pipe_resize, t_pipe_cvt_color, t_pipe_normalize_autocast,
         t_pipe_to_planar, t_pipe_dnn_single_kernel, t_pipe_batch_list,
         t_pipe_batch_tensor_equals_list, t_functional_eager,
         t_pipe_out_and_reuse, t_pipe_out_validation,
         t_pipe_source_thread_fusion, t_pipe_height1_roundtrip,
         t_pipe_errors], "highlevel-api")
