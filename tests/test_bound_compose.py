"""COMPOSE-TIME BUFFER BINDING tests: TensorRead(x)/TensorWrite(out) bind
concrete buffers at compose(); the kernel compiles EAGERLY and runs with no
arguments (k() / k(stream=s)). Explicit args still override the binding,
validated against the compiled signature. Plus DeviceBuffer.from_ptr
(non-owning raw-pointer wrap) bound straight into a chain.
"""
import ctypes

from harness import dev_f32, dev_u8, unf32, unu8, f32, check, check_true, run
import fkl


def t_bound_argless():
    """read+write bound at compose -> k() runs with no arguments."""
    W, H = 33, 17  # non-pow2 to catch pitch errors
    src = [float((y * W + x) % 97) for y in range(H) for x in range(W)]
    x = dev_f32(src, W, H)
    out = fkl.DeviceBuffer(W, H, "float32")
    k = fkl.compose(fkl.TensorRead(x), fkl.Mul(2.0), fkl.Add(1.0),
                    fkl.TensorWrite(out))
    ret = k()
    check_true("bound: k() returns the bound output", ret is out)
    check("bound: argless call computes", unf32(out.copy_to_host(), W * H),
          [v * 2 + 1 for v in src])
    # the binding is by POINTER: update the input in place, rerun argless
    src2 = [v + 1 for v in src]
    x.copy_from_host(f32(src2))
    k()
    check("bound: rerun sees updated input data",
          unf32(out.copy_to_host(), W * H), [v * 2 + 1 for v in src2])


def t_bound_stream():
    """k(stream=s) on a bound kernel: async launch on a caller-owned
    driver-API stream, caller syncs."""
    W = 64
    src = [float(i % 53) for i in range(W)]
    x, out = dev_f32(src, W), fkl.DeviceBuffer(W, 1, "float32")
    k = fkl.compose(fkl.TensorRead(x), fkl.Mul(4.0), fkl.TensorWrite(out))

    cu = fkl.DeviceBuffer._cuda
    cu.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cu.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    cu.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    s = ctypes.c_void_p()
    check_true("bound: cuStreamCreate", cu.cuStreamCreate(ctypes.byref(s), 0) == 0)
    k(stream=int(s.value))          # async: caller owns the stream
    cu.cuStreamSynchronize(s)
    check("bound: k(stream=s) computes", unf32(out.copy_to_host(), W),
          [v * 4 for v in src])
    cu.cuStreamDestroy_v2(s)


def t_eager_compile():
    """Binding compiles at compose() time; a second compose of the same
    chain signature is a pure cache hit (same .so)."""
    W = 32
    src = [float(i) for i in range(W)]
    x, out = dev_f32(src, W), fkl.DeviceBuffer(W, 1, "float32")
    k = fkl.compose(fkl.TensorRead(x), fkl.Mul(3.0), fkl.TensorWrite(out))
    check_true("eager: compiled inside compose() (before any call)",
               len(k._variants) == 1)
    so1 = list(k._variants.values())[0][3]
    import time
    t0 = time.perf_counter()
    k2 = fkl.compose(fkl.TensorRead(x), fkl.Mul(7.0), fkl.TensorWrite(out))
    ms = (time.perf_counter() - t0) * 1e3
    so2 = list(k2._variants.values())[0][3]
    check_true("eager: second compose reuses the cached .so",
               so1 == so2, f"compose took {ms:.1f} ms")
    k2()
    check("eager: cached kernel computes with its own values",
          unf32(out.copy_to_host(), W), [v * 7 for v in src])


def t_read_only_bound():
    """Binding only the read: k() auto-allocates the output per call."""
    W = 16
    src = [float(i) for i in range(W)]
    k = fkl.compose(fkl.TensorRead(dev_f32(src, W)), fkl.Mul(2.0),
                    fkl.TensorWrite())
    out = k()
    check("bound: read-only binding auto-allocates output",
          unf32(out.copy_to_host(), W), [v * 2 for v in src])


def t_write_only_bound():
    """Binding only the write: pipe(x) uses the bound buffer as default out
    (lazy compile as usual — input signature arrives with x)."""
    W = 16
    src = [float(i) for i in range(W)]
    out = fkl.DeviceBuffer(W, 1, "float32")
    k = fkl.compose(fkl.TensorRead(), fkl.Sub(1.0), fkl.TensorWrite(out))
    check_true("bound: write-only binding stays lazy", len(k._variants) == 0)
    ret = k(dev_f32(src, W))
    check_true("bound: write-only binding is the default out", ret is out)
    check("bound: write-only values", unf32(out.copy_to_host(), W),
          [v - 1 for v in src])


def t_override_at_call():
    """Explicit args override the binding (same compiled signature)."""
    W = 24
    a = [float(i % 19) for i in range(W)]
    b = [float(i * 3 % 23) for i in range(W)]
    x, out = dev_f32(a, W), fkl.DeviceBuffer(W, 1, "float32")
    k = fkl.compose(fkl.TensorRead(x), fkl.Mul(2.0), fkl.TensorWrite(out))
    y = dev_f32(b, W)
    r1 = k(y)                       # input override -> still writes bound out
    check_true("override: k(y) writes the bound output", r1 is out)
    check("override: k(y) values", unf32(out.copy_to_host(), W),
          [v * 2 for v in b])
    z = fkl.DeviceBuffer(W, 1, "float32")
    r2 = k(y, out=z)                # both overridden
    check_true("override: k(y, out=z) returns z", r2 is z)
    check("override: k(y, out=z) values", unf32(z.copy_to_host(), W),
          [v * 2 for v in b])
    k()                             # bindings intact after overrides
    check("override: k() still uses the bindings",
          unf32(out.copy_to_host(), W), [v * 2 for v in a])


def t_bound_uchar3_split():
    """Bound DNN-ingest chain: uchar3 -> normalize -> planar CHW, argless.
    TensorSplit's destination layout is (C, H, W) of the base dtype."""
    W, H = 8, 4
    n = W * H * 3
    src = [(i * 7) % 256 for i in range(n)]
    x = dev_u8(src, W, H, ch=3)
    out = fkl.DeviceBuffer(W, H, "float32", planes=3)  # 3 planes of HxW
    k = fkl.compose(fkl.TensorRead(x), fkl.Cast("float32"),
                    fkl.Div((255.0,) * 3), fkl.TensorSplit(out))
    k()
    got = unf32(out.copy_to_host(), n)
    exp = [src[p * 3 + c] / 255.0
           for c in range(3) for p in range(W * H)]
    check("bound: uchar3 -> planar CHW argless", got, exp)


def t_from_ptr_roundtrip():
    """from_ptr wraps a DeviceBuffer's own pointer: same bytes, non-owning,
    usable as a bound chain input."""
    W, H = 8, 4
    src = [float((y * W + x) % 31) for y in range(H) for x in range(W)]
    owner = dev_f32(src, W, H)
    view = fkl.DeviceBuffer.from_ptr(owner.ptr, (H, W), "float32")
    check_true("from_ptr: same pointer, non-owning",
               view.ptr == owner.ptr and not view._owns)
    check("from_ptr: reads the owner's bytes",
          unf32(view.copy_to_host(), W * H), src)
    out = fkl.DeviceBuffer(W, H, "float32")
    fkl.compose(fkl.TensorRead(view), fkl.Mul(2.0), fkl.TensorWrite(out))()
    check("from_ptr: kernel reads through the wrapper",
          unf32(out.copy_to_host(), W * H), [v * 2 for v in src])
    del view  # non-owning: must NOT free the owner's memory
    check("from_ptr: owner memory survives wrapper deletion",
          unf32(owner.copy_to_host(), W * H), src)


def t_from_ptr_output_and_vector():
    """from_ptr as the bound OUTPUT (raw destination pointer) and with a
    vector dtype spec ('uint8x3' == shape (H, W, 3))."""
    W, H = 4, 4
    n = W * H * 3
    src = [(i * 5) % 256 for i in range(n)]
    x = dev_u8(src, W, H, ch=3)
    dst = fkl.DeviceBuffer(W, H, "uint8", channels=3)
    wrap = fkl.DeviceBuffer.from_ptr(dst.ptr, (H, W), "uint8x3")
    check_true("from_ptr: vector dtype spec folds channels",
               wrap.dtype.ctype == "uchar3" and wrap.width == W
               and wrap.height == H)
    fkl.compose(fkl.TensorRead(x), fkl.VectorReorder(2, 1, 0),
                fkl.TensorWrite(wrap))()
    got = [float(v) for v in unu8(dst.copy_to_host(), n)]
    exp = [float(src[p * 3 + (2 - c)]) for p in range(W * H) for c in range(3)]
    check("from_ptr: raw-pointer output receives results", got, exp)


def t_from_ptr_guards():
    """from_ptr argument validation: null pointers, ambiguous vector-dtype +
    trailing-channel-dim specs, and CAI v3 stream spelling."""
    W, H = 8, 4
    owner = fkl.DeviceBuffer(W, H, "uint8", channels=3)

    # NULL is never a valid device pointer — fail at the call site, not
    # later in .ptr / copy_to_host
    for bad in (0, None):
        try:
            fkl.DeviceBuffer.from_ptr(bad, (H, W), "uint8x3")
            check_true(f"from_ptr: null ptr ({bad!r}) rejected", False)
        except ValueError as e:
            check_true(f"from_ptr: null ptr ({bad!r}) rejected",
                       "non-null" in str(e), str(e)[:60])

    # vector dtype AND the trailing channel dim would double-count the
    # channels ((H, W, 3) x uchar3 = 3x the real allocation) — reject
    try:
        fkl.DeviceBuffer.from_ptr(owner.ptr, (H, W, 3), "uint8x3")
        check_true("from_ptr: vector dtype + trailing dim rejected", False)
    except TypeError as e:
        check_true("from_ptr: vector dtype + trailing dim rejected",
                   "ambiguous" in str(e), str(e)[:60])

    # CAI v3 stream key: 0 (legacy default stream) must be spelled 1,
    # None means "no synchronization needed" (key omitted)
    cai = fkl.DeviceBuffer.from_ptr(owner.ptr, (H, W), "uint8x3",
                                    stream=0).__cuda_array_interface__
    check_true("from_ptr: stream=0 advertised as legacy default (1)",
               cai.get("stream") == 1)
    cai = fkl.DeviceBuffer.from_ptr(owner.ptr, (H, W), "uint8x3",
                                    stream=None).__cuda_array_interface__
    check_true("from_ptr: stream=None omits the CAI stream key",
               "stream" not in cai)
    cai = fkl.DeviceBuffer.from_ptr(owner.ptr, (H, W), "uint8x3",
                                    stream=1234).__cuda_array_interface__
    check_true("from_ptr: explicit stream handle advertised as-is",
               cai.get("stream") == 1234)


def t_errors():
    W = 16
    src = [float(i) for i in range(W)]
    x, out = dev_f32(src, W), fkl.DeviceBuffer(W, 1, "float32")

    # argless call without a bound input
    k = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite())
    try:
        k()
        check_true("err: argless call without binding raises", False)
    except TypeError as e:
        check_true("err: argless call without binding raises",
                   "bound" in str(e), str(e)[:60])

    kb = fkl.compose(fkl.TensorRead(x), fkl.Mul(2.0), fkl.TensorWrite(out))
    # dtype mismatch on input override
    try:
        kb(fkl.DeviceBuffer(W, 1, "int32"))
        check_true("err: input override dtype mismatch", False)
    except ValueError as e:
        check_true("err: input override dtype mismatch",
                   "mismatch" in str(e), str(e)[:60])
    # shape mismatch on input override
    try:
        kb(fkl.DeviceBuffer(W * 2, 1, "float32"))
        check_true("err: input override shape mismatch", False)
    except ValueError:
        check_true("err: input override shape mismatch", True)
    # out= override mismatch on a bound kernel
    try:
        kb(out=fkl.DeviceBuffer(W // 2, 1, "float32"))
        check_true("err: out= override size mismatch", False)
    except ValueError:
        check_true("err: out= override size mismatch", True)
    # batch (list) override of an eagerly-compiled bound kernel
    try:
        kb([x, x])
        check_true("err: batch override of bound kernel", False)
    except ValueError:
        check_true("err: batch override of bound kernel", True)
    # bound output validated at compose() time
    try:
        fkl.compose(fkl.TensorRead(x), fkl.Mul(2.0),
                    fkl.TensorWrite(fkl.DeviceBuffer(W // 2, 1, "float32")))
        check_true("err: bound output size mismatch at compose", False)
    except ValueError:
        check_true("err: bound output size mismatch at compose", True)
    # binding is GPU-only (CPU chains take numpy arrays per call)
    try:
        fkl.compose(fkl.TensorRead(x), fkl.Mul(2.0), fkl.TensorWrite(),
                    target="cpu")
        check_true("err: cpu target rejects binding", False)
    except ValueError:
        check_true("err: cpu target rejects binding", True)
    # a batch (list) cannot be bound — pass it at call time instead
    try:
        fkl.compose(fkl.TensorRead([x, x]), fkl.Mul(2.0), fkl.TensorWrite())
        check_true("err: list binding rejected at compose", False)
    except TypeError:
        check_true("err: list binding rejected at compose", True)
    # divergent chains do not take bindings (batch arrives at call time)
    try:
        fkl.compose_divergent(
            [1, 1],
            [fkl.TensorRead(x), fkl.Mul(2.0), fkl.TensorWrite()])
        check_true("err: compose_divergent rejects binding", False)
    except ValueError:
        check_true("err: compose_divergent rejects binding", True)
    # unbound behavior unchanged: lazy compile, explicit args required
    check_true("unbound: compose stays lazy", len(k._variants) == 0)
    o = k(dev_f32(src, W))
    check("unbound: pipe(x) path intact", unf32(o.copy_to_host(), W),
          [v * 2 for v in src])


if __name__ == "__main__":
    run([t_bound_argless, t_bound_stream, t_eager_compile,
         t_read_only_bound, t_write_only_bound, t_override_at_call,
         t_bound_uchar3_split, t_from_ptr_roundtrip,
         t_from_ptr_output_and_vector, t_from_ptr_guards, t_errors],
        "bound-compose")
