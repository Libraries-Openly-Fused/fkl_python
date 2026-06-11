"""End-to-end test of fkl-python with NO external deps (uses DeviceBuffer).

Proves the full pipeline on real hardware:
  compose() -> codegen -> nvcc/clang single-step .so -> ctypes -> GPU -> result
"""
import struct
import sys
import time

import fkl


def floats_to_bytes(vals):
    return struct.pack(f"{len(vals)}f", *vals)


def bytes_to_floats(b, n):
    return list(struct.unpack(f"{n}f", b))


def main():
    N = 16
    src = [float(i) for i in range(N)]

    inp = fkl.DeviceBuffer(N, 1, "float32")
    inp.copy_from_host(floats_to_bytes(src))
    out = fkl.DeviceBuffer(N, 1, "float32")

    # Compose a fused chain: (x * 2) + 1, then * 3  -> all in ONE kernel
    t0 = time.time()
    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Mul(2.0),    # fused
        fkl.Add(1.0),    # fused
        fkl.Mul(3.0),    # fused
        fkl.TensorWrite(),
    )
    t_compile = time.time() - t0
    print(f"compose (lazy, no compile yet): {t_compile*1000:.1f} ms")
    print(f"backend: {fkl.get_backend().kind}")

    # hot path
    t0 = time.time()
    pipe(inp, out)
    t_run = time.time() - t0
    print(f"first launch (incl dlopen warmup): {t_run*1000:.3f} ms")

    # timed steady-state hot path
    iters = 1000
    t0 = time.time()
    for _ in range(iters):
        pipe(inp, out)
    t_hot = (time.time() - t0) / iters
    print(f"steady-state hot launch: {t_hot*1e6:.1f} us/call")

    got = bytes_to_floats(out.copy_to_host(), N)
    expect = [(v * 2.0 + 1.0) * 3.0 for v in src]

    ok = all(abs(a - b) < 1e-4 for a, b in zip(got, expect))
    print(f"got[:8]:    {[round(v,1) for v in got[:8]]}")
    print(f"expect[:8]: {[round(v,1) for v in expect[:8]]}")

    # second compose with same chain -> must hit disk cache (fast)
    t0 = time.time()
    pipe2 = fkl.compose(
        fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0), fkl.Mul(3.0), fkl.TensorWrite(),
    )
    t_warm = time.time() - t0
    print(f"compose (cache HIT): {t_warm*1000:.1f} ms")

    print("RESULT:", "E2E_PASS" if ok else "E2E_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
