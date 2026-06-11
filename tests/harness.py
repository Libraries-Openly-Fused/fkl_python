"""Shared test harness (dependency-free; uses DeviceBuffer + struct)."""
import struct

import fkl

PASS, FAIL = [], []


def f32(vals):
    return struct.pack(f"{len(vals)}f", *vals)


def unf32(b, n):
    return list(struct.unpack(f"{n}f", b[:4 * n]))


def u8(vals):
    return struct.pack(f"{len(vals)}B", *[int(v) & 0xFF for v in vals])


def unu8(b, n):
    return list(struct.unpack(f"{n}B", b[:n]))


def i32(vals):
    return struct.pack(f"{len(vals)}i", *[int(v) for v in vals])


def uni32(b, n):
    return list(struct.unpack(f"{n}i", b[:4 * n]))


def dev_f32(vals, w, h=1, ch=1, planes=1):
    buf = fkl.DeviceBuffer(w, h, "float32", channels=ch, planes=planes)
    buf.copy_from_host(f32(vals))
    return buf


def dev_u8(vals, w, h=1, ch=1, planes=1):
    buf = fkl.DeviceBuffer(w, h, "uint8", channels=ch, planes=planes)
    buf.copy_from_host(u8(vals))
    return buf


def check(name, got, expect, tol=1e-3):
    ok = len(got) == len(expect) and all(
        abs(float(a) - float(b)) <= tol for a, b in zip(got, expect))
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"    got    [:8]: {[round(float(v), 3) for v in got[:8]]} (n={len(got)})")
        print(f"    expect [:8]: {[round(float(v), 3) for v in expect[:8]]} (n={len(expect)})")


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))


def run(tests, label):
    import sys
    for t in tests:
        try:
            t()
        except Exception as e:
            FAIL.append(t.__name__)
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n[{label}] {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
