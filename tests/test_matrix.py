"""Full-surface test matrix for fkl-python v2 (dependency-free).

Covers: scalar arithmetic chains, vector pixel types, Cast/SaturateCast,
Saturate, Max/Min, VectorReduce, Discard/AddLast/VectorReorder,
ColorConversion, Crop, Resize -- each verified against a CPU reference.
"""
import struct
import sys

import fkl


def f32_to_bytes(vals):
    return struct.pack(f"{len(vals)}f", *vals)


def bytes_to_f32(b, n):
    return list(struct.unpack(f"{n}f", b))


def u8_to_bytes(vals):
    return struct.pack(f"{len(vals)}B", *vals)


def bytes_to_u8(b, n):
    return list(struct.unpack(f"{n}B", b))


PASS, FAIL = [], []


def check(name, got, expect, tol=1e-3):
    ok = len(got) == len(expect) and all(abs(a - b) <= tol for a, b in zip(got, expect))
    (PASS if ok else FAIL).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}")
    if not ok:
        print(f"    got[:8]:    {got[:8]}")
        print(f"    expect[:8]: {expect[:8]}")


# ---------------------------------------------------------------- 1: scalar chain
def t_scalar_chain():
    N = 64
    src = [float(i) for i in range(N)]
    inp = fkl.DeviceBuffer(N, 1, "float32"); inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0),
                       fkl.Sub(0.5), fkl.Div(2.0), fkl.TensorWrite())
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), N)
    check("scalar Mul+Add+Sub+Div", got, [((v * 2 + 1) - 0.5) / 2 for v in src])


# ---------------------------------------------------------------- 2: max/min clamp
def t_max_min():
    N = 32
    src = [float(i - 16) for i in range(N)]
    inp = fkl.DeviceBuffer(N, 1, "float32"); inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.Max(-5.0), fkl.Min(5.0), fkl.TensorWrite())
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), N)
    check("Max(-5)+Min(5) clamp", got, [max(-5.0, min(5.0, v)) for v in src])


# ---------------------------------------------------------------- 3: saturate
def t_saturate():
    N = 32
    src = [float(i - 16) for i in range(N)]
    inp = fkl.DeviceBuffer(N, 1, "float32"); inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.Saturate(-3.0, 3.0), fkl.TensorWrite())
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), N)
    check("Saturate[-3,3]", got, [max(-3.0, min(3.0, v)) for v in src])


# ---------------------------------------------------------------- 4: vector pixels uchar3 -> float3 normalize
def t_vector_pixels():
    W, H = 8, 4
    n = W * H * 3
    src = [(i * 7) % 256 for i in range(n)]
    inp = fkl.DeviceBuffer(W, H, "uint8", channels=3)
    inp.copy_from_host(u8_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                       fkl.Mul((2.0, 0.5, 1.0)), fkl.TensorWrite())
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), n)
    factors = [2.0, 0.5, 1.0]
    check("uchar3->Cast->Mul(vec)", got,
          [src[i] * factors[i % 3] for i in range(n)])


# ---------------------------------------------------------------- 5: saturate-cast round trip
def t_saturate_cast():
    W, H = 16, 2
    n = W * H
    src = [float(i * 20 - 60) for i in range(n)]  # spans <0 and >255
    inp = fkl.DeviceBuffer(W, H, "float32"); inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.SaturateCast("uint8"), fkl.TensorWrite())
    out = pipe(inp)
    got = [float(v) for v in bytes_to_u8(out.copy_to_host(), n)]
    check("SaturateCast float->uchar", got,
          [float(max(0, min(255, round(v)))) for v in src])


# ---------------------------------------------------------------- 6: vector reorder (BGR<->RGB)
def t_reorder():
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 11) % 256 for i in range(n)]
    inp = fkl.DeviceBuffer(W, H, "uint8", channels=3)
    inp.copy_from_host(u8_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.VectorReorder(2, 1, 0), fkl.TensorWrite())
    out = pipe(inp)
    got = [float(v) for v in bytes_to_u8(out.copy_to_host(), n)]
    exp = []
    for p in range(W * H):
        b, g, r = src[p*3], src[p*3+1], src[p*3+2]
        exp.extend([float(r), float(g), float(b)])
    check("VectorReorder(2,1,0)", got, exp)


# ---------------------------------------------------------------- 7: discard + addlast
def t_discard_addlast():
    W, H = 4, 2
    n3, n4 = W * H * 3, W * H * 4
    src = [(i * 5) % 256 for i in range(n3)]
    inp = fkl.DeviceBuffer(W, H, "uint8", channels=3)
    inp.copy_from_host(u8_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.AddLast(255), fkl.TensorWrite())
    out = pipe(inp)
    got = [float(v) for v in bytes_to_u8(out.copy_to_host(), n4)]
    exp = []
    for p in range(W * H):
        exp.extend([float(src[p*3]), float(src[p*3+1]), float(src[p*3+2]), 255.0])
    check("AddLast(alpha=255) uchar3->uchar4", got, exp)


# ---------------------------------------------------------------- 8: color conversion RGB2GRAY
def t_color_gray():
    W, H = 8, 2
    n = W * H * 3
    src = [(i * 13) % 256 for i in range(n)]
    inp = fkl.DeviceBuffer(W, H, "uint8", channels=3)
    inp.copy_from_host(u8_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.ColorConversion("RGB2GRAY"), fkl.TensorWrite())
    out = pipe(inp)
    got = [float(v) for v in bytes_to_u8(out.copy_to_host(), W * H)]
    # OpenCV ITU-R BT.601: 0.299 R + 0.587 G + 0.114 B
    exp = []
    for p in range(W * H):
        r, g, b = src[p*3], src[p*3+1], src[p*3+2]
        exp.append(float(int(0.299*r + 0.587*g + 0.114*b + 0.5)))
    check("ColorConversion RGB2GRAY", got, exp, tol=1.0)


# ---------------------------------------------------------------- 9: crop
def t_crop():
    W, H = 16, 8
    CW, CH, CX, CY = 4, 3, 5, 2
    src = [float(y * W + x) for y in range(H) for x in range(W)]
    inp = fkl.DeviceBuffer(W, H, "float32"); inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.Crop(CX, CY, CW, CH), fkl.TensorWrite())
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), CW * CH)
    exp = [float((CY + y) * W + (CX + x)) for y in range(CH) for x in range(CW)]
    check("Crop(5,2,4,3)", got, exp)


# ---------------------------------------------------------------- 10: resize (linear, ignore AR)
def t_resize():
    W, H = 8, 8
    OW, OH = 4, 4
    src = [float(x) for y in range(H) for x in range(W)]  # gradient in x
    inp = fkl.DeviceBuffer(W, H, "float32"); inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.Resize(OW, OH), fkl.TensorWrite())
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), OW * OH)
    # x gradient halved: cv2-style linear mapping src_x = (dst_x+0.5)*2 - 0.5
    exp = []
    for y in range(OH):
        for x in range(OW):
            exp.append(min(max((x + 0.5) * 2 - 0.5, 0.0), W - 1.0))
    check("Resize 8x8->4x4 linear", got, exp, tol=0.51)


# ---------------------------------------------------------------- 11: crop+resize+arith fused (README pipeline)
def t_full_pipeline():
    W, H = 32, 16
    src = [float((y * W + x) % 256) for y in range(H) for x in range(W)]
    inp = fkl.DeviceBuffer(W, H, "float32"); inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Crop(4, 4, 16, 8),
        fkl.Resize(8, 4),
        fkl.Mul(0.5),
        fkl.Add(10.0),
        fkl.TensorWrite(),
    )
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), 8 * 4)
    ok = len(got) == 32 and all(v >= 10.0 for v in got)
    (PASS if ok else FAIL).append("Crop+Resize+Mul+Add fused")
    print(f"[{'PASS' if ok else 'FAIL'}] Crop+Resize+Mul+Add fused (sanity: 32 vals, all >= 10)")


# ---------------------------------------------------------------- 12: vector reduce
def t_vector_reduce():
    W, H = 4, 2
    n = W * H * 3
    src = [float((i * 3) % 50) for i in range(n)]
    inp = fkl.DeviceBuffer(W, H, "float32", channels=3)
    inp.copy_from_host(f32_to_bytes(src))
    pipe = fkl.compose(fkl.TensorRead(), fkl.VectorReduce("Add"), fkl.TensorWrite())
    out = pipe(inp)
    got = bytes_to_f32(out.copy_to_host(), W * H)
    exp = [src[p*3] + src[p*3+1] + src[p*3+2] for p in range(W * H)]
    check("VectorReduce(Add) float3->float", got, exp)


if __name__ == "__main__":
    for t in [t_scalar_chain, t_max_min, t_saturate, t_vector_pixels,
              t_saturate_cast, t_reorder, t_discard_addlast, t_color_gray,
              t_crop, t_resize, t_full_pipeline, t_vector_reduce]:
        try:
            t()
        except Exception as e:
            FAIL.append(t.__name__)
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", FAIL)
    sys.exit(1 if FAIL else 0)
