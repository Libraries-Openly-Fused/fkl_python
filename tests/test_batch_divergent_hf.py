"""Batch-of-images Horizontal Fusion + DIVERGENT HF (paper's 4th technique)
+ 16-bit dtypes + TensorSplit E2E.
"""
import struct
from harness import (dev_f32, dev_u8, unf32, unu8, f32, check, check_true, run)
import fkl


# ============ HF over a batch of separate images =========================

def t_batch_images_basic():
    """3 separate images -> ONE kernel (BatchRead) -> Tensor with 3 planes."""
    W, H = 16, 8
    imgs, expected = [], []
    for k in range(3):
        vals = [float((k + 1) * 100 + (y * W + x) % 50)
                for y in range(H) for x in range(W)]
        imgs.append(dev_f32(vals, W, H))
        expected.extend(v * 2.0 + 1.0 for v in vals)
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0),
                       fkl.TensorWrite())
    out = pipe(imgs)                      # list input => batch HF
    got = unf32(out.copy_to_host(), W * H * 3)
    check("HF-batch: 3 images Mul+Add -> Tensor(3)", got, expected)
    check_true("HF-batch: one signature", len(pipe._variants) == 1)


def t_batch_images_uchar3():
    """Batch of uchar3 images -> normalize -> float3 Tensor."""
    W, H = 8, 4
    n = W * H * 3
    imgs, expected = [], []
    for k in range(4):
        vals = [((k * 31 + i * 7) % 256) for i in range(n)]
        imgs.append(dev_u8(vals, W, H, ch=3))
        expected.extend(v / 255.0 for v in vals)
    pipe = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                       fkl.Div(255.0), fkl.TensorWrite())
    out = pipe(imgs)
    check("HF-batch: 4 uchar3 images normalized", 
          unf32(out.copy_to_host(), n * 4), expected, tol=1e-5)


def t_batch_size_is_type():
    """B is part of the C++ type: B=2 vs B=3 compile separately; same B
    reuses; the single-image path stays independent."""
    W, H = 8, 4
    vals = [float(i) for i in range(W * H)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite())
    pipe(dev_f32(vals, W, H))                                 # single
    pipe([dev_f32(vals, W, H), dev_f32(vals, W, H)])          # B=2
    pipe([dev_f32(vals, W, H)] * 3)                           # B=3
    pipe([dev_f32(vals, W, H), dev_f32(vals, W, H)])          # B=2 again
    check_true("HF-batch: 3 signatures (single, B=2, B=3)",
               len(pipe._variants) == 3, f"{len(pipe._variants)} variants")


def t_batch_plus_crop():
    """Batch images + per-thread compute + write: each plane is an
    independent image but ALL processed by one kernel."""
    W, H = 12, 6
    img_a = [float(y * W + x) for y in range(H) for x in range(W)]
    img_b = [float(1000 + y * W + x) for y in range(H) for x in range(W)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Sub(1.0), fkl.TensorWrite())
    out = pipe([dev_f32(img_a, W, H), dev_f32(img_b, W, H)])
    got = unf32(out.copy_to_host(), W * H * 2)
    check("HF-batch: plane independence", got,
          [v - 1 for v in img_a] + [v - 1 for v in img_b])


# ============ DIVERGENT HF (4th fusion technique) =========================

def t_divergent_two_sequences():
    """Plane 0 runs Mul(2); planes 1,2 run Add(100). ONE kernel."""
    W, H = 8, 4
    base = [float(i) for i in range(W * H)]
    imgs = [dev_f32(base, W, H) for _ in range(3)]
    pipe = fkl.compose_divergent(
        [1, 2, 2],
        [fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite()],
        [fkl.TensorRead(), fkl.Add(100.0), fkl.TensorWrite()],
    )
    out = pipe(imgs)
    got = unf32(out.copy_to_host(), W * H * 3)
    exp = ([v * 2 for v in base] +      # plane 0 -> seq 1
           [v + 100 for v in base] +    # plane 1 -> seq 2
           [v + 100 for v in base])     # plane 2 -> seq 2
    check("DivergentHF: [Mul2 | Add100 | Add100]", got, exp)


def t_divergent_three_sequences():
    W, H = 4, 4
    base = [float(i % 17) for i in range(W * H)]
    imgs = [dev_f32(base, W, H) for _ in range(4)]
    pipe = fkl.compose_divergent(
        [1, 2, 3, 1],
        [fkl.TensorRead(), fkl.Mul(3.0), fkl.TensorWrite()],
        [fkl.TensorRead(), fkl.Sub(1.0), fkl.TensorWrite()],
        [fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(5.0), fkl.TensorWrite()],
    )
    out = pipe(imgs)
    got = unf32(out.copy_to_host(), W * H * 4)
    exp = ([v * 3 for v in base] + [v - 1 for v in base] +
           [v * 2 + 5 for v in base] + [v * 3 for v in base])
    check("DivergentHF: 3 sequences over 4 planes", got, exp)


def t_divergent_deep_chains():
    """Divergent with chains of different LENGTHS (deeper VF in one branch)."""
    W, H = 8, 2
    base = [float(i + 1) for i in range(W * H)]
    imgs = [dev_f32(base, W, H) for _ in range(2)]
    pipe = fkl.compose_divergent(
        [1, 2],
        [fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0), fkl.Sub(0.5),
         fkl.Div(2.0), fkl.TensorWrite()],
        [fkl.TensorRead(), fkl.Max(5.0), fkl.TensorWrite()],
    )
    out = pipe(imgs)
    got = unf32(out.copy_to_host(), W * H * 2)
    exp = ([((v * 2 + 1) - 0.5) / 2 for v in base] +
           [max(5.0, v) for v in base])
    check("DivergentHF: different chain depths", got, exp)


# ============ 16-bit dtypes ===============================================

def t_uint16_chain():
    W, H = 8, 4
    n = W * H
    src = [(i * 257) % 65536 for i in range(n)]
    buf = fkl.DeviceBuffer(W, H, "uint16")
    buf.copy_from_host(struct.pack(f"{n}H", *src))
    out = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                      fkl.Mul(0.5), fkl.TensorWrite())(buf)
    check("uint16 -> float chain", unf32(out.copy_to_host(), n),
          [v * 0.5 for v in src])


def t_int16_chain():
    N = 32
    src = [(i * 100) - 1600 for i in range(N)]
    buf = fkl.DeviceBuffer(N, 1, "int16")
    buf.copy_from_host(struct.pack(f"{N}h", *src))
    out = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                      fkl.Add(10.0), fkl.TensorWrite())(buf)
    check("int16 (negative) -> float chain", unf32(out.copy_to_host(), N),
          [float(v) + 10 for v in src])


def t_uint16x3_vector():
    W, H = 4, 2
    n = W * H * 3
    src = [(i * 1000) % 65536 for i in range(n)]
    buf = fkl.DeviceBuffer(W, H, "uint16", channels=3)
    buf.copy_from_host(struct.pack(f"{n}H", *src))
    out = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                      fkl.Div(65535.0), fkl.TensorWrite())(buf)
    check("ushort3 -> float3 normalize", unf32(out.copy_to_host(), n),
          [v / 65535.0 for v in src], tol=1e-6)


# ============ TensorSplit E2E (HF batch + planar output) ===================

def t_tensorsplit_after_batch():
    """The DNN-ingest pattern: batch of uchar3 images -> normalize ->
    TensorSplit -> planar float planes per image."""
    W, H = 4, 2
    n = W * H * 3
    imgs, srcs = [], []
    for k in range(2):
        vals = [((k * 11 + i * 3) % 256) for i in range(n)]
        srcs.append(vals)
        imgs.append(dev_u8(vals, W, H, ch=3))
    pipe = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                       fkl.Div(255.0), fkl.TensorSplit())
    out = pipe(imgs)
    got = unf32(out.copy_to_host(), n * 2)
    exp = []
    for vals in srcs:                 # per image: 3 planes (CHW)
        for c in range(3):
            for p in range(W * H):
                exp.append(vals[p * 3 + c] / 255.0)
    check("TensorSplit E2E: batch->planar CHW x2 images", got, exp, tol=1e-6)


if __name__ == "__main__":
    run([t_batch_images_basic, t_batch_images_uchar3, t_batch_size_is_type,
         t_batch_plus_crop, t_divergent_two_sequences,
         t_divergent_three_sequences, t_divergent_deep_chains,
         t_uint16_chain, t_int16_chain, t_uint16x3_vector,
         t_tensorsplit_after_batch], "batch-divergent-hf")
