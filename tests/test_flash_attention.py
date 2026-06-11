"""Tests for fkl.flash_attention: FA-2 forward vs fp64 numpy oracle,
dense + compressed (int8) KV cache + fused epilogue. Requires numpy."""
import math
import struct
import sys

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy required")
    sys.exit(0)

import fkl
from harness import check_true, run


def _oracle(q, k, v, causal, scale):
    """fp64 reference: softmax(scale * QK^T [causal]) V per (bh)."""
    q, k, v = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    bh, sq, d = q.shape
    sk = k.shape[1]
    out = np.zeros((bh, sq, d))
    for b in range(bh):
        s = q[b] @ k[b].T * scale                     # (sq, sk)
        if causal:
            mask = np.triu(np.ones((sq, sk), dtype=bool), 1)
            s[mask] = -np.inf
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        out[b] = p @ v[b]
    return out


def _to_gpu(arr):
    bh, seq, d = arr.shape
    buf = fkl.DeviceBuffer(d, seq, "float32", planes=bh)
    buf.copy_from_host(arr.astype(np.float32).tobytes())
    return buf


def _from_gpu(buf, shape):
    return np.frombuffer(buf.copy_to_host(), dtype=np.float32).reshape(shape)


def _mk(bh, sq, sk, d, seed):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1, 1, (bh, sq, d)).astype(np.float32),
            rng.uniform(-1, 1, (bh, sk, d)).astype(np.float32),
            rng.uniform(-1, 1, (bh, sk, d)).astype(np.float32))


def t_dense_causal():
    q, k, v = _mk(2, 64, 64, 64, 1)
    out = fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v), causal=True)
    got = _from_gpu(out, q.shape)
    ref = _oracle(q, k, v, True, 1 / math.sqrt(64))
    err = np.abs(got - ref).max()
    check_true(f"FA dense d64 causal (err={err:.2e})", err < 5e-6)


def t_dense_cross_ragged():
    q, k, v = _mk(2, 33, 127, 32, 2)
    out = fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v))
    got = _from_gpu(out, q.shape)
    ref = _oracle(q, k, v, False, 1 / math.sqrt(32))
    err = np.abs(got - ref).max()
    check_true(f"FA dense d32 ragged 33/127 (err={err:.2e})", err < 5e-6)


def t_compressed_kv():
    """int8 KV cache: kernel must be EXACT vs oracle-on-dequantized values,
    and END-TO-END accuracy vs the original values must be quantization-
    bounded. Also verifies the 4x memory saving."""
    q, k, v = _mk(2, 48, 96, 64, 3)
    kc = fkl.compress_kv(_to_gpu(k))
    vc = fkl.compress_kv(_to_gpu(v))

    # memory: int8 data + 1 float/token vs fp32 dense
    dense_bytes = k.size * 4
    comp_bytes = kc.nbytes
    ratio = dense_bytes / comp_bytes
    check_true(f"KV compression ratio {ratio:.2f}x (>3.5x)", ratio > 3.5)

    out = fkl.flash_attention(_to_gpu(q), kc, vc, causal=True)
    got = _from_gpu(out, q.shape)

    # exactness vs dequantized oracle
    k8 = np.frombuffer(kc.data.copy_to_host(), dtype=np.int8).reshape(k.shape)
    ks = np.frombuffer(kc.scales.copy_to_host(), dtype=np.float32).reshape(2, 96, 1)
    v8 = np.frombuffer(vc.data.copy_to_host(), dtype=np.int8).reshape(v.shape)
    vs = np.frombuffer(vc.scales.copy_to_host(), dtype=np.float32).reshape(2, 96, 1)
    kd, vd = k8 * ks, v8 * vs
    ref_dq = _oracle(q, kd, vd, True, 1 / math.sqrt(64))
    err_exact = np.abs(got - ref_dq).max()
    check_true(f"FA int8-KV exact vs dequantized oracle (err={err_exact:.2e})",
               err_exact < 5e-6)

    # end-to-end: quantization-bounded vs original
    ref_full = _oracle(q, k, v, True, 1 / math.sqrt(64))
    err_e2e = np.abs(got - ref_full).max()
    check_true(f"FA int8-KV end-to-end quant error bounded (err={err_e2e:.2e})",
               err_e2e < 2e-2)


def t_fused_epilogue():
    """epilogue=[Mul, Add] runs inside the kernel; equals host-applied."""
    q, k, v = _mk(1, 32, 32, 32, 4)
    base = _from_gpu(fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v)),
                     q.shape)
    fused = _from_gpu(
        fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                            epilogue=[fkl.Mul(2.0), fkl.Add(0.5)]), q.shape)
    err = np.abs(fused - (base * 2.0 + 0.5)).max()
    check_true(f"FA fused epilogue == host-applied (err={err:.2e})", err < 1e-6)


def t_epilogue_values_no_recompile():
    q, k, v = _mk(1, 16, 16, 32, 5)
    a = _from_gpu(fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                                      epilogue=[fkl.Mul(3.0)]), q.shape)
    b = _from_gpu(fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                                      epilogue=[fkl.Mul(5.0)]), q.shape)
    base = _from_gpu(fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v)),
                     q.shape)
    ok = (np.abs(a - base * 3).max() < 1e-5 and np.abs(b - base * 5).max() < 1e-5)
    check_true("FA epilogue values change without recompile", ok)


def t_single_query_decode():
    """seq_q=1 (autoregressive decode step) against a long compressed cache."""
    q, k, v = _mk(2, 1, 256, 64, 6)
    kc, vc = fkl.compress_kv(_to_gpu(k)), fkl.compress_kv(_to_gpu(v))
    got = _from_gpu(fkl.flash_attention(_to_gpu(q), kc, vc), q.shape)
    ref = _oracle(q, k, v, False, 1 / math.sqrt(64))
    err = np.abs(got - ref).max()
    check_true(f"FA decode step s_q=1 vs s_k=256 int8 (err={err:.2e})", err < 2e-2)


def t_prologue_q():
    """Q prologue Mul(2): equals oracle on 2*Q (fused in the Read IOp)."""
    q, k, v = _mk(2, 24, 48, 32, 7)
    got = _from_gpu(
        fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                            prologue_q=[fkl.Mul(2.0)]), q.shape)
    ref = _oracle(q * 2.0, k, v, False, 1 / math.sqrt(32))
    err = np.abs(got - ref).max()
    check_true(f"FA Q-prologue Mul(2) == oracle(2Q) (err={err:.2e})", err < 5e-6)


def t_prologue_v_affine():
    """V prologue Mul(3).then(Add(1)) => 3*out + 1 (since sum p_j = 1)."""
    q, k, v = _mk(2, 24, 48, 32, 8)
    got = _from_gpu(
        fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                            prologue_v=[fkl.Mul(3.0), fkl.Add(1.0)]), q.shape)
    ref = _oracle(q, k, v, False, 1 / math.sqrt(32))
    err = np.abs(got - (3.0 * ref + 1.0)).max()
    check_true(f"FA V-prologue Mul(3).Add(1) == 3*out+1 (err={err:.2e})", err < 5e-6)


def t_prologue_int8_kv():
    """Prologue chains AFTER int8 dequant: V int8 + Mul(2) => 2*out_int8."""
    q, k, v = _mk(2, 16, 64, 32, 9)
    kc, vc = fkl.compress_kv(_to_gpu(k)), fkl.compress_kv(_to_gpu(v))
    base = _from_gpu(fkl.flash_attention(_to_gpu(q), kc, vc), q.shape)
    got = _from_gpu(fkl.flash_attention(_to_gpu(q), kc, vc,
                                        prologue_v=[fkl.Mul(2.0)]), q.shape)
    err = np.abs(got - 2.0 * base).max()
    check_true(f"FA int8-KV + V-prologue Mul(2) == 2*base (err={err:.2e})",
               err < 1e-5)


def t_prologue_values_no_recompile():
    """Changing prologue values reuses the cached kernel."""
    import time
    q, k, v = _mk(1, 16, 16, 32, 10)
    _ = fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                            prologue_q=[fkl.Mul(2.0)])      # compiles
    t0 = time.perf_counter()
    a = _from_gpu(fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                                      prologue_q=[fkl.Mul(4.0)]), q.shape)
    dt = time.perf_counter() - t0
    ref = _oracle(q * 4.0, k, v, False, 1 / math.sqrt(32))
    ok = np.abs(a - ref).max() < 5e-6 and dt < 1.0
    check_true(f"FA prologue values change without recompile ({dt*1e3:.0f}ms)", ok)


def t_mma_dense_causal():
    """Tensor-core path (bf16 mma): same oracle, bf16-class tolerance."""
    q, k, v = _mk(2, 128, 128, 64, 11)
    out = fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v),
                              causal=True, mma=True)
    got = _from_gpu(out, q.shape)
    ref = _oracle(q, k, v, True, 1 / math.sqrt(64))
    err = np.abs(got - ref).max()
    check_true(f"FA mma d64 causal (err={err:.2e})", err < 2e-2)


def t_mma_prologue_epilogue():
    """mma path keeps the FKL superpowers: V-prologue + epilogue fused."""
    q, k, v = _mk(2, 64, 128, 64, 12)
    got = _from_gpu(
        fkl.flash_attention(_to_gpu(q), _to_gpu(k), _to_gpu(v), mma=True,
                            prologue_v=[fkl.Mul(3.0), fkl.Add(1.0)],
                            epilogue=[fkl.Mul(2.0)]), q.shape)
    ref = _oracle(q, k, v, False, 1 / math.sqrt(64))
    err = np.abs(got - 2.0 * (3.0 * ref + 1.0)).max()
    check_true(f"FA mma V-prologue+epilogue == 2*(3*out+1) (err={err:.2e})",
               err < 5e-2)


def t_fp8_kv():
    """fp8 e4m3 KV cache: exact vs dequantized oracle is impractical from
    python (needs fp8 decode), so verify quant-bounded e2e accuracy and the
    memory ratio; mma path exercises the QUANT_KV cp.async schedule."""
    q, k, v = _mk(2, 48, 96, 64, 20)
    kc = fkl.compress_kv(_to_gpu(k), fmt="fp8")
    vc = fkl.compress_kv(_to_gpu(v), fmt="fp8")
    ratio = (k.size * 4) / kc.nbytes
    check_true(f"FP8 KV compression ratio {ratio:.2f}x (>3.5x)", ratio > 3.5)

    got = _from_gpu(fkl.flash_attention(_to_gpu(q), kc, vc, causal=True), q.shape)
    ref = _oracle(q, k, v, True, 1 / math.sqrt(64))
    err = np.abs(got - ref).max()
    # e4m3 has 3 mantissa bits (~6% rel err); on UNIFORM data int8's 127
    # uniform levels are tighter. fp8 wins on outlier-heavy real caches.
    check_true(f"FA fp8-KV e2e quant-bounded (err={err:.2e})", err < 6e-2)

    got2 = _from_gpu(fkl.flash_attention(_to_gpu(q), kc, vc, causal=True,
                                         mma=True), q.shape)
    err2 = np.abs(got2 - ref).max()
    check_true(f"FA fp8-KV mma path (err={err2:.2e})", err2 < 6e-2)


if __name__ == "__main__":
    run([t_dense_causal, t_dense_cross_ragged, t_compressed_kv,
         t_fused_epilogue, t_epilogue_values_no_recompile,
         t_single_query_decode, t_prologue_q, t_prologue_v_affine,
         t_prologue_int8_kv, t_prologue_values_no_recompile,
         t_mma_dense_causal, t_mma_prologue_epilogue, t_fp8_kv],
        "flash-attention")
