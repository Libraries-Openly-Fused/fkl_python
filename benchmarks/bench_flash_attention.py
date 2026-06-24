"""Benchmark + accuracy: fkl.flash_attention vs PyTorch SDPA (FlashAttention-2
backend) and, if installed, the official flash-attn package.

Run with the torch venv:
    .venv-torch/bin/python benchmarks/bench_flash_attention.py

Measures (batch*heads=32, head_dim=64, fp32 for fkl / fp16 for references):
  - accuracy of every implementation vs an fp64 oracle
  - latency + effective TFLOPS
  - the FUSION advantage: attention + post-op as ONE fkl kernel vs
    SDPA + separate post-op kernel (what handmade FA forces you to do)
  - compressed KV cache: memory + decode-step latency
"""
import math
import time

import numpy as np
import torch

import fkl

DEV = "cuda"
BH, D = 32, 64


def oracle(q, k, v, causal, scale):
    q64 = q.double().cpu().numpy()
    k64 = k.double().cpu().numpy()
    v64 = v.double().cpu().numpy()
    out = np.zeros_like(q64)
    for b in range(q64.shape[0]):
        s = q64[b] @ k64[b].T * scale
        if causal:
            s[np.triu_indices_from(s, 1)] = -np.inf
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        out[b] = p @ v64[b]
    return out


def timeit(fn, iters=50):
    fn()  # warmup + compile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def tflops(ms, seq_q, seq_k, causal):
    flops = 4.0 * BH * seq_q * seq_k * D * (0.5 if causal else 1.0)
    return flops / (ms * 1e-3) / 1e12


def main():
    torch.manual_seed(7)
    print(f"config: batch*heads={BH}, head_dim={D}, RTX PRO 6000 (sm_120)\n")

    # ---------- accuracy ----------
    print("== accuracy vs fp64 oracle (seq 256, causal) ==")
    q = torch.empty(BH, 256, D, device=DEV).uniform_(-1, 1)
    k = torch.empty(BH, 256, D, device=DEV).uniform_(-1, 1)
    v = torch.empty(BH, 256, D, device=DEV).uniform_(-1, 1)
    scale = 1 / math.sqrt(D)
    ref = oracle(q, k, v, True, scale)

    out_fkl = torch.from_dlpack(fkl.flash_attention(q, k, v, causal=True))
    err_fkl = np.abs(out_fkl.cpu().numpy().reshape(ref.shape) - ref).max()
    print(f"fkl  fp32 dense : {err_fkl:.3e}")

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        out_sdpa16 = torch.nn.functional.scaled_dot_product_attention(
            q.half(), k.half(), v.half(), is_causal=True)
    err_sdpa16 = np.abs(out_sdpa16.float().cpu().numpy() - ref).max()
    print(f"sdpa fp16 flash : {err_sdpa16:.3e}   (PyTorch FA-2 kernel)")

    kc, vc = fkl.compress_kv(k), fkl.compress_kv(v)
    out_q = torch.from_dlpack(fkl.flash_attention(q, kc, vc, causal=True))
    err_q = np.abs(out_q.cpu().numpy().reshape(ref.shape) - ref).max()
    print(f"fkl  int8-KV    : {err_q:.3e}   (4x smaller KV cache)")

    try:
        from flash_attn import flash_attn_func
        # flash-attn wants (batch, seq, heads, dim) fp16
        q4 = q.half().view(4, 8, 256, D).permute(0, 2, 1, 3).contiguous()
        k4 = k.half().view(4, 8, 256, D).permute(0, 2, 1, 3).contiguous()
        v4 = v.half().view(4, 8, 256, D).permute(0, 2, 1, 3).contiguous()
        out_official = flash_attn_func(q4, k4, v4, causal=True)
        oo = out_official.permute(0, 2, 1, 3).reshape(BH, 256, D)
        err_off = np.abs(oo.float().cpu().numpy() - ref).max()
        print(f"flash-attn fp16 : {err_off:.3e}   (official Dao-AILab)")
        have_official = True
    except ImportError:
        print("flash-attn      : not installed (skipping)")
        have_official = False

    # ---------- performance ----------
    print("\n== latency / TFLOPS (seq 1024, causal) ==")
    q = torch.empty(BH, 1024, D, device=DEV).uniform_(-1, 1)
    k = torch.empty(BH, 1024, D, device=DEV).uniform_(-1, 1)
    v = torch.empty(BH, 1024, D, device=DEV).uniform_(-1, 1)
    qh, kh, vh = q.half(), k.half(), v.half()

    fa = fkl.FlashAttention(D)               # reuse compiled kernel
    out_buf = fkl.DeviceBuffer(D, 1024, "float32", planes=BH)
    ms_fkl = timeit(lambda: fa(q, k, v, out=out_buf, causal=True))
    print(f"fkl  fp32 SIMT  : {ms_fkl:7.3f} ms  {tflops(ms_fkl,1024,1024,True):6.2f} TFLOPS")

    def run_sdpa():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    ms_sdpa = timeit(run_sdpa)
    print(f"sdpa fp16 flash : {ms_sdpa:7.3f} ms  {tflops(ms_sdpa,1024,1024,True):6.2f} TFLOPS")

    if have_official:
        q4 = qh.view(4, 8, 1024, D).permute(0, 2, 1, 3).contiguous()
        k4 = kh.view(4, 8, 1024, D).permute(0, 2, 1, 3).contiguous()
        v4 = vh.view(4, 8, 1024, D).permute(0, 2, 1, 3).contiguous()
        ms_off = timeit(lambda: flash_attn_func(q4, k4, v4, causal=True))
        print(f"flash-attn fp16 : {ms_off:7.3f} ms  {tflops(ms_off,1024,1024,True):6.2f} TFLOPS")

    # ---------- the fusion advantage ----------
    print("\n== fusion: attention + Mul(2)+Add(0.5) post-op (seq 1024) ==")
    ms_fused = timeit(lambda: fa(q, k, v, out=out_buf, causal=True))
    fa_ep = fkl.FlashAttention(D, epilogue=[fkl.Mul(2.0), fkl.Add(0.5)])
    ms_fused_ep = timeit(lambda: fa_ep(q, k, v, out=out_buf, causal=True))
    print(f"fkl  FA alone          : {ms_fused:7.3f} ms")
    print(f"fkl  FA+epilogue FUSED : {ms_fused_ep:7.3f} ms  "
          f"(overhead {100*(ms_fused_ep-ms_fused)/ms_fused:+.1f}%)")

    def sdpa_then_postop():
        o = run_sdpa()
        return o * 2.0 + 0.5                      # extra kernel(s) + DRAM trip
    ms_sdpa_post = timeit(sdpa_then_postop)
    print(f"sdpa + separate post-op: {ms_sdpa_post:7.3f} ms  "
          f"(post-op cost {100*(ms_sdpa_post-ms_sdpa)/ms_sdpa:+.1f}% vs sdpa alone)")

    # ---------- compressed KV decode ----------
    print("\n== decode step (seq_q=1) vs 8192-token KV cache ==")
    kl = torch.empty(BH, 8192, D, device=DEV).uniform_(-1, 1)
    vl = torch.empty(BH, 8192, D, device=DEV).uniform_(-1, 1)
    q1 = torch.empty(BH, 1, D, device=DEV).uniform_(-1, 1)
    dense_mb = kl.numel() * 4 * 2 / 2**20
    kc, vc = fkl.compress_kv(kl), fkl.compress_kv(vl)
    comp_mb = (kc.nbytes + vc.nbytes) / 2**20
    print(f"KV cache: dense fp32 {dense_mb:.0f} MB -> int8 {comp_mb:.0f} MB "
          f"({dense_mb/comp_mb:.2f}x)")

    fa_dec = fkl.FlashAttention(D)
    out1 = fkl.DeviceBuffer(D, 1, "float32", planes=BH)
    ms_dense = timeit(lambda: fa_dec(q1, kl, vl, out=out1), iters=200)
    fa_decq = fkl.FlashAttention(D, kv_layout="int8")
    ms_int8 = timeit(lambda: fa_decq(q1, kc, vc, out=out1), iters=200)
    print(f"decode dense fp32 : {ms_dense*1e3:7.1f} us")
    print(f"decode int8   KV  : {ms_int8*1e3:7.1f} us  "
          f"({ms_dense/ms_int8:.2f}x faster — bandwidth-bound)")


if __name__ == "__main__":
    main()
