"""Torch interop for the high-level API (requires torch+cuda; skips cleanly
otherwise): fkl.Tensor over torch tensors, .torch() zero-copy views both
ways, torch-backed Image/ImageBatch pipelines, out= into torch tensors.

Run with a torch-enabled python:
    .venv-torch/bin/python tests/test_highlevel_torch.py
"""
import sys
import fkl

try:
    import torch
    assert torch.cuda.is_available()
except Exception as e:
    print(f"SKIP: torch+cuda not available ({e})")
    sys.exit(0)

from harness import check, check_true, run


def t_tensor_wraps_torch():
    x = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8)
    t = fkl.Tensor(x)
    check_true("Tensor(torch): shape/dtype/device/ptr",
               t.shape == (4, 8) and t.dtype.base == "float32"
               and t.device == 0 and t.ptr == x.data_ptr())
    check_true("Tensor(torch).torch() is the SAME tensor", t.torch() is x)
    v = t.reshape(8, 4).torch()
    check_true("reshaped .torch() is a zero-copy view",
               tuple(v.shape) == (8, 4) and v.data_ptr() == x.data_ptr())


def t_empty_torch_view():
    import struct
    t = fkl.Tensor.empty((4, 4), "float32")
    v = t.torch()                       # DLPack: ownership moves to torch
    v.fill_(3.5)
    check_true("Tensor.empty().torch(): cached zero-copy view",
               t.torch() is v and tuple(v.shape) == (4, 4) and v.is_cuda)
    check("Tensor.empty(): DeviceBuffer sees torch writes (zero-copy)",
          list(struct.unpack("16f", t.base.copy_to_host())), [3.5] * 16)


def t_torch_image_pipeline():
    H, W = 24, 32
    MEAN, STD = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
    x = (torch.arange(H * W * 3, device="cuda", dtype=torch.float32)
         .mul(7).remainder(256).to(torch.uint8).reshape(H, W, 3).contiguous())
    out = (fkl.pipe(fkl.Image(x)).normalize(MEAN, STD).to_planar("CHW")
           .run())
    v = out.torch()
    ref = ((x.float() - torch.tensor(MEAN, device="cuda"))
           / torch.tensor(STD, device="cuda")).permute(2, 0, 1).contiguous()
    check_true("torch pipeline: (C, H, W) float out",
               tuple(v.shape) == (3, H, W) and v.dtype == torch.float32)
    check_true("torch pipeline: values match torch reference",
               torch.allclose(v, ref, atol=1e-3))


def t_torch_batch_pipeline():
    N, H, W = 3, 10, 12
    x = (torch.arange(N * H * W * 3, device="cuda", dtype=torch.float32)
         .mul(13).remainder(256).to(torch.uint8)
         .reshape(N, H, W, 3).contiguous())
    b = fkl.ImageBatch(x)
    check_true("ImageBatch(torch NHWC): plane views",
               len(b) == N and b[0].ptr == x.data_ptr())
    out = fkl.pipe(b).resize((6, 5)).to_planar("CHW").run()
    v = out.torch()
    check_true("torch batch pipeline: (N, C, H, W) out",
               tuple(v.shape) == (N, 3, 5, 6))
    single = fkl.pipe(fkl.Image(x[0])).resize((6, 5)).to_planar("CHW").run()
    check_true("torch batch plane 0 == single-image pipeline",
               torch.allclose(v[0], single.torch(), atol=1e-4))


def t_torch_out_prealloc():
    x = torch.full((8, 8), 2.0, device="cuda")
    pre = torch.empty(8, 8, device="cuda")
    out = fkl.pipe(x).add(3.0).run(out=pre)
    check_true("pipe: out= torch tensor filled in place",
               torch.allclose(pre, torch.full_like(pre, 5.0))
               and out.ptr == pre.data_ptr())
    bad = 0
    try:    # too small: must be rejected BEFORE the launch
        fkl.pipe(x).add(3.0).run(out=torch.empty(2, 2, device="cuda"))
    except ValueError:
        bad += 1
    try:    # matching element count, wrong dtype
        fkl.pipe(x).add(3.0).run(
            out=torch.empty(8, 8, device="cuda", dtype=torch.int32))
    except ValueError:
        bad += 1
    check_true("pipe: out= torch tensor validated (size + dtype)", bad == 2)


def t_torch_half_rejected():
    for dt in (torch.float16, torch.bfloat16):
        try:
            fkl.Tensor(torch.zeros(4, 4, device="cuda", dtype=dt))
            ok = False
        except TypeError:
            ok = True
        check_true(f"Tensor(torch {dt}): clear TypeError (no FKL surface)",
                   ok)


def t_torch_out_allocation_follows_source():
    # batched torch tensor: the kernel args are fkl-internal plane views,
    # but the auto-allocated output must still be torch (torch in ->
    # torch out); same for a reshaped Tensor over a torch base.
    x = (torch.arange(2 * 4 * 6 * 3, device="cuda", dtype=torch.float32)
         .remainder(256).to(torch.uint8).reshape(2, 4, 6, 3).contiguous())
    out = fkl.pipe(fkl.ImageBatch(x)).cast("float32").to_planar("CHW").run()
    check_true("ImageBatch(torch tensor): auto-allocated out is torch",
               type(out.base).__module__.split(".")[0] == "torch"
               and tuple(out.shape) == (2, 3, 4, 6))
    y = torch.arange(16, device="cuda", dtype=torch.float32)
    r = fkl.pipe(fkl.Tensor(y).reshape(4, 4)).mul(2.0).run()
    check_true("reshaped torch source: auto-allocated out is torch",
               type(r.base).__module__.split(".")[0] == "torch"
               and torch.allclose(r.torch(), y.reshape(4, 4) * 2))


if __name__ == "__main__":
    run([t_tensor_wraps_torch, t_empty_torch_view, t_torch_image_pipeline,
         t_torch_batch_pipeline, t_torch_out_prealloc, t_torch_half_rejected,
         t_torch_out_allocation_follows_source], "highlevel-torch")
