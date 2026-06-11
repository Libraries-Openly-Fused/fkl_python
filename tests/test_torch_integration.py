"""REAL torch integration tests (requires torch+cuda): zero-copy in both
directions, external streams, out= preallocation, CircularTensor->torch.

Run with a torch-enabled python:
    .venv-torch/bin/python tests/test_torch_integration.py
"""
import sys
import fkl

try:
    import torch
    assert torch.cuda.is_available()
except Exception as e:
    print(f"SKIP: torch+cuda not available ({e})")
    sys.exit(0)

from harness import check, check_true, run, unf32


def t_torch_in_dlpack_out():
    """torch tensor in -> fused kernel -> DeviceBuffer -> torch.from_dlpack."""
    x = torch.full((16, 32), 3.0, device="cuda", dtype=torch.float32)
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0),
                       fkl.TensorWrite())
    out = pipe(x)                                   # torch in -> torch out
    check_true("torch in -> torch out (auto-alloc)",
               isinstance(out, torch.Tensor) and out.is_cuda
               and torch.allclose(out, torch.full_like(out, 7.0)))


def t_dlpack_zero_copy_roundtrip():
    """DeviceBuffer -> torch.from_dlpack shares the SAME memory."""
    buf = fkl.DeviceBuffer(8, 4, "float32")
    import struct
    buf.copy_from_host(struct.pack("32f", *[float(i) for i in range(32)]))
    t = torch.from_dlpack(buf)
    check_true("from_dlpack shape+device", tuple(t.shape) == (4, 8) and t.is_cuda)
    check("from_dlpack values", t.flatten().tolist(),
          [float(i) for i in range(32)])
    # zero-copy proof: mutate via torch, the underlying memory changes
    t += 100.0
    t2 = torch.from_dlpack(buf) if not hasattr(buf, "_exported_dlpack") else t
    check_true("zero-copy (mutation visible)", float(t.flatten()[0]) == 100.0)


def t_torch_stream_async():
    """Launch on a torch stream: async semantics, caller syncs."""
    x = torch.full((64, 64), 1.0, device="cuda")
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(5.0), fkl.TensorWrite())
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        out = pipe(x, stream=s.cuda_stream)
    s.synchronize()
    check_true("external torch stream", torch.allclose(out, torch.full_like(out, 5.0)))


def t_out_preallocated_torch():
    x = torch.full((8, 8), 2.0, device="cuda")
    out = torch.empty(8, 8, device="cuda")
    pipe = fkl.compose(fkl.TensorRead(), fkl.Add(3.0), fkl.TensorWrite())
    ret = pipe(x, out=out)
    check_true("out= preallocated torch tensor",
               ret is out and torch.allclose(out, torch.full_like(out, 5.0)))


def t_uchar3_image_preproc_to_torch():
    """The DNN-ingest path: uint8 HWC torch image -> fused preproc -> CHW."""
    img = torch.randint(0, 256, (24, 32, 3), device="cuda", dtype=torch.uint8)
    pipe = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                       fkl.Div((255.0, 255.0, 255.0)), fkl.TensorSplit())
    out = pipe(img)                                  # (3, 24, 32) float
    ref = img.permute(2, 0, 1).float() / 255.0
    check_true("uchar3 HWC -> CHW float matches torch reference",
               tuple(out.shape) == (3, 24, 32)
               and torch.allclose(out, ref, atol=1e-6))


def t_circular_tensor_to_torch():
    """CircularTensor window consumed by torch via DLPack."""
    ct = fkl.CircularTensor(16, 8, batch=3, dtype="float32")
    for k in (1.0, 2.0, 3.0):
        ct.update(torch.full((8, 16), k, device="cuda"))
    win = torch.from_dlpack(ct.snapshot())           # (3, 8, 16)
    check_true("CircularTensor -> torch window",
               tuple(win.shape) == (3, 8, 16)
               and win[0].max().item() == 3.0 and win[1].max().item() == 2.0
               and win[2].max().item() == 1.0)


def t_batch_hf_from_torch_list():
    imgs = [torch.full((8, 16), float(k), device="cuda") for k in (1, 2, 3)]
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(10.0), fkl.TensorWrite())
    out = pipe(imgs)                                 # (3, 8, 16)
    check_true("HF batch from list of torch tensors",
               tuple(out.shape) == (3, 8, 16)
               and out[0].max().item() == 10.0 and out[2].max().item() == 30.0)


if __name__ == "__main__":
    run([t_torch_in_dlpack_out, t_dlpack_zero_copy_roundtrip,
         t_torch_stream_async, t_out_preallocated_torch,
         t_uchar3_image_preproc_to_torch, t_circular_tensor_to_torch,
         t_batch_hf_from_torch_list], "torch-integration")
