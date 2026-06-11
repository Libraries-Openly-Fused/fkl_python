"""PyTorch interop example (run on a machine with torch+cuda).
compose once, then call on torch cuda tensors with zero copy and zero stall.
"""
import torch
import fkl

# Compose a fused chain. Kernel is generated + compiled ONCE here.
pipe = fkl.compose(
    fkl.TensorRead(),
    fkl.Mul(2.0),    # all three fused into ONE kernel,
    fkl.Add(1.0),    # one DRAM read + one DRAM write,
    fkl.Mul(3.0),    # intermediates stay in registers (VF)
    fkl.TensorWrite(),
    dtype="float32",
)

x = torch.arange(16, device="cuda", dtype=torch.float32)

# Hot path: zero-copy (torch's __cuda_array_interface__), single C call,
# runs on torch's current stream (async, no implicit sync).
s = torch.cuda.current_stream()
out = pipe(x, stream=s)
torch.cuda.synchronize()

print("in :", x.tolist())
print("out:", out.tolist())
expected = (x * 2.0 + 1.0) * 3.0
assert torch.allclose(out, expected), (out, expected)
print("OK: torch zero-copy fused kernel matches reference")
