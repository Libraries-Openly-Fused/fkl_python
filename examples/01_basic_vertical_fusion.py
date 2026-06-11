"""Example 01 — Vertical Fusion basics.

Chain arbitrary point operations between a read and a write: FKL fuses
them into ONE kernel at C++ compile time. Intermediates never touch DRAM,
they live in registers.

    result = max(min((x * 2 + 10) / 4, 50), 5)

Key ideas demonstrated:
  * compose() is LAZY: nothing compiles until the first call.
  * The first call compiles a .so for this chain signature (~seconds),
    cached forever in ~/.cache/fkl/.
  * Changing the VALUES (factors) later reuses the same kernel.
"""
import fkl
from _util import gpu_image_f32, to_floats

W, H = 64, 32
x = [float(i % 100) for i in range(W * H)]

pipe = fkl.compose(
    fkl.TensorRead(),
    fkl.Mul(2.0),
    fkl.Add(10.0),
    fkl.Div(4.0),
    fkl.Min(50.0),     # min(v, 50)
    fkl.Max(5.0),      # max(v, 5)
    fkl.TensorWrite(),
)

out = pipe(gpu_image_f32(x, W, H))           # first call: compiles + runs
got = to_floats(out, W * H)

ref = [max(min((v * 2 + 10) / 4, 50.0), 5.0) for v in x]
assert all(abs(a - b) < 1e-5 for a, b in zip(got, ref)), "mismatch!"
print(f"OK  {W}x{H} px through 5 fused ops in one kernel")
print(f"    in[:5]  = {x[:5]}")
print(f"    out[:5] = {got[:5]}")

# Second pipeline with DIFFERENT values -> same cached .so, no recompile
pipe2 = fkl.compose(fkl.TensorRead(), fkl.Mul(3.0), fkl.Add(1.0),
                    fkl.Div(2.0), fkl.Min(99.0), fkl.Max(0.0),
                    fkl.TensorWrite())
out2 = pipe2(gpu_image_f32(x, W, H))
print("OK  same chain shape with new values reused the cached kernel")
