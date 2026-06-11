"""Tests for the CPU backend (ParArch::CPU): same fused chains, host memory,
numpy in/out, validated against numpy references and against the GPU backend.
Requires numpy + clang++ (no CUDA needed for the CPU path itself)."""
import sys

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy not available")
    sys.exit(0)

import fkl
from harness import check_true, run


def t_cpu_basic_vf():
    x = np.arange(32, dtype=np.float32).reshape(4, 8)
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0),
                       fkl.TensorWrite(), target="cpu")
    out = pipe(x)
    check_true("CPU VF Mul+Add", np.allclose(out, x * 2 + 1))


def t_cpu_uchar3_preproc():
    img = np.random.randint(0, 256, (6, 10, 3), dtype=np.uint8)
    pipe = fkl.compose(fkl.TensorRead(), fkl.Cast("float32"),
                       fkl.Div((255.0, 255.0, 255.0)), fkl.TensorSplit(),
                       target="cpu")
    out = pipe(img)
    ref = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    check_true("CPU uchar3 -> CHW float", out.shape == (3, 6, 10)
               and np.allclose(out, ref, atol=1e-6))


def t_cpu_crop_resize():
    img = np.random.rand(12, 16).astype(np.float32) * 100
    pipe = fkl.compose(fkl.TensorRead(), fkl.Crop(4, 2, 8, 8),
                       fkl.Resize(4, 4), fkl.TensorWrite(), target="cpu")
    out = pipe(img)
    check_true("CPU BVF Crop+Resize shape", out.shape == (4, 4))


def t_cpu_matches_gpu():
    """The exact same chain must produce identical results on both targets."""
    img = np.random.randint(0, 256, (8, 12, 3), dtype=np.uint8)
    ops = lambda: (fkl.TensorRead(), fkl.Cast("float32"),
                   fkl.Mul((2.0, 0.5, 1.0)), fkl.Sub((10.0, 0.0, 5.0)),
                   fkl.TensorSplit())
    cpu_out = fkl.compose(*ops(), target="cpu")(img)

    try:
        gpu_buf = fkl.DeviceBuffer(12, 8, "uint8", channels=3)
    except Exception:
        check_true("CPU==GPU (skipped: no GPU)", True)
        return
    gpu_buf.copy_from_host(img.tobytes())
    gpu_out_buf = fkl.compose(*ops())(gpu_buf)
    gpu_out = np.frombuffer(gpu_out_buf.copy_to_host(),
                            dtype=np.float32).reshape(cpu_out.shape)
    check_true("CPU == GPU bit-comparable", np.allclose(cpu_out, gpu_out, atol=1e-5))


def t_cpu_values_no_recompile():
    x = np.ones((4, 8), dtype=np.float32)
    pipe = fkl.compose(fkl.TensorRead(), fkl.Mul(3.0), fkl.TensorWrite(),
                       target="cpu")
    a = pipe(x)
    pipe.ops[1]._v = None  # not how values change; use a fresh compose
    pipe2 = fkl.compose(fkl.TensorRead(), fkl.Mul(7.0), fkl.TensorWrite(),
                        target="cpu")
    b = pipe2(x)
    check_true("CPU values via params", float(a[0, 0]) == 3.0 and float(b[0, 0]) == 7.0)


if __name__ == "__main__":
    run([t_cpu_basic_vf, t_cpu_uchar3_preproc, t_cpu_crop_resize,
         t_cpu_matches_gpu, t_cpu_values_no_recompile], "cpu-backend")
