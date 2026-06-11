"""ThreadFusion (TF::ENABLED) tests: same results, vectorized accesses."""
import struct
from harness import dev_f32, dev_u8, unf32, check, check_true, run
import fkl


def t_tf_correctness_f32():
    W, H = 256, 32                       # wide → thread-fusable
    src = [float(i % 977) for i in range(W * H)]
    ops = (fkl.TensorRead(), fkl.Mul(2.0), fkl.Add(1.0), fkl.TensorWrite())
    base = fkl.compose(*ops)(dev_f32(src, W, H))
    tf = fkl.compose(*ops, thread_fusion=True)(dev_f32(src, W, H))
    check("TF f32 result == baseline", unf32(tf.copy_to_host(), W * H),
          unf32(base.copy_to_host(), W * H))


def t_tf_odd_width():
    """Non-divisible width exercises the THREAD_DIVISIBLE=false tail path."""
    W, H = 251, 8                        # prime width
    src = [float(i) for i in range(W * H)]
    ops = (fkl.TensorRead(), fkl.Mul(3.0), fkl.TensorWrite())
    tf = fkl.compose(*ops, thread_fusion=True)(dev_f32(src, W, H))
    check("TF odd width exact", unf32(tf.copy_to_host(), W * H),
          [v * 3.0 for v in src])


def t_tf_uchar3():
    W, H = 128, 16
    n = W * H * 3
    vals = [(i * 7) % 256 for i in range(n)]
    ops = (fkl.TensorRead(), fkl.Cast("float32"), fkl.Div((255.0,) * 3),
           fkl.TensorSplit())
    base = fkl.compose(*ops)(dev_u8(vals, W, H, ch=3))
    tf = fkl.compose(*ops, thread_fusion=True)(dev_u8(vals, W, H, ch=3))
    check("TF uchar3 CHW == baseline", unf32(tf.copy_to_host(), n),
          unf32(base.copy_to_host(), n))


def t_tf_distinct_cache_entries():
    ops = (fkl.TensorRead(), fkl.Mul(1.5), fkl.TensorWrite())
    a = fkl.compose(*ops)
    b = fkl.compose(*ops, thread_fusion=True)
    x = dev_f32([1.0] * 64 * 4, 64, 4)
    a(x); b(x)
    so_a = list(a._variants.values())[0][3]
    so_b = list(b._variants.values())[0][3]
    check_true("TF compiles a distinct .so", so_a != so_b)


def t_tf_cpu_rejected_gracefully():
    """thread_fusion on CPU target silently disables (no CUDA TF on CPU)."""
    p = fkl.compose(fkl.TensorRead(), fkl.Mul(2.0), fkl.TensorWrite(),
                    target="cpu", thread_fusion=True)
    check_true("TF ignored for CPU target", p.thread_fusion is False)


if __name__ == "__main__":
    run([t_tf_correctness_f32, t_tf_odd_width, t_tf_uchar3,
         t_tf_distinct_cache_entries, t_tf_cpu_rejected_gracefully],
        "thread-fusion")
