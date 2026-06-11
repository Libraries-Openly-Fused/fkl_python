"""Tests for fkl.CircularTensor — stateful temporal window (video).

CPU reference: a deque-like rolling list of preprocessed frames.
"""
import struct
from harness import dev_f32, dev_u8, unf32, unu8, check, check_true, run
import fkl


def _frame_f32(value, W, H):
    return dev_f32([float(value)] * (W * H), W, H)


def _frame_pattern(k, W, H):
    """Position+frame encoded pixels so wrong slots/rotation show up."""
    return dev_f32([float(k * 1000 + i) for i in range(W * H)], W, H)


def t_basic_rotation_newest_first():
    """Push 1..6 into a window of 4; newest_first => plane0 = latest."""
    W, H, B = 8, 4, 4
    ct = fkl.CircularTensor(W, H, batch=B, dtype="float32",
                            order="newest_first")
    pushed = []
    for k in range(1, 7):                       # 6 frames into window of 4
        ct.update(_frame_f32(k, W, H))
        pushed.append(float(k))
    snap = unf32(ct.snapshot().copy_to_host(), W * H * B)
    # newest first: planes = [6, 5, 4, 3]
    expect_per_plane = [6.0, 5.0, 4.0, 3.0]
    got_per_plane = [snap[p * W * H] for p in range(B)]
    check("CT newest_first plane order after 6 pushes",
          got_per_plane, expect_per_plane)
    # every plane must be constant (whole frame copied, not just px 0)
    ok = all(snap[p*W*H + i] == expect_per_plane[p]
             for p in range(B) for i in range(W * H))
    check_true("CT planes fully populated", ok)


def t_basic_rotation_oldest_first():
    W, H, B = 6, 4, 3
    ct = fkl.CircularTensor(W, H, batch=B, dtype="float32",
                            order="oldest_first")
    for k in range(1, 5):                       # 4 pushes into window of 3
        ct.update(_frame_f32(k, W, H))
    snap = unf32(ct.snapshot().copy_to_host(), W * H * B)
    got = [snap[p * W * H] for p in range(B)]
    check("CT oldest_first plane order", got, [2.0, 3.0, 4.0])


def t_preproc_chain_fused():
    """update() runs the preproc chain INSIDE the same kernel."""
    W, H, B = 8, 2, 3
    ct = fkl.CircularTensor(W, H, batch=B, dtype="float32")
    for k in (1, 2, 3):
        ct.update(_frame_f32(k, W, H), ops=[fkl.Mul(10.0), fkl.Add(0.5)])
    snap = unf32(ct.snapshot().copy_to_host(), W * H * B)
    got = [snap[p * W * H] for p in range(B)]
    check("CT fused preproc Mul(10)+Add(0.5)", got, [30.5, 20.5, 10.5])


def t_pixel_content_preserved():
    """Full-frame content (not just constants) survives rotation intact."""
    W, H, B = 6, 5, 3
    ct = fkl.CircularTensor(W, H, batch=B, dtype="float32")
    for k in range(1, 5):                       # frames 1..4, window keeps 4,3,2
        ct.update(_frame_pattern(k, W, H))
    snap = unf32(ct.snapshot().copy_to_host(), W * H * B)
    exp = []
    for k in (4, 3, 2):                         # newest first
        exp.extend(float(k * 1000 + i) for i in range(W * H))
    check("CT pixel-exact content across rotation", snap, exp)


def t_values_change_no_recompile():
    """Same chain shape with different VALUES reuses the compiled .so."""
    W, H, B = 8, 2, 2     # W must be >4: trailing dim 2..4 folds to channels
    ct = fkl.CircularTensor(W, H, batch=B, dtype="float32")
    ct.update(_frame_f32(1, W, H), ops=[fkl.Mul(2.0)])
    so_before = ct._chain_key
    ct.update(_frame_f32(2, W, H), ops=[fkl.Mul(5.0)])   # new value, same op
    check_true("CT same chain shape => no recompile",
               ct._chain_key == so_before and ct.frames_pushed == 2)
    snap = unf32(ct.snapshot().copy_to_host(), W * H * B)
    check("CT values still correct", [snap[0], snap[W * H]], [10.0, 2.0])


def t_uchar3_planar_dnn_window():
    """The video-DNN ingest: uchar3 frames -> normalize -> planar CHW
    window of B frames, ready for a temporal model."""
    W, H, B = 4, 2, 2
    n = W * H * 3
    ct = fkl.CircularTensor(W, H, batch=B, dtype="uint8", channels=3,
                            layout="planar", out_dtype="float32")
    srcs = []
    for k in (1, 2, 3):
        vals = [((k * 37 + i * 11) % 256) for i in range(n)]
        srcs.append(vals)
        ct.update(dev_u8(vals, W, H, ch=3),
                  ops=[fkl.Cast("float32"), fkl.Div(255.0)])
    snap = unf32(ct.snapshot().copy_to_host(), n * B)
    exp = []
    for vals in (srcs[2], srcs[1]):             # newest first
        for c in range(3):                       # planar CHW per frame
            for p in range(W * H):
                exp.append(vals[p * 3 + c] / 255.0)
    check("CT uchar3 -> planar float window (DNN ingest)", snap, exp,
          tol=1e-6)


def t_wrong_input_rejected():
    W, H = 6, 4
    ct = fkl.CircularTensor(W, H, batch=2, dtype="float32")
    try:
        ct.update(_frame_f32(1, 8, 8))
        check_true("CT rejects wrong frame size", False)
    except ValueError:
        check_true("CT rejects wrong frame size", True)
    try:
        ct2 = fkl.CircularTensor(W, H, batch=1, dtype="float32")
        check_true("CT rejects batch<2", False)
    except ValueError:
        check_true("CT rejects batch<2", True)


if __name__ == "__main__":
    run([t_basic_rotation_newest_first, t_basic_rotation_oldest_first,
         t_preproc_chain_fused, t_pixel_content_preserved,
         t_values_change_no_recompile, t_uchar3_planar_dnn_window,
         t_wrong_input_rejected], "circular-tensor")
