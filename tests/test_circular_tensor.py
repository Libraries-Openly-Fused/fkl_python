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


def t_race_audit():
    """Static audit of the divergent launch: within one update() kernel the
    temp slot WRITTEN by seq1 must never be among the slots READ by seq2,
    for both orders and every rotation index. This is what makes the
    single-kernel update race-free (no __threadfence/global sync needed)."""
    B = 4
    for order in ("newest_first", "oldest_first"):
        for idx in range(B):
            if order == "newest_first":
                # selector: z==0 -> seq1 ; CTRead Descendent: slot = idx - z
                seq1_z = {0}
                write_slot = (0 + idx) % B          # CTWrite Ascendent @ z=0
                read_slots = {(idx - z) % B for z in range(B) if z not in seq1_z}
            else:
                # selector: z==B-1 -> seq1 ; CTRead Ascendent: slot = idx + z
                seq1_z = {B - 1}
                write_slot = (B - 1 + idx) % B
                read_slots = {(idx + z) % B for z in range(B) if z not in seq1_z}
            check_true(f"race audit {order} idx={idx}",
                       write_slot not in read_slots,
                       f"w={write_slot} r={sorted(read_slots)}")


def t_temp_data_consistency():
    """After every update, temp must hold ALL frames of the window (in slot
    space) and data must be temp re-ordered by the rotation — i.e. the
    double-buffer never tears across many rotations (3 full wraps)."""
    import ctypes
    W, H, B = 8, 2, 3
    ct = fkl.CircularTensor(W, H, batch=B, dtype="float32")
    n = W * H
    for k in range(1, 3 * B + 1):                  # 9 pushes = 3 wraps
        ct.update(_frame_f32(k, W, H))
        data = unf32(ct.snapshot().copy_to_host(), n * B)
        tmp_buf = fkl.DeviceBuffer(W, H, "float32", planes=B)
        ct._lib.ct_snapshot_temp(ct._handle, ctypes.c_void_p(tmp_buf.ptr), None)
        temp = unf32(tmp_buf.copy_to_host(), n * B)

        idx = ct.frames_pushed % B                 # next write slot
        # newest_first: data plane z == temp slot (idx-1-z) mod B
        expect_data = []
        for z in range(B):
            slot = (idx - 1 - z) % B
            expect_data.extend(temp[slot * n:(slot + 1) * n])
        if data != expect_data:
            check_true(f"temp/data coherent after push {k}", False)
            return
    check_true("temp/data coherent across 3 wraps (9 pushes)", True)


def t_generate_source_no_compile():
    """generate_source() returns valid-looking CUDA without compiling."""
    ct = fkl.CircularTensor(16, 8, batch=2, dtype="float32")
    src = ct.generate_source([fkl.Mul(2.0)])
    needed = ("PyCT", "CircularTensorWrite", "CircularTensorRead",
              "SequenceSelectorType", "launchDivergentBatchTransformDPP_Kernel",
              "ct_update", "ct_snapshot", "ct_destroy")
    missing = [s for s in needed if s not in src]
    check_true("generate_source has all structural pieces",
               not missing and ct._lib is None, f"missing={missing}")


def t_size_not_in_signature_regression():
    """REGRESSION (real bug): two CTs with identical signature
    (dtype/B/order/layout/chain) but DIFFERENT sizes. The generated .so must
    serve both — sizes are runtime values held in PyCT, never baked
    literals. The bug: cache-hit on the first CT's .so with the first CT's
    W/H hardcoded -> wrong grid + wrong snapshot byte count."""
    a = fkl.CircularTensor(6, 5, batch=3, dtype="float32")
    for k in (1, 2, 3):
        a.update(_frame_f32(k, 6, 5))
    b = fkl.CircularTensor(8, 2, batch=3, dtype="float32")   # same sig, new size
    for k in (10, 20, 30):
        b.update(_frame_f32(k, 8, 2))

    va = unf32(a.snapshot().copy_to_host(), 6 * 5 * 3)
    vb = unf32(b.snapshot().copy_to_host(), 8 * 2 * 3)
    ok_a = [va[p * 30] for p in range(3)] == [3.0, 2.0, 1.0] and all(
        va[p * 30 + i] == va[p * 30] for p in range(3) for i in range(30))
    ok_b = [vb[p * 16] for p in range(3)] == [30.0, 20.0, 10.0] and all(
        vb[p * 16 + i] == vb[p * 16] for p in range(3) for i in range(16))
    check_true("same-signature CTs with different sizes coexist",
               ok_a and ok_b, f"a={[va[p*30] for p in range(3)]} b={[vb[p*16] for p in range(3)]}")


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
         t_race_audit, t_temp_data_consistency, t_generate_source_no_compile,
         t_size_not_in_signature_regression,
         t_wrong_input_rejected], "circular-tensor")
