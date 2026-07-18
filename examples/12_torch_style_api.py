"""Example 12 — PyTorch-style API: Tensor / Image / ImageBatch + fkl.pipe().

The classic DNN-ingest chain, written the way a torch user expects:

    batch of HWC uint8 frames -> resize -> normalize -> NCHW float planes

...and it STILL lowers to ONE fused FKL kernel: the fluent methods only
append the same symbolic ops compose() takes; .run() JIT-compiles once per
signature (disk cache) and then costs a single launch. resize fuses into the
read (BVF), normalize runs in registers (VF), the batch is horizontal fusion
(HF), to_planar is the planar TensorSplit write.

Out of scope by design: autograd, broadcasting, >4-channel images,
multi-input graphs — this is a preprocessing API layer, not a framework.
"""
import fkl
from _util import gpu_image_u8, to_floats, synthetic_rgb

SRC_W, SRC_H = 320, 240          # camera frames
NET_W, NET_H = 64, 64            # model input
MEAN = (123.675, 116.28, 103.53)     # ImageNet mean (RGB)
STD = (58.395, 57.12, 57.375)

# three same-size frames (deterministic, shifted patterns)
frames = []
for k in range(3):
    pat = synthetic_rgb(SRC_W, SRC_H)
    frames.append(gpu_image_u8([(v + 40 * k) % 256 for v in pat],
                               SRC_W, SRC_H, channels=3))

batch = fkl.ImageBatch(frames)       # also takes ONE (N, H, W, C) tensor
print(f"OK  {batch!r}")

pre = (fkl.pipe(batch)
       .resize((NET_W, NET_H))       # bilinear, fused into the read (BVF)
       .normalize(MEAN, STD)         # (x - mean) / std, in registers (VF)
       .to_planar("CHW"))            # planar NCHW write (TensorSplit)

out = pre.run()                      # ONE kernel launch for all 3 images
print(f"OK  3x {SRC_W}x{SRC_H} HWC uint8 -> {out!r}")
assert out.shape == (3, 3, NET_H, NET_W)

plane = to_floats(out.base, 16)
print(f"    img0 R-plane[:4] = {[round(v, 3) for v in plane[:4]]}")

# Zero-copy hand-off to a DNN runtime: out.torch() gives a (3, 3, 64, 64)
# cuda tensor over the SAME memory when torch is installed (DLPack).

# Reusing the Pipeline object skips codegen entirely (the kernel handle is
# cached on it); a NEW pipeline with different VALUES (sizes, mean/std,
# rects keep the same types) still reuses the compiled .so from disk.
pre.run()
print("OK  second run: cached kernel, single ctypes call")

# Single images work the same, and one-op eager calls live in fkl.F:
roi = (fkl.pipe(fkl.Image(frames[0]))
       .crop(40, 30, 240, 180)
       .resize((NET_W, NET_H))
       .normalize(MEAN, STD)
       .to_planar("CHW")
       .run())
print(f"OK  single image crop->resize->normalize->CHW: {roi.shape}")

gray = fkl.F.cvt_color(fkl.Image(frames[0]), "RGB2GRAY")   # eager one-op
print(f"OK  fkl.F.cvt_color: {gray.shape} {gray.dtype.base}")
