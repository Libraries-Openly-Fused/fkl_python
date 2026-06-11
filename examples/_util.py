"""Shared helpers for the fkl-python examples.

Dependency-free: builds GPU inputs with fkl.DeviceBuffer + struct, so the
examples run on any machine with CUDA — no numpy/torch needed. In your own
code you would normally pass cuda torch tensors or cupy arrays directly.
"""
import struct
import fkl


def gpu_image_f32(values, width, height):
    """1-channel float32 image from a flat list (row-major)."""
    buf = fkl.DeviceBuffer(width, height, "float32")
    buf.copy_from_host(struct.pack(f"{len(values)}f", *values))
    return buf


def gpu_image_u8(values, width, height, channels=1):
    """uint8 image (channels=3 -> uchar3 pixels) from a flat list."""
    buf = fkl.DeviceBuffer(width, height, "uint8", channels=channels)
    buf.copy_from_host(bytes(values))
    return buf


def to_floats(buf, count):
    return list(struct.unpack(f"{count}f", buf.copy_to_host()[:count * 4]))


def to_bytes(buf, count):
    return list(buf.copy_to_host()[:count])


def synthetic_rgb(width, height):
    """Deterministic RGB test pattern: value = (pixel*3 + channel*7) % 256."""
    return [((p * 3 + c * 7) % 256) for p in range(width * height)
            for c in range(3)]
