"""FlexAttention-style score mods and block-sparse masks for fkl.

SCORE MODS (the PyTorch FlexAttention idea, fused at COMPILE TIME):
a score mod transforms each attention score after scaling and
causal/bounds masking, before the online softmax:

    s' = mod(s, q_idx, kv_idx)

Built-ins map 1:1 to the C++ functors in flash_attention_mma.h:

    fkl.flash_attention(q, k, v, mma=True, score_mod=fkl.ALiBi(slope=0.0625))
    fkl.flash_attention(q, k, v, mma=True, score_mod=fkl.SoftCap(20.0))
    fkl.flash_attention(q, k, v, mma=True, score_mod=fkl.SlidingWindow(256))

Mod VALUES (slope, cap, window) are runtime kernel arguments — changing
them never recompiles. Only the mod TYPE is part of the compile key.

BLOCK SPARSITY: BlockMask wraps a (bh, nQB, nKB) uint8 numpy array (or
GPU buffer); inactive tiles are skipped entirely (loads AND math):

    mask = fkl.BlockMask(dense_mask_uint8, block_q=128, block_kv=128)
    fkl.flash_attention(q, k, v, mma=True, block_mask=mask)
"""

from __future__ import annotations

import ctypes

# NOTE: numpy is imported lazily inside the methods that need it. The core
# package contract is "import fkl works with zero python deps" (see
# pyproject: dependencies = []); numpy is only required if you actually
# construct a BlockMask from a host array.
from .tensor import DeviceBuffer, as_device_view


# ---------------------------------------------------------------------------
# score mods
# ---------------------------------------------------------------------------
class ScoreMod:
    """Base. Subclasses define: name (C++ functor), values (runtime params)."""
    name = "NoScoreMod"
    cpp_ctor = "NoScoreMod{}"

    @property
    def values(self):
        return []

    def token(self) -> str:
        return self.name


class ALiBi(ScoreMod):
    """s - slope * (q_idx - kv_idx). One slope per kernel call; for
    per-head slopes launch per-head or use the C++ API directly."""
    name = "ALiBiScoreMod"

    def __init__(self, slope: float):
        self.slope = float(slope)

    @property
    def cpp_ctor(self):
        return "ALiBiScoreMod{ modParams[0] }"

    @property
    def values(self):
        return [self.slope]


class SoftCap(ScoreMod):
    """Gemma-2 style logit soft capping: cap * tanh(s / cap)."""
    name = "SoftCapScoreMod"

    def __init__(self, cap: float):
        self.cap = float(cap)

    @property
    def cpp_ctor(self):
        return "SoftCapScoreMod{ modParams[0] }"

    @property
    def values(self):
        return [self.cap]


class SlidingWindow(ScoreMod):
    """mask_mod: keep only the last `window` keys for each query."""
    name = "SlidingWindowMask"

    def __init__(self, window: int):
        self.window = int(window)

    @property
    def cpp_ctor(self):
        return "SlidingWindowMask{ (int)modParams[0] }"

    @property
    def values(self):
        return [float(self.window)]


# ---------------------------------------------------------------------------
# block sparsity
# ---------------------------------------------------------------------------
class BlockMask:
    """Block-sparse attention mask at (block_q x block_kv) granularity.

    mask: numpy uint8/bool array (bh, nQB, nKB) — 1 = attend, 0 = skip —
    or anything with __cuda_array_interface__ already on the GPU.
    Inactive tiles skip global loads AND tensor-core math; runtime scales
    with density. block_q/block_kv must be multiples of the kernel tiles
    (128 is always safe for the defaults)."""

    def __init__(self, mask, block_q: int = 128, block_kv: int = 128):
        import numpy as np
        if isinstance(mask, np.ndarray):
            if mask.ndim != 3:
                raise ValueError("mask must be (bh, nQBlocks, nKVBlocks)")
            m = np.ascontiguousarray(mask.astype(np.uint8))
            self.bh, self.n_q_blocks, self.n_kv_blocks = m.shape
            self._buf = DeviceBuffer(self.n_kv_blocks, self.n_q_blocks,
                                     "uint8", planes=self.bh)
            self._buf.copy_from_host(m.tobytes())
            self.ptr = as_device_view(self._buf).ptr
        else:
            v = as_device_view(mask)
            self._buf = mask  # keep alive
            self.bh, self.n_q_blocks, self.n_kv_blocks = v.planes, v.height, v.width
            self.ptr = v.ptr
        self.block_q = int(block_q)
        self.block_kv = int(block_kv)

    @staticmethod
    def causal(bh: int, seq: int, block: int = 128) -> "BlockMask":
        """Convenience: block-causal mask (tile fully above diagonal -> skip)."""
        import numpy as np
        n = (seq + block - 1) // block
        m = np.zeros((bh, n, n), dtype=np.uint8)
        for qb in range(n):
            m[:, qb, :qb + 1] = 1
        return BlockMask(m, block, block)

    def density(self) -> float:
        return -1.0  # unknown without download; informational only
