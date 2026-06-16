from __future__ import annotations

from typing import List

import numpy as np

from constants import CHUNK_SAMPLES


class AudioBuffer:
    def __init__(self, chunk_samples: int = CHUNK_SAMPLES) -> None:
        self.chunk_samples = chunk_samples
        self.pending_blocks: List[np.ndarray] = []
        self.pending_samples = 0

    def append(self, mono_block: np.ndarray) -> None:
        block = mono_block.astype(np.float32, copy=True)
        self.pending_blocks.append(block)
        self.pending_samples += block.shape[0]

    def has_chunk(self) -> bool:
        return self.pending_samples >= self.chunk_samples

    def pop_ready_chunks(self) -> list[np.ndarray]:
        if not self.has_chunk():
            return []

        joined = np.concatenate(self.pending_blocks)
        chunks: list[np.ndarray] = []
        offset = 0
        while joined.shape[0] - offset >= self.chunk_samples:
            chunks.append(joined[offset : offset + self.chunk_samples])
            offset += self.chunk_samples

        remainder = joined[offset:]
        self.pending_blocks = [remainder] if remainder.size else []
        self.pending_samples = remainder.shape[0]
        return chunks
