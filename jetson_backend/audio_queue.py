from __future__ import annotations

import queue

import numpy as np


class AudioQueue:
    def __init__(self) -> None:
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def push(self, block: np.ndarray) -> None:
        self._queue.put(block.copy())

    def pop(self) -> np.ndarray:
        return self._queue.get()
