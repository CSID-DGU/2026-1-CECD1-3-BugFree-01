from __future__ import annotations

import numpy as np

from audio_math import rms_dbfs


class AudioLevelMeter:
    def calculate_dbfs(self, chunk: np.ndarray) -> float:
        return rms_dbfs(chunk)
