from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_math import enhance_threshold_amplitude, rms_dbfs
from constants import DB_EPSILON


@dataclass(frozen=True)
class PreprocessResult:
    processed_audio: np.ndarray
    enhanced_dbfs: float
    clipped: bool
    quiet_gain: float
    loud_gain: float


class AudioPreprocessor:
    def __init__(
        self,
        *,
        enhance_threshold_db: float,
        noise_reduction_db: float,
        main_gain_db: float,
        enhance_sharpness: float,
    ) -> None:
        self.enhance_threshold_db = float(enhance_threshold_db)
        self.noise_reduction_db = float(noise_reduction_db)
        self.main_gain_db = float(main_gain_db)
        self.enhance_sharpness = float(enhance_sharpness)

    def enhance(self, waveform: np.ndarray) -> PreprocessResult:
        waveform = np.asarray(waveform, dtype=np.float32)
        threshold = enhance_threshold_amplitude(self.enhance_threshold_db)
        quiet_gain = float(10.0 ** (-abs(self.noise_reduction_db) / 20.0))
        loud_gain = float(10.0 ** (self.main_gain_db / 20.0))
        sharpness = max(self.enhance_sharpness, 0.1)

        relative_level = np.abs(waveform) / max(threshold, DB_EPSILON)
        loud_weight = np.power(relative_level, sharpness)
        loud_weight = loud_weight / (1.0 + loud_weight)
        gain = quiet_gain + (loud_gain - quiet_gain) * loud_weight

        enhanced = waveform * gain.astype(np.float32, copy=False)
        clipped = bool(np.any(np.abs(enhanced) > 1.0))
        enhanced = np.clip(enhanced, -1.0, 1.0).astype(np.float32, copy=False)
        return PreprocessResult(
            processed_audio=enhanced,
            enhanced_dbfs=rms_dbfs(enhanced),
            clipped=clipped,
            quiet_gain=quiet_gain,
            loud_gain=loud_gain,
        )
