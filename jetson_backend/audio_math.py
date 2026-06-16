from __future__ import annotations

import sys
from typing import Dict

import numpy as np

from constants import ANSI_RESET, DB_EPSILON


def rms_dbfs(waveform: np.ndarray) -> float:
    waveform = np.asarray(waveform, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    return 20.0 * np.log10(max(rms, DB_EPSILON))


def db_gate_threshold(min_db: float) -> float:
    if min_db > 0:
        return -min_db
    return min_db


def enhance_threshold_amplitude(enhance_threshold_db: float) -> float:
    dbfs = db_gate_threshold(enhance_threshold_db)
    return float(10.0 ** (dbfs / 20.0))


def format_scores(scores: Dict[str, float]) -> str:
    return ", ".join(f"{label}={probability:.1%}" for label, probability in scores.items())


def colorize(text: str, color_code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color_code}{text}{ANSI_RESET}"
