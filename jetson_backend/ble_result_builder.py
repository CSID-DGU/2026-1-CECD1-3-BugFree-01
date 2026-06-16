from __future__ import annotations

from typing import Dict


def build_ble_result(
    *,
    timestamp: str,
    best_label: str,
    best_probability: float,
    scores: Dict[str, float],
    status_key: str,
    status_text: str,
    chunk_dbfs: float,
    enhanced_dbfs: float | None,
    quiet_gain: float | None,
    loud_gain: float | None,
    clipped: bool,
    raw_line: str,
) -> dict:
    return {
        "source": "live_inference_refactored_ble_independent",
        "time": timestamp,
        "label": best_label,
        "score": round(float(best_probability), 6),
        "status": status_key,
        "status_text": status_text,
        "level_dbfs": round(float(chunk_dbfs), 2),
        "enhanced_dbfs": None if enhanced_dbfs is None else round(float(enhanced_dbfs), 2),
        "quiet_gain": None if quiet_gain is None else round(float(quiet_gain), 4),
        "loud_gain": None if loud_gain is None else round(float(loud_gain), 4),
        "clipped": bool(clipped),
        "scores": {label: round(float(score), 6) for label, score in scores.items()},
        "raw": raw_line,
    }


def build_skip_result(*, timestamp: str, chunk_dbfs: float, threshold_dbfs: float, raw_line: str) -> dict:
    return {
        "source": "live_inference_refactored_ble_independent",
        "time": timestamp,
        "label": "low_signal",
        "score": 0.0,
        "status": "low_signal",
        "status_text": f"소리작음 {chunk_dbfs:+.1f}<{threshold_dbfs:+.1f}dBFS",
        "level_dbfs": round(float(chunk_dbfs), 2),
        "enhanced_dbfs": None,
        "quiet_gain": None,
        "loud_gain": None,
        "clipped": False,
        "scores": {},
        "raw": raw_line,
    }
