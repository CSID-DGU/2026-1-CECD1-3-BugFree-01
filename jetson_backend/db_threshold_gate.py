from __future__ import annotations

from audio_math import db_gate_threshold


class DbThresholdGate:
    def __init__(self, min_db: float) -> None:
        self.min_db = float(min_db)
        self.min_dbfs = db_gate_threshold(self.min_db)

    def is_over_threshold(self, dbfs: float) -> bool:
        return float(dbfs) >= self.min_dbfs

    def low_signal_message(self, dbfs: float) -> str:
        return f"소리작음 {dbfs:+.1f}<{self.min_dbfs:+.1f}dBFS"
