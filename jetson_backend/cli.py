from __future__ import annotations

import argparse
from pathlib import Path

from constants import (
    DEFAULT_ENHANCE_SHARPNESS,
    DEFAULT_ENHANCE_THRESHOLD_DB,
    DEFAULT_MAIN_GAIN_DB,
    DEFAULT_MIN_DB,
    DEFAULT_MIN_SCORE,
    DEFAULT_NOISE_REDUCTION_DB,
    MIC_CHANNEL_INDEX,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime EfficientAT inference from ReSpeaker Array V3."
    )
    parser.add_argument(
        "--efficientat-dir",
        default=str(Path(__file__).resolve().parent / "EfficientAT"),
        help="Path to cloned fschmid56/EfficientAT repository.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional sounddevice input device index. Defaults to automatic ReSpeaker search.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=MIC_CHANNEL_INDEX,
        help=(
            "Input channel to use. ReSpeaker 4 Mic Array 6-channel firmware is usually "
            "ch0=processed audio, ch1-4=raw microphones, ch5=playback."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input devices and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print waveform, mel, logits, and top AudioSet sigmoid scores for each chunk.",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=DEFAULT_MIN_DB,
        help=(
            "Mark chunks quieter than this level as low signal. Positive values are treated as "
            "dB below full scale, so 30 means -30 dBFS. Use 0 or a negative value "
            "to pass an explicit dBFS threshold."
        ),
    )
    parser.add_argument(
        "--enhance-threshold-db",
        type=float,
        default=DEFAULT_ENHANCE_THRESHOLD_DB,
        help=(
            "Sample-level enhancement threshold. Positive values are treated as "
            "dB below full scale, so 35 means -35 dBFS."
        ),
    )
    parser.add_argument(
        "--noise-reduction-db",
        type=float,
        default=DEFAULT_NOISE_REDUCTION_DB,
        help="Reduce quieter waveform parts by this many dB before inference.",
    )
    parser.add_argument(
        "--main-gain-db",
        type=float,
        default=DEFAULT_MAIN_GAIN_DB,
        help="Boost louder waveform parts by this many dB before inference.",
    )
    parser.add_argument("--gain-db", type=float, dest="main_gain_db", help=argparse.SUPPRESS)
    parser.add_argument(
        "--enhance-sharpness",
        type=float,
        default=DEFAULT_ENHANCE_SHARPNESS,
        help="Higher values separate quiet noise and loud events more aggressively.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Mark predictions whose best custom sigmoid score is below this value as low confidence.",
    )
    parser.add_argument(
        "--skip-low-db",
        action="store_true",
        help="Skip model inference when chunk dBFS is below --min-db threshold.",
    )
    parser.add_argument(
        "--no-ble",
        action="store_true",
        help="Run console inference only without opening the BLE GATT server.",
    )
    parser.add_argument(
        "--ble-name",
        default="JHello",
        help="BLE advertising name. Keep JHello to match the Flutter scanner.",
    )
    parser.add_argument(
        "--ble-chunk-bytes",
        type=int,
        default=180,
        help="Maximum bytes per BLE notification frame. Flutter usually requests MTU 247; 180 is safe.",
    )
    return parser.parse_args()
