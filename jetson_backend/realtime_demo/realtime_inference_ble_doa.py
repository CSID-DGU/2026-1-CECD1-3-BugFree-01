#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio

import jetson_ef.realtime_demo.realtime_inference_ble as ble
from jetson_ef.realtime_demo.realtime_inference import (
    ANSI_GREEN,
    ANSI_RED,
    CHUNK_SAMPLES,
    CHUNK_SECONDS,
    DEFAULT_ENHANCE_SHARPNESS,
    DEFAULT_ENHANCE_THRESHOLD_DB,
    DEFAULT_MAIN_GAIN_DB,
    DEFAULT_MIN_DB,
    DEFAULT_MIN_SCORE,
    DEFAULT_NOISE_REDUCTION_DB,
    MIC_CHANNEL_INDEX,
    MODEL_INPUT_SECONDS,
    MODEL_SAMPLE_RATE,
    REQUIRED_INPUT_CHANNELS,
    SAMPLE_RATE,
    build_custom_label_indices,
    colorize,
    db_gate_threshold,
    enhance_chunk,
    find_respeaker_device,
    format_scores,
    load_efficientat,
    predict_chunk,
    print_input_devices,
    rms_dbfs,
)

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )

if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


CARDINAL_SUFFIX = {
    "북": "북쪽",
    "동": "동쪽",
    "남": "남쪽",
    "서": "서쪽",
}

DANGER_LABELS = {
    "gunshot",
    "alarm_siren",
    "horn",
    "glass_shatter",
}

CAUTION_LABELS = {
    "construction",
    "water",
    "knock",
    "appliances",
    "baby_cry",
    "animal_cry",
}

RESPEAKER_USB_VENDOR_ID = 0x2886
RESPEAKER_USB_PRODUCT_ID = 0x0018
SPEED_OF_SOUND_M_S = 343.0
RAW_DOA_CHANNELS = (1, 2, 3, 4)
RAW_DOA_MIC_POSITIONS_M = np.asarray(
    [
        (-0.032, 0.000),
        (0.000, -0.032),
        (0.032, 0.000),
        (0.000, 0.032),
    ],
    dtype=np.float32,
)


def parse_optional_db_gate(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"off", "none", "disable", "disabled", "all"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a number, or one of: off, none, disabled, all"
        ) from exc


def format_optional_db_gate(value: float | None) -> str:
    if value is None:
        return "off"
    return f"{db_gate_threshold(value):+.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime EfficientAT inference from ReSpeaker Array V3 over BLE, "
            "with ReSpeaker DOA fields for EdgeAudioRecognition."
        )
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
        "--ble-name",
        default="JHello",
        help="BLE advertising name. Keep JHello to match the Flutter scanner.",
    )
    parser.add_argument(
        "--ble-chunk-bytes",
        type=int,
        default=244,
        help=(
            "Expected maximum BLE notify bytes. This file sends one App-compatible JSON "
            "notification instead of the framed protocol used by realtime_inference_ble.py."
        ),
    )
    parser.add_argument(
        "--north-offset",
        type=float,
        default=0.0,
        help="Raw DOA angle that should be treated as North. Default: 0.",
    )
    parser.add_argument(
        "--disable-doa",
        action="store_true",
        help="Run BLE inference without trying to read ReSpeaker DOA.",
    )
    parser.add_argument(
        "--doa-source",
        choices=("auto", "audio", "usb"),
        default="auto",
        help=(
            "DOA source. auto uses raw mic audio when available and falls back to "
            "the ReSpeaker USB DSP angle. Default: auto."
        ),
    )
    parser.add_argument(
        "--doa-poll-interval",
        type=float,
        default=0.1,
        help="Seconds between background USB DSP DOA polls. Default: 0.1.",
    )
    parser.add_argument(
        "--audio-doa-min-db",
        type=parse_optional_db_gate,
        default=None,
        help=(
            "Minimum raw mic level for audio-based DOA. Positive values are treated "
            "as dB below full scale, so 45 means -45 dBFS. Use 'off' to calculate "
            "DOA even for quiet raw mic windows. Default: off."
        ),
    )
    parser.add_argument(
        "--audio-doa-window-ms",
        type=float,
        default=250.0,
        help="Loudest raw mic window length used for audio-based DOA. Default: 250.",
    )
    parser.add_argument(
        "--db-offset",
        type=float,
        default=80.0,
        help=(
            "Convert internal dBFS to the positive dB value expected by the app: "
            "app_db=max(0, dBFS + offset). Default: 80."
        ),
    )
    parser.add_argument(
        "--full-packet",
        action="store_true",
        help="Include raw/items fields. Requires a BLE MTU large enough for the bigger JSON.",
    )
    return parser.parse_args()


def angle_to_cardinal(angle: float) -> str:
    corrected = float(angle) % 360.0
    if corrected < 45.0 or corrected >= 315.0:
        return "북"
    if corrected < 135.0:
        return "동"
    if corrected < 225.0:
        return "남"
    return "서"


def corrected_angle(raw_angle: float, north_offset: float) -> int:
    return int(round((float(raw_angle) - float(north_offset)) % 360.0)) % 360


@dataclass(frozen=True)
class DOAReading:
    raw_angle: int | None
    source: str
    status: str


class DOAReader:
    def __init__(
        self,
        enabled: bool = True,
        poll_interval: float = 0.1,
        disabled_reason: str = "disabled",
    ):
        self.ok = False
        self.tuning = None
        self.status = "disabled"
        self.poll_interval = max(0.02, float(poll_interval))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_angle: int | None = None
        self._last_voice: bool | None = None
        self._last_read_at: float | None = None
        self._last_error: str | None = None

        if not enabled:
            print(f"[DOA] USB reader {disabled_reason}", file=sys.stderr, flush=True)
            return

        try:
            import usb.core  # type: ignore
            from tuning import Tuning  # type: ignore

            dev = usb.core.find(
                idVendor=RESPEAKER_USB_VENDOR_ID,
                idProduct=RESPEAKER_USB_PRODUCT_ID,
            )
            if dev is None:
                self.status = "device_not_found"
                print("[DOA] ReSpeaker USB control device not found.", file=sys.stderr, flush=True)
                return

            self.tuning = Tuning(dev)
            self.ok = True
            self.status = "enabled"
            self._poll_once()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            if self.status == "enabled":
                print("[DOA] ReSpeaker USB DOA reader enabled.", file=sys.stderr, flush=True)
            else:
                usb_status = self.status
                if self._last_error:
                    usb_status = f"{usb_status}:{self._last_error}"
                print(
                    f"[DOA] ReSpeaker USB control found, but reads failed: {usb_status}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            self.status = "unavailable"
            print(f"[DOA] disabled: {exc!r}", file=sys.stderr, flush=True)

    def _read_device_locked(self) -> Tuple[int | None, bool | None]:
        if self.tuning is None:
            return None, None

        voice = None
        try:
            voice_value = self.tuning.is_voice()
            if voice_value is not None:
                voice = bool(int(voice_value))
        except Exception:
            voice = None

        angle = self.tuning.direction
        if angle is None:
            return None, voice
        return int(float(angle)) % 360, voice

    def _poll_once(self) -> None:
        if not self.ok or self.tuning is None:
            return

        try:
            with self._lock:
                angle, voice = self._read_device_locked()
            now = time.monotonic()
            self._last_read_at = now
            self._last_voice = voice
            if angle is not None:
                self._last_angle = angle
            self.status = "enabled"
            self._last_error = None
        except Exception as exc:
            self.status = "read_error"
            self._last_error = type(exc).__name__

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            self._poll_once()

    def read_angle(self) -> int | None:
        return self.snapshot().raw_angle

    def snapshot(self) -> DOAReading:
        if not self.ok:
            return DOAReading(None, "usb", self.status)

        self._poll_once()
        angle = self._last_angle
        if angle is None:
            status = self.status
            if self._last_error:
                status = f"{status}:{self._last_error}"
            if status == "enabled":
                status = "usb_no_angle"
            return DOAReading(None, "usb", status)

        now = time.monotonic()
        age = None if self._last_read_at is None else now - self._last_read_at
        if age is not None and age > max(1.0, self.poll_interval * 5.0):
            return DOAReading(None, "usb", "usb_stale")

        if self._last_voice is False:
            status = "usb_no_voice"
        else:
            status = "usb_active"
        return DOAReading(angle, "usb", status)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.tuning is not None:
            try:
                self.tuning.close()
            except Exception:
                pass


class AudioDOAEstimator:
    def __init__(
        self,
        *,
        enabled: bool,
        stream_channels: int,
        sample_rate: int,
        min_db: float | None,
        window_ms: float,
    ):
        self.enabled = enabled
        self.stream_channels = stream_channels
        self.sample_rate = sample_rate
        self.min_dbfs = None if min_db is None else db_gate_threshold(min_db)
        self.window_samples = max(256, int(round(sample_rate * max(20.0, window_ms) / 1000.0)))
        self.status = "disabled"
        self.channel_indices = tuple(RAW_DOA_CHANNELS)
        self.mic_positions = RAW_DOA_MIC_POSITIONS_M
        self.pairs = [
            (left, right)
            for left in range(len(self.channel_indices))
            for right in range(left + 1, len(self.channel_indices))
        ]
        self.expected_taus = self._build_expected_taus()

        if not enabled:
            return
        if stream_channels <= max(self.channel_indices):
            self.status = "audio_channels_unavailable"
            return
        self.status = "audio_enabled"

    def _build_expected_taus(self) -> np.ndarray:
        compass_degrees = np.arange(360, dtype=np.float32)
        radians = np.deg2rad(compass_degrees)
        directions = np.stack((np.sin(radians), np.cos(radians)), axis=1)
        expected = []
        for left, right in self.pairs:
            delta = self.mic_positions[left] - self.mic_positions[right]
            expected.append(-(directions @ delta) / SPEED_OF_SOUND_M_S)
        return np.stack(expected, axis=1).astype(np.float32)

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        result = 1
        while result < value:
            result <<= 1
        return result

    def _gcc_phat(
        self,
        sig: np.ndarray,
        refsig: np.ndarray,
        *,
        max_tau: float,
        interp: int = 16,
    ) -> float | None:
        n = self._next_power_of_two(sig.size + refsig.size)
        sig_fft = np.fft.rfft(sig, n=n)
        ref_fft = np.fft.rfft(refsig, n=n)
        cross_power = sig_fft * np.conj(ref_fft)
        cross_power /= np.abs(cross_power) + 1e-12
        cc = np.fft.irfft(cross_power, n=interp * n)

        max_shift = min(int(round(interp * self.sample_rate * max_tau)), (interp * n) // 2)
        if max_shift < 1:
            return None

        cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
        shift = int(np.argmax(cc)) - max_shift
        return float(shift) / float(interp * self.sample_rate)

    def _select_loudest_window(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] <= self.window_samples:
            return audio

        step = max(128, self.window_samples // 4)
        best_start = 0
        best_power = -1.0
        last_start = audio.shape[0] - self.window_samples
        for start in range(0, last_start + 1, step):
            window = audio[start : start + self.window_samples]
            power = float(np.mean(np.square(window)))
            if power > best_power:
                best_power = power
                best_start = start
        return audio[best_start : best_start + self.window_samples]

    def estimate(self, chunk: np.ndarray) -> DOAReading:
        if not self.enabled:
            return DOAReading(None, "audio", "audio_disabled")
        if self.status == "audio_channels_unavailable":
            return DOAReading(None, "audio", self.status)
        if chunk.ndim != 2 or chunk.shape[1] <= max(self.channel_indices):
            self.status = "audio_channels_unavailable"
            return DOAReading(None, "audio", self.status)

        raw = chunk[:, self.channel_indices].astype(np.float32, copy=True)
        raw = self._select_loudest_window(raw)
        raw_dbfs = rms_dbfs(raw.reshape(-1))
        if self.min_dbfs is not None and raw_dbfs < self.min_dbfs:
            self.status = f"audio_low_signal {raw_dbfs:+.1f}dBFS"
            return DOAReading(None, "audio", self.status)

        raw -= np.mean(raw, axis=0, keepdims=True)
        channel_rms = np.sqrt(np.mean(np.square(raw), axis=0) + 1e-12)
        usable_channels = channel_rms > 1e-5
        if int(np.count_nonzero(usable_channels)) < 2:
            self.status = "audio_not_enough_active_channels"
            return DOAReading(None, "audio", self.status)

        raw /= channel_rms.reshape(1, -1)
        raw *= np.hanning(raw.shape[0]).astype(np.float32).reshape(-1, 1)

        measured_taus = []
        pair_indices = []
        for pair_index, (left, right) in enumerate(self.pairs):
            if not (usable_channels[left] and usable_channels[right]):
                continue
            max_tau = (
                float(np.linalg.norm(self.mic_positions[left] - self.mic_positions[right]))
                / SPEED_OF_SOUND_M_S
            )
            tau = self._gcc_phat(raw[:, left], raw[:, right], max_tau=max_tau)
            if tau is None:
                continue
            measured_taus.append(tau)
            pair_indices.append(pair_index)

        if len(measured_taus) < 2:
            self.status = "audio_not_enough_tdoa_pairs"
            return DOAReading(None, "audio", self.status)

        measured = np.asarray(measured_taus, dtype=np.float32)
        expected = self.expected_taus[:, pair_indices]
        errors = np.mean(np.square(expected - measured.reshape(1, -1)), axis=1)
        angle = int(np.argmin(errors)) % 360
        error_us = float(np.sqrt(np.min(errors)) * 1_000_000.0)
        self.status = f"audio_active {raw_dbfs:+.1f}dBFS err={error_us:.1f}us"
        return DOAReading(angle, "audio", self.status)


def choose_doa_reading(
    *,
    doa_source: str,
    usb_reader: DOAReader,
    audio_estimator: AudioDOAEstimator,
    chunk: np.ndarray,
) -> DOAReading:
    if doa_source == "audio":
        return audio_estimator.estimate(chunk)
    if doa_source == "usb":
        return usb_reader.snapshot()

    audio_reading = audio_estimator.estimate(chunk)
    if audio_reading.raw_angle is not None:
        return audio_reading

    usb_reading = usb_reader.snapshot()
    if usb_reading.raw_angle is not None:
        return usb_reading

    return DOAReading(
        None,
        "none",
        f"{audio_reading.status};{usb_reading.status}",
    )


class AppInferenceCharacteristic(ble.InferenceCharacteristic):
    def _notify_latest(self) -> bool:
        if not self.notifying:
            return False

        self.sequence += 1
        payload_bytes = self.latest_payload.encode("utf-8")
        if len(payload_bytes) > self.chunk_bytes:
            print(
                f"warning: BLE JSON is {len(payload_bytes)} bytes; "
                f"larger than --ble-chunk-bytes={self.chunk_bytes}. "
                "If the app does not receive packets, increase Android MTU or omit --full-packet.",
                file=sys.stderr,
                flush=True,
            )

        self.PropertiesChanged(
            ble.GATT_CHRC_IFACE,
            ble.dbus.Dictionary({"Value": ble.byte_array(payload_bytes)}, signature="sv"),
            ble.dbus.Array([], signature="s"),
        )
        print(
            f"sent EdgeAudioRecognition notification seq={self.sequence} "
            f"bytes={len(payload_bytes)}",
            flush=True,
        )
        return False


class AppBleInferenceServer(ble.BleInferenceServer):
    def publish(self, data: dict) -> None:
        if self.characteristic is None:
            return
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        self.characteristic.notify_text(payload)


def install_app_compatible_ble_characteristic() -> None:
    ble.InferenceCharacteristic = AppInferenceCharacteristic


def risk_level(label: str) -> str:
    key = str(label).strip().lower()
    if key in DANGER_LABELS:
        return "danger"
    if key in CAUTION_LABELS:
        return "caution"
    return "info"


def app_db_from_dbfs(dbfs: float, offset: float) -> float:
    return round(max(0.0, float(dbfs) + float(offset)), 1)


def build_app_sound_packet(
    *,
    timestamp: str,
    label: str,
    score: float,
    scores: Dict[str, float],
    infer_sec: float,
    total_sec: float,
    chunk_dbfs: float,
    app_db: float,
    status_text: str,
    raw_line: str,
    raw_angle: int | None,
    north_offset: float,
    doa_status: str,
    doa_source: str,
    full_packet: bool,
) -> dict:
    if raw_angle is None:
        angle = None
        direction = ""
        direction_text = ""
    else:
        angle = corrected_angle(raw_angle, north_offset)
        direction = angle_to_cardinal(angle)
        direction_text = f"{CARDINAL_SUFFIX[direction]} {angle}도"

    packet = {
        "status": "ok",
        "time": timestamp,
        "label": label,
        "score": round(float(score), 6),
        "infer_sec": round(float(infer_sec), 3),
        "total_sec": round(float(total_sec), 3),
        "db": app_db,
        "level": risk_level(label),
        "direction": direction,
        "angle": float(angle) if angle is not None else None,
        "angle_raw": float(raw_angle) if raw_angle is not None else None,
        "direction_text": direction_text,
        "doa_status": doa_status,
        "doa_source": doa_source,
        "has_doa": raw_angle is not None,
    }

    if full_packet:
        packet["display_label"] = label
        packet["dbfs"] = round(float(chunk_dbfs), 2)
        packet["status_text"] = status_text
        packet["raw"] = raw_line
        packet["items"] = [
            {
                "label": item_label,
                "display_label": item_label,
                "score": round(float(item_score), 6),
                "direction": direction,
            }
            for item_label, item_score in scores.items()
        ]

    return packet


def run_stream_ble_doa(
    *,
    device_index: int,
    device_info: dict,
    stream_channels: int,
    channel_index: int,
    model: torch.nn.Module,
    mel: torch.nn.Module,
    resampler: torch.nn.Module | None,
    custom_indices: Dict[str, List[int]],
    audioset_labels: Sequence[str],
    device: torch.device,
    debug: bool,
    min_db: float,
    enhance_threshold_db: float,
    noise_reduction_db: float,
    main_gain_db: float,
    enhance_sharpness: float,
    min_score: float,
    ble_server: AppBleInferenceServer,
    usb_doa_reader: DOAReader,
    audio_doa_estimator: AudioDOAEstimator,
    doa_source: str,
    north_offset: float,
    db_offset: float,
    full_packet: bool,
) -> None:
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    if channel_index < 0 or channel_index >= stream_channels:
        raise RuntimeError(
            f"Selected channel index is out of range: "
            f"channel={channel_index}, available=0..{stream_channels - 1}"
        )

    if stream_channels < REQUIRED_INPUT_CHANNELS:
        print(
            f"Warning: selected input device reports only {stream_channels} input channels. "
            "Continuing with the available channel.",
            file=sys.stderr,
            flush=True,
        )

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"Audio input status: {status}", file=sys.stderr, flush=True)
        audio_queue.put(indata.copy())

    print(
        f"Input device: [{device_index}] {device_info.get('name')} | "
        f"channels={stream_channels}, mic_sr={SAMPLE_RATE}, "
        f"model_sr={MODEL_SAMPLE_RATE}, chunk={CHUNK_SECONDS}s, "
        f"model_input={MODEL_INPUT_SECONDS}s, channel={channel_index}, "
        f"min_dbfs={db_gate_threshold(min_db):+.1f}, "
        f"enhance_threshold_dbfs={db_gate_threshold(enhance_threshold_db):+.1f}, "
        f"noise_reduction_db={noise_reduction_db:.1f}, main_gain_db={main_gain_db:+.1f}, "
        f"min_score={min_score:.1%}, doa_source={doa_source}, "
        f"audio_doa_min_dbfs={format_optional_db_gate(audio_doa_estimator.min_dbfs)}, "
        f"usb_doa={usb_doa_reader.status}, audio_doa={audio_doa_estimator.status}"
    )
    print("Ctrl+C to stop.")

    pending_blocks: List[np.ndarray] = []
    pending_samples = 0

    with ble.sd.InputStream(
        device=device_index,
        samplerate=SAMPLE_RATE,
        channels=stream_channels,
        dtype="float32",
        callback=callback,
    ):
        while True:
            block = audio_queue.get()
            pending_blocks.append(block.astype(np.float32, copy=True))
            pending_samples += block.shape[0]

            if pending_samples < CHUNK_SAMPLES:
                continue

            joined = np.concatenate(pending_blocks, axis=0)
            offset = 0
            while joined.shape[0] - offset >= CHUNK_SAMPLES:
                chunk_started = time.perf_counter()
                chunk_multi = joined[offset: offset + CHUNK_SAMPLES]
                chunk = chunk_multi[:, channel_index].astype(np.float32, copy=True)
                offset += CHUNK_SAMPLES

                timestamp = datetime.now().strftime("%H:%M:%S")
                chunk_dbfs = rms_dbfs(chunk)
                min_dbfs = db_gate_threshold(min_db)

                inference_chunk, clipped, quiet_gain, loud_gain = enhance_chunk(
                    chunk,
                    enhance_threshold_db,
                    noise_reduction_db,
                    main_gain_db,
                    enhance_sharpness,
                )
                enhanced_dbfs = rms_dbfs(inference_chunk)

                try:
                    infer_started = time.perf_counter()
                    best_label, best_probability, scores = predict_chunk(
                        inference_chunk,
                        model,
                        mel,
                        resampler,
                        custom_indices,
                        audioset_labels,
                        device,
                        debug=debug,
                    )
                    infer_sec = time.perf_counter() - infer_started
                except Exception as exc:
                    print(
                        f"[{timestamp}] inference error: {exc} | skipping chunk",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                status_reasons = []
                if chunk_dbfs < min_dbfs:
                    status_reasons.append(f"low_signal {chunk_dbfs:+.1f}<{min_dbfs:+.1f}dBFS")
                if best_probability < min_score:
                    status_reasons.append(f"low_score {best_probability:.1%}<{min_score:.1%}")

                if status_reasons:
                    status_text = "low(" + ", ".join(status_reasons) + ")"
                    line_color = ANSI_RED
                else:
                    status_text = "detected"
                    line_color = ANSI_GREEN

                doa_reading = choose_doa_reading(
                    doa_source=doa_source,
                    usb_reader=usb_doa_reader,
                    audio_estimator=audio_doa_estimator,
                    chunk=chunk_multi,
                )
                raw_angle = doa_reading.raw_angle
                angle = corrected_angle(raw_angle, north_offset) if raw_angle is not None else None
                direction = angle_to_cardinal(angle) if angle is not None else ""
                if angle is None:
                    doa_text = f" | DOA=unavailable source={doa_reading.source} status={doa_reading.status}"
                else:
                    doa_text = (
                        f" | DOA={direction} {angle}deg raw={raw_angle} "
                        f"source={doa_reading.source} status={doa_reading.status}"
                    )

                line = (
                    f"[{timestamp}] predict: {best_label} ({best_probability:.1%}) | "
                    f"status={status_text} | "
                    f"level={chunk_dbfs:+.1f} dBFS | "
                    f"app_db={app_db_from_dbfs(chunk_dbfs, db_offset):.1f} dB | "
                    f"enhanced={enhanced_dbfs:+.1f} dBFS | "
                    f"infer={infer_sec:.3f}s | "
                    f"quiet_gain={quiet_gain:.2f}x loud_gain={loud_gain:.2f}x"
                    f"{' clipped' if clipped else ''}{doa_text} | all: {format_scores(scores)}"
                )
                print(colorize(line, line_color), flush=True)

                total_sec = CHUNK_SECONDS + (time.perf_counter() - chunk_started)
                ble_server.publish(
                    build_app_sound_packet(
                        timestamp=timestamp,
                        label=best_label,
                        score=best_probability,
                        scores=scores,
                        infer_sec=infer_sec,
                        total_sec=total_sec,
                        chunk_dbfs=chunk_dbfs,
                        app_db=app_db_from_dbfs(chunk_dbfs, db_offset),
                        status_text=status_text,
                        raw_line=line,
                        raw_angle=raw_angle,
                        north_offset=north_offset,
                        doa_status=doa_reading.status,
                        doa_source=doa_reading.source,
                        full_packet=full_packet,
                    )
                )

            remainder = joined[offset:]
            pending_blocks = [remainder] if remainder.size else []
            pending_samples = remainder.shape[0]


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    try:
        device_index, device_info, stream_channels = find_respeaker_device(args.device_index)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        print_input_devices()
        return 1

    install_app_compatible_ble_characteristic()
    ble_server = AppBleInferenceServer(args.ble_name, args.ble_chunk_bytes)
    try:
        ble_server.start()
    except Exception as exc:
        print(f"BLE startup error: {exc}", file=sys.stderr, flush=True)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    resampler = None
    if SAMPLE_RATE != MODEL_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(device).eval()

    try:
        model, mel, audioset_labels = load_efficientat(Path(args.efficientat_dir), device)
        custom_indices = build_custom_label_indices(audioset_labels)
    except Exception as exc:
        print(f"Model initialization error: {exc}", file=sys.stderr, flush=True)
        ble_server.stop()
        return 1

    usb_doa_reader = DOAReader(
        enabled=not args.disable_doa and args.doa_source in ("auto", "usb"),
        poll_interval=args.doa_poll_interval,
        disabled_reason=(
            "disabled by --disable-doa"
            if args.disable_doa
            else "disabled because --doa-source=audio"
        ),
    )
    audio_doa_estimator = AudioDOAEstimator(
        enabled=not args.disable_doa and args.doa_source in ("auto", "audio"),
        stream_channels=stream_channels,
        sample_rate=SAMPLE_RATE,
        min_db=args.audio_doa_min_db,
        window_ms=args.audio_doa_window_ms,
    )
    stop_requested = False

    def stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        run_stream_ble_doa(
            device_index=device_index,
            device_info=device_info,
            stream_channels=stream_channels,
            channel_index=args.channel_index,
            model=model,
            mel=mel,
            resampler=resampler,
            custom_indices=custom_indices,
            audioset_labels=audioset_labels,
            device=device,
            debug=args.debug,
            min_db=args.min_db,
            enhance_threshold_db=args.enhance_threshold_db,
            noise_reduction_db=args.noise_reduction_db,
            main_gain_db=args.main_gain_db,
            enhance_sharpness=args.enhance_sharpness,
            min_score=args.min_score,
            ble_server=ble_server,
            usb_doa_reader=usb_doa_reader,
            audio_doa_estimator=audio_doa_estimator,
            doa_source=args.doa_source,
            north_offset=args.north_offset,
            db_offset=args.db_offset,
            full_packet=args.full_packet,
        )
    except KeyboardInterrupt:
        print("\nStopping.")
        return 0
    except Exception as exc:
        print(f"Audio stream error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if stop_requested:
            print("Stopping BLE server...", flush=True)
        usb_doa_reader.stop()
        ble_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
