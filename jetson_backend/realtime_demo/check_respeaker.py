#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
DURATION_SECONDS = 2.0
REQUIRED_INPUT_CHANNELS = 6
MIC_NAME_KEYWORDS = ("respeaker", "re speaker", "seeed", "array v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ReSpeaker Array V3 audio input.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Use a specific sounddevice input device index.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION_SECONDS,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=SAMPLE_RATE,
        help="Input sample rate.",
    )
    return parser.parse_args()


def input_devices() -> List[Tuple[int, dict]]:
    return [
        (index, dict(device))
        for index, device in enumerate(sd.query_devices())
        if int(device.get("max_input_channels", 0)) > 0
    ]


def print_input_devices() -> None:
    devices = input_devices()
    if not devices:
        print("No input devices found.")
        return

    print("Input devices:")
    for index, device in devices:
        print(
            f"  [{index}] {device.get('name')} | "
            f"inputs={device.get('max_input_channels')} | "
            f"default_sr={device.get('default_samplerate')}"
        )


def find_respeaker_device(device_index: int | None) -> Tuple[int, dict, int]:
    devices = input_devices()

    if device_index is not None:
        for index, device in devices:
            if index == device_index:
                channels = min(REQUIRED_INPUT_CHANNELS, int(device["max_input_channels"]))
                return index, device, channels
        raise RuntimeError(f"Input device index not found: {device_index}")

    candidates = []
    for index, device in devices:
        name = str(device.get("name", "")).lower()
        max_channels = int(device.get("max_input_channels", 0))
        if max_channels <= 0:
            continue
        if any(keyword in name for keyword in MIC_NAME_KEYWORDS):
            candidates.append((index, device))

    if candidates:
        candidates.sort(
            key=lambda item: (
                int(item[1].get("default_samplerate", 0)) != SAMPLE_RATE,
                int(item[1].get("max_input_channels", 0)) < REQUIRED_INPUT_CHANNELS,
                item[0],
            )
        )
        index, device = candidates[0]
        channels = min(REQUIRED_INPUT_CHANNELS, int(device["max_input_channels"]))
        return index, device, channels

    raise RuntimeError("Could not auto-detect ReSpeaker Array V3 input device.")


def print_channel_stats(audio: np.ndarray, samplerate: int) -> None:
    print()
    print(f"Recorded shape: samples={audio.shape[0]}, channels={audio.shape[1]}")
    print(f"Sample rate: {samplerate} Hz")
    print()
    print("Channel stats:")

    for channel in range(audio.shape[1]):
        samples = audio[:, channel]
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        mean = float(np.mean(samples))
        clipped = int(np.count_nonzero(np.abs(samples) >= 0.999))
        silent = rms < 1e-4

        status = []
        if silent:
            status.append("very quiet")
        if clipped:
            status.append(f"clipped={clipped}")
        status_text = f" | {'; '.join(status)}" if status else ""

        print(
            f"  ch{channel}: rms={rms:.6f}, peak={peak:.6f}, "
            f"mean={mean:+.6f}{status_text}"
        )

    channel0 = audio[:, 0]
    print()
    print(
        "Channel 0 quick check: "
        f"rms={np.sqrt(np.mean(np.square(channel0))):.6f}, "
        f"peak={np.max(np.abs(channel0)):.6f}"
    )


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    try:
        device_index, device_info, channels = find_respeaker_device(args.device_index)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print_input_devices()
        return 1

    if channels < REQUIRED_INPUT_CHANNELS:
        print(
            f"WARNING: device reports only {channels} input channel(s), "
            f"expected {REQUIRED_INPUT_CHANNELS}.",
            file=sys.stderr,
        )

    print(
        f"Selected input device: [{device_index}] {device_info.get('name')} | "
        f"channels={channels}, samplerate={args.samplerate}"
    )

    try:
        sd.check_input_settings(
            device=device_index,
            channels=channels,
            samplerate=args.samplerate,
            dtype="float32",
        )
    except Exception as exc:
        print(f"ERROR: input setting check failed: {exc}", file=sys.stderr)
        print_input_devices()
        return 1

    frames = int(args.duration * args.samplerate)
    print(f"Recording {args.duration:.1f}s. Make some sound near the mic...")

    try:
        audio = sd.rec(
            frames,
            samplerate=args.samplerate,
            channels=channels,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"ERROR: recording failed: {exc}", file=sys.stderr)
        return 1

    print_channel_stats(np.asarray(audio, dtype=np.float32), args.samplerate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
