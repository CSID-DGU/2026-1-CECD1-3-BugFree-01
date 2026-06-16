#!/usr/bin/env python3
from __future__ import annotations

import signal
import sys
from pathlib import Path

import torch
import torchaudio

from audio_buffer import AudioBuffer
from audio_level_meter import AudioLevelMeter
from audio_preprocessor import AudioPreprocessor
from audio_queue import AudioQueue
from audio_stream_controller import AudioStreamController
from cli import parse_args
from constants import MODEL_SAMPLE_RATE, SAMPLE_RATE
from db_threshold_gate import DbThresholdGate
from device_finder import InputDeviceFinder
from efficientat_loader import EfficientATLoader
from io_setup import configure_utf8_stdio
from label_mapper import CustomLabelMapper
from microphone_module import MicrophoneModule
from model_inference import ModelInferenceEngine


def main() -> int:
    configure_utf8_stdio()
    args = parse_args()

    device_finder = InputDeviceFinder()
    if args.list_devices:
        device_finder.print_input_devices()
        return 0

    try:
        device_index, device_info, stream_channels = device_finder.find_respeaker_device(
            args.device_index
        )
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        device_finder.print_input_devices()
        return 1

    ble_server = None
    if not args.no_ble:
        try:
            # Import here so console-only mode can run on machines without python-dbus/BlueZ.
            from ble_inference_server import BleInferenceServer

            ble_server = BleInferenceServer(args.ble_name, args.ble_chunk_bytes)
            ble_server.start()
        except Exception as exc:
            print(f"BLE 초기화 오류: {exc}", file=sys.stderr)
            return 1

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"추론 디바이스: {torch_device}")

    resampler = None
    if SAMPLE_RATE != MODEL_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(torch_device).eval()

    try:
        model, mel, audioset_labels = EfficientATLoader(
            Path(args.efficientat_dir),
            torch_device,
        ).load()
        custom_indices = CustomLabelMapper().build_indices(audioset_labels)
    except Exception as exc:
        print(f"모델 초기화 오류: {exc}", file=sys.stderr)
        if ble_server is not None:
            ble_server.stop()
        return 1

    audio_queue = AudioQueue()
    microphone = MicrophoneModule(
        device_index=device_index,
        device_info=device_info,
        stream_channels=stream_channels,
        channel_index=args.channel_index,
        audio_queue=audio_queue,
    )

    inference_engine = ModelInferenceEngine(
        model=model,
        mel=mel,
        resampler=resampler,
        custom_indices=custom_indices,
        audioset_labels=audioset_labels,
        device=torch_device,
        debug=args.debug,
    )

    controller = AudioStreamController(
        microphone=microphone,
        audio_queue=audio_queue,
        audio_buffer=AudioBuffer(),
        level_meter=AudioLevelMeter(),
        threshold_gate=DbThresholdGate(args.min_db),
        preprocessor=AudioPreprocessor(
            enhance_threshold_db=args.enhance_threshold_db,
            noise_reduction_db=args.noise_reduction_db,
            main_gain_db=args.main_gain_db,
            enhance_sharpness=args.enhance_sharpness,
        ),
        inference_engine=inference_engine,
        min_score=args.min_score,
        skip_low_db=args.skip_low_db,
        publisher=ble_server,
    )

    stop_requested = False

    def stop(_signum, _frame) -> None:  # noqa: ANN001
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        return 0
    except Exception as exc:
        print(f"오디오 스트림 오류: {exc}", file=sys.stderr)
        return 1
    finally:
        if stop_requested and ble_server is not None:
            print("Stopping BLE server...", flush=True)
        if ble_server is not None:
            ble_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
