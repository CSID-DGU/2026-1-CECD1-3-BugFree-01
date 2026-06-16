from __future__ import annotations

from typing import List, Tuple

import sounddevice as sd

from constants import MIC_CHANNEL_INDEX, MIC_NAME_KEYWORDS, REQUIRED_INPUT_CHANNELS, SAMPLE_RATE


class InputDeviceFinder:
    def input_devices(self) -> List[Tuple[int, dict]]:
        return [
            (index, dict(device))
            for index, device in enumerate(sd.query_devices())
            if int(device.get("max_input_channels", 0)) > 0
        ]

    def print_input_devices(self) -> None:
        devices = self.input_devices()
        if not devices:
            print("사용 가능한 입력 디바이스가 없습니다.")
            return

        print("사용 가능한 입력 디바이스:")
        for index, device in devices:
            print(
                f"  [{index}] {device.get('name')} | "
                f"inputs={device.get('max_input_channels')} | "
                f"default_sr={device.get('default_samplerate')}"
            )

    def find_respeaker_device(self, device_index: int | None) -> Tuple[int, dict, int]:
        devices = self.input_devices()

        if device_index is not None:
            for index, device in devices:
                if index == device_index:
                    channels = min(REQUIRED_INPUT_CHANNELS, int(device["max_input_channels"]))
                    return index, device, channels
            raise RuntimeError(f"입력 디바이스 index {device_index}를 찾을 수 없습니다.")

        candidates = []
        for index, device in devices:
            name = str(device.get("name", "")).lower()
            max_channels = int(device.get("max_input_channels", 0))
            if max_channels <= MIC_CHANNEL_INDEX:
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

        raise RuntimeError("ReSpeaker Array V3 입력 디바이스를 자동 탐색하지 못했습니다.")
