from __future__ import annotations

import sys
from types import TracebackType
from typing import Callable

import numpy as np
import sounddevice as sd

from audio_queue import AudioQueue
from constants import REQUIRED_INPUT_CHANNELS, SAMPLE_RATE


class MicrophoneModule:
    def __init__(
        self,
        *,
        device_index: int,
        device_info: dict,
        stream_channels: int,
        channel_index: int,
        audio_queue: AudioQueue,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.device_index = device_index
        self.device_info = device_info
        self.stream_channels = stream_channels
        self.channel_index = channel_index
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self._stream: sd.InputStream | None = None

    def validate(self) -> None:
        if self.channel_index < 0 or self.channel_index >= self.stream_channels:
            raise RuntimeError(
                f"선택한 채널 index가 범위를 벗어났습니다: "
                f"channel={self.channel_index}, available=0..{self.stream_channels - 1}"
            )

        if self.stream_channels < REQUIRED_INPUT_CHANNELS:
            print(
                f"경고: 선택한 디바이스가 {self.stream_channels}개 입력 채널만 보고합니다. "
                "사용 가능한 채널로 계속 진행합니다.",
                file=sys.stderr,
            )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"오디오 입력 상태: {status}", file=sys.stderr)
        self.audio_queue.push(indata)

    def open(self) -> None:
        self.validate()
        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.stream_channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "MicrophoneModule":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def extract_mono(self, block: np.ndarray) -> np.ndarray:
        return block[:, self.channel_index].astype(np.float32, copy=True)
