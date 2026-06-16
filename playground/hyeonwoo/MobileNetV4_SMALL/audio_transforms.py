from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio가 필요합니다. 설치 예: pip install torchaudio"
        ) from exc
    return torchaudio


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration_sec: float = 5.0
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 50.0
    f_max: float = 8_000.0
    image_size: int = 224

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)


def fix_audio_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    if waveform.numel() > target_samples:
        return waveform[:target_samples]
    if waveform.numel() < target_samples:
        return F.pad(waveform, (0, target_samples - waveform.numel()))
    return waveform


def _load_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio.load가 실패했고 soundfile fallback도 없습니다. "
            "pip install soundfile 또는 torchcodec 설치가 필요합니다."
        ) from exc

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T)
    return waveform, int(sample_rate)


def load_audio(path: Path, config: AudioConfig) -> torch.Tensor:
    torchaudio = require_torchaudio()
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception:
        waveform, sample_rate = _load_with_soundfile(path)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.float()

    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)

    mono = waveform.mean(dim=0)
    mono = fix_audio_length(mono, config.num_samples)
    return mono.clamp(-1.0, 1.0)


class LogMelImageTransform(nn.Module):
    """ESC-50 wav를 MobileNetV4 입력용 3채널 log-mel 이미지로 변환한다."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        torchaudio = require_torchaudio()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
            normalized=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = fix_audio_length(waveform, self.config.num_samples)
        mel = self.mel(waveform.unsqueeze(0))
        log_mel = self.to_db(mel)

        min_value = log_mel.amin()
        max_value = log_mel.amax()
        denom = (max_value - min_value).clamp_min(1e-6)
        image = (log_mel - min_value) / denom

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image = image.repeat(3, 1, 1)
        image = (image - self.imagenet_mean) / self.imagenet_std
        return image

    def load_file(self, path: Path) -> torch.Tensor:
        waveform = load_audio(path, self.config)
        return self.forward(waveform)
