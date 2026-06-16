import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_wav_mono(path: str, target_sr: int) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_sr:
        # rational resampling without librosa/torchaudio
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = resample_poly(data, up, down).astype(np.float32)

    return data.astype(np.float32)


def crop_or_pad(wav: np.ndarray, target_len: int) -> np.ndarray:
    wav = wav.astype(np.float32)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)), mode="constant")
    elif len(wav) > target_len:
        wav = wav[:target_len]
    return wav.astype(np.float32)


def _pytorch_reflect_pad_1d(x: np.ndarray, pad: int) -> np.ndarray:
    # torch.stft(center=True) uses reflect padding by default.
    return np.pad(x, (pad, pad), mode="reflect")


def waveform_to_mel_tensor(wav: np.ndarray, cfg: dict, mel_basis: np.ndarray) -> np.ndarray:
    sr = int(cfg["sample_rate"])
    clip_samples = int(cfg["clip_samples"])
    n_fft = int(cfg["n_fft"])
    win_length = int(cfg["win_length"])
    hop = int(cfg["hop_size"])
    n_mels = int(cfg["n_mels"])
    time_frames = int(cfg["time_frames"])
    preemph = float(cfg.get("preemphasis", 0.97))
    log_offset = float(cfg.get("log_offset", 1e-5))
    norm_add = float(cfg.get("norm_add", 4.5))
    norm_div = float(cfg.get("norm_div", 5.0))

    wav = crop_or_pad(wav, clip_samples)

    # EfficientAT AugmentMelSTFT: conv1d([[-0.97, 1]]) => x[1:] - 0.97*x[:-1]
    y = wav[1:] - preemph * wav[:-1]

    # torch.stft with center=True pads n_fft//2 on both sides using reflect mode.
    pad = n_fft // 2
    y = _pytorch_reflect_pad_1d(y, pad)

    # torch.hann_window(win_length, periodic=False), padded to n_fft at center.
    window = np.hanning(win_length).astype(np.float32)
    full_window = np.zeros(n_fft, dtype=np.float32)
    left = (n_fft - win_length) // 2
    full_window[left:left + win_length] = window

    power = np.empty((n_fft // 2 + 1, time_frames), dtype=np.float32)
    for t in range(time_frames):
        start = t * hop
        frame = y[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)), mode="constant")
        frame = frame * full_window
        spec = np.fft.rfft(frame, n=n_fft)
        power[:, t] = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)

    mel = np.matmul(mel_basis.astype(np.float32), power)
    mel = np.log(mel + log_offset)
    mel = (mel + norm_add) / norm_div
    mel = mel.reshape(1, 1, n_mels, time_frames).astype(np.float32)
    return np.ascontiguousarray(mel)
