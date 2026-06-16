#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import sys
import wave
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torchaudio


SAMPLE_RATE = 32000
N_MELS = 128
WINDOW_SIZE = 800
HOP_SIZE = 320
N_FFT = 1024
MODEL_NAME = "mn10_as"


def describe(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float().cpu()
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():+.4f}, max={tensor.max().item():+.4f}, "
        f"mean={tensor.mean().item():+.4f}"
    )


def main() -> int:
    repo_dir = Path(__file__).resolve().parent / "EfficientAT"
    if not repo_dir.exists():
        print(f"EfficientAT repo not found: {repo_dir}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(repo_dir))
    old_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        from helpers.utils import NAME_TO_WIDTH, labels  # type: ignore
        from models.mn.model import get_model  # type: ignore
        from models.preprocess import AugmentMelSTFT  # type: ignore

        with redirect_stdout(io.StringIO()):
            model = get_model(
                width_mult=NAME_TO_WIDTH(MODEL_NAME),
                pretrained_name=MODEL_NAME,
                strides=(2, 2, 2, 2),
                head_type="mlp",
            )

        final_bias = model.classifier[-1].bias.detach().float()
        final_weight = model.classifier[-1].weight.detach().float()
        print(
            "classifier.5.bias "
            f"shape={tuple(final_bias.shape)}, min={final_bias.min().item():+.4f}, "
            f"max={final_bias.max().item():+.4f}, mean={final_bias.mean().item():+.4f}"
        )
        print(
            "classifier.5.weight "
            f"shape={tuple(final_weight.shape)}, min={final_weight.min().item():+.4f}, "
            f"max={final_weight.max().item():+.4f}, mean={final_weight.mean().item():+.4f}"
        )

        mel = AugmentMelSTFT(
            n_mels=N_MELS,
            sr=SAMPLE_RATE,
            win_length=WINDOW_SIZE,
            hopsize=HOP_SIZE,
            n_fft=N_FFT,
            freqm=0,
            timem=0,
        )
        model.eval()
        mel.eval()

        examples = {
            "silence_2s": torch.zeros(1, SAMPLE_RATE * 2),
            "silence_10s": torch.zeros(1, SAMPLE_RATE * 10),
            "noise_low_2s": torch.randn(1, SAMPLE_RATE * 2) * 0.005,
            "noise_low_10s": torch.randn(1, SAMPLE_RATE * 10) * 0.005,
            "noise_loud_2s": torch.randn(1, SAMPLE_RATE * 2) * 0.1,
        }

        wav_path = repo_dir / "resources" / "metro_station-paris.wav"
        if wav_path.exists():
            try:
                with wave.open(str(wav_path), "rb") as wav_file:
                    sr = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frames = wav_file.readframes(wav_file.getnframes())

                if sample_width != 2:
                    raise RuntimeError(f"expected 16-bit PCM wav, got sample_width={sample_width}")

                wav_np = torch.frombuffer(frames, dtype=torch.int16).float() / 32768.0
                wav_np = wav_np.reshape(-1, channels).mean(dim=1)
                wav = wav_np.unsqueeze(0)
                if sr != SAMPLE_RATE:
                    wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
                examples["metro_station_resource"] = wav
            except Exception as exc:
                print(f"Skipping resource wav load: {exc}", file=sys.stderr)

        for name, waveform in examples.items():
            with torch.no_grad():
                spec = mel(waveform)
                logits, _ = model(spec.unsqueeze(0))
                probs = torch.sigmoid(logits.float()).squeeze(0)

            print(f"\n== {name} ==")
            describe("waveform", waveform)
            describe("mel", spec)
            describe("logits", logits)
            describe("probs", probs)

            top = torch.argsort(probs, descending=True)[:10].tolist()
            print("top10:")
            for index in top:
                print(f"  {labels[index]}: prob={probs[index].item():.4f}, logit={logits.squeeze(0)[index].item():+.3f}")
    finally:
        os.chdir(old_cwd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
