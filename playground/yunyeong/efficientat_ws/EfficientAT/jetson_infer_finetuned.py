#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single WAV inference on Jetson Nano with a fine-tuned EfficientAT checkpoint."""

from __future__ import print_function

import argparse
import contextlib
import json
import os
import sys
import time
import wave

import numpy as np
import torch

TARGET_SR_DEFAULT = 32000


class nullcontext(object):
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def amp_context(use_cuda, use_amp):
    if use_cuda and use_amp and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast()
    return nullcontext()


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_wav_select_channel(path, channel_index):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("Only 16-bit or 32-bit PCM WAV is supported.")

    if channels > 1:
        audio = audio.reshape(-1, channels)
        if channel_index == -1:
            audio = audio.mean(axis=1)
        else:
            if channel_index < 0 or channel_index >= channels:
                raise RuntimeError("channel_index is out of range: %s for %s channels" % (channel_index, channels))
            audio = audio[:, channel_index]
    return audio.astype(np.float32), sr, channels


def resample_linear(audio, src_sr, dst_sr):
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return audio.astype(np.float32)
    duration = len(audio) / float(src_sr)
    old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_len = int(round(duration * dst_sr))
    new_x = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def prepare_clip(audio, sample_rate, clip_seconds, normalize=True, crop_mode="center"):
    audio = np.asarray(audio, dtype=np.float32)
    clip_len = int(round(float(clip_seconds) * int(sample_rate)))
    if clip_len <= 0:
        raise RuntimeError("clip_seconds must be positive")
    if audio.size == 0:
        audio = np.zeros((clip_len,), dtype=np.float32)

    if normalize:
        audio = audio - float(audio.mean())
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 1e-5:
            peak = 1e-5
        audio = np.clip(audio / peak, -1.0, 1.0).astype(np.float32)

    n = int(audio.shape[0])
    if n < clip_len:
        audio = np.pad(audio, (0, clip_len - n), mode="constant")
    elif n > clip_len:
        if crop_mode == "start":
            start = 0
        elif crop_mode == "end":
            start = n - clip_len
        else:
            start = (n - clip_len) // 2
        audio = audio[start:start + clip_len]
    return audio.astype(np.float32)


def make_windows(audio, sample_rate, clip_seconds, hop_seconds, normalize=True):
    clip_len = int(round(float(clip_seconds) * int(sample_rate)))
    hop_len = int(round(float(hop_seconds) * int(sample_rate)))
    if hop_len <= 0:
        hop_len = clip_len
    if len(audio) <= clip_len:
        return [prepare_clip(audio, sample_rate, clip_seconds, normalize=normalize)]
    windows = []
    start = 0
    while start < len(audio):
        chunk = audio[start:start + clip_len]
        windows.append(prepare_clip(chunk, sample_rate, clip_seconds, normalize=normalize, crop_mode="start"))
        if start + clip_len >= len(audio):
            break
        start += hop_len
    return windows


def ensure_repo_on_path(efficientat_dir):
    efficientat_dir = os.path.abspath(efficientat_dir)
    if not os.path.isdir(efficientat_dir):
        raise RuntimeError("EfficientAT directory not found: %s" % efficientat_dir)
    if efficientat_dir not in sys.path:
        sys.path.insert(0, efficientat_dir)
    os.chdir(efficientat_dir)


def infer_num_classes_from_state(state_dict):
    for key in ["classifier.5.bias", "classifier.1.bias"]:
        if key in state_dict:
            return int(state_dict[key].numel())
    candidates = []
    for k, v in state_dict.items():
        if k.startswith("classifier") and hasattr(v, "ndim") and v.ndim == 1:
            candidates.append((k, int(v.numel())))
    if candidates:
        candidates.sort()
        return candidates[-1][1]
    raise RuntimeError("Could not infer num_classes from checkpoint.")


def load_finetuned_model(checkpoint_path, efficientat_dir, device, override_model_name=None, override_head_type=None, quiet=True):
    ensure_repo_on_path(efficientat_dir)
    from helpers.utils import NAME_TO_WIDTH  # noqa
    from models.mn.model import get_model as get_mobilenet  # noqa
    from models.preprocess import AugmentMelSTFT  # noqa

    ckpt = safe_torch_load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {}) or {}
        labels = ckpt.get("labels", None)
    else:
        state_dict = ckpt
        cfg = {}
        labels = None

    num_classes = len(labels) if labels else infer_num_classes_from_state(state_dict)
    if not labels:
        labels = ["class_%d" % i for i in range(num_classes)]

    model_name = override_model_name or cfg.get("model_name", "mn04_as")
    head_type = override_head_type or cfg.get("head_type", "mlp")
    sample_rate = int(cfg.get("sample_rate", TARGET_SR_DEFAULT))
    clip_seconds = float(cfg.get("clip_seconds", 10.0))

    if quiet:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            model = get_mobilenet(
                num_classes=num_classes,
                pretrained_name=None,
                width_mult=NAME_TO_WIDTH(model_name),
                head_type=head_type,
            )
    else:
        model = get_mobilenet(
            num_classes=num_classes,
            pretrained_name=None,
            width_mult=NAME_TO_WIDTH(model_name),
            head_type=head_type,
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    mel = AugmentMelSTFT(
        n_mels=128,
        sr=sample_rate,
        win_length=800,
        hopsize=320,
        freqm=0,
        timem=0,
    )
    mel.to(device)
    mel.eval()

    meta = {
        "model_name": model_name,
        "head_type": head_type,
        "sample_rate": sample_rate,
        "clip_seconds": clip_seconds,
        "num_classes": num_classes,
        "labels": labels,
    }
    return model, mel, labels, meta


def predict_batch(model, mel, clips, device, use_cuda, use_amp):
    waves = torch.from_numpy(np.stack(clips, axis=0)).to(device)
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), amp_context(use_cuda, use_amp):
        spec = mel(waves)
        out = model(spec.unsqueeze(1))
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits.float(), dim=-1)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    return probs.detach().cpu().numpy(), t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--efficientat-dir", default="/workspace/EfficientAT")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--head-type", default=None)
    parser.add_argument("--channel-index", type=int, default=-1, help="-1=average channels")
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--hop-seconds", type=float, default=5.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--show-model", action="store_true")
    args = parser.parse_args()

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = False

    model, mel, labels, meta = load_finetuned_model(
        args.checkpoint,
        args.efficientat_dir,
        device,
        override_model_name=args.model_name,
        override_head_type=args.head_type,
        quiet=(not args.show_model),
    )
    sample_rate = int(meta["sample_rate"])
    clip_seconds = float(args.clip_seconds if args.clip_seconds is not None else meta["clip_seconds"])

    audio, src_sr, detected_channels = read_wav_select_channel(args.audio_path, args.channel_index)
    audio = resample_linear(audio, src_sr, sample_rate)
    clips = make_windows(audio, sample_rate, clip_seconds, args.hop_seconds, normalize=(not args.no_normalize))
    probs, infer_time = predict_batch(model, mel, clips, device, use_cuda, args.amp)
    avg_probs = probs.mean(axis=0)
    sorted_idx = np.argsort(avg_probs)[::-1]

    result = {
        "audio_path": args.audio_path,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "model_name": meta["model_name"],
        "head_type": meta["head_type"],
        "sample_rate": sample_rate,
        "clip_seconds": clip_seconds,
        "num_windows": len(clips),
        "inference_time_sec": infer_time,
        "topk": [
            {"label": labels[int(i)], "score": float(avg_probs[int(i)])}
            for i in sorted_idx[:args.topk]
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
