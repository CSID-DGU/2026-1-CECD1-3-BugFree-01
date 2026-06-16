#!/usr/bin/env python3
"""ESC-50 subset training, ONNX export, and edge inference for Jetson Orin Nano.

All default paths intentionally point inside this script's playground/hyeonwoo
directory. In this repository that is:
    /home/bugless/EdgeAudioRecognition/playground/hyeonwoo

Examples:
    python esc50_mobilenet_workflow.py train
    python esc50_mobilenet_workflow.py export
    python esc50_mobilenet_workflow.py infer --wav /playground/hyeonwoo/test.wav
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 1. Target Class Mapping (ESC-50 Subset)
# =============================================================================

WORKSPACE = Path(__file__).resolve().parent
ESC50_ROOT = WORKSPACE / "ESC-50-master"
CHECKPOINT_PATH = WORKSPACE / "best_model.pth"
ONNX_PATH = WORKSPACE / "model.onnx"
LABEL_MAP_PATH = WORKSPACE / "label_map.json"

TARGET_CLASS_MAPPING: dict[str, list[str]] = {
    "construction": ["jackhammer", "drilling"],
    "gunshot": ["gunshot"],
    "alarm_siren": ["siren", "clock_alarm"],
    "horn": ["car_horn"],
    "water": ["rain", "pouring_water", "water_drops"],
    "knock": ["door_wood_knock"],
    "appliances": ["washing_machine", "vacuum_cleaner"],
    "baby_cry": ["crying_baby"],
    "animal_cry": ["dog", "cat"],
    "glass_shatter": ["glass_breaking"],
}

CLASS_NAMES = list(TARGET_CLASS_MAPPING.keys())
ESC50_TO_TARGET = {
    esc50_category: target_class
    for target_class, esc50_categories in TARGET_CLASS_MAPPING.items()
    for esc50_category in esc50_categories
}


def require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise SystemExit(
            "torchaudio is required for audio loading and Mel-Spectrogram extraction.\n"
            "Install a Jetson-compatible build in this venv, for example:\n"
            "  pip install torchaudio==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126"
        ) from exc
    return torchaudio


# =============================================================================
# 2. Audio Preprocessing & Dataset Pipeline
# =============================================================================

@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration_sec: float = 5.0
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 8000.0

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_inside_workspace(path: Path, workspace: Path) -> Path:
    """Prevent accidental writes outside the required workspace."""
    resolved_path = path.expanduser().resolve()
    resolved_workspace = workspace.expanduser().resolve()
    if resolved_path != resolved_workspace and resolved_workspace not in resolved_path.parents:
        raise ValueError(f"Path must be inside {resolved_workspace}: {resolved_path}")
    return resolved_path


class LogMelExtractor(nn.Module):
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return normalized log-mel features with shape [1, n_mels, frames]."""
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        waveform = fix_audio_length(waveform, self.config.num_samples)
        mel = self.mel(waveform.unsqueeze(0))
        log_mel = self.to_db(mel)
        return (log_mel + 80.0) / 80.0


def fix_audio_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    if waveform.numel() > target_samples:
        return waveform[:target_samples]
    if waveform.numel() < target_samples:
        return F.pad(waveform, (0, target_samples - waveform.numel()))
    return waveform


def load_audio(path: Path, config: AudioConfig) -> torch.Tensor:
    torchaudio = require_torchaudio()
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except ImportError as exc:
        if "TorchCodec" not in str(exc) and "torchcodec" not in str(exc):
            raise
        try:
            import soundfile as sf
        except ImportError as sf_exc:
            raise ImportError(
                "torchaudio.load requires torchcodec in this environment, and the "
                "soundfile fallback is not installed. Install one of them:\n"
                "  pip install torchcodec\n"
                "or\n"
                "  pip install soundfile"
            ) from sf_exc
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T)
    waveform = waveform.mean(dim=0)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)
    return fix_audio_length(waveform, config.num_samples)


def read_filtered_metadata(esc50_root: Path, val_fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_path = esc50_root / "meta" / "esc50.csv"
    audio_dir = esc50_root / "audio"
    if not metadata_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 audio directory not found: {audio_dir}")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            esc50_category = row["category"]
            if esc50_category not in ESC50_TO_TARGET:
                continue

            target_class = ESC50_TO_TARGET[esc50_category]
            item = {
                "audio_path": audio_dir / row["filename"],
                "target_class": target_class,
                "label": class_to_index[target_class],
                "esc50_category": esc50_category,
                "fold": int(row["fold"]),
            }
            if item["fold"] == val_fold:
                val_rows.append(item)
            else:
                train_rows.append(item)

    if not train_rows or not val_rows:
        raise RuntimeError(
            f"No usable ESC-50 subset rows found. Check dataset at {esc50_root} and val_fold={val_fold}."
        )
    return train_rows, val_rows


def summarize_target_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for row in rows:
        counts[row["target_class"]] += 1
    return counts


def warn_missing_target_classes(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    train_counts = summarize_target_counts(train_rows)
    val_counts = summarize_target_counts(val_rows)
    missing_train = [class_name for class_name, count in train_counts.items() if count == 0]
    missing_val = [class_name for class_name, count in val_counts.items() if count == 0]

    if missing_train or missing_val:
        print("WARNING: Some target classes have no ESC-50 samples in this split.")
        if missing_train:
            print(f"  Missing from train: {', '.join(missing_train)}")
        if missing_val:
            print(f"  Missing from val/test: {', '.join(missing_val)}")
        print("  A classifier cannot learn or evaluate classes with zero samples.")


class ESC50SubsetDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], audio_config: AudioConfig, train: bool):
        self.rows = rows
        self.audio_config = audio_config
        self.train = train
        self.extractor = LogMelExtractor(audio_config)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        waveform = load_audio(row["audio_path"], self.audio_config)
        if self.train:
            waveform = self._augment_waveform(waveform)
        features = self.extractor(waveform)
        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            gain = torch.empty(1).uniform_(0.75, 1.25).item()
            waveform = waveform * gain
        if torch.rand(()) < 0.5:
            shift = int(torch.randint(-1600, 1601, (1,)).item())
            waveform = torch.roll(waveform, shifts=shift)
        if torch.rand(()) < 0.25:
            noise = torch.randn_like(waveform) * 0.003
            waveform = waveform + noise
        return waveform.clamp(-1.0, 1.0)


# =============================================================================
# 3. Lightweight Model Architecture (For Jetson Orin Nano)
# =============================================================================

def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = torchvision.models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Could not load pretrained MobileNetV3 weights ({exc}). "
            "Training will continue from random initialization.",
            RuntimeWarning,
        )
        model = torchvision.models.mobilenet_v3_small(weights=None)

    first_conv = model.features[0][0]
    first_conv_weight = first_conv.weight.detach().clone() if pretrained else None
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )
    if first_conv_weight is not None:
        model.features[0][0].weight.data.copy_(first_conv_weight.mean(dim=1, keepdim=True))
        if first_conv.bias is not None:
            model.features[0][0].bias.data.copy_(first_conv.bias.data)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_dummy_input(audio_config: AudioConfig) -> torch.Tensor:
    extractor = LogMelExtractor(audio_config)
    waveform = torch.zeros(audio_config.num_samples)
    features = extractor(waveform)
    return features.unsqueeze(0)


# =============================================================================
# 4. Training & Validation Loop
# =============================================================================

@dataclass
class TrainConfig:
    workspace: Path = WORKSPACE
    esc50_root: Path = ESC50_ROOT
    checkpoint_path: Path = CHECKPOINT_PATH
    label_map_path: Path = LABEL_MAP_PATH
    epochs: int = 40
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    val_fold: int = 5
    seed: int = 42
    fp32: bool = False
    pretrained: bool = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)
    for step, (features, labels) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    rows: list[dict[str, Any]],
    class_names: list[str],
    device: torch.device,
    use_amp: bool,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    row_offset = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(features)
        probabilities = torch.softmax(logits.float(), dim=1)
        scores, predicted = torch.max(probabilities, dim=1)

        for batch_index in range(labels.size(0)):
            row = rows[row_offset + batch_index]
            true_index = int(labels[batch_index].item())
            pred_index = int(predicted[batch_index].item())
            confusion[true_index, pred_index] += 1
            predictions.append(
                {
                    "audio_path": str(row["audio_path"]),
                    "esc50_category": row["esc50_category"],
                    "fold": row["fold"],
                    "true_label": class_names[true_index],
                    "pred_label": class_names[pred_index],
                    "pred_score": float(scores[batch_index].item()),
                    "correct": true_index == pred_index,
                }
            )
        row_offset += labels.size(0)

    return predictions, confusion


def print_test_report(predictions: list[dict[str, Any]], confusion: torch.Tensor, class_names: list[str]) -> None:
    correct = sum(1 for row in predictions if row["correct"])
    total = len(predictions)
    accuracy = correct / total if total else 0.0

    print(f"Test rows: {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("\nPer-class accuracy:")
    for index, class_name in enumerate(class_names):
        class_total = int(confusion[index].sum().item())
        class_correct = int(confusion[index, index].item())
        class_acc = class_correct / class_total if class_total else 0.0
        print(f"  {class_name}: {class_acc:.4f} ({class_correct}/{class_total})")

    print("\nConfusion matrix: rows=true, cols=pred")
    header = "true\\pred," + ",".join(class_names)
    print(header)
    for index, class_name in enumerate(class_names):
        counts = ",".join(str(int(value)) for value in confusion[index].tolist())
        print(f"{class_name},{counts}")


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "esc50_category", "fold", "true_label", "pred_label", "pred_score", "correct"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def run_train(args: argparse.Namespace) -> None:
    workspace = ensure_inside_workspace(Path(args.workspace), Path(args.workspace))
    checkpoint_path = ensure_inside_workspace(Path(args.checkpoint_path), workspace)
    label_map_path = ensure_inside_workspace(Path(args.label_map_path), workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        workspace=workspace,
        esc50_root=Path(args.esc50_root),
        checkpoint_path=checkpoint_path,
        label_map_path=label_map_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_fold=args.val_fold,
        seed=args.seed,
        fp32=args.fp32,
        pretrained=args.pretrained,
    )
    audio_config = AudioConfig(
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    seed_everything(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not train_config.fp32

    train_rows, val_rows = read_filtered_metadata(train_config.esc50_root, train_config.val_fold)
    warn_missing_target_classes(train_rows, val_rows)
    train_dataset = ESC50SubsetDataset(train_rows, audio_config, train=True)
    val_dataset = ESC50SubsetDataset(val_rows, audio_config, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=train_config.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    label_payload = {
        "class_names": CLASS_NAMES,
        "target_class_mapping": TARGET_CLASS_MAPPING,
        "esc50_to_target": ESC50_TO_TARGET,
        "audio_config": asdict(audio_config),
    }
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_payload, f, indent=2, ensure_ascii=False)

    best_val_acc = -math.inf
    print(f"Device: {device}")
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)} | Classes: {len(CLASS_NAMES)}")
    print(
        f"Batch size: {train_config.batch_size} | "
        f"Grad accum steps: {train_config.grad_accum_steps} | "
        f"Effective batch size: {train_config.batch_size * train_config.grad_accum_steps}"
    )
    print(f"Checkpoint: {checkpoint_path}")

    for epoch in range(1, train_config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            train_config.grad_accum_steps,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)

        print(
            f"Epoch {epoch:03d}/{train_config.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "target_class_mapping": TARGET_CLASS_MAPPING,
                    "audio_config": asdict(audio_config),
                    "train_config": {
                        **asdict(train_config),
                        "workspace": str(train_config.workspace),
                        "esc50_root": str(train_config.esc50_root),
                        "checkpoint_path": str(train_config.checkpoint_path),
                        "label_map_path": str(train_config.label_map_path),
                    },
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: val_acc={best_val_acc:.4f}")


# =============================================================================
# 5. ONNX Export Script
# =============================================================================

def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, AudioConfig, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    audio_config = AudioConfig(**checkpoint.get("audio_config", asdict(AudioConfig())))
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, audio_config, class_names


def export_onnx(checkpoint_path: Path, onnx_path: Path, workspace: Path, opset: int) -> None:
    checkpoint_path = ensure_inside_workspace(checkpoint_path, workspace)
    onnx_path = ensure_inside_workspace(onnx_path, workspace)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, audio_config, _ = load_checkpoint_model(checkpoint_path, device)
    dummy_input = get_dummy_input(audio_config).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["log_mel"],
        output_names=["logits"],
        dynamic_axes={"log_mel": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX: {onnx_path}")
    print(f"Input shape: {tuple(dummy_input.shape)}")


def run_export(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).expanduser().resolve()
    export_onnx(Path(args.checkpoint_path), Path(args.onnx_path), workspace, args.opset)


# =============================================================================
# 6. PyTorch Checkpoint Test Script
# =============================================================================

def get_rows_for_split(esc50_root: Path, val_fold: int, split: str) -> list[dict[str, Any]]:
    train_rows, val_rows = read_filtered_metadata(esc50_root, val_fold)
    if split == "train":
        return train_rows
    if split == "val":
        return val_rows
    if split == "all":
        return train_rows + val_rows
    raise ValueError(f"Unsupported split: {split}")


def run_test(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).expanduser().resolve()
    checkpoint_path = ensure_inside_workspace(Path(args.checkpoint_path), workspace)
    predictions_csv = (
        ensure_inside_workspace(Path(args.predictions_csv), workspace)
        if args.predictions_csv is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    audio_config = AudioConfig(**checkpoint.get("audio_config", asdict(AudioConfig())))
    checkpoint_train_config = checkpoint.get("train_config", {})
    val_fold = args.val_fold if args.val_fold is not None else int(checkpoint_train_config.get("val_fold", 5))
    use_amp = device.type == "cuda" and not args.fp32

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    rows = get_rows_for_split(Path(args.esc50_root), val_fold, args.split)
    train_rows, val_rows = read_filtered_metadata(Path(args.esc50_root), val_fold)
    warn_missing_target_classes(train_rows, val_rows)
    dataset = ESC50SubsetDataset(rows, audio_config, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint best_val_acc: {checkpoint.get('best_val_acc', 'unknown')}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"ESC-50 split: {args.split} | val_fold: {val_fold}")

    predictions, confusion = collect_predictions(model, loader, rows, class_names, device, use_amp)
    print_test_report(predictions, confusion, class_names)

    if predictions_csv is not None:
        write_predictions_csv(predictions_csv, predictions)
        print(f"\nSaved predictions CSV: {predictions_csv}")


# =============================================================================
# 7. Edge Inference Script (Jetson Orin Nano)
# =============================================================================

def get_onnx_providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


@torch.inference_mode()
def preprocess_wav_for_onnx(wav_path: Path, audio_config: AudioConfig) -> torch.Tensor:
    waveform = load_audio(wav_path, audio_config)
    extractor = LogMelExtractor(audio_config)
    features = extractor(waveform)
    return features.unsqueeze(0)


def run_infer(args: argparse.Namespace) -> None:
    import numpy as np
    import onnxruntime as ort

    workspace = Path(args.workspace).expanduser().resolve()
    onnx_path = ensure_inside_workspace(Path(args.onnx_path), workspace)
    label_map_path = ensure_inside_workspace(Path(args.label_map_path), workspace)

    with label_map_path.open("r", encoding="utf-8") as f:
        label_payload = json.load(f)
    class_names = label_payload["class_names"]
    audio_config = AudioConfig(**label_payload["audio_config"])

    features = preprocess_wav_for_onnx(Path(args.wav), audio_config).numpy().astype(np.float32)
    providers = get_onnx_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    logits = session.run(["logits"], {"log_mel": features})[0][0]
    logits_tensor = torch.from_numpy(logits)
    probabilities = torch.softmax(logits_tensor, dim=0)
    topk = torch.topk(probabilities, k=min(args.topk, len(class_names)))

    print(f"ONNX providers: {session.get_providers()}")
    print(f"WAV: {args.wav}")
    for rank, (index, score) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        print(f"{rank}. {class_names[index]}: {score:.4f}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESC-50 subset MobileNetV3 workflow for Jetson Orin Nano.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train MobileNetV3-small on the ESC-50 target subset.")
    train.add_argument("--workspace", type=Path, default=WORKSPACE)
    train.add_argument("--esc50-root", type=Path, default=ESC50_ROOT)
    train.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    train.add_argument("--label-map-path", type=Path, default=LABEL_MAP_PATH)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients to keep an effective larger batch while using a smaller CUDA batch.",
    )
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--val-fold", type=int, default=5, choices=[1, 2, 3, 4, 5])
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--fp32", action="store_true", help="Disable CUDA AMP mixed precision.")
    train.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize MobileNetV3-small from ImageNet weights and adapt the first conv to 1-channel log-mel input.",
    )
    train.add_argument("--sample-rate", type=int, default=16_000)
    train.add_argument("--duration-sec", type=float, default=5.0)
    train.add_argument("--n-fft", type=int, default=1024)
    train.add_argument("--hop-length", type=int, default=320)
    train.add_argument("--n-mels", type=int, default=64)
    train.set_defaults(func=run_train)

    export = subparsers.add_parser("export", help="Export the best PyTorch checkpoint to ONNX.")
    export.add_argument("--workspace", type=Path, default=WORKSPACE)
    export.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    export.add_argument("--onnx-path", type=Path, default=ONNX_PATH)
    export.add_argument("--opset", type=int, default=17)
    export.set_defaults(func=run_export)

    test = subparsers.add_parser("test", help="Evaluate best_model.pth on an ESC-50 fold split.")
    test.add_argument("--workspace", type=Path, default=WORKSPACE)
    test.add_argument("--esc50-root", type=Path, default=ESC50_ROOT)
    test.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    test.add_argument("--split", choices=["val", "train", "all"], default="val")
    test.add_argument(
        "--val-fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Fold to evaluate as validation/test. Defaults to the fold saved in the checkpoint.",
    )
    test.add_argument("--batch-size", type=int, default=32)
    test.add_argument("--num-workers", type=int, default=2)
    test.add_argument("--predictions-csv", type=Path, default=WORKSPACE / "test_predictions.csv")
    test.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    test.add_argument("--fp32", action="store_true", help="Disable CUDA AMP mixed precision.")
    test.set_defaults(func=run_test)

    infer = subparsers.add_parser("infer", help="Run ONNX inference on one raw 5-second WAV file.")
    infer.add_argument("--workspace", type=Path, default=WORKSPACE)
    infer.add_argument("--onnx-path", type=Path, default=ONNX_PATH)
    infer.add_argument("--label-map-path", type=Path, default=LABEL_MAP_PATH)
    infer.add_argument("--wav", type=Path, required=True)
    infer.add_argument("--topk", type=int, default=5)
    infer.set_defaults(func=run_infer)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
