from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_transforms import AudioConfig
from esc50_dataset import DEFAULT_ESC50_ROOT, ESC50LogMelDataset, load_esc50_metadata, resolve_path
from mobilenetv4_model import MODEL_NAME, count_parameters, create_mobilenetv4_small


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/esc-50_tuning_model.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV4-small log-mel image classifier on ESC-50."
    )
    parser.add_argument("--esc50-root", type=Path, default=DEFAULT_ESC50_ROOT)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--val-fold", type=int, choices=[1, 2, 3, 4, 5], default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda 또는 cuda:0 형식. 기본값: auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="ImageNet pretrained weight 없이 랜덤 초기화로 학습합니다.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="CUDA mixed precision을 끕니다.",
    )
    return parser.parse_args()


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (SCRIPT_DIR / path).resolve()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device를 요청했지만 torch.cuda.is_available()가 False입니다.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def split_metadata(metadata: pd.DataFrame, val_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = metadata["fold"].astype(int)
    train_df = metadata[~folds.eq(val_fold)].copy().reset_index(drop=True)
    val_df = metadata[folds.eq(val_fold)].copy().reset_index(drop=True)
    if train_df.empty:
        raise ValueError(f"val_fold={val_fold}를 제외한 학습 샘플이 없습니다.")
    if val_df.empty:
        raise ValueError(f"val_fold={val_fold} validation 샘플이 없습니다.")
    return train_df, val_df


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with make_autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_count += batch_size

        progress.set_postfix(
            loss=f"{total_loss / max(total_count, 1):.4f}",
            acc=f"{total_correct / max(total_count, 1):.4f}",
        )

    return total_loss / total_count, total_correct / total_count


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc=f"Val epoch {epoch}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).long()

        with make_autocast(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_count += batch_size

        progress.set_postfix(
            loss=f"{total_loss / max(total_count, 1):.4f}",
            acc=f"{total_correct / max(total_count, 1):.4f}",
        )

    return total_loss / total_count, total_correct / total_count


def json_ready_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_names: list[str],
    audio_config: AudioConfig,
    args: argparse.Namespace,
    epoch: int,
    best_val_accuracy: float,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "model_name": MODEL_NAME,
            "audio_config": asdict(audio_config),
            "train_args": json_ready_args(args),
            "epoch": int(epoch),
            "best_val_accuracy": float(best_val_accuracy),
            "history": history,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs는 1 이상이어야 합니다.")
    if args.batch_size < 1:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    seed_everything(args.seed)
    checkpoint_path = resolve_output_path(args.output_checkpoint)
    metadata_bundle = load_esc50_metadata(
        esc50_root=args.esc50_root,
        metadata_csv=args.metadata_csv,
        fold=None,
    )
    train_df, val_df = split_metadata(metadata_bundle.dataframe, args.val_fold)

    audio_config = AudioConfig()
    train_dataset = ESC50LogMelDataset(train_df, metadata_bundle.audio_dir, audio_config)
    val_dataset = ESC50LogMelDataset(val_df, metadata_bundle.audio_dir, audio_config)
    device = get_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = create_mobilenetv4_small(
        num_classes=len(metadata_bundle.class_names),
        pretrained=not args.no_pretrained,
        model_name=MODEL_NAME,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_grad_scaler(use_amp)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"ESC-50 root: {metadata_bundle.esc50_root}")
    print(f"Train rows: {len(train_dataset)} | Val rows: {len(val_dataset)} | Val fold: {args.val_fold}")
    print(f"Classes: {len(metadata_bundle.class_names)} | Parameters: {count_parameters(model):,}")
    print(f"Checkpoint output: {checkpoint_path}")

    best_val_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
        )
        val_loss, val_accuracy = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
        )
        scheduler.step()

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                class_names=metadata_bundle.class_names,
                audio_config=audio_config,
                args=args,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                history=history,
            )
            print(f"Saved best checkpoint: {checkpoint_path} | val_acc={best_val_accuracy:.4f}")

    history_path = checkpoint_path.with_suffix(".history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary_path = checkpoint_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "history_csv": str(history_path),
                "best_val_accuracy": best_val_accuracy,
                "val_fold": args.val_fold,
                "epochs": args.epochs,
                "model_name": MODEL_NAME,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved training history: {history_path}")
    print(f"Saved training summary: {summary_path}")


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
