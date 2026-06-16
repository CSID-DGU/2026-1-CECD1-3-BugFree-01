#!/usr/bin/env python3
"""EfficientAT pretrained inference restricted to the project target labels.

EfficientAT AudioSet checkpoints output 527 AudioSet probabilities. This script
keeps the pretrained checkpoint unchanged, then aggregates selected AudioSet
labels into the 10 consolidated labels used by the ESC-50 subset workflow.

Example:
    python efficientat_target_infer.py --audio-path test.wav --cuda
    python efficientat_target_infer.py --audio-dir ESC-50-master/audio --output-csv results.csv --cuda
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch import autocast
except ImportError:
    autocast = None


WORKSPACE = Path(__file__).resolve().parent


def _is_efficientat_root(path: Path) -> bool:
    return (path / "models").is_dir() and (path / "helpers").is_dir() and (path / "metadata").is_dir()


def resolve_efficientat_root() -> Path:
    """Find the EfficientAT source tree in local checkout or Jetson workspace layouts."""
    candidates: list[Path] = []
    env_root = os.environ.get("EFFICIENTAT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    for parent in [WORKSPACE, *WORKSPACE.parents]:
        candidates.extend(
            [
                parent,
                parent / "EfficientAT",
                parent / "ywkim" / "EfficientAT",
                parent / "yunyeong" / "efficientat_ws" / "EfficientAT",
                parent / "playground" / "ywkim" / "EfficientAT",
                parent / "playground" / "yunyeong" / "efficientat_ws" / "EfficientAT",
                parent / "efficientat_ws" / "EfficientAT",
            ]
        )
    candidates.extend(
        [
            Path("/workspace/EfficientAT"),
            Path("/plyground/ywkim/EfficientAT"),
            Path("/plyground/yunyeong/efficientat_ws/EfficientAT"),
            Path("/playground/ywkim/EfficientAT"),
            Path("/playground/yunyeong/efficientat_ws/EfficientAT"),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_efficientat_root(resolved):
            return resolved
    return candidates[0].expanduser().resolve() if candidates else Path("/workspace/EfficientAT")


EFFICIENTAT_ROOT = resolve_efficientat_root()

TARGET_AUDIOSET_MAPPING: dict[str, list[str]] = {
    "construction": ["Jackhammer", "Drill"],
    "gunshot": ["Gunshot, gunfire"],
    "alarm_siren": ["Siren", "Alarm", "Alarm clock"],
    "horn": ["Vehicle horn, car horn, honking"],
    "water": ["Rain", "Raindrop", "Rain on surface", "Water tap, faucet", "Pour"],
    "knock": ["Knock"],
    "appliances": ["Vacuum cleaner"],
    "baby_cry": ["Baby cry, infant cry"],
    "animal_cry": ["Dog", "Cat", "Caterwaul"],
    "glass_shatter": ["Glass", "Shatter"],
}

ESC50_TO_TARGET: dict[str, str] = {
    "jackhammer": "construction",
    "drilling": "construction",
    "gunshot": "gunshot",
    "siren": "alarm_siren",
    "clock_alarm": "alarm_siren",
    "car_horn": "horn",
    "rain": "water",
    "pouring_water": "water",
    "water_drops": "water",
    "door_wood_knock": "knock",
    "washing_machine": "appliances",
    "vacuum_cleaner": "appliances",
    "crying_baby": "baby_cry",
    "dog": "animal_cry",
    "cat": "animal_cry",
    "glass_breaking": "glass_shatter",
}


def setup_efficientat_imports() -> None:
    if not EFFICIENTAT_ROOT.exists():
        raise FileNotFoundError(f"EfficientAT directory not found: {EFFICIENTAT_ROOT}")
    os.chdir(EFFICIENTAT_ROOT)
    root = str(EFFICIENTAT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_target_mapping(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return TARGET_AUDIOSET_MAPPING
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Target mapping JSON must be an object: {target_label: [audioset_label, ...]}")
    return {str(key): [str(value) for value in values] for key, values in payload.items()}


def build_label_indexes(target_mapping: dict[str, list[str]], audioset_labels: list[str]) -> dict[str, list[int]]:
    label_to_index = {label: index for index, label in enumerate(audioset_labels)}
    indexes: dict[str, list[int]] = {}
    missing: dict[str, list[str]] = {}

    for target_label, candidates in target_mapping.items():
        matched = [label_to_index[label] for label in candidates if label in label_to_index]
        indexes[target_label] = matched
        not_found = [label for label in candidates if label not in label_to_index]
        if not_found:
            missing[target_label] = not_found

    if missing:
        print("Warning: these AudioSet labels were not found and will be ignored:")
        for target_label, labels in missing.items():
            print(f"  {target_label}: {', '.join(labels)}")
    empty_targets = [label for label, matched in indexes.items() if not matched]
    if empty_targets:
        raise ValueError(f"No AudioSet labels matched for target labels: {', '.join(empty_targets)}")
    return indexes


def load_model(args: argparse.Namespace, device: torch.device):
    setup_efficientat_imports()
    from helpers.utils import NAME_TO_WIDTH, labels
    from models.dymn.model import get_model as get_dymn
    from models.ensemble import get_ensemble_model
    from models.mn.model import get_model as get_mobilenet
    from models.preprocess import AugmentMelSTFT

    if args.ensemble:
        with redirect_stdout(io.StringIO()):
            model = get_ensemble_model(args.ensemble)
    elif args.model_name.startswith("dymn"):
        with redirect_stdout(io.StringIO()):
            model = get_dymn(
                width_mult=NAME_TO_WIDTH(args.model_name),
                pretrained_name=args.model_name,
                strides=args.strides,
            )
    else:
        with redirect_stdout(io.StringIO()):
            model = get_mobilenet(
                width_mult=NAME_TO_WIDTH(args.model_name),
                pretrained_name=args.model_name,
                strides=args.strides,
                head_type=args.head_type,
            )

    model.to(device).eval()
    mel = AugmentMelSTFT(
        n_mels=args.n_mels,
        sr=args.sample_rate,
        win_length=args.window_size,
        hopsize=args.hop_size,
    )
    mel.to(device).eval()
    return model, mel, list(labels)


def load_waveform(path: Path, sample_rate: int, duration_sec: float | None) -> torch.Tensor:
    import librosa

    waveform, _ = librosa.core.load(path, sr=sample_rate, mono=True)
    if duration_sec is not None:
        target_len = int(sample_rate * duration_sec)
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]
    return torch.from_numpy(waveform[None, :])


@torch.inference_mode()
def predict_audio(
    path: Path,
    model: torch.nn.Module,
    mel: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    waveform = load_waveform(path, args.sample_rate, args.duration_sec).to(device)
    ctx = autocast(device_type=device.type) if device.type == "cuda" and autocast is not None else nullcontext()
    with ctx:
        spec = mel(waveform)
        logits, _ = model(spec.unsqueeze(0))
    return torch.sigmoid(logits.float()).squeeze().cpu().numpy()


def aggregate_targets(
    probs: np.ndarray,
    target_indexes: dict[str, list[int]],
    audioset_labels: list[str],
    method: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for target_label, indexes in target_indexes.items():
        scores = probs[indexes]
        best_pos = int(np.argmax(scores))
        if method == "mean":
            score = float(np.mean(scores))
        else:
            score = float(scores[best_pos])
        best_index = indexes[best_pos]
        output[target_label] = {
            "score": score,
            "best_audioset_label": audioset_labels[best_index],
            "best_audioset_score": float(probs[best_index]),
        }
    return output


def resolve_input_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (base_dir / path).expanduser().resolve()


def iter_audio_files(args: argparse.Namespace, base_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if args.audio_path:
        paths.append(resolve_input_path(Path(args.audio_path), base_dir))
    if args.audio_dir:
        audio_dir = resolve_input_path(Path(args.audio_dir), base_dir)
        for suffix in ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a"):
            paths.extend(audio_dir.rglob(suffix))
    unique_paths = sorted({path.expanduser().resolve() for path in paths})
    if not unique_paths:
        raise ValueError("Provide --audio-path or --audio-dir with at least one audio file.")
    return unique_paths


def write_csv(path: Path, rows: list[dict[str, Any]], target_labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "top_target", "top_score"]
    fieldnames += [f"score_{label}" for label in target_labels]
    fieldnames += [f"best_audioset_{label}" for label in target_labels]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_esc50_truth(meta_path: Path) -> dict[str, dict[str, str]]:
    truth: dict[str, dict[str, str]] = {}
    with meta_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            if category not in ESC50_TO_TARGET:
                continue
            truth[row["filename"]] = {
                "esc50_category": category,
                "target_label": ESC50_TO_TARGET[category],
                "fold": row["fold"],
            }
    return truth


def print_confusion_matrix(confusion: dict[str, dict[str, int]], labels: list[str]) -> None:
    print("\nConfusion matrix: rows=true, cols=pred")
    print("true\\pred," + ",".join(labels))
    for true_label in labels:
        counts = [str(confusion.get(true_label, {}).get(pred_label, 0)) for pred_label in labels]
        print(f"{true_label}," + ",".join(counts))


def run_evaluate_csv(args: argparse.Namespace) -> None:
    csv_path = args.csv_path.expanduser().resolve()
    meta_path = args.esc50_meta.expanduser().resolve()
    truth = load_esc50_truth(meta_path)
    labels = list(TARGET_AUDIOSET_MAPPING.keys())
    confusion = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    per_class_total = {label: 0 for label in labels}
    per_class_correct = {label: 0 for label in labels}
    total = 0
    correct = 0
    skipped = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = Path(row["audio_path"]).name
            item = truth.get(filename)
            if item is None:
                skipped += 1
                continue
            if args.fold is not None and int(item["fold"]) != args.fold:
                continue

            true_label = item["target_label"]
            pred_label = row["top_target"]
            if pred_label not in labels:
                skipped += 1
                continue

            total += 1
            is_correct = true_label == pred_label
            correct += int(is_correct)
            per_class_total[true_label] += 1
            per_class_correct[true_label] += int(is_correct)
            confusion[true_label][pred_label] += 1

    accuracy = correct / total if total else 0.0
    fold_text = f"fold {args.fold}" if args.fold is not None else "all matched folds"
    print(f"CSV: {csv_path}")
    print(f"ESC-50 meta: {meta_path}")
    print(f"Evaluated: {total} rows ({fold_text})")
    if skipped:
        print(f"Skipped: {skipped} rows without target-label truth")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    print("\nPer-class accuracy:")
    for label in labels:
        class_total = per_class_total[label]
        class_correct = per_class_correct[label]
        class_acc = class_correct / class_total if class_total else 0.0
        print(f"  {label}: {class_acc:.4f} ({class_correct}/{class_total})")

    print_confusion_matrix(confusion, labels)


def run_infer(args: argparse.Namespace) -> None:
    global EFFICIENTAT_ROOT
    original_cwd = Path.cwd()
    output_csv = resolve_input_path(args.output_csv, original_cwd) if args.output_csv else None
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.efficientat_dir is not None:
        EFFICIENTAT_ROOT = args.efficientat_dir.expanduser().resolve()
    target_mapping = load_target_mapping(args.target_mapping)
    model, mel, audioset_labels = load_model(args, device)
    target_indexes = build_label_indexes(target_mapping, audioset_labels)
    audio_paths = iter_audio_files(args, original_cwd)

    rows: list[dict[str, Any]] = []
    for audio_path in audio_paths:
        probs = predict_audio(audio_path, model, mel, device, args)
        targets = aggregate_targets(probs, target_indexes, audioset_labels, args.aggregate)
        ordered = sorted(targets.items(), key=lambda item: item[1]["score"], reverse=True)

        print(f"\nAudio: {audio_path}")
        for target_label, payload in ordered[: args.topk]:
            print(
                f"{target_label}: {payload['score']:.4f} "
                f"(best AudioSet: {payload['best_audioset_label']}={payload['best_audioset_score']:.4f})"
            )

        top_target, top_payload = ordered[0]
        row: dict[str, Any] = {
            "audio_path": str(audio_path),
            "top_target": top_target,
            "top_score": top_payload["score"],
        }
        for target_label, payload in targets.items():
            row[f"score_{target_label}"] = payload["score"]
            row[f"best_audioset_{target_label}"] = payload["best_audioset_label"]
        rows.append(row)

    if output_csv:
        write_csv(output_csv, rows, list(target_mapping.keys()))
        print(f"\nSaved CSV: {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EfficientAT and collect only the project target labels.")
    subparsers = parser.add_subparsers(dest="command")

    infer = subparsers.add_parser("infer", help="Run EfficientAT target-label inference.")
    infer.add_argument("--audio-path", type=Path, help="One audio file to infer.")
    infer.add_argument("--audio-dir", type=Path, help="Directory of audio files to infer recursively.")
    infer.add_argument("--output-csv", type=Path, help="Optional CSV path for batch results.")
    infer.add_argument("--target-mapping", type=Path, help="Optional JSON mapping: target_label -> AudioSet labels.")
    infer.add_argument("--efficientat-dir", type=Path, help="EfficientAT source dir. Default: auto-detect.")
    infer.add_argument("--aggregate", choices=["max", "mean"], default="max")
    infer.add_argument("--topk", type=int, default=10)
    infer.add_argument("--model-name", type=str, default="mn10_as")
    infer.add_argument("--strides", nargs=4, default=[2, 2, 2, 2], type=int)
    infer.add_argument("--head-type", type=str, default="mlp")
    infer.add_argument("--cuda", action="store_true")
    infer.add_argument("--duration-sec", type=float, default=10.0)
    infer.add_argument("--sample-rate", type=int, default=32000)
    infer.add_argument("--window-size", type=int, default=800)
    infer.add_argument("--hop-size", type=int, default=320)
    infer.add_argument("--n-mels", type=int, default=128)
    infer.add_argument("--ensemble", nargs="+", default=[])
    infer.set_defaults(func=run_infer)

    evaluate = subparsers.add_parser("evaluate-csv", help="Evaluate a saved EfficientAT target CSV against ESC-50 metadata.")
    evaluate.add_argument("--csv-path", type=Path, required=True)
    evaluate.add_argument("--esc50-meta", type=Path, default=WORKSPACE / "ESC-50-master" / "meta" / "esc50.csv")
    evaluate.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], help="Evaluate only one ESC-50 fold.")
    evaluate.set_defaults(func=run_evaluate_csv)
    return parser


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] not in {"infer", "evaluate-csv", "-h", "--help"}:
        argv = ["infer", *argv]
    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):
        build_parser().print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
