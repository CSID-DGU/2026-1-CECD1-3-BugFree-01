#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import matplotlib
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


WAVEFORM_SAMPLE_RATE = 16000
MODEL_SAMPLE_RATE = 32000
MODEL_INPUT_SECONDS = 10
MODEL_INPUT_SAMPLES = MODEL_SAMPLE_RATE * MODEL_INPUT_SECONDS
AUDIOSET_CLASS_COUNT = 527

N_MELS = 128
WINDOW_SIZE = 800
HOP_SIZE = 320
N_FFT = 1024
FMAX = MODEL_SAMPLE_RATE // 2 - 1000

LABEL_MAPPING = {
    "construction": ["Jackhammer", "Drill"],
    "gunshot":      ["Gunshot, gunfire"],
    "alarm_siren":  ["Siren", "Alarm", "Alarm clock"],
    "horn":         ["Vehicle horn, car horn, honking"],
    "water":        ["Rain", "Raindrop", "Water tap, faucet", "Pour"],
    "knock":        ["Knock"],
    "appliances":   ["Vacuum cleaner"],
    "baby_cry":     ["Baby cry, infant cry"],
    "animal_cry":   ["Dog", "Cat", "Caterwaul"],
    "glass_shatter":["Glass", "Shatter"],
}

ESC50_TO_CUSTOM = {
    # construction
    "chainsaw":          "construction",

    # gunshot
    "gun_shot":          "gunshot",

    # alarm_siren
    "siren":             "alarm_siren",
    "alarm_clock":       "alarm_siren",

    # horn
    "car_horn":          "horn",

    # water
    "rain":              "water",
    "water_drops":       "water",
    "pouring_water":     "water",

    # knock
    "door_wood_knock":   "knock",

    # appliances
    "vacuum_cleaner":    "appliances",

    # baby_cry
    "crying_baby":       "baby_cry",

    # animal_cry
    "dog":               "animal_cry",
    "cat":               "animal_cry",

    # glass_shatter
    "glass_breaking":    "glass_shatter",
}

CLASS_ORDER = list(LABEL_MAPPING.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientAT on ESC-50 mapped to 10 custom classes."
    )
    parser.add_argument(
        "--esc50_dir",
        default="./ESC-50",
        help="ESC-50 root directory. Expected audio/ and meta/esc50.csv inside.",
    )
    parser.add_argument(
        "--model",
        default="mn10_as",
        help="EfficientAT model name.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Root directory for versioned evaluation outputs.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def load_efficientat(model_name: str, device: torch.device):
    repo_dir = Path(__file__).resolve().parent / "EfficientAT"
    if not repo_dir.exists():
        raise RuntimeError(
            f"EfficientAT repository not found: {repo_dir}\n"
            "Clone it first: git clone https://github.com/fschmid56/EfficientAT.git"
        )

    sys.path.insert(0, str(repo_dir.resolve()))
    old_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        from helpers.utils import NAME_TO_WIDTH, labels  # type: ignore
        from models.mn.model import get_model as get_mn  # type: ignore
        from models.preprocess import AugmentMelSTFT  # type: ignore

        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Don't use ConvNormActivation directly.*",
                category=UserWarning,
                module="torchvision\\.ops\\.misc",
            )
            model = get_mn(
                width_mult=NAME_TO_WIDTH(model_name),
                pretrained_name=model_name,
                strides=(2, 2, 2, 2),
                head_type="mlp",
            )

        mel = AugmentMelSTFT(
            n_mels=N_MELS,
            sr=MODEL_SAMPLE_RATE,
            win_length=WINDOW_SIZE,
            hopsize=HOP_SIZE,
            n_fft=N_FFT,
            fmax=FMAX,
            freqm=0,
            timem=0,
        )
    finally:
        os.chdir(old_cwd)

    model.to(device).eval()
    mel.to(device).eval()
    return model, mel, list(labels)


def build_custom_label_indices(labels: Sequence[str]) -> Dict[str, List[int]]:
    if len(labels) != AUDIOSET_CLASS_COUNT:
        raise RuntimeError(
            f"Expected {AUDIOSET_CLASS_COUNT} AudioSet labels, got {len(labels)}."
        )

    label_to_index = {label: index for index, label in enumerate(labels)}
    custom_indices: Dict[str, List[int]] = {}
    missing: Dict[str, List[str]] = {}

    for custom_label, audioset_labels in LABEL_MAPPING.items():
        matched = [label_to_index[label] for label in audioset_labels if label in label_to_index]
        not_found = [label for label in audioset_labels if label not in label_to_index]
        custom_indices[custom_label] = matched
        if not_found:
            missing[custom_label] = not_found

    if missing:
        details = "; ".join(
            f"{custom_label}: {', '.join(labels)}"
            for custom_label, labels in missing.items()
        )
        raise RuntimeError(f"Missing AudioSet labels: {details}")

    return custom_indices


def load_esc50_metadata(esc50_dir: Path) -> pd.DataFrame:
    meta_path = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    if not meta_path.exists():
        raise RuntimeError(f"ESC-50 metadata not found: {meta_path}")
    if not audio_dir.exists():
        raise RuntimeError(f"ESC-50 audio directory not found: {audio_dir}")

    df = pd.read_csv(meta_path)
    required_columns = {"filename", "fold", "target", "category", "esc10", "src_file", "take"}
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"esc50.csv is missing columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["true_label"] = df["category"].map(ESC50_TO_CUSTOM)
    mapped = df[df["true_label"].notna()].copy()
    mapped["audio_path"] = mapped["filename"].apply(lambda name: str(audio_dir / name))
    return mapped


def prepare_waveform(
    audio_path: Path,
    resampler: torch.nn.Module | None,
    device: torch.device,
) -> torch.Tensor:
    waveform, _ = librosa.load(
        str(audio_path),
        sr=WAVEFORM_SAMPLE_RATE,
        mono=True,
        dtype=np.float32,
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.clip(waveform, -1.0, 1.0)

    tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    if resampler is not None:
        tensor = resampler(tensor)

    if tensor.shape[1] < MODEL_INPUT_SAMPLES:
        tensor = torch.nn.functional.pad(tensor, (0, MODEL_INPUT_SAMPLES - tensor.shape[1]))
    elif tensor.shape[1] > MODEL_INPUT_SAMPLES:
        tensor = tensor[:, :MODEL_INPUT_SAMPLES]

    return tensor


def predict_custom_scores(
    waveform: torch.Tensor,
    model: torch.nn.Module,
    mel: torch.nn.Module,
    custom_indices: Dict[str, List[int]],
    device: torch.device,
) -> Tuple[str, Dict[str, float]]:
    amp_context = torch.amp.autocast("cuda", enabled=True) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), amp_context:
        spec = mel(waveform)
        logits, _ = model(spec.unsqueeze(0))
        probabilities = torch.sigmoid(logits.float()).squeeze(0).detach().cpu().numpy()

    if probabilities.shape[0] != AUDIOSET_CLASS_COUNT:
        raise RuntimeError(
            f"Expected {AUDIOSET_CLASS_COUNT} model outputs, got {probabilities.shape[0]}."
        )

    scores = {
        custom_label: float(np.max(probabilities[label_indices]))
        for custom_label, label_indices in custom_indices.items()
    }
    pred_label = max(scores, key=scores.get)
    return pred_label, scores


def save_confusion_matrix(y_true: List[str], y_pred: List[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ESC-50 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_next_results_dir(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)

    version = 1
    while True:
        output_dir = results_root / f"ver{version}"
        try:
            output_dir.mkdir()
            return output_dir
        except FileExistsError:
            version += 1


def main() -> int:
    args = parse_args()
    esc50_dir = Path(args.esc50_dir).resolve()
    results_root = Path(args.results_dir).resolve()

    try:
        device = resolve_device(args.device)
        mapped_df = load_esc50_metadata(esc50_dir)
        model, mel, audioset_labels = load_efficientat(args.model, device)
        custom_indices = build_custom_label_indices(audioset_labels)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    resampler = None
    if WAVEFORM_SAMPLE_RATE != MODEL_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=WAVEFORM_SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(device).eval()

    print("=== ESC-50 Evaluation (매핑된 샘플만) ===")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"총 샘플 수: {len(mapped_df)} / 2000 (매핑됨)")
    print()

    sample_counts = mapped_df["true_label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    print("클래스별 샘플 수:")
    for label, count in sample_counts.items():
        print(f"  {label}: {count}")
    print()

    results = []
    y_true: List[str] = []
    y_pred: List[str] = []
    skipped = 0

    for row in tqdm(
        mapped_df.itertuples(index=False),
        total=len(mapped_df),
        desc="[진행상황]",
        unit="file",
    ):
        audio_path = Path(row.audio_path)
        true_label = str(row.true_label)

        try:
            waveform = prepare_waveform(audio_path, resampler, device)
            pred_label, scores = predict_custom_scores(
                waveform,
                model,
                mel,
                custom_indices,
                device,
            )
        except Exception as exc:
            skipped += 1
            warnings.warn(f"Skipping {row.filename}: {exc}")
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

        result_row = {
            "filename": row.filename,
            "true_label": true_label,
            "pred_label": pred_label,
        }
        result_row.update(scores)
        results.append(result_row)

    if not results:
        print("ERROR: No samples were evaluated.", file=sys.stderr)
        return 1

    print()
    print(f"[진행상황] {len(results)}/{len(mapped_df)} 완료")
    if skipped:
        print(f"로드/추론 실패로 스킵된 샘플: {skipped}")

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )

    output_dir = create_next_results_dir(results_root)
    eval_results_path = output_dir / "eval_results.csv"
    confusion_matrix_path = output_dir / "confusion_matrix.png"
    classification_report_path = output_dir / "classification_report.txt"

    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_results_path, index=False)

    save_confusion_matrix(y_true, y_pred, confusion_matrix_path)

    with open(classification_report_path, "w", encoding="utf-8") as report_file:
        report_file.write("=== ESC-50 Evaluation (mapped samples only) ===\n")
        report_file.write(f"Model: {args.model}\n")
        report_file.write(f"Device: {device}\n")
        report_file.write(f"Results directory: {output_dir}\n")
        report_file.write(f"Mapped samples: {len(mapped_df)} / 2000\n")
        report_file.write(f"Evaluated samples: {len(results)}\n")
        report_file.write(f"Skipped samples: {skipped}\n")
        report_file.write(f"Accuracy: {accuracy:.6f}\n\n")
        report_file.write("Class sample counts:\n")
        for label, count in sample_counts.items():
            report_file.write(f"{label}: {count}\n")
        report_file.write("\nClassification report:\n")
        report_file.write(report)
        report_file.write("\n")

    print("=== 결과 ===")
    print(f"전체 Accuracy: {accuracy:.1%}")
    print()
    print("클래스별 결과:")
    print(report)
    print(f"Results directory -> {output_dir}")
    print(f"Confusion Matrix -> {confusion_matrix_path} 저장 완료")
    print(f"상세 결과 -> {eval_results_path} 저장 완료")
    print(f"Classification Report -> {classification_report_path} 저장 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
