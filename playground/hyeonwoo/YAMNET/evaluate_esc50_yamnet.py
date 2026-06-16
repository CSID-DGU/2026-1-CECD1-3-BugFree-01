from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm


YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16_000
SCRIPT_DIR = Path(__file__).resolve().parent


# ESC-50 category names and YAMNet labels do not use the same taxonomy.
# This mapping is intentionally small and easy to edit.
#
# How to add or fix mappings:
# - Key: ESC-50 category value from meta/esc50.csv, for example "dog".
# - Value: one or more YAMNet AudioSet display labels that should count as
#   correct for that ESC-50 category.
# - The evaluation below uses top-1 accuracy only. A file is correct when the
#   top-1 YAMNet label is included in the mapped label list for that category.
# - Use results/yamnet_raw_predictions.csv to inspect the actual YAMNet labels
#   before expanding this dictionary.
#
# Examples:
#   "dog": ["Dog", "Bark"],
#   "sea_waves": ["Waves, surf"],
ESC50_TO_YAMNET_LABELS: dict[str, list[str]] = {
    "dog": ["Dog", "Bark"],
    "cat": ["Cat", "Meow"],
    "rooster": ["Rooster"],
    "crying_baby": ["Baby cry, infant cry"],
    "sneezing": ["Sneeze"],
    "clapping": ["Clapping"],
    "coughing": ["Cough"],
    "laughing": ["Laughter"],
    "toilet_flush": ["Toilet flush"],
    "thunderstorm": ["Thunderstorm"],
    "sea_waves": ["Waves, surf"],
    "vacuum_cleaner": ["Vacuum cleaner"],
    "clock_alarm": ["Alarm clock"],
    "helicopter": ["Helicopter"],
    "chainsaw": ["Chainsaw"],
    "siren": ["Siren"],
    "church_bells": ["Church bell"],
    "fireworks": ["Fireworks"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained YAMNet predictions on the ESC-50 dataset."
    )
    parser.add_argument(
        "--esc50_dir",
        type=Path,
        default=Path("../ESC-50-master"),
        help="ESC-50 dataset directory. Default: ../ESC-50-master",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of YAMNet predictions to save for each wav file. Default: 5",
    )
    return parser.parse_args()


def resolve_esc50_dir(path: Path) -> Path:
    if path.is_absolute():
        return path

    script_relative = (SCRIPT_DIR / path).resolve()
    cwd_relative = path.resolve()

    if script_relative.exists() or not cwd_relative.exists():
        return script_relative
    return cwd_relative


def validate_esc50_paths(esc50_dir: Path) -> tuple[Path, Path]:
    metadata_path = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    if not esc50_dir.exists():
        raise FileNotFoundError(
            f"ESC-50 directory not found: {esc50_dir}\n"
            f"Expected metadata file: {metadata_path}\n"
            f"Expected audio directory: {audio_dir}"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata file not found: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 audio directory not found: {audio_dir}")

    return metadata_path, audio_dir


def load_yamnet_model() -> tuple[tf.Module, Path]:
    model_dir = Path(hub.resolve(YAMNET_MODEL_URL))
    model = hub.load(str(model_dir))
    return model, model_dir


def load_yamnet_class_names(model: tf.Module) -> list[str]:
    if not hasattr(model, "class_map_path"):
        raise RuntimeError("Loaded YAMNet model does not expose class_map_path().")

    class_map_path_value = model.class_map_path().numpy()
    if isinstance(class_map_path_value, bytes):
        class_map_path_value = class_map_path_value.decode("utf-8")

    class_map_path = Path(class_map_path_value)
    if not class_map_path.exists():
        raise FileNotFoundError(f"YAMNet class map asset was not found: {class_map_path}")

    class_map = pd.read_csv(class_map_path)
    return class_map["display_name"].tolist()


def get_path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size

    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def count_parameters(variables: list[tf.Variable]) -> int:
    return int(sum(np.prod(variable.shape.as_list()) for variable in variables))


def load_wav_16k_mono(wav_path: Path) -> tf.Tensor:
    audio_binary = tf.io.read_file(str(wav_path))
    waveform, sample_rate = tf.audio.decode_wav(audio_binary)

    waveform = tf.reduce_mean(waveform, axis=1)
    waveform_np = tf.cast(waveform, tf.float32).numpy()

    original_sample_rate = int(sample_rate.numpy())
    if original_sample_rate != YAMNET_SAMPLE_RATE:
        common_divisor = math.gcd(original_sample_rate, YAMNET_SAMPLE_RATE)
        up = YAMNET_SAMPLE_RATE // common_divisor
        down = original_sample_rate // common_divisor
        waveform_np = signal.resample_poly(waveform_np, up=up, down=down).astype(np.float32)

    return tf.convert_to_tensor(waveform_np, dtype=tf.float32)


def predict_topk(
    model: tf.Module,
    class_names: list[str],
    wav_path: Path,
    topk: int,
) -> tuple[list[tuple[str, float]], float]:
    waveform = load_wav_16k_mono(wav_path)
    start_time = time.perf_counter()
    scores, _, _ = model(waveform)
    inference_time_seconds = time.perf_counter() - start_time

    clip_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_indices = np.argsort(clip_scores)[::-1][:topk]

    top_predictions = [(class_names[index], float(clip_scores[index])) for index in top_indices]
    return top_predictions, inference_time_seconds


def normalize_label(label: str) -> str:
    return label.strip().casefold()


def is_correct_mapped_prediction(esc50_category: str, top1_label: str) -> bool | None:
    mapped_labels = ESC50_TO_YAMNET_LABELS.get(esc50_category)
    if not mapped_labels:
        return None

    normalized_top1 = normalize_label(top1_label)
    normalized_mapped_labels = {normalize_label(label) for label in mapped_labels}
    return normalized_top1 in normalized_mapped_labels


def build_raw_row(
    filename: str,
    esc50_category: str,
    esc50_target: int,
    top_predictions: list[tuple[str, float]],
    topk: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "filename": filename,
        "esc50_category": esc50_category,
        "esc50_target": int(esc50_target),
    }

    for rank in range(1, topk + 1):
        label, score = top_predictions[rank - 1]
        row[f"top{rank}_label"] = label
        row[f"top{rank}_score"] = score

    return row


def evaluate(args: argparse.Namespace) -> None:
    if args.topk < 1:
        raise ValueError("--topk must be 1 or greater.")

    esc50_dir = resolve_esc50_dir(args.esc50_dir)
    metadata_path, audio_dir = validate_esc50_paths(esc50_dir)

    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_predictions_path = results_dir / "yamnet_raw_predictions.csv"
    summary_path = results_dir / "yamnet_eval_summary.json"
    model_info_path = results_dir / "yamnet_model_info.json"

    metadata = pd.read_csv(metadata_path)
    required_columns = {"filename", "target", "category"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ESC-50 metadata is missing required columns: {missing}")

    model, model_dir = load_yamnet_model()
    class_names = load_yamnet_class_names(model)

    topk = min(args.topk, len(class_names))
    raw_rows: list[dict[str, object]] = []
    mapped_files = 0
    correct = 0
    inference_times_seconds: list[float] = []

    warmup_waveform = tf.zeros([YAMNET_SAMPLE_RATE], dtype=tf.float32)
    model(warmup_waveform)

    iterator = metadata[["filename", "target", "category"]].itertuples(index=False)
    for item in tqdm(iterator, total=len(metadata), desc="Evaluating ESC-50"):
        wav_path = audio_dir / item.filename
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file listed in metadata was not found: {wav_path}")

        top_predictions, inference_time_seconds = predict_topk(
            model,
            class_names,
            wav_path,
            topk,
        )
        inference_times_seconds.append(inference_time_seconds)
        raw_rows.append(
            build_raw_row(
                filename=item.filename,
                esc50_category=item.category,
                esc50_target=item.target,
                top_predictions=top_predictions,
                topk=topk,
            )
        )

        mapped_result = is_correct_mapped_prediction(item.category, top_predictions[0][0])
        if mapped_result is not None:
            mapped_files += 1
            correct += int(mapped_result)

    pd.DataFrame(raw_rows).to_csv(raw_predictions_path, index=False)

    inference_times_ms = [seconds * 1000.0 for seconds in inference_times_seconds]
    summary = {
        "total_files": int(len(metadata)),
        "mapped_files": int(mapped_files),
        "correct": int(correct),
        "accuracy": (correct / mapped_files) if mapped_files else None,
        "mapped_categories": sorted(
            category for category, labels in ESC50_TO_YAMNET_LABELS.items() if labels
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    variables = list(getattr(model, "variables", []))
    trainable_variables = list(getattr(model, "trainable_variables", []))
    model_size_bytes = get_path_size_bytes(model_dir)
    model_info = {
        "model_name": "YAMNet",
        "model_source": YAMNET_MODEL_URL,
        "model_cache_dir": str(model_dir),
        "model_size_bytes": int(model_size_bytes),
        "model_size_mb": round(model_size_bytes / (1024 * 1024), 3),
        "input_sample_rate_hz": YAMNET_SAMPLE_RATE,
        "class_count": int(len(class_names)),
        "esc50_dir": str(esc50_dir),
        "total_files": int(len(metadata)),
        "topk": int(topk),
        "parameter_count": count_parameters(variables),
        "trainable_parameter_count": count_parameters(trainable_variables),
        "inference_time_note": (
            "Measured model(waveform) call only. Audio file loading and resampling are excluded. "
            "One warm-up inference is excluded."
        ),
        "average_inference_time_ms": round(float(np.mean(inference_times_ms)), 3),
        "median_inference_time_ms": round(float(np.median(inference_times_ms)), 3),
        "min_inference_time_ms": round(float(np.min(inference_times_ms)), 3),
        "max_inference_time_ms": round(float(np.max(inference_times_ms)), 3),
        "total_measured_inference_time_seconds": round(
            float(np.sum(inference_times_seconds)),
            6,
        ),
    }
    model_info_path.write_text(
        json.dumps(model_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved raw predictions: {raw_predictions_path}")
    print(f"Saved evaluation summary: {summary_path}")
    print(f"Saved model information: {model_info_path}")
    print(
        "Mapped top-1 accuracy: "
        f"{summary['accuracy'] if summary['accuracy'] is not None else 'N/A'} "
        f"({correct}/{mapped_files})"
    )


def main() -> int:
    args = parse_args()
    try:
        evaluate(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
