from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_transforms import AudioConfig
from esc50_dataset import DEFAULT_ESC50_ROOT, ESC50LogMelDataset, load_esc50_metadata, resolve_path
from metrics import analyze_predictions, save_metric_artifacts
from mobilenetv4_model import MODEL_NAME, count_parameters, create_mobilenetv4_small, load_checkpoint
from visualize import save_all_plots, save_html_report


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV4-small log-mel image classifier on ESC-50."
    )
    parser.add_argument("--esc50-root", type=Path, default=DEFAULT_ESC50_ROOT)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="ESC-50 fold to evaluate. Defaults to fold 5, matching train.py's held-out fold.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Evaluate all ESC-50 folds instead of a single held-out fold.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda 또는 cuda:0 형식. 기본값: auto",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
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


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch_to_rows(
    batch: dict[str, object],
    probabilities: torch.Tensor,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    class_names: list[str],
    inference_time_ms_per_sample: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    batch_size = int(probabilities.shape[0])
    filenames = list(batch["filename"])
    categories = list(batch["category"])
    folds = [int(value) for value in batch["fold"]]
    targets = [int(value) for value in batch["target"]]

    for batch_index in range(batch_size):
        true_index = targets[batch_index]
        row: dict[str, object] = {
            "filename": filenames[batch_index],
            "fold": folds[batch_index],
            "esc50_category": categories[batch_index],
            "esc50_target": true_index,
            "true_index": true_index,
            "true_label": class_names[true_index],
            "inference_time_ms": inference_time_ms_per_sample,
        }
        for rank in range(1, 6):
            pred_index = int(top_indices[batch_index, rank - 1].item())
            row[f"top{rank}_index"] = pred_index
            row[f"top{rank}_label"] = class_names[pred_index]
            row[f"top{rank}_score"] = float(top_scores[batch_index, rank - 1].item())
        rows.append(row)
    return rows


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    topk = min(5, len(class_names))

    for batch in tqdm(loader, desc="Evaluating ESC-50"):
        images = batch["image"].to(device, non_blocking=True)

        synchronize_if_needed(device)
        start_time = time.perf_counter()
        logits = model(images)
        synchronize_if_needed(device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        probabilities = torch.softmax(logits.float(), dim=1).cpu()
        top_scores, top_indices = torch.topk(probabilities, k=topk, dim=1)

        rows.extend(
            batch_to_rows(
                batch=batch,
                probabilities=probabilities,
                top_indices=top_indices,
                top_scores=top_scores,
                class_names=class_names,
                inference_time_ms_per_sample=elapsed_ms / max(int(images.shape[0]), 1),
            )
        )

    return pd.DataFrame(rows)


def evaluate(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    output_dir = resolve_output_dir(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    eval_fold = None if args.all_folds else args.fold
    metadata_bundle = load_esc50_metadata(
        esc50_root=args.esc50_root,
        metadata_csv=args.metadata_csv,
        fold=eval_fold,
    )
    audio_config = AudioConfig()
    dataset = ESC50LogMelDataset(
        metadata=metadata_bundle.dataframe,
        audio_dir=metadata_bundle.audio_dir,
        audio_config=audio_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = get_device(args.device)
    model = create_mobilenetv4_small(
        num_classes=len(metadata_bundle.class_names),
        pretrained=args.checkpoint is None,
        model_name=MODEL_NAME,
    )
    checkpoint_info: dict[str, object] | None = None
    if args.checkpoint is not None:
        checkpoint_path = resolve_path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint 파일이 없습니다: {checkpoint_path}")
        checkpoint_info = load_checkpoint(model, checkpoint_path, device=torch.device("cpu"), strict=False)

    model.to(device)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"ESC-50 root: {metadata_bundle.esc50_root}")
    print(f"Metadata: {metadata_bundle.metadata_path}")
    print(f"Rows: {len(dataset)} | Classes: {len(metadata_bundle.class_names)} | Fold: {eval_fold or 'all'}")
    print(f"Checkpoint: {checkpoint_info['checkpoint_path'] if checkpoint_info else 'none'}")
    print(f"Parameters: {count_parameters(model):,}")

    predictions = run_inference(model, loader, metadata_bundle.class_names, device)
    artifacts = analyze_predictions(predictions, metadata_bundle.class_names, max_rank=5)
    save_metric_artifacts(artifacts, output_dir)
    plot_paths = save_all_plots(
        analyzed_predictions=artifacts.analyzed_predictions,
        overall_metrics=artifacts.overall_metrics,
        overall_df=artifacts.overall_df,
        per_category_df=artifacts.per_category_df,
        confusion_normalized=artifacts.confusion_normalized,
        top_confusions_df=artifacts.top_confusions_df,
        plots_dir=plots_dir,
    )
    report_path = save_html_report(
        overall_metrics=artifacts.overall_metrics,
        overall_df=artifacts.overall_df,
        per_category_df=artifacts.per_category_df,
        top_confusions_df=artifacts.top_confusions_df,
        plots_dir=plots_dir,
        output_dir=output_dir,
    )

    print(f"Saved predictions: {output_dir / 'predictions.csv'}")
    print(f"Saved overall metrics: {output_dir / 'overall_metrics.json'}")
    print(f"Saved per-category metrics: {output_dir / 'per_category_metrics.csv'}")
    print(f"Saved normalized confusion matrix: {output_dir / 'confusion_matrix.csv'}")
    print(f"Saved top confusions: {output_dir / 'top_confusions.csv'}")
    print(f"Saved plots: {plots_dir}")
    print(f"Saved HTML report: {report_path}")
    print(
        "Summary: "
        f"top1_accuracy={artifacts.overall_metrics['top1_accuracy']:.4f}, "
        f"macro_f1={artifacts.overall_metrics['macro_f1']:.4f}, "
        f"weighted_f1={artifacts.overall_metrics['weighted_f1']:.4f}, "
        f"mrr={artifacts.overall_metrics['mean_reciprocal_rank']:.4f}"
    )
    for _, path in plot_paths.items():
        print(f"  - {path.name}")


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
