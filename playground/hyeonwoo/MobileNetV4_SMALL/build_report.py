from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from metrics import analyze_predictions, save_metric_artifacts
from visualize import save_all_plots, save_html_report


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MobileNetV4-small ESC-50 HTML report from saved CSV/JSON/PNG results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="ESC-50 fold to display in the report. Defaults to fold 5.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Display all folds instead of filtering the report to one fold.",
    )
    return parser.parse_args()


def resolve_results_dir(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (SCRIPT_DIR / path).resolve()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required report input not found: {path}")
    return path


def build_class_names(predictions: pd.DataFrame) -> list[str]:
    required_columns = {"true_index", "true_label"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")

    target_to_label: dict[int, str] = {}
    for row in predictions[["true_index", "true_label"]].drop_duplicates().itertuples(index=False):
        target = int(row.true_index)
        label = str(row.true_label)
        existing = target_to_label.get(target)
        if existing is not None and existing != label:
            raise ValueError(f"Conflicting labels for target {target}: {existing}, {label}")
        target_to_label[target] = label

    if not target_to_label:
        raise ValueError("Prediction CSV has no class labels.")

    max_target = max(target_to_label)
    return [target_to_label.get(index, f"class_{index}") for index in range(max_target + 1)]


def load_report_predictions(results_dir: Path) -> tuple[pd.DataFrame, Path]:
    all_folds_path = results_dir / "predictions_all_folds.csv"
    predictions_path = results_dir / "predictions.csv"
    if all_folds_path.exists():
        return pd.read_csv(all_folds_path), all_folds_path
    return pd.read_csv(require_file(predictions_path)), predictions_path


def preserve_all_fold_predictions(
    predictions: pd.DataFrame,
    source_path: Path,
    results_dir: Path,
) -> None:
    all_folds = sorted(int(value) for value in predictions["fold"].unique())
    all_folds_path = results_dir / "predictions_all_folds.csv"
    if source_path.name == "predictions.csv" and len(all_folds) > 1 and not all_folds_path.exists():
        predictions.to_csv(all_folds_path, index=False)


def rebuild_report_artifacts(results_dir: Path, fold: int | None) -> Path:
    plots_dir = results_dir / "plots"
    predictions, source_path = load_report_predictions(results_dir)
    preserve_all_fold_predictions(predictions, source_path, results_dir)

    if fold is not None:
        predictions = predictions[predictions["fold"].astype(int).eq(int(fold))].copy()
        if predictions.empty:
            raise ValueError(f"No prediction rows found for fold {fold}.")

    class_names = build_class_names(predictions)
    artifacts = analyze_predictions(predictions, class_names, max_rank=5)
    save_metric_artifacts(artifacts, results_dir)
    save_all_plots(
        analyzed_predictions=artifacts.analyzed_predictions,
        overall_metrics=artifacts.overall_metrics,
        overall_df=artifacts.overall_df,
        per_category_df=artifacts.per_category_df,
        confusion_normalized=artifacts.confusion_normalized,
        top_confusions_df=artifacts.top_confusions_df,
        plots_dir=plots_dir,
    )

    return save_html_report(
        overall_metrics=artifacts.overall_metrics,
        overall_df=artifacts.overall_df,
        per_category_df=artifacts.per_category_df,
        top_confusions_df=artifacts.top_confusions_df,
        plots_dir=plots_dir,
        output_dir=results_dir,
    )


def build_report(results_dir: Path, fold: int | None = 5) -> Path:
    results_dir = resolve_results_dir(results_dir)
    plots_dir = results_dir / "plots"
    if (results_dir / "predictions.csv").exists() or (results_dir / "predictions_all_folds.csv").exists():
        return rebuild_report_artifacts(results_dir, fold)

    overall_metrics = json.loads(
        require_file(results_dir / "overall_metrics.json").read_text(encoding="utf-8")
    )
    overall_df = pd.read_csv(require_file(results_dir / "overall_metrics.csv"))
    per_category_df = pd.read_csv(require_file(results_dir / "per_category_metrics.csv"))
    top_confusions_df = pd.read_csv(require_file(results_dir / "top_confusions.csv"))

    return save_html_report(
        overall_metrics=overall_metrics,
        overall_df=overall_df,
        per_category_df=per_category_df,
        top_confusions_df=top_confusions_df,
        plots_dir=plots_dir,
        output_dir=results_dir,
    )


def main() -> int:
    args = parse_args()
    try:
        report_path = build_report(
            results_dir=args.results_dir,
            fold=None if args.all_folds else args.fold,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved HTML report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
