from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass
class MetricArtifacts:
    analyzed_predictions: pd.DataFrame
    overall_metrics: dict[str, Any]
    overall_df: pd.DataFrame
    per_category_df: pd.DataFrame
    confusion_counts: pd.DataFrame
    confusion_normalized: pd.DataFrame
    top_confusions_df: pd.DataFrame


def safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(numeric) or np.isinf(numeric):
        return 0.0
    return numeric


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return safe_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def get_available_ranks(predictions: pd.DataFrame, max_rank: int = 5) -> list[int]:
    ranks = []
    for rank in range(1, max_rank + 1):
        if f"top{rank}_index" in predictions.columns and f"top{rank}_score" in predictions.columns:
            ranks.append(rank)
    if not ranks:
        raise ValueError("prediction CSV에 top-k 컬럼이 없습니다.")
    return ranks


def add_ranking_columns(predictions: pd.DataFrame, ranks: list[int]) -> pd.DataFrame:
    analyzed = predictions.copy()
    best_correct_ranks: list[int | None] = []

    for _, row in analyzed.iterrows():
        true_index = int(row["true_index"])
        best_rank = None
        for rank in ranks:
            if int(row[f"top{rank}_index"]) == true_index:
                best_rank = rank
                break
        best_correct_ranks.append(best_rank)

    analyzed["best_correct_rank"] = best_correct_ranks
    analyzed["reciprocal_rank"] = analyzed["best_correct_rank"].map(
        lambda rank: 0.0 if pd.isna(rank) else 1.0 / float(rank)
    )
    analyzed["top1_is_correct"] = analyzed["best_correct_rank"].eq(1)
    for rank in ranks:
        analyzed[f"hit@{rank}"] = analyzed["best_correct_rank"].map(
            lambda best_rank: bool(pd.notna(best_rank) and best_rank <= rank)
        )
    return analyzed


def build_top_confusions(
    counts_matrix: pd.DataFrame,
    limit: int = 20,
) -> pd.DataFrame:
    rows = []
    row_sums = counts_matrix.sum(axis=1).replace(0, np.nan)
    for true_category in counts_matrix.index:
        for predicted_category in counts_matrix.columns:
            count = int(counts_matrix.loc[true_category, predicted_category])
            if count == 0 or true_category == predicted_category:
                continue
            rows.append(
                {
                    "true_category": true_category,
                    "predicted_category": predicted_category,
                    "count": count,
                    "true_category_error_rate": safe_float(count / row_sums.loc[true_category]),
                    "pair": f"{true_category} -> {predicted_category}",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "true_category",
                "predicted_category",
                "count",
                "true_category_error_rate",
                "pair",
            ]
        )
    return pd.DataFrame(rows).sort_values(["count", "true_category_error_rate"], ascending=False).head(limit)


def analyze_predictions(
    predictions: pd.DataFrame,
    class_names: list[str],
    max_rank: int = 5,
) -> MetricArtifacts:
    required_columns = {
        "filename",
        "fold",
        "esc50_category",
        "true_index",
        "top1_index",
        "top1_label",
        "top1_score",
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"prediction 결과에 필수 컬럼이 없습니다: {missing}")
    if predictions.empty:
        raise ValueError("평가할 prediction row가 없습니다.")

    ranks = get_available_ranks(predictions, max_rank=max_rank)
    analyzed = add_ranking_columns(predictions, ranks)

    labels = list(range(len(class_names)))
    y_true = analyzed["true_index"].astype(int).to_numpy()
    y_pred = analyzed["top1_index"].astype(int).to_numpy()

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    topk_hit_rates = {
        f"hit@{rank}": safe_float(analyzed[f"hit@{rank}"].mean()) for rank in ranks
    }
    inference_ms = (
        analyzed["inference_time_ms"].astype(float)
        if "inference_time_ms" in analyzed.columns
        else pd.Series(dtype=float)
    )

    overall_metrics: dict[str, Any] = {
        "total_files": int(len(analyzed)),
        "folds": sorted(int(value) for value in analyzed["fold"].unique()),
        "class_count": int(len(class_names)),
        "topk_analyzed": int(max(ranks)),
        "top1_accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "macro_precision": safe_float(precision_macro),
        "macro_recall": safe_float(recall_macro),
        "macro_f1": safe_float(f1_macro),
        "weighted_f1": safe_float(f1_weighted),
        "mean_reciprocal_rank": safe_float(analyzed["reciprocal_rank"].mean()),
        "average_top1_score": safe_float(analyzed["top1_score"].mean()),
        "average_top1_score_when_correct": safe_float(
            analyzed.loc[analyzed["top1_is_correct"], "top1_score"].mean()
        ),
        "average_top1_score_when_incorrect": safe_float(
            analyzed.loc[~analyzed["top1_is_correct"], "top1_score"].mean()
        ),
        "average_inference_time_ms": safe_float(inference_ms.mean()) if not inference_ms.empty else None,
        "median_inference_time_ms": safe_float(inference_ms.median()) if not inference_ms.empty else None,
        "topk_hit_rates": topk_hit_rates,
    }

    per_category_rows = []
    for index, category in enumerate(class_names):
        category_df = analyzed[analyzed["true_index"].astype(int).eq(index)]
        row = {
            "category": category,
            "target": index,
            "support": int(per_support[index]),
            "precision": safe_float(per_precision[index]),
            "recall": safe_float(per_recall[index]),
            "f1_score": safe_float(per_f1[index]),
            "mean_reciprocal_rank": safe_float(category_df["reciprocal_rank"].mean()),
            "average_top1_score": safe_float(category_df["top1_score"].mean()),
        }
        for rank in ranks:
            row[f"hit@{rank}"] = safe_float(category_df[f"hit@{rank}"].mean())
        per_category_rows.append(row)
    per_category_df = pd.DataFrame(per_category_rows)

    overall_rows = [
        ("Top-1 accuracy", overall_metrics["top1_accuracy"]),
        ("Macro precision", overall_metrics["macro_precision"]),
        ("Macro recall", overall_metrics["macro_recall"]),
        ("Macro F1", overall_metrics["macro_f1"]),
        ("Weighted F1", overall_metrics["weighted_f1"]),
        ("MRR", overall_metrics["mean_reciprocal_rank"]),
    ]
    overall_rows.extend(
        (f"Hit@{rank}", overall_metrics["topk_hit_rates"][f"hit@{rank}"]) for rank in ranks
    )
    overall_df = pd.DataFrame(overall_rows, columns=["metric", "value"])

    counts = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_counts = pd.DataFrame(counts, index=class_names, columns=class_names)
    row_sums = confusion_counts.sum(axis=1).replace(0, np.nan)
    confusion_normalized = confusion_counts.div(row_sums, axis=0).fillna(0.0)
    top_confusions_df = build_top_confusions(confusion_counts)

    return MetricArtifacts(
        analyzed_predictions=analyzed,
        overall_metrics=to_builtin(overall_metrics),
        overall_df=overall_df,
        per_category_df=per_category_df,
        confusion_counts=confusion_counts,
        confusion_normalized=confusion_normalized,
        top_confusions_df=top_confusions_df,
    )


def save_metric_artifacts(artifacts: MetricArtifacts, output_dir: Path) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.analyzed_predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(artifacts.overall_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts.overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    artifacts.per_category_df.to_csv(output_dir / "per_category_metrics.csv", index=False)
    artifacts.confusion_normalized.to_csv(output_dir / "confusion_matrix.csv", index_label="true_category")
    artifacts.top_confusions_df.to_csv(output_dir / "top_confusions.csv", index=False)
