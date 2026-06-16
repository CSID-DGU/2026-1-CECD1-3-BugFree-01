from __future__ import annotations

import argparse
import ast
import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_PREDICTIONS_PATH = Path("results/yamnet_raw_predictions.csv")
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_EVALUATE_SCRIPT_PATH = Path("evaluate_esc50_yamnet.py")
UNMAPPED_PREDICTION = "__unmapped_prediction__"
AMBIGUOUS_PREDICTION = "__ambiguous_prediction__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual YAMNet ESC-50 analysis reports from raw predictions."
    )
    parser.add_argument(
        "--raw_predictions",
        type=Path,
        default=DEFAULT_RAW_PREDICTIONS_PATH,
        help="Path to results/yamnet_raw_predictions.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where report and images are saved.",
    )
    parser.add_argument(
        "--evaluate_script",
        type=Path,
        default=DEFAULT_EVALUATE_SCRIPT_PATH,
        help="Script containing ESC50_TO_YAMNET_LABELS.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Maximum top-k rank to analyze. Default: use all top-k columns in the CSV.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def normalize_label(label: object) -> str:
    if pd.isna(label):
        return ""
    return str(label).strip().casefold()


def load_mapping_from_evaluate_script(script_path: Path) -> dict[str, list[str]]:
    if not script_path.exists():
        raise FileNotFoundError(f"Mapping source script not found: {script_path}")

    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in tree.body:
        target_name = None
        value_node = None

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_name = target.id
                    value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value

        if target_name == "ESC50_TO_YAMNET_LABELS" and value_node is not None:
            mapping = ast.literal_eval(value_node)
            return {
                str(category): [str(label) for label in labels]
                for category, labels in mapping.items()
                if labels
            }

    raise ValueError(f"ESC50_TO_YAMNET_LABELS was not found in {script_path}")


def get_available_ranks(df: pd.DataFrame, requested_topk: int | None) -> list[int]:
    rank_pattern = re.compile(r"^top(\d+)_label$")
    ranks = []
    for column in df.columns:
        match = rank_pattern.match(column)
        if match:
            rank = int(match.group(1))
            if f"top{rank}_score" in df.columns:
                ranks.append(rank)

    ranks = sorted(ranks)
    if not ranks:
        raise ValueError("No top-k prediction columns were found in the raw CSV.")

    if requested_topk is not None:
        if requested_topk < 1:
            raise ValueError("--topk must be 1 or greater.")
        ranks = [rank for rank in ranks if rank <= requested_topk]
        if not ranks:
            raise ValueError("No top-k columns are available for the requested --topk.")

    return ranks


def validate_raw_predictions(df: pd.DataFrame) -> None:
    required_columns = {"filename", "esc50_category", "esc50_target", "top1_label", "top1_score"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Raw prediction CSV is missing required columns: {missing}")


def build_inverse_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    inverse: dict[str, set[str]] = {}
    for category, labels in mapping.items():
        for label in labels:
            inverse.setdefault(normalize_label(label), set()).add(category)
    return {label: sorted(categories) for label, categories in inverse.items()}


def label_to_esc50_category(label: object, inverse_mapping: dict[str, list[str]]) -> str:
    categories = inverse_mapping.get(normalize_label(label), [])
    if not categories:
        return UNMAPPED_PREDICTION
    if len(categories) > 1:
        return AMBIGUOUS_PREDICTION
    return categories[0]


def get_accepted_label_set(mapping: dict[str, list[str]]) -> dict[str, set[str]]:
    return {
        category: {normalize_label(label) for label in labels}
        for category, labels in mapping.items()
    }


def get_correct_rank(row: pd.Series, ranks: list[int], accepted_labels: set[str]) -> int | None:
    for rank in ranks:
        if normalize_label(row[f"top{rank}_label"]) in accepted_labels:
            return rank
    return None


def add_analysis_columns(
    df: pd.DataFrame,
    ranks: list[int],
    mapping: dict[str, list[str]],
    inverse_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    analyzed = df.copy()
    accepted_labels_by_category = get_accepted_label_set(mapping)

    analyzed["is_true_category_mapped"] = analyzed["esc50_category"].isin(mapping)
    analyzed["top1_predicted_esc50_category"] = analyzed["top1_label"].map(
        lambda label: label_to_esc50_category(label, inverse_mapping)
    )
    analyzed["top1_prediction_status"] = np.where(
        analyzed["top1_predicted_esc50_category"].eq(UNMAPPED_PREDICTION),
        "unmapped_yamnet_label",
        np.where(
            analyzed["top1_predicted_esc50_category"].eq(AMBIGUOUS_PREDICTION),
            "ambiguous_yamnet_label",
            "mapped_yamnet_label",
        ),
    )

    correct_ranks: list[int | None] = []
    for _, row in analyzed.iterrows():
        category = row["esc50_category"]
        accepted_labels = accepted_labels_by_category.get(category)
        if not accepted_labels:
            correct_ranks.append(None)
            continue
        correct_ranks.append(get_correct_rank(row, ranks, accepted_labels))

    analyzed["best_correct_rank"] = correct_ranks
    analyzed["top1_is_correct_for_mapping"] = analyzed["best_correct_rank"].eq(1)
    analyzed["topk_is_correct_for_mapping"] = analyzed["best_correct_rank"].notna()
    analyzed["reciprocal_rank"] = analyzed["best_correct_rank"].map(
        lambda rank: 0.0 if pd.isna(rank) else 1.0 / float(rank)
    )

    for rank in ranks:
        analyzed[f"hit_at_{rank}"] = analyzed["best_correct_rank"].map(
            lambda correct_rank: bool(pd.notna(correct_rank) and correct_rank <= rank)
        )

    return analyzed


def safe_float(value: float) -> float:
    if pd.isna(value) or np.isnan(value):
        return 0.0
    return float(value)


def build_metrics(
    analyzed: pd.DataFrame,
    mapped_categories: list[str],
    ranks: list[int],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    mapped_df = analyzed[analyzed["is_true_category_mapped"]].copy()
    if mapped_df.empty:
        raise ValueError(
            "No rows have a mapped ESC-50 category. Add labels to ESC50_TO_YAMNET_LABELS first."
        )

    y_true = mapped_df["esc50_category"]
    y_pred = mapped_df["top1_predicted_esc50_category"]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=mapped_categories,
        average="macro",
        zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=mapped_categories,
        average="micro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=mapped_categories,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=mapped_categories,
            average=None,
            zero_division=0,
        )
    )

    topk_hit_rates = {
        f"hit@{rank}": safe_float(mapped_df[f"hit_at_{rank}"].mean()) for rank in ranks
    }

    mapped_top1_prediction_count = int(
        mapped_df["top1_prediction_status"].eq("mapped_yamnet_label").sum()
    )
    unmapped_top1_prediction_count = int(
        mapped_df["top1_prediction_status"].eq("unmapped_yamnet_label").sum()
    )
    ambiguous_top1_prediction_count = int(
        mapped_df["top1_prediction_status"].eq("ambiguous_yamnet_label").sum()
    )

    metrics = {
        "total_files": int(len(analyzed)),
        "mapped_true_files": int(len(mapped_df)),
        "unmapped_true_files": int(len(analyzed) - len(mapped_df)),
        "true_mapping_coverage": safe_float(len(mapped_df) / len(analyzed)),
        "mapped_categories": mapped_categories,
        "mapped_category_count": int(len(mapped_categories)),
        "topk_analyzed": int(max(ranks)),
        "top1_direct_mapping_accuracy": safe_float(
            mapped_df["top1_is_correct_for_mapping"].mean()
        ),
        "top1_predicted_category_accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(np.mean(per_class_recall)),
        "macro_precision": safe_float(precision_macro),
        "macro_recall": safe_float(recall_macro),
        "macro_f1": safe_float(f1_macro),
        "micro_precision": safe_float(precision_micro),
        "micro_recall": safe_float(recall_micro),
        "micro_f1": safe_float(f1_micro),
        "weighted_precision": safe_float(precision_weighted),
        "weighted_recall": safe_float(recall_weighted),
        "weighted_f1": safe_float(f1_weighted),
        "mean_reciprocal_rank": safe_float(mapped_df["reciprocal_rank"].mean()),
        "topk_hit_rates": topk_hit_rates,
        "mapped_top1_prediction_count": mapped_top1_prediction_count,
        "unmapped_top1_prediction_count": unmapped_top1_prediction_count,
        "ambiguous_top1_prediction_count": ambiguous_top1_prediction_count,
        "top1_prediction_mapping_coverage": safe_float(
            mapped_top1_prediction_count / len(mapped_df)
        ),
        "average_top1_score": safe_float(mapped_df["top1_score"].mean()),
        "average_top1_score_when_top1_correct": safe_float(
            mapped_df.loc[mapped_df["top1_is_correct_for_mapping"], "top1_score"].mean()
        ),
        "average_top1_score_when_top1_incorrect": safe_float(
            mapped_df.loc[~mapped_df["top1_is_correct_for_mapping"], "top1_score"].mean()
        ),
    }

    per_category_rows = []
    for index, category in enumerate(mapped_categories):
        category_df = mapped_df[mapped_df["esc50_category"].eq(category)]
        row = {
            "category": category,
            "support": int(per_class_support[index]),
            "precision": safe_float(per_class_precision[index]),
            "recall": safe_float(per_class_recall[index]),
            "f1_score": safe_float(per_class_f1[index]),
            "top1_direct_recall": safe_float(
                category_df["top1_is_correct_for_mapping"].mean()
            ),
            "mean_reciprocal_rank": safe_float(category_df["reciprocal_rank"].mean()),
            "average_top1_score": safe_float(category_df["top1_score"].mean()),
        }
        for rank in ranks:
            row[f"hit@{rank}"] = safe_float(category_df[f"hit_at_{rank}"].mean())
        per_category_rows.append(row)

    per_category_df = pd.DataFrame(per_category_rows)

    overall_rows = [
        ("Top-1 accuracy", metrics["top1_direct_mapping_accuracy"]),
        ("Balanced accuracy", metrics["balanced_accuracy"]),
        ("Macro precision", metrics["macro_precision"]),
        ("Macro recall", metrics["macro_recall"]),
        ("Macro F1", metrics["macro_f1"]),
        ("Micro F1", metrics["micro_f1"]),
        ("Weighted F1", metrics["weighted_f1"]),
        ("MRR", metrics["mean_reciprocal_rank"]),
        ("Mapping coverage", metrics["true_mapping_coverage"]),
    ]
    overall_df = pd.DataFrame(overall_rows, columns=["metric", "value"])
    return metrics, overall_df, per_category_df


def build_confusion_matrices(
    analyzed: pd.DataFrame,
    mapped_categories: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_df = analyzed[analyzed["is_true_category_mapped"]].copy()
    prediction_columns = mapped_categories.copy()
    if mapped_df["top1_predicted_esc50_category"].eq(UNMAPPED_PREDICTION).any():
        prediction_columns.append(UNMAPPED_PREDICTION)
    if mapped_df["top1_predicted_esc50_category"].eq(AMBIGUOUS_PREDICTION).any():
        prediction_columns.append(AMBIGUOUS_PREDICTION)

    counts = pd.crosstab(
        mapped_df["esc50_category"],
        mapped_df["top1_predicted_esc50_category"],
    )
    counts = counts.reindex(index=mapped_categories, columns=prediction_columns, fill_value=0)

    row_sums = counts.sum(axis=1).replace(0, np.nan)
    normalized = counts.div(row_sums, axis=0).fillna(0.0)
    return counts, normalized


def require_plotting() -> None:
    if plt is None or sns is None:
        raise RuntimeError(
            "matplotlib and seaborn are required for visual reports. "
            "Install them with: pip install -r requirements.txt"
        )


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 11


def save_metrics_dashboard(
    metrics: dict[str, object],
    overall_df: pd.DataFrame,
    output_path: Path,
) -> None:
    topk_df = pd.DataFrame(
        {
            "rank": [label.replace("hit@", "@") for label in metrics["topk_hit_rates"].keys()],
            "hit_rate": list(metrics["topk_hit_rates"].values()),
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})

    overview_df = overall_df[
        overall_df["metric"].isin(
            [
                "Top-1 accuracy",
                "Macro precision",
                "Macro recall",
                "Macro F1",
                "Weighted F1",
                "MRR",
                "Mapping coverage",
            ]
        )
    ].copy()
    overview_df = overview_df.sort_values("value", ascending=True)
    sns.barplot(data=overview_df, x="value", y="metric", ax=axes[0], color="#4C78A8")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("")
    axes[0].set_title("Overall Evaluation Metrics")
    for container in axes[0].containers:
        axes[0].bar_label(container, labels=[f"{value:.3f}" for value in overview_df["value"]])

    sns.lineplot(data=topk_df, x="rank", y="hit_rate", marker="o", linewidth=3, ax=axes[1])
    sns.barplot(data=topk_df, x="rank", y="hit_rate", alpha=0.25, ax=axes[1], color="#59A14F")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Hit rate")
    axes[1].set_title("Top-k Hit Rate")
    for index, row in topk_df.iterrows():
        axes[1].text(index, row["hit_rate"] + 0.025, f"{row['hit_rate']:.3f}", ha="center")

    fig.suptitle("YAMNet ESC-50 Analysis Dashboard", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_per_category_heatmap(per_category_df: pd.DataFrame, output_path: Path) -> None:
    metric_columns = [
        "precision",
        "recall",
        "f1_score",
        "top1_direct_recall",
        "mean_reciprocal_rank",
    ]
    hit_columns = [column for column in per_category_df.columns if column.startswith("hit@")]
    metric_columns.extend(hit_columns)

    plot_df = per_category_df.set_index("category")[metric_columns].sort_values(
        "f1_score",
        ascending=False,
    )
    fig_height = max(8, 0.45 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title("Per-category Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("ESC-50 category")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_per_category_f1_bar(per_category_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = per_category_df.sort_values("f1_score", ascending=True)
    fig_height = max(7, 0.38 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.barplot(data=plot_df, x="f1_score", y="category", ax=ax, color="#F58518")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_ylabel("")
    ax.set_title("F1 Score by ESC-50 Category")
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{value:.2f}" for value in plot_df["f1_score"]], padding=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    fig_width = max(13.0, min(30.0, 0.6 * len(matrix.columns) + 5.0))
    fig_height = max(10.0, min(26.0, 0.48 * len(matrix.index) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalized else "d",
        cmap="Blues",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Ratio" if normalized else "Count"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted ESC-50 category from YAMNet top-1")
    ax.set_ylabel("True ESC-50 category")
    ax.tick_params(axis="x", rotation=55)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_top_confusions(counts_matrix: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    rows = []
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
                    "pair": f"{true_category} -> {predicted_category}",
                }
            )
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(limit)


def save_top_confusions(top_confusions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, max(6, 0.45 * len(top_confusions) + 2)))
    if top_confusions.empty:
        ax.text(0.5, 0.5, "No off-diagonal confusions", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = top_confusions.sort_values("count", ascending=True)
        sns.barplot(data=plot_df, x="count", y="pair", ax=ax, color="#E15759")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.set_title("Most Frequent Confusions")
        for container in ax.containers:
            ax.bar_label(container, padding=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_confidence_distribution(analyzed: pd.DataFrame, output_path: Path) -> None:
    mapped_df = analyzed[analyzed["is_true_category_mapped"]].copy()
    mapped_df["result"] = np.where(
        mapped_df["top1_is_correct_for_mapping"],
        "top-1 correct",
        "top-1 incorrect",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=mapped_df,
        x="top1_score",
        hue="result",
        bins=30,
        kde=True,
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("YAMNet top-1 score")
    ax.set_ylabel("Density")
    ax.set_title("Top-1 Score Distribution")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def format_score(value: object) -> str:
    return f"{float(value):.3f}"


def df_to_html_table(df: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    display_df = df.copy()
    for column in display_df.columns:
        if column in percent_columns:
            display_df[column] = display_df[column].map(format_percent)
        elif pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(format_score)
    return display_df.to_html(index=False, classes="data-table", escape=False)


def build_report_html(
    metrics: dict[str, object],
    overall_df: pd.DataFrame,
    per_category_df: pd.DataFrame,
    top_confusions: pd.DataFrame,
    image_paths: list[Path],
    output_dir: Path,
) -> str:
    cards = [
        ("Total files", f"{metrics['total_files']:,}"),
        ("Mapped files", f"{metrics['mapped_true_files']:,}"),
        ("Mapping coverage", format_percent(metrics["true_mapping_coverage"])),
        ("Top-1 accuracy", format_percent(metrics["top1_direct_mapping_accuracy"])),
        ("Macro F1", format_score(metrics["macro_f1"])),
        ("Weighted F1", format_score(metrics["weighted_f1"])),
        ("MRR", format_score(metrics["mean_reciprocal_rank"])),
        (f"Hit@{metrics['topk_analyzed']}", format_percent(metrics["topk_hit_rates"][f"hit@{metrics['topk_analyzed']}"])),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    image_html = "\n".join(
        f'<section><h2>{html.escape(path.stem.replace("_", " ").title())}</h2>'
        f'<img src="{html.escape(path.relative_to(output_dir).as_posix())}" alt="{html.escape(path.stem)}"></section>'
        for path in image_paths
    )

    top_confusions_table = (
        "<p>No off-diagonal confusions were found.</p>"
        if top_confusions.empty
        else df_to_html_table(top_confusions[["true_category", "predicted_category", "count"]])
    )

    per_category_display = per_category_df.sort_values("f1_score", ascending=False)
    percent_columns = {
        "value",
        "precision",
        "recall",
        "f1_score",
        "top1_direct_recall",
        "mean_reciprocal_rank",
        "average_top1_score",
        *{column for column in per_category_df.columns if column.startswith("hit@")},
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YAMNet ESC-50 Analysis Report</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      font-family: Arial, Helvetica, sans-serif;
      color: #1f2933;
      background: #f6f8fb;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    h2 {{
      margin: 36px 0 14px;
      font-size: 22px;
    }}
    .subtitle {{
      margin: 0 0 24px;
      color: #52606d;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 24px 0 28px;
    }}
    .card {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 16px;
    }}
    .card-label {{
      color: #627d98;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .card-value {{
      color: #102a43;
      font-size: 25px;
      font-weight: 700;
    }}
    section {{
      margin: 24px 0;
      padding: 22px;
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
    }}
    img {{
      width: 100%;
      max-width: 100%;
      display: block;
      border: 1px solid #edf2f7;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: white;
    }}
    .data-table th, .data-table td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      text-align: left;
    }}
    .data-table th {{
      background: #eef2f7;
    }}
    .note {{
      padding: 14px 16px;
      background: #fff8e6;
      border: 1px solid #f0d98c;
      border-radius: 8px;
      color: #594214;
    }}
  </style>
</head>
<body>
<main>
  <h1>YAMNet ESC-50 Analysis Report</h1>
  <p class="subtitle">Metrics are computed only for ESC-50 categories mapped in <code>ESC50_TO_YAMNET_LABELS</code>.</p>
  <div class="cards">
    {card_html}
  </div>
  <p class="note">YAMNet AudioSet labels and ESC-50 categories are different taxonomies. Improve the mapping dictionary before treating these numbers as final model performance.</p>
  {image_html}
  <section>
    <h2>Overall Metrics</h2>
    {df_to_html_table(overall_df, percent_columns={"value"})}
  </section>
  <section>
    <h2>Per-category Metrics</h2>
    {df_to_html_table(per_category_display, percent_columns=percent_columns)}
  </section>
  <section>
    <h2>Top Confusions</h2>
    {top_confusions_table}
  </section>
</main>
</body>
</html>
"""


def save_visual_report(
    analyzed: pd.DataFrame,
    metrics: dict[str, object],
    overall_df: pd.DataFrame,
    per_category_df: pd.DataFrame,
    counts_matrix: pd.DataFrame,
    normalized_matrix: pd.DataFrame,
    output_dir: Path,
) -> Path:
    require_plotting()
    set_plot_style()

    dashboard_path = output_dir / "yamnet_metrics_dashboard.png"
    per_category_heatmap_path = output_dir / "yamnet_per_category_metrics_heatmap.png"
    per_category_f1_path = output_dir / "yamnet_per_category_f1.png"
    counts_confusion_path = output_dir / "yamnet_confusion_matrix_counts.png"
    normalized_confusion_path = output_dir / "yamnet_confusion_matrix_normalized.png"
    top_confusions_path = output_dir / "yamnet_top_confusions.png"
    confidence_path = output_dir / "yamnet_confidence_distribution.png"
    report_path = output_dir / "yamnet_analysis_report.html"

    top_confusions = build_top_confusions(counts_matrix)
    save_metrics_dashboard(metrics, overall_df, dashboard_path)
    save_per_category_heatmap(per_category_df, per_category_heatmap_path)
    save_per_category_f1_bar(per_category_df, per_category_f1_path)
    save_confusion_heatmap(
        counts_matrix,
        counts_confusion_path,
        "YAMNet ESC-50 Confusion Matrix (Counts)",
        normalized=False,
    )
    save_confusion_heatmap(
        normalized_matrix,
        normalized_confusion_path,
        "YAMNet ESC-50 Confusion Matrix (Normalized)",
        normalized=True,
    )
    save_top_confusions(top_confusions, top_confusions_path)
    save_confidence_distribution(analyzed, confidence_path)

    image_paths = [
        dashboard_path,
        per_category_heatmap_path,
        per_category_f1_path,
        normalized_confusion_path,
        counts_confusion_path,
        top_confusions_path,
        confidence_path,
    ]
    report_html = build_report_html(
        metrics,
        overall_df,
        per_category_df,
        top_confusions,
        image_paths,
        output_dir,
    )
    report_path.write_text(report_html, encoding="utf-8")
    return report_path


def analyze(args: argparse.Namespace) -> None:
    raw_predictions_path = resolve_path(args.raw_predictions)
    output_dir = resolve_path(args.output_dir)
    evaluate_script_path = resolve_path(args.evaluate_script)

    if not raw_predictions_path.exists():
        raise FileNotFoundError(f"Raw prediction CSV not found: {raw_predictions_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping_from_evaluate_script(evaluate_script_path)
    mapped_categories = sorted(mapping)
    inverse_mapping = build_inverse_mapping(mapping)

    raw_predictions = pd.read_csv(raw_predictions_path)
    validate_raw_predictions(raw_predictions)
    ranks = get_available_ranks(raw_predictions, args.topk)

    analyzed = add_analysis_columns(raw_predictions, ranks, mapping, inverse_mapping)
    metrics, overall_df, per_category_df = build_metrics(analyzed, mapped_categories, ranks)
    counts_matrix, normalized_matrix = build_confusion_matrices(analyzed, mapped_categories)

    report_path = save_visual_report(
        analyzed,
        metrics,
        overall_df,
        per_category_df,
        counts_matrix,
        normalized_matrix,
        output_dir,
    )

    print(f"Saved visual analysis report: {report_path}")
    print(f"Saved seaborn chart images to: {output_dir}")
    print(
        "Macro F1: "
        f"{metrics['macro_f1']:.4f}, "
        "Weighted F1: "
        f"{metrics['weighted_f1']:.4f}, "
        "Top-1 direct accuracy: "
        f"{metrics['top1_direct_mapping_accuracy']:.4f}"
    )


def main() -> int:
    args = parse_args()
    try:
        analyze(args)
    except (FileNotFoundError, RuntimeError, ValueError, SyntaxError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
