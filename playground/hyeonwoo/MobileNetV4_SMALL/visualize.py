from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


def require_plotting() -> None:
    if plt is None or sns is None:
        raise RuntimeError(
            "matplotlib과 seaborn이 필요합니다. 설치 예: pip install matplotlib seaborn"
        )


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 11


def save_metrics_dashboard(
    overall_metrics: dict[str, Any],
    overall_df: pd.DataFrame,
    output_path: Path,
) -> None:
    topk_df = pd.DataFrame(
        {
            "rank": [key.replace("hit@", "@") for key in overall_metrics["topk_hit_rates"].keys()],
            "hit_rate": list(overall_metrics["topk_hit_rates"].values()),
        }
    )
    overview_names = [
        "Top-1 accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
        "Weighted F1",
        "MRR",
    ]
    overview_df = overall_df[overall_df["metric"].isin(overview_names)].copy()
    overview_df = overview_df.sort_values("value", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})
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
        axes[1].text(index, min(row["hit_rate"] + 0.025, 0.98), f"{row['hit_rate']:.3f}", ha="center")

    fig.suptitle("MobileNetV4-small ESC-50 Analysis Dashboard", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_per_category_heatmap(per_category_df: pd.DataFrame, output_path: Path) -> None:
    metric_columns = [
        "precision",
        "recall",
        "f1_score",
        "mean_reciprocal_rank",
        "average_top1_score",
    ]
    hit_columns = [column for column in per_category_df.columns if column.startswith("hit@")]
    metric_columns.extend(hit_columns)

    plot_df = per_category_df.set_index("category")[metric_columns].sort_values(
        "f1_score",
        ascending=False,
    )
    fig_height = max(10, 0.45 * len(plot_df) + 2)
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


def save_per_category_f1(per_category_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = per_category_df.sort_values("f1_score", ascending=True)
    fig_height = max(10, 0.38 * len(plot_df) + 2)
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


def save_confusion_matrix(confusion_normalized: pd.DataFrame, output_path: Path) -> None:
    fig_width = max(18.0, min(34.0, 0.55 * len(confusion_normalized.columns) + 6.0))
    fig_height = max(16.0, min(30.0, 0.45 * len(confusion_normalized.index) + 6.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        confusion_normalized,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.15,
        linecolor="white",
        cbar_kws={"label": "Row-normalized ratio"},
        ax=ax,
    )
    ax.set_title("MobileNetV4-small ESC-50 Confusion Matrix (Normalized)")
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.tick_params(axis="x", rotation=70, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_top_confusions(top_confusions_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, max(6, 0.45 * len(top_confusions_df) + 2)))
    if top_confusions_df.empty:
        ax.text(0.5, 0.5, "No off-diagonal confusions", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = top_confusions_df.sort_values("count", ascending=True)
        sns.barplot(data=plot_df, x="count", y="pair", ax=ax, color="#E15759")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.set_title("Most Frequent Confusions")
        for container in ax.containers:
            ax.bar_label(container, padding=3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_confidence_distribution(analyzed_predictions: pd.DataFrame, output_path: Path) -> None:
    plot_df = analyzed_predictions.copy()
    plot_df["result"] = np.where(plot_df["top1_is_correct"], "top-1 correct", "top-1 incorrect")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=plot_df,
        x="top1_score",
        hue="result",
        bins=30,
        kde=True,
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-1 softmax score")
    ax.set_ylabel("Density")
    ax.set_title("Top-1 Confidence Distribution")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_all_plots(
    analyzed_predictions: pd.DataFrame,
    overall_metrics: dict[str, Any],
    overall_df: pd.DataFrame,
    per_category_df: pd.DataFrame,
    confusion_normalized: pd.DataFrame,
    top_confusions_df: pd.DataFrame,
    plots_dir: Path,
) -> dict[str, Path]:
    require_plotting()
    set_plot_style()
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "dashboard": plots_dir / "mobilenetv4_metrics_dashboard.png",
        "per_category_heatmap": plots_dir / "mobilenetv4_per_category_metrics_heatmap.png",
        "per_category_f1": plots_dir / "mobilenetv4_per_category_f1.png",
        "confusion_matrix": plots_dir / "mobilenetv4_confusion_matrix_normalized.png",
        "top_confusions": plots_dir / "mobilenetv4_top_confusions.png",
        "confidence_distribution": plots_dir / "mobilenetv4_confidence_distribution.png",
    }
    save_metrics_dashboard(overall_metrics, overall_df, paths["dashboard"])
    save_per_category_heatmap(per_category_df, paths["per_category_heatmap"])
    save_per_category_f1(per_category_df, paths["per_category_f1"])
    save_confusion_matrix(confusion_normalized, paths["confusion_matrix"])
    save_top_confusions(top_confusions_df, paths["top_confusions"])
    save_confidence_distribution(analyzed_predictions, paths["confidence_distribution"])
    return paths


def format_percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def format_score(value: object) -> str:
    return f"{float(value):.3f}"


def df_to_html_table(
    df: pd.DataFrame,
    percent_columns: set[str] | None = None,
    table_id: str | None = None,
    sortable: bool = False,
) -> str:
    percent_columns = percent_columns or set()
    display_df = df.copy()
    for column in display_df.columns:
        if column in percent_columns:
            display_df[column] = display_df[column].map(format_percent)
        elif pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(format_score)

    classes = ["data-table"]
    if sortable:
        classes.append("sortable-table")
    return display_df.to_html(
        index=False,
        classes=" ".join(classes),
        table_id=table_id,
        escape=False,
    )


def build_report_html(
    overall_metrics: dict[str, Any],
    overall_df: pd.DataFrame,
    per_category_df: pd.DataFrame,
    top_confusions_df: pd.DataFrame,
    image_paths: list[Path],
    output_dir: Path,
) -> str:
    hit_key = f"hit@{overall_metrics.get('topk_analyzed', 5)}"
    topk_hit_rates = overall_metrics.get("topk_hit_rates", {})
    hit_value = topk_hit_rates.get(hit_key, 0.0)
    folds = overall_metrics.get("folds", [])
    fold_text = ", ".join(str(fold) for fold in folds) if folds else "unknown"

    cards = [
        ("Total files", f"{int(overall_metrics.get('total_files', 0)):,}"),
        ("Classes", f"{int(overall_metrics.get('class_count', 0)):,}"),
        ("Folds", fold_text),
        ("Top-1 accuracy", format_percent(overall_metrics.get("top1_accuracy", 0.0))),
        ("Macro F1", format_score(overall_metrics.get("macro_f1", 0.0))),
        ("Weighted F1", format_score(overall_metrics.get("weighted_f1", 0.0))),
        ("MRR", format_score(overall_metrics.get("mean_reciprocal_rank", 0.0))),
        (f"Hit@{overall_metrics.get('topk_analyzed', 5)}", format_percent(hit_value)),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    image_html = "\n".join(
        f'<section><h2>{html.escape(path.stem.replace("_", " ").title())}</h2>'
        f'<img src="{html.escape(path.relative_to(output_dir).as_posix())}" '
        f'alt="{html.escape(path.stem)}"></section>'
        for path in image_paths
        if path.exists()
    )

    top_confusions_table = (
        "<p>No off-diagonal confusions were found.</p>"
        if top_confusions_df.empty
        else df_to_html_table(top_confusions_df[["true_category", "predicted_category", "count"]])
    )

    per_category_display = per_category_df.sort_values("f1_score", ascending=False)
    percent_columns = {
        "value",
        "precision",
        "recall",
        "f1_score",
        "mean_reciprocal_rank",
        "average_top1_score",
        *{column for column in per_category_df.columns if column.startswith("hit@")},
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MobileNetV4-small ESC-50 Analysis Report</title>
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
    .sortable-table th {{
      cursor: pointer;
      padding-right: 24px;
      position: relative;
      user-select: none;
      white-space: nowrap;
    }}
    .sortable-table th:focus {{
      outline: 2px solid #3b82f6;
      outline-offset: -2px;
    }}
    .sort-indicator {{
      color: #627d98;
      font-size: 11px;
      margin-left: 6px;
    }}
    .note {{
      padding: 14px 16px;
      background: #fff8e6;
      border: 1px solid #f0d98c;
      border-radius: 8px;
      color: #594214;
    }}
  </style>
  <script>
    function parseSortValue(text) {{
      var cleaned = text.trim().replace(/,/g, "");
      if (cleaned.endsWith("%")) {{
        cleaned = cleaned.slice(0, -1);
      }}
      if (cleaned !== "" && !Number.isNaN(Number(cleaned))) {{
        return Number(cleaned);
      }}
      return text.trim().toLowerCase();
    }}

    function sortTable(table, columnIndex, direction) {{
      var tbody = table.tBodies[0];
      var rows = Array.from(tbody.rows).map(function(row, index) {{
        return {{
          row: row,
          index: index,
          value: parseSortValue(row.cells[columnIndex].textContent)
        }};
      }});
      var numeric = rows.every(function(item) {{
        return typeof item.value === "number";
      }});

      rows.sort(function(left, right) {{
        var comparison;
        if (numeric) {{
          comparison = left.value - right.value;
        }} else {{
          comparison = String(left.value).localeCompare(String(right.value), undefined, {{
            numeric: true,
            sensitivity: "base"
          }});
        }}
        if (comparison === 0) {{
          comparison = left.index - right.index;
        }}
        return direction === "asc" ? comparison : -comparison;
      }});

      rows.forEach(function(item) {{
        tbody.appendChild(item.row);
      }});
    }}

    document.addEventListener("DOMContentLoaded", function() {{
      document.querySelectorAll("table.sortable-table").forEach(function(table) {{
        table.querySelectorAll("thead th").forEach(function(header, columnIndex) {{
          var indicator = document.createElement("span");
          indicator.className = "sort-indicator";
          indicator.setAttribute("aria-hidden", "true");
          header.appendChild(indicator);
          header.tabIndex = 0;
          header.setAttribute("role", "button");
          header.setAttribute("aria-sort", "none");

          function activateSort() {{
            var direction = header.dataset.sortDirection === "asc" ? "desc" : "asc";
            table.querySelectorAll("thead th").forEach(function(otherHeader) {{
              otherHeader.dataset.sortDirection = "";
              otherHeader.setAttribute("aria-sort", "none");
              var otherIndicator = otherHeader.querySelector(".sort-indicator");
              if (otherIndicator) {{
                otherIndicator.textContent = "";
              }}
            }});
            header.dataset.sortDirection = direction;
            header.setAttribute("aria-sort", direction === "asc" ? "ascending" : "descending");
            indicator.textContent = direction === "asc" ? "ASC" : "DESC";
            sortTable(table, columnIndex, direction);
          }}

          header.addEventListener("click", activateSort);
          header.addEventListener("keydown", function(event) {{
            if (event.key === "Enter" || event.key === " ") {{
              event.preventDefault();
              activateSort();
            }}
          }});
        }});
      }});
    }});
  </script>
</head>
<body>
<main>
  <h1>MobileNetV4-small ESC-50 Analysis Report</h1>
  <p class="subtitle">Log-mel spectrogram images are evaluated with a timm MobileNetV4-small classifier.</p>
  <div class="cards">
    {card_html}
  </div>
  <p class="note">If this report was generated without a fine-tuned checkpoint, the ESC-50 classifier head is random and the numbers are only a pipeline baseline.</p>
  {image_html}
  <section>
    <h2>Overall Metrics</h2>
    {df_to_html_table(overall_df, percent_columns={"value"})}
  </section>
  <section>
    <h2>Per-category Metrics</h2>
    {df_to_html_table(
        per_category_display,
        percent_columns=percent_columns,
        table_id="per-category-metrics",
        sortable=True,
    )}
  </section>
  <section>
    <h2>Top Confusions</h2>
    {top_confusions_table}
  </section>
</main>
</body>
</html>
"""


def save_html_report(
    overall_metrics: dict[str, Any],
    overall_df: pd.DataFrame,
    per_category_df: pd.DataFrame,
    top_confusions_df: pd.DataFrame,
    plots_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [
        plots_dir / "mobilenetv4_metrics_dashboard.png",
        plots_dir / "mobilenetv4_per_category_metrics_heatmap.png",
        plots_dir / "mobilenetv4_per_category_f1.png",
        plots_dir / "mobilenetv4_confusion_matrix_normalized.png",
        plots_dir / "mobilenetv4_top_confusions.png",
        plots_dir / "mobilenetv4_confidence_distribution.png",
    ]
    report_html = build_report_html(
        overall_metrics=overall_metrics,
        overall_df=overall_df,
        per_category_df=per_category_df,
        top_confusions_df=top_confusions_df,
        image_paths=image_paths,
        output_dir=output_dir,
    )
    report_path = output_dir / "mobilenetv4_analysis_report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path
