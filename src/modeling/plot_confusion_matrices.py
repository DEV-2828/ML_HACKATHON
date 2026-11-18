"""Generate publication-quality confusion matrix plots for every trained baseline model.

Reads actual experimental results from `results/metrics/confusion_matrices.json` and
`results/metrics/model_comparison.csv`, saving per-model PNGs and a 2x3 overview grid
into `PLOT/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _build_annotations(cm: np.ndarray) -> np.ndarray:
    """Annotate each confusion matrix cell with raw count + row percentage."""
    row_sums = cm.sum(axis=1, keepdims=True)
    pcts = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f"{cm[i, j]:,d}\n({pcts[i, j]:.1f}%)"
    return labels


def _plot_single_cm(
    ax: plt.Axes,
    cm: np.ndarray,
    model_name: str,
    metrics_row: pd.Series | None,
    cbar: bool = True,
) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    annot = _build_annotations(cm)

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=cbar,
        cbar_kws={"label": "Row-Normalised Rate"} if cbar else None,
        linewidths=1.0,
        linecolor="#d0d7de",
        square=True,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        xticklabels=["Pred: Non-Match (0)", "Pred: Match (1)"],
        yticklabels=["True: Non-Match (0)", "True: Match (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Reference Label", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", rotation=0)

    if metrics_row is not None:
        title = (
            f"{model_name} ({metrics_row['Algorithm Family']})\n"
            f"Test Macro F1={metrics_row['Student-Test Macro F1-score']:.4f} | "
            f"Top-1={metrics_row['Top-1 Accuracy']:.4f} | "
            f"P={metrics_row['Precision']:.3f}, R={metrics_row['Recall']:.3f}"
        )
    else:
        title = f"{model_name} — Student-Test Confusion Matrix"
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)


def generate_confusion_matrix_plots(
    metrics_dir: str | Path = "results/metrics",
    output_dir: str | Path = "PLOT",
) -> list[Path]:
    m_dir = Path(metrics_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (m_dir / "confusion_matrices.json").open("r", encoding="utf-8") as f:
        cms = json.load(f)

    comp_path = m_dir / "model_comparison.csv"
    comp_df = pd.read_csv(comp_path).set_index("Model") if comp_path.exists() else None

    # Order models by CV Macro F1 descending if comparison CSV exists
    model_order = (
        [m for m in comp_df.index.tolist() if m in cms]
        if comp_df is not None
        else list(cms.keys())
    )

    saved_paths: list[Path] = []
    sns.set_theme(style="white")

    # 1. Individual high-res confusion matrix per model
    for model_name in model_order:
        cm = np.asarray(cms[model_name]["matrix"], dtype=np.int64)
        row = comp_df.loc[model_name] if comp_df is not None and model_name in comp_df.index else None

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        _plot_single_cm(ax, cm, model_name, row, cbar=True)
        fig.tight_layout()

        out_path = out_dir / f"cm_{model_name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"Saved {out_path}")

    # 2. Combined 2x3 grid for easy side-by-side review & report comparison
    n_models = len(model_order)
    ncols = 3
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(19.5, 5.8 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, model_name in enumerate(model_order):
        cm = np.asarray(cms[model_name]["matrix"], dtype=np.int64)
        row = comp_df.loc[model_name] if comp_df is not None and model_name in comp_df.index else None
        _plot_single_cm(axes_flat[idx], cm, model_name, row, cbar=False)
        if idx % ncols != 0:
            axes_flat[idx].set_ylabel("")

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        "PathSnap — Student-Test Confusion Matrices Across All 6 Models (10 Trips, 341,116 Labeled Candidates)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.5, h_pad=2.5)
    grid_path = out_dir / "cm_all_models_grid.png"
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(grid_path)
    print(f"Saved {grid_path}")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot confusion matrices for all PathSnap models.")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--out-dir", default="PLOT")
    args = parser.parse_args()
    generate_confusion_matrix_plots(args.metrics_dir, args.out_dir)
