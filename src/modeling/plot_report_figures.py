"""Generate all compulsory Report Section 3.3 & Section 4 plots from real experimental CSVs.

Produces 300-DPI PNG figures in `PLOT/`:
  1. cv_macro_f1_comparison.png      — 5-fold GroupKFold CV Macro F1 (±std) across models
  2. student_test_top1_and_f1.png    — Student-Test Top-1 Accuracy & Macro F1 grouped bars
  3. log_loss_comparison.png         — Student-Test Log Loss across models (lower is better)
  4. learning_curves_iterative.png   — Genuine Train vs Val Log Loss ONLY for XGBoost & MLP
  5. optuna_bayesian_history.png     — Optuna TPE trial history + best-so-far & param importance
  6. untuned_vs_tuned_xgboost.png    — Untuned vs Bayesian-Tuned XGBoost side-by-side metrics
  7. cm_XGBoost_tuned.png            — Final Tuned XGBoost Student-Test Confusion Matrix

Never fabricates curves or numbers: reads directly from `results/metrics/*.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def _load_combined_models(m_dir: Path) -> pd.DataFrame:
    comp = pd.read_csv(m_dir / "model_comparison.csv")
    uvt_path = m_dir / "untuned_vs_tuned.csv"
    if uvt_path.exists():
        uvt = pd.read_csv(uvt_path)
        tuned = uvt[uvt["Version"].str.contains("Tuned", case=False, na=False)]
        tuned = tuned[~tuned["Version"].str.contains("Delta", case=False, na=False)]
        if not tuned.empty:
            r = tuned.iloc[0]
            tuned_row = pd.DataFrame(
                [
                    {
                        "Model": "XGBoost (Tuned)",
                        "Algorithm Family": "Boosting (Bayesian Tuned)",
                        "CV Macro F1-score": r["CV Macro F1-score"],
                        "CV Macro F1 Std": r["CV Macro F1 Std"],
                        "Top-1 Accuracy": r["Student-Test Top-1 Accuracy"],
                        "Student-Test Macro F1-score": r["Student-Test Macro F1-score"],
                        "Route Continuity": r["Route Continuity"],
                        "Log Loss": r["Log Loss"],
                        "Precision": r["Precision"],
                        "Recall": r["Recall"],
                        "Inference Time / Trip (ms)": r["Inference Time / Trip (ms)"],
                    }
                ]
            )
            comp = pd.concat([tuned_row, comp], ignore_index=True)
    return comp.sort_values("CV Macro F1-score", ascending=False).reset_index(drop=True)


def plot_cv_macro_f1(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [
        "#1f77b4" if "Tuned" in m else "#5b8fd9" if i <= 2 else "#9bb7e0"
        for i, m in enumerate(df["Model"])
    ]
    bars = ax.barh(
        df["Model"][::-1],
        df["CV Macro F1-score"][::-1],
        xerr=df["CV Macro F1 Std"][::-1],
        color=colors[::-1],
        edgecolor="#1b2a4a",
        capsize=4,
        height=0.62,
    )
    for bar, val, std in zip(
        bars, df["CV Macro F1-score"][::-1], df["CV Macro F1 Std"][::-1]
    ):
        ax.text(
            val + std + 0.006,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f} ± {std:.4f}",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlim(0.65, min(1.03, float((df["CV Macro F1-score"] + df["CV Macro F1 Std"]).max()) + 0.08))
    ax.set_xlabel("5-Fold GroupKFold CV Macro F1-score (80 Train Trips)", fontsize=10.5, fontweight="bold")
    ax.set_title("Cross-Validation Macro F1-score Comparison Across Models", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, out_dir / "cv_macro_f1_comparison.png")


def plot_student_test_top1_and_f1(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    x = np.arange(len(df))
    w = 0.36
    b1 = ax.bar(
        x - w / 2,
        df["Student-Test Macro F1-score"],
        width=w,
        label="Student-Test Macro F1",
        color="#1f77b4",
        edgecolor="#1b2a4a",
    )
    b2 = ax.bar(
        x + w / 2,
        df["Top-1 Accuracy"],
        width=w,
        label="Observation-Level Top-1 Accuracy (Overall)",
        color="#ff7f0e",
        edgecolor="#4a2800",
    )
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.008,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Score", fontsize=10.5, fontweight="bold")
    ax.set_title("Student-Test Macro F1-score & Observation-Level Top-1 Accuracy (10 Trips)", fontsize=12.5, fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, out_dir / "student_test_top1_and_f1.png")


def plot_log_loss(df: pd.DataFrame, out_dir: Path) -> Path:
    df_ll = df.sort_values("Log Loss", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = ["#2ca02c" if i == 0 else "#6baed6" for i in range(len(df_ll))]
    bars = ax.barh(
        df_ll["Model"][::-1],
        df_ll["Log Loss"][::-1],
        color=colors[::-1],
        edgecolor="#1b2a4a",
        height=0.62,
    )
    for bar, val in zip(bars, df_ll["Log Loss"][::-1]):
        ax.text(
            val + 0.004,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.5f}",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlim(0.0, float(df_ll["Log Loss"].max()) * 1.18)
    ax.set_xlabel("Student-Test Binary Log Loss (Lower = Better Probability Calibration)", fontsize=10.5, fontweight="bold")
    ax.set_title("Student-Test Log Loss Comparison (Probability Quality for HMM Emissions)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, out_dir / "log_loss_comparison.png")


def plot_iterative_learning_curves(m_dir: Path, out_dir: Path) -> Path | None:
    xgb_p = m_dir / "history_XGBoost.csv"
    xgb_tuned_p = m_dir / "history_XGBoost_tuned.csv"
    mlp_p = m_dir / "history_MLP.csv"

    panels = []
    if xgb_p.exists():
        panels.append(("XGBoost (Untuned)", pd.read_csv(xgb_p), "round", "Boosting Round"))
    if xgb_tuned_p.exists():
        panels.append(("XGBoost (Bayesian Tuned)", pd.read_csv(xgb_tuned_p), "round", "Boosting Round"))
    if mlp_p.exists():
        panels.append(("MLP (Neural Network)", pd.read_csv(mlp_p), "epoch", "Training Epoch"))

    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.6))
    axes_list = np.atleast_1d(axes).ravel()

    for ax, (title, hdf, xcol, xlabel) in zip(axes_list, panels):
        ax.plot(hdf[xcol], hdf["train_log_loss"], label="Train Log Loss", color="#1f77b4", linewidth=2.0)
        ax.plot(hdf[xcol], hdf["val_log_loss"], label="Val Log Loss (Trip-Grouped)", color="#d62728", linewidth=2.0, linestyle="--")
        best_idx = int(hdf["val_log_loss"].idxmin())
        best_x = hdf.loc[best_idx, xcol]
        best_y = hdf.loc[best_idx, "val_log_loss"]
        ax.scatter([best_x], [best_y], color="#2ca02c", zorder=5, s=55, label=f"Best Val={best_y:.4f} (@ {int(best_x)})")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax.set_ylabel("Binary Log Loss", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Genuine Iterative Training & Validation Loss Curves (Iterative Models Only)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir / "learning_curves_iterative.png")


def plot_optuna_history(m_dir: Path, out_dir: Path) -> Path | None:
    trials_p = m_dir / "optuna_trials.csv"
    if not trials_p.exists():
        print("[INFO] Skipping optuna_bayesian_history.png (optuna_trials.csv not found).")
        return None

    tdf = pd.read_csv(trials_p)
    tdf = tdf[tdf["state"] == "COMPLETE"].sort_values("number").reset_index(drop=True)
    if tdf.empty:
        return None

    tdf["trial_idx"] = tdf["number"] + 1
    tdf["best_so_far"] = tdf["value"].cummax()

    param_cols = [c for c in tdf.columns if c.startswith("params_")]
    corrs = {}
    for col in param_cols:
        vals = pd.to_numeric(tdf[col], errors="coerce")
        if vals.nunique(dropna=True) > 1:
            corrs[col.replace("params_", "")] = abs(float(vals.corr(tdf["value"], method="spearman")))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Left: optimisation history
    ax0 = axes[0]
    ax0.scatter(tdf["trial_idx"], tdf["value"], color="#5b8fd9", alpha=0.8, s=45, label="Trial CV Macro F1")
    ax0.plot(tdf["trial_idx"], tdf["best_so_far"], color="#d62728", linewidth=2.3, label="Best-So-Far CV Macro F1")
    best_i = int(tdf["value"].idxmax())
    ax0.scatter(
        [tdf.loc[best_i, "trial_idx"]],
        [tdf.loc[best_i, "value"]],
        color="#2ca02c",
        s=90,
        zorder=6,
        label=f"Best Trial #{int(tdf.loc[best_i, 'trial_idx'])} ({tdf.loc[best_i, 'value']:.5f})",
    )
    ax0.set_xlabel("Optuna Bayesian Trial #", fontsize=10, fontweight="bold")
    ax0.set_ylabel("5-Fold GroupKFold CV Macro F1", fontsize=10, fontweight="bold")
    ax0.set_title("Bayesian Hyperparameter Optimisation History (XGBoost)", fontsize=11.5, fontweight="bold")
    ax0.legend(fontsize=8.5, frameon=True, loc="lower right")
    ax0.grid(True, linestyle="--", alpha=0.4)

    # Right: parameter impact (|Spearman rho| with CV Macro F1)
    ax1 = axes[1]
    if corrs:
        s_corr = pd.Series(corrs).sort_values(ascending=True)
        bars = ax1.barh(s_corr.index, s_corr.values, color="#ff7f0e", edgecolor="#4a2800", height=0.62)
        for bar, v in zip(bars, s_corr.values):
            ax1.text(v + 0.015, bar.get_y() + bar.get_height() / 2.0, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
        ax1.set_xlim(0.0, min(1.08, float(s_corr.max()) + 0.14))
    ax1.set_xlabel("|Spearman Correlation| with CV Macro F1", fontsize=10, fontweight="bold")
    ax1.set_title("Hyperparameter Sensitivity Across Optuna Trials", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return _save(fig, out_dir / "optuna_bayesian_history.png")


def plot_untuned_vs_tuned(m_dir: Path, out_dir: Path) -> Path | None:
    uvt_p = m_dir / "untuned_vs_tuned.csv"
    if not uvt_p.exists():
        print("[INFO] Skipping untuned_vs_tuned_xgboost.png (untuned_vs_tuned.csv not found).")
        return None

    uvt = pd.read_csv(uvt_p)
    sub = uvt[~uvt["Version"].str.contains("Delta", case=False, na=False)].reset_index(drop=True)
    if len(sub) < 2:
        return None

    metrics = [
        ("CV Macro F1-score", "CV Macro F1"),
        ("Student-Test Macro F1-score", "Test Macro F1"),
        ("Student-Test Top-1 Accuracy", "Top-1 Accuracy"),
        ("Route Continuity", "Route Continuity"),
        ("Log Loss", "Log Loss\n(Lower Better)"),
    ]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    v0 = [float(sub.iloc[0][k]) for k, _ in metrics]
    v1 = [float(sub.iloc[1][k]) for k, _ in metrics]

    b0 = ax.bar(x - w / 2, v0, width=w, label=sub.iloc[0]["Version"], color="#7f9fc4", edgecolor="#1b2a4a")
    b1 = ax.bar(x + w / 2, v1, width=w, label=sub.iloc[1]["Version"], color="#2ca02c", edgecolor="#124212")

    for bars in (b0, b1):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.015,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Metric Value", fontsize=10.5, fontweight="bold")
    ax.set_title("XGBoost: Untuned Baseline vs Bayesian-Tuned Performance (Same Splits & Metrics)", fontsize=12, fontweight="bold", pad=16)
    ax.legend(frameon=True, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, out_dir / "untuned_vs_tuned_xgboost.png")


def plot_tuned_confusion_matrix(m_dir: Path, out_dir: Path) -> Path | None:
    cm_p = m_dir / "cm_XGBoost_tuned.csv"
    if not cm_p.exists():
        print("[INFO] Skipping cm_XGBoost_tuned.png (cm_XGBoost_tuned.csv not found).")
        return None

    cm_df = pd.read_csv(cm_p, index_col=0)
    cm = cm_df.to_numpy(dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,d}\n({cm_norm[i, j] * 100.0:.1f}%)"

    subtitle = ""
    uvt_p = m_dir / "untuned_vs_tuned.csv"
    if uvt_p.exists():
        uvt = pd.read_csv(uvt_p)
        t = uvt[uvt["Version"].str.contains("Tuned", case=False, na=False)]
        t = t[~t["Version"].str.contains("Delta", case=False, na=False)]
        if not t.empty:
            r = t.iloc[0]
            subtitle = (
                f"\nTest Macro F1={r['Student-Test Macro F1-score']:.4f} | "
                f"Top-1={r['Student-Test Top-1 Accuracy']:.4f} | "
                f"LogLoss={r['Log Loss']:.5f}"
            )

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Row-Normalised Rate"},
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
    ax.set_title(f"XGBoost (Bayesian Tuned) — Student-Test Confusion Matrix{subtitle}", fontsize=10.5, fontweight="bold", pad=10)
    fig.tight_layout()
    return _save(fig, out_dir / "cm_XGBoost_tuned.png")


def generate_all_report_plots(
    metrics_dir: str | Path = "results/metrics",
    output_dir: str | Path = "PLOT",
) -> List[Path]:
    m_dir = Path(metrics_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = _load_combined_models(m_dir)
    saved: List[Path] = []
    saved.append(plot_cv_macro_f1(df, out_dir))
    saved.append(plot_student_test_top1_and_f1(df, out_dir))
    saved.append(plot_log_loss(df, out_dir))
    for fn in (
        plot_iterative_learning_curves,
        plot_optuna_history,
        plot_untuned_vs_tuned,
        plot_tuned_confusion_matrix,
    ):
        p = fn(m_dir, out_dir)
        if p is not None:
            saved.append(p)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all compulsory PathSnap report plots.")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--out-dir", default="PLOT")
    args = parser.parse_args()
    generate_all_report_plots(args.metrics_dir, args.out_dir)
