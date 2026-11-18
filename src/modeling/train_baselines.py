"""Step 3: Train, cross-validate, evaluate, and log 6 diverse baseline models for PathSnap.

Models across 5 distinct algorithm families:
  1. LogisticRegression       (Linear / Probabilistic)
  2. DecisionTree             (Tree-based)
  3. RandomForest             (Ensemble - Bagging)
  4. ExtraTrees               (Ensemble - Randomized Trees)
  5. XGBoost                  (Boosting) — logs real round-by-round train/val log loss
  6. MLP                      (Neural Network) — logs real epoch-by-epoch train/val log loss

Outputs written to results/metrics/:
  - model_comparison.csv
  - model_configs.json
  - confusion_matrices.json
  - experiment_summary.json
  - history_XGBoost.csv
  - history_MLP.csv
  - student_test_probs_<ModelName>.parquet  (ready for Member 3's HMM/Viterbi)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.modeling.data_loader import CandidateDataset, load_train_and_student_test
from src.modeling.evaluate import (
    RANDOM_SEED,
    build_pipeline,
    compute_candidate_metrics,
    cross_validate_model,
    evaluate_predictions,
    get_positive_proba,
)


@dataclass
class ModelSpec:
    name: str
    family: str
    key_config_str: str
    loss_objective: str
    scale: bool
    hyperparams: Dict[str, Any]
    factory: Callable[[np.ndarray], Any]


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return float(n_neg / max(n_pos, 1))


def get_baseline_model_specs(seed: int = RANDOM_SEED) -> List[ModelSpec]:
    """Define 6 diverse models spanning 5 algorithm families."""
    return [
        ModelSpec(
            name="LogisticRegression",
            family="Linear / Probabilistic",
            key_config_str="C=1.0, lbfgs, class_weight=balanced, StandardScaler",
            loss_objective="L2-regularised Binary Cross-Entropy (Log Loss)",
            scale=True,
            hyperparams={
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": seed,
            },
            factory=lambda y_tr: LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="DecisionTree",
            family="Tree-based",
            key_config_str="max_depth=12, min_samples_leaf=20, class_weight=balanced",
            loss_objective="Gini Impurity",
            scale=False,
            hyperparams={
                "criterion": "gini",
                "max_depth": 12,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
                "random_state": seed,
            },
            factory=lambda y_tr: DecisionTreeClassifier(
                criterion="gini",
                max_depth=12,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="RandomForest",
            family="Ensemble (Bagging)",
            key_config_str="n_estimators=200, max_depth=16, min_samples_leaf=5",
            loss_objective="Gini Impurity (Bagged Ensemble)",
            scale=False,
            hyperparams={
                "n_estimators": 200,
                "max_depth": 16,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": seed,
            },
            factory=lambda y_tr: RandomForestClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="ExtraTrees",
            family="Ensemble (Randomized Trees)",
            key_config_str="n_estimators=200, max_depth=18, min_samples_leaf=5",
            loss_objective="Gini Impurity (Extremely Randomized Splits)",
            scale=False,
            hyperparams={
                "n_estimators": 200,
                "max_depth": 18,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": seed,
            },
            factory=lambda y_tr: ExtraTreesClassifier(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="XGBoost",
            family="Boosting",
            key_config_str="n_estimators=300, max_depth=6, lr=0.1, subsample=0.8, hist",
            loss_objective="Binary Logistic Loss (binary:logistic / logloss)",
            scale=False,
            hyperparams={
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "n_jobs": -1,
                "random_state": seed,
            },
            factory=lambda y_tr: XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                scale_pos_weight=_compute_scale_pos_weight(y_tr),
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="MLP",
            family="Neural Network",
            key_config_str="hidden=(128,64), relu, alpha=1e-3, lr=1e-3, max_iter=25",
            loss_objective="L2-regularised Binary Cross-Entropy (Adam optimiser)",
            scale=True,
            hyperparams={
                "hidden_layer_sizes": [128, 64],
                "activation": "relu",
                "solver": "adam",
                "alpha": 1e-3,
                "batch_size": 256,
                "learning_rate_init": 1e-3,
                "max_iter": 25,
                "early_stopping": False,
                "random_state": seed,
            },
            factory=lambda y_tr: MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=25,
                early_stopping=False,
                random_state=seed,
            ),
        ),
    ]


def _trip_grouped_val_split(
    train_ds: CandidateDataset,
    val_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a single trip-grouped 80/20 split inside train for genuine learning curves."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(
        gss.split(train_ds.X, train_ds.y, train_ds.meta["trip_id"].to_numpy())
    )
    return tr_idx, va_idx


def record_xgboost_history(
    train_ds: CandidateDataset,
    out_csv: Path,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Capture genuine boosting-round train vs validation log-loss curve for XGBoost."""
    tr_idx, va_idx = _trip_grouped_val_split(train_ds, val_size=0.2, seed=seed)
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train_ds.X.iloc[tr_idx])
    y_tr = train_ds.y[tr_idx]
    X_va = imputer.transform(train_ds.X.iloc[va_idx])
    y_va = train_ds.y[va_idx]

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=_compute_scale_pos_weight(y_tr),
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)
    res = clf.evals_result()
    hist_df = pd.DataFrame(
        {
            "round": np.arange(1, len(res["validation_0"]["logloss"]) + 1),
            "train_log_loss": res["validation_0"]["logloss"],
            "val_log_loss": res["validation_1"]["logloss"],
        }
    )
    hist_df.to_csv(out_csv, index=False)
    return hist_df


def record_mlp_history(
    train_ds: CandidateDataset,
    out_csv: Path,
    epochs: int = 25,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Capture genuine epoch-wise train vs validation log-loss curve for MLP."""
    tr_idx, va_idx = _trip_grouped_val_split(train_ds, val_size=0.2, seed=seed)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr = scaler.fit_transform(imputer.fit_transform(train_ds.X.iloc[tr_idx]))
    y_tr = train_ds.y[tr_idx]
    X_va = scaler.transform(imputer.transform(train_ds.X.iloc[va_idx]))
    y_va = train_ds.y[va_idx]

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=min(256, max(16, len(y_tr) // 4)),
        learning_rate_init=1e-3,
        max_iter=1,
        warm_start=True,
        random_state=seed,
    )

    rows: List[Dict[str, float]] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            mlp.fit(X_tr, y_tr)
            tr_m = compute_candidate_metrics(y_tr, get_positive_proba(mlp, X_tr))
            va_m = compute_candidate_metrics(y_va, get_positive_proba(mlp, X_va))
            rows.append(
                {
                    "epoch": epoch,
                    "train_log_loss": tr_m["log_loss"],
                    "val_log_loss": va_m["log_loss"],
                    "train_macro_f1": tr_m["macro_f1"],
                    "val_macro_f1": va_m["macro_f1"],
                }
            )

    hist_df = pd.DataFrame(rows)
    hist_df.to_csv(out_csv, index=False)
    return hist_df


def run_baseline_benchmark(
    train_path: str | Path,
    student_test_path: str | Path,
    stats_path: Optional[str | Path] = None,
    raw_data_dir: Optional[str | Path] = "Student folder",
    output_dir: str | Path = "results/metrics",
    artifacts_dir: str | Path = "artifacts",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Train, cross-validate, evaluate, and log all 6 baseline models."""
    out_dir = Path(output_dir)
    art_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = load_train_and_student_test(
        train_path=train_path,
        student_test_path=student_test_path,
        stats_path=stats_path,
    )

    specs = get_baseline_model_specs(seed=seed)
    comparison_rows: List[Dict[str, Any]] = []
    configs_log: Dict[str, Any] = {
        "random_seed": seed,
        "cv_strategy": f"GroupKFold(n_splits={n_splits}, group=trip_id)",
        "train_trips": train_ds.n_trips,
        "student_test_trips": test_ds.n_trips,
        "feature_cols": train_ds.feature_cols,
        "models": {},
    }
    confusion_matrices: Dict[str, Any] = {}
    full_summary: Dict[str, Any] = {
        "random_seed": seed,
        "feature_cols": train_ds.feature_cols,
        "candidate_miss_rate_train": train_ds.candidate_miss_rate,
        "candidate_miss_rate_student_test": test_ds.candidate_miss_rate,
        "models": {},
    }

    best_model_name: Optional[str] = None
    best_cv_f1 = -1.0
    best_fitted_pipe: Optional[Any] = None

    for spec in specs:
        print(f"\n>>> Training & Evaluating: {spec.name} ({spec.family})")

        # 1. Trip-grouped CV on the 80 training trips
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cv_res = cross_validate_model(
                estimator_factory=spec.factory,
                X=train_ds.X,
                y=train_ds.y,
                meta=train_ds.meta,
                n_splits=n_splits,
                scale=spec.scale,
                raw_data_dir=raw_data_dir,
            )

        # 2. Fit on all 80 training trips, evaluate on 10 student_test trips
        pipe = build_pipeline(spec.factory(train_ds.y), scale=spec.scale)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            t0 = time.perf_counter()
            pipe.fit(train_ds.X, train_ds.y)
            full_train_sec = time.perf_counter() - t0

        train_prob = get_positive_proba(pipe, train_ds.X)
        train_metrics = compute_candidate_metrics(train_ds.y, train_prob)

        t1 = time.perf_counter()
        test_prob = get_positive_proba(pipe, test_ds.X)
        test_infer_sec = time.perf_counter() - t1

        test_metrics, _ = evaluate_predictions(
            meta=test_ds.meta,
            y_true=test_ds.y,
            y_prob=test_prob,
            total_gps_observations=test_ds.total_gps_observations,
            raw_data_dir=raw_data_dir,
        )

        # 3. Record real epoch/round learning curves ONLY for iterative models
        history_file: Optional[str] = None
        if spec.name == "XGBoost":
            hist_path = out_dir / "history_XGBoost.csv"
            record_xgboost_history(train_ds, hist_path, seed=seed)
            history_file = str(hist_path)
        elif spec.name == "MLP":
            hist_path = out_dir / "history_MLP.csv"
            record_mlp_history(train_ds, hist_path, epochs=25, seed=seed)
            history_file = str(hist_path)

        # 4. Save student_test candidate probabilities for Member 3 (HMM/Viterbi)
        prob_df = test_ds.meta[["trip_id", "obs_idx", "arc_id", "label"]].copy()
        if "from_node" in test_ds.meta.columns and "to_node" in test_ds.meta.columns:
            prob_df["from_node"] = test_ds.meta["from_node"].to_numpy()
            prob_df["to_node"] = test_ds.meta["to_node"].to_numpy()
        prob_df["prob"] = test_prob
        prob_df.to_parquet(out_dir / f"student_test_probs_{spec.name}.parquet", index=False)

        # 5. Track best baseline model by CV Macro F1
        cv_f1 = float(cv_res["val_macro_f1_mean"])
        if cv_f1 > best_cv_f1:
            best_cv_f1 = cv_f1
            best_model_name = spec.name
            best_fitted_pipe = pipe

        # 6. Collect logs aligned with Deliverables.pdf Section 3.1
        row = {
            "Model": spec.name,
            "Algorithm Family": spec.family,
            "Key Configuration": spec.key_config_str,
            "Train Macro F1": round(train_metrics["macro_f1"], 5),
            "CV Macro F1-score": round(cv_f1, 5),
            "CV Macro F1 Std": round(float(cv_res["val_macro_f1_std"]), 5),
            "Top-1 Accuracy": round(float(test_metrics["top1_accuracy"]), 5),
            "Top-1 Accuracy (Covered)": round(float(test_metrics["top1_accuracy_covered"]), 5),
            "Route Continuity": (
                round(float(test_metrics["route_continuity"]), 5)
                if test_metrics["route_continuity"] is not None
                else None
            ),
            "Student-Test Macro F1-score": round(test_metrics["macro_f1"], 5),
            "Student-Test Accuracy": round(test_metrics["accuracy"], 5),
            "Precision": round(test_metrics["precision"], 5),
            "Recall": round(test_metrics["recall"], 5),
            "Log Loss": round(test_metrics["log_loss"], 5),
            "Candidate Miss Rate": round(float(test_metrics["candidate_miss_rate"]), 5),
            "Training Time (s)": round(full_train_sec, 4),
            "Inference Time (s)": round(test_infer_sec, 4),
        }
        comparison_rows.append(row)

        configs_log["models"][spec.name] = {
            "family": spec.family,
            "loss_objective": spec.loss_objective,
            "preprocessing": (
                "SimpleImputer(median) + StandardScaler"
                if spec.scale
                else "SimpleImputer(median)"
            ),
            "hyperparameters": spec.hyperparams,
            "history_file": history_file,
        }
        confusion_matrices[spec.name] = {
            "labels": [0, 1],
            "matrix": test_metrics["confusion_matrix"],
        }
        full_summary["models"][spec.name] = {
            "cv": cv_res,
            "train_full": train_metrics,
            "student_test": test_metrics,
            "train_time_sec": full_train_sec,
            "inference_time_sec": test_infer_sec,
        }

        rc_str = (
            f"{test_metrics['route_continuity']:.4f}"
            if test_metrics["route_continuity"] is not None
            else "N/A"
        )
        print(
            f"    Train F1={train_metrics['macro_f1']:.4f} | "
            f"CV F1={cv_f1:.4f}±{cv_res['val_macro_f1_std']:.4f} | "
            f"Test F1={test_metrics['macro_f1']:.4f} | "
            f"Top-1={test_metrics['top1_accuracy']:.4f} | "
            f"LogLoss={test_metrics['log_loss']:.4f} | "
            f"Continuity={rc_str} | "
            f"Fit={full_train_sec:.2f}s | Infer={test_infer_sec:.3f}s"
        )

    comp_df = pd.DataFrame(comparison_rows).sort_values(
        by="CV Macro F1-score", ascending=False
    )
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)

    with (out_dir / "model_configs.json").open("w", encoding="utf-8") as f:
        json.dump(configs_log, f, indent=2)
    with (out_dir / "confusion_matrices.json").open("w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=2)
    with (out_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    # Save best untuned baseline artifact immediately so Member 3 can integrate early
    if best_fitted_pipe is not None and best_model_name is not None:
        joblib.dump(best_fitted_pipe, art_dir / "baseline_best_model.joblib")
        with (art_dir / "baseline_best_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": best_model_name,
                    "selection_criterion": "CV Macro F1-score (untuned baseline)",
                    "cv_macro_f1": best_cv_f1,
                    "feature_cols": train_ds.feature_cols,
                    "random_seed": seed,
                },
                f,
                indent=2,
            )

    print("\n" + "=" * 72)
    print(f"BENCHMARK COMPLETE — Best Baseline by CV Macro F1: {best_model_name} ({best_cv_f1:.4f})")
    print(f"Saved comparison table -> {out_dir / 'model_comparison.csv'}")
    print("=" * 72)
    return comp_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate 6 baseline models for PathSnap (Member 2)."
    )
    parser.add_argument(
        "--train",
        default="data/processed/train_candidates.parquet",
        help="Path to train candidate table (.parquet or .csv)",
    )
    parser.add_argument(
        "--test",
        default="data/processed/student_test_candidates.parquet",
        help="Path to student_test candidate table (.parquet or .csv)",
    )
    parser.add_argument(
        "--stats",
        default="data/processed/candidate_stats.json",
        help="Optional JSON with total_gps_observations and candidate_miss_rate per split",
    )
    parser.add_argument(
        "--raw-dir",
        default="Student folder",
        help="Path to raw Student folder (fallback for .arcs endpoints if not in candidate table)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/metrics",
        help="Directory for metrics CSV/JSON and student_test probability outputs",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for saved baseline model bundle",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Global random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats_file = Path(args.stats) if args.stats and Path(args.stats).exists() else None
    run_baseline_benchmark(
        train_path=args.train,
        student_test_path=args.test,
        stats_path=stats_file,
        raw_data_dir=args.raw_dir,
        output_dir=args.out_dir,
        artifacts_dir=args.artifacts_dir,
        n_splits=args.folds,
        seed=args.seed,
    )
