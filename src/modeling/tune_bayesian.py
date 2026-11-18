"""Step 5 (Phase 2): Bayesian Hyperparameter Optimisation (Optuna TPE) for XGBoost
and final deployment artifact export for Member 3 (Laptop 1 FastAPI + HMM).

Strict Leakage Rules:
  - Optuna optimizes 5-fold GroupKFold CV (grouped strictly by trip_id) ONLY on the
    80 training trips.
  - Preprocessing (SimpleImputer(median)) is fit strictly inside each training fold.
  - student_test (10 trips) is NEVER used inside the Optuna objective; it is evaluated
    only once after refitting the tuned model on all 80 training trips.
  - evaluator_test (10 trips) is completely untouched.

Outputs written:
  - results/metrics/optuna_trials.csv
  - results/metrics/bayesian_tuning_summary.json
  - results/metrics/untuned_vs_tuned.csv
  - results/metrics/cm_XGBoost_tuned.csv
  - results/metrics/history_XGBoost_tuned.csv
  - results/student_test_probs_tuned.parquet  (for Member 3 HMM/Viterbi)
  - artifacts/final_model.joblib              (CPU-portable Pipeline for Laptop 1)
  - artifacts/model_metadata.json             (feature_cols, params, metrics)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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

# Search space documented explicitly for Deliverables.pdf Section 4
SEARCH_SPACE_SPEC: Dict[str, Dict[str, Any]] = {
    "scale_pos_weight": {
        "type": "float",
        "low": 1.0,
        "high": 6.0,
        "log": False,
        "rationale": "Untuned full ratio (~14.2) over-shifted probabilities upward; tuning [1.0, 6.0] balances recall with calibrated precision & log loss for HMM emissions.",
    },
    "max_depth": {
        "type": "int",
        "low": 4,
        "high": 10,
        "log": False,
        "rationale": "Controls interaction depth across geometric/kinematic features (distance x heading x road length).",
    },
    "learning_rate": {
        "type": "float",
        "low": 0.02,
        "high": 0.25,
        "log": True,
        "rationale": "Step-size shrinkage controlling convergence speed and generalisation.",
    },
    "n_estimators": {
        "type": "int",
        "low": 150,
        "high": 500,
        "log": False,
        "rationale": "Untuned XGBoost was still improving at round 200; expanding up to 500 captures remaining signal.",
    },
    "min_child_weight": {
        "type": "int",
        "low": 1,
        "high": 20,
        "log": False,
        "rationale": "Minimum Hessian weight per leaf to prevent overfitting noisy GPS outliers.",
    },
    "subsample": {
        "type": "float",
        "low": 0.6,
        "high": 1.0,
        "log": False,
        "rationale": "Row subsampling per tree for variance reduction.",
    },
    "colsample_bytree": {
        "type": "float",
        "low": 0.6,
        "high": 1.0,
        "log": False,
        "rationale": "Feature subsampling per tree across the 12 geometric & kinematic predictors.",
    },
    "reg_alpha": {
        "type": "float",
        "low": 1e-4,
        "high": 10.0,
        "log": True,
        "rationale": "L1 leaf weight regularisation.",
    },
    "reg_lambda": {
        "type": "float",
        "low": 1e-3,
        "high": 10.0,
        "log": True,
        "rationale": "L2 leaf weight regularisation for smoother, better-calibrated log-odds.",
    },
}


def _detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        X_tmp = np.random.default_rng(0).standard_normal((128, 4), dtype=np.float32)
        y_tmp = np.array([0, 1] * 64, dtype=np.int8)
        clf = XGBClassifier(
            n_estimators=2, tree_method="hist", device="cuda", eval_metric="logloss"
        )
        clf.fit(X_tmp, y_tmp)
        return "cuda"
    except Exception:
        return "cpu"


def _make_xgb(params: Dict[str, Any], device: str, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=int(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        scale_pos_weight=float(params["scale_pos_weight"]),
        tree_method="hist",
        device=device,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )


def _precompute_leakage_free_folds(
    X: pd.DataFrame,
    y: np.ndarray,
    trip_ids: np.ndarray,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute GroupKFold splits with median imputation fit STRICTLY per training fold."""
    gkf = GroupKFold(n_splits=n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in gkf.split(X, y, trip_ids):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X.iloc[tr_idx]).astype(np.float32)
        X_va = imp.transform(X.iloc[va_idx]).astype(np.float32)
        folds.append((X_tr, y[tr_idx], X_va, y[va_idx]))
    return folds


def _record_tuned_history(
    train_ds: CandidateDataset,
    best_params: Dict[str, Any],
    device: str,
    out_csv: Path,
    seed: int,
) -> pd.DataFrame:
    """Capture genuine round-by-round train/val log-loss curve for tuned XGBoost."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(
        gss.split(train_ds.X, train_ds.y, train_ds.meta["trip_id"].to_numpy())
    )
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(train_ds.X.iloc[tr_idx]).astype(np.float32)
    y_tr = train_ds.y[tr_idx]
    X_va = imp.transform(train_ds.X.iloc[va_idx]).astype(np.float32)
    y_va = train_ds.y[va_idx]

    clf = _make_xgb(best_params, device=device, seed=seed)
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


def run_bayesian_tuning(
    train_path: str | Path = "data_geo/outputs/data/train_candidates.parquet",
    student_test_path: str | Path = "data_geo/outputs/data/student_test_candidates.parquet",
    metadata_path: Optional[str | Path] = "data_geo/outputs/data/metadata.json",
    raw_data_dir: Optional[str | Path] = "Student folder",
    output_dir: str | Path = "results/metrics",
    artifacts_dir: str | Path = "artifacts",
    n_trials: int = 30,
    n_splits: int = 5,
    device: str = "auto",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    art_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _detect_device(device)
    print("=" * 78)
    print(f"BAYESIAN HYPERPARAMETER OPTIMISATION — XGBoost (Optuna TPE)")
    print(f"  Device: {resolved_device} | Trials: {n_trials} | CV: {n_splits}-fold GroupKFold(trip_id) | Seed: {seed}")
    print("=" * 78)

    train_ds, test_ds = load_train_and_student_test(
        train_path=train_path,
        student_test_path=student_test_path,
        metadata_path=metadata_path,
    )

    print(f"\nPrecomputing {n_splits} leakage-free GroupKFold splits (in-fold median imputation)...")
    t_prep = time.perf_counter()
    folds = _precompute_leakage_free_folds(
        train_ds.X, train_ds.y, train_ds.meta["trip_id"].to_numpy(), n_splits=n_splits
    )
    print(f"Folds ready in {time.perf_counter() - t_prep:.2f}s.\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name="pathsnap_xgboost_bayesian",
        direction="maximize",
        sampler=sampler,
    )

    t_opt_start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 6.0),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        fold_f1s: List[float] = []
        fold_lls: List[float] = []
        t_trial = time.perf_counter()

        for X_tr, y_tr, X_va, y_va in folds:
            clf = _make_xgb(params, device=resolved_device, seed=seed)
            clf.fit(X_tr, y_tr)
            probs = np.clip(clf.predict_proba(X_va)[:, 1], 1e-15, 1.0 - 1e-15)
            preds = (probs >= 0.5).astype(np.int8)
            fold_f1s.append(float(f1_score(y_va, preds, average="macro", zero_division=0)))
            fold_lls.append(float(log_loss(y_va, probs, labels=[0, 1])))

        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        mean_ll = float(np.mean(fold_lls))
        dur = time.perf_counter() - t_trial

        trial.set_user_attr("cv_macro_f1_std", std_f1)
        trial.set_user_attr("cv_log_loss_mean", mean_ll)
        trial.set_user_attr("trial_duration_sec", dur)

        best_so_far = max(mean_f1, study.best_value) if trial.number > 0 else mean_f1
        elapsed = time.perf_counter() - t_opt_start
        print(
            f"  [Trial {trial.number + 1:2d}/{n_trials}] "
            f"CV Macro F1={mean_f1:.5f}±{std_f1:.4f} | CV LogLoss={mean_ll:.5f} | "
            f"Best={best_so_far:.5f} | {dur:.1f}s (elapsed {elapsed:.0f}s) | "
            f"spw={params['scale_pos_weight']:.2f}, d={params['max_depth']}, "
            f"lr={params['learning_rate']:.3f}, n={params['n_estimators']}"
        )
        return mean_f1

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_wall_sec = time.perf_counter() - t_opt_start

    best_params = dict(study.best_params)
    print("\n" + "=" * 78)
    print(f"OPTUNA COMPLETE in {opt_wall_sec:.1f}s ({opt_wall_sec / 60.0:.2f} min)")
    print(f"  Best Trial #{study.best_trial.number + 1} | CV Macro F1 = {study.best_value:.5f}")
    print(f"  Best Hyperparameters: {json.dumps(best_params, indent=2)}")
    print("=" * 78)

    # Save full Optuna trial log
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)

    # Full unified evaluation of tuned model (5-fold CV + refit on all 80 train trips -> student_test)
    print("\nRunning full unified CV + refit + student_test evaluation on best hyperparameters...")
    tuned_cv = cross_validate_model(
        estimator_factory=lambda _y_tr: _make_xgb(best_params, device=resolved_device, seed=seed),
        X=train_ds.X,
        y=train_ds.y,
        meta=train_ds.meta,
        n_splits=n_splits,
        scale=False,
        raw_data_dir=raw_data_dir,
    )

    # Refit final model on all 80 training trips
    final_pipe = build_pipeline(
        _make_xgb(best_params, device=resolved_device, seed=seed), scale=False
    )
    t0 = time.perf_counter()
    final_pipe.fit(train_ds.X, train_ds.y)
    tuned_fit_sec = time.perf_counter() - t0

    # Switch to CPU for portable FastAPI inference on Laptop 1 and measure CPU latency honestly
    final_pipe.named_steps["model"].set_params(device="cpu", n_jobs=-1)

    train_prob = get_positive_proba(final_pipe, train_ds.X)
    train_metrics = compute_candidate_metrics(train_ds.y, train_prob)

    t1 = time.perf_counter()
    test_prob = get_positive_proba(final_pipe, test_ds.X)
    tuned_infer_sec = time.perf_counter() - t1

    test_metrics, _ = evaluate_predictions(
        meta=test_ds.meta,
        y_true=test_ds.y,
        y_prob=test_prob,
        total_gps_observations=test_ds.total_gps_observations,
        candidate_miss_rate=test_ds.candidate_miss_rate,
        raw_data_dir=raw_data_dir,
        inference_time_sec=tuned_infer_sec,
    )

    # Record tuned XGBoost round-by-round learning curve
    _record_tuned_history(
        train_ds, best_params, device=resolved_device, out_csv=out_dir / "history_XGBoost_tuned.csv", seed=seed
    )

    # Save tuned confusion matrix CSV
    pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(out_dir / "cm_XGBoost_tuned.csv")

    # Save tuned student_test candidate probabilities for Member 3 (HMM/Viterbi)
    prob_df = test_ds.meta[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    if "from_node" in test_ds.meta.columns and "to_node" in test_ds.meta.columns:
        prob_df["from_node"] = test_ds.meta["from_node"].to_numpy()
        prob_df["to_node"] = test_ds.meta["to_node"].to_numpy()
    prob_df["prob"] = test_prob
    prob_df.to_parquet(out_dir / "student_test_probs_XGBoost_tuned.parquet", index=False)
    prob_df.to_parquet(out_dir.parent / "student_test_probs_tuned.parquet", index=False)

    # Build Untuned vs Tuned comparison table
    tuned_row = {
        "Version": "XGBoost (Bayesian Tuned)",
        "Train Macro F1": round(train_metrics["macro_f1"], 5),
        "CV Macro F1-score": round(float(tuned_cv["val_macro_f1_mean"]), 5),
        "CV Macro F1 Std": round(float(tuned_cv["val_macro_f1_std"]), 5),
        "Student-Test Top-1 Accuracy": round(float(test_metrics["top1_accuracy"]), 5),
        "Top-1 Accuracy (Covered)": round(float(test_metrics["top1_accuracy_covered"]), 5),
        "Student-Test Macro F1-score": round(test_metrics["macro_f1"], 5),
        "Precision": round(test_metrics["precision"], 5),
        "Recall": round(test_metrics["recall"], 5),
        "Log Loss": round(test_metrics["log_loss"], 5),
        "Route Continuity": round(float(test_metrics["route_continuity"]), 5),
        "Candidate Miss Rate": round(float(test_metrics["candidate_miss_rate"]), 5),
        "Training Time (s)": round(tuned_fit_sec, 4),
        "Inference Time (s)": round(tuned_infer_sec, 4),
        "Inference Time / Trip (ms)": round(float(test_metrics["inference_time_per_trip_ms"]), 2),
    }

    rows: List[Dict[str, Any]] = []
    comp_path = out_dir / "model_comparison.csv"
    untuned_row: Optional[Dict[str, Any]] = None
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        xgb_match = comp_df[comp_df["Model"] == "XGBoost"]
        if not xgb_match.empty:
            r = xgb_match.iloc[0]
            untuned_row = {
                "Version": "XGBoost (Untuned Baseline)",
                "Train Macro F1": float(r["Train Macro F1"]),
                "CV Macro F1-score": float(r["CV Macro F1-score"]),
                "CV Macro F1 Std": float(r["CV Macro F1 Std"]),
                "Student-Test Top-1 Accuracy": float(r["Top-1 Accuracy"]),
                "Top-1 Accuracy (Covered)": float(r["Top-1 Accuracy (Covered)"]),
                "Student-Test Macro F1-score": float(r["Student-Test Macro F1-score"]),
                "Precision": float(r["Precision"]),
                "Recall": float(r["Recall"]),
                "Log Loss": float(r["Log Loss"]),
                "Route Continuity": float(r["Route Continuity"]),
                "Candidate Miss Rate": float(r["Candidate Miss Rate"]),
                "Training Time (s)": float(r["Training Time (s)"]),
                "Inference Time (s)": float(r["Inference Time (s)"]),
                "Inference Time / Trip (ms)": float(r["Inference Time / Trip (ms)"]),
            }
            rows.append(untuned_row)

    rows.append(tuned_row)

    if untuned_row is not None:
        delta_row: Dict[str, Any] = {"Version": "Delta (Tuned - Untuned)"}
        for k in tuned_row:
            if k == "Version":
                continue
            delta_row[k] = round(float(tuned_row[k]) - float(untuned_row[k]), 5)
        rows.append(delta_row)

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out_dir / "untuned_vs_tuned.csv", index=False)

    # Export final deployment bundle for Member 3 (Laptop 1 FastAPI)
    joblib.dump(final_pipe, art_dir / "final_model.joblib")
    model_metadata = {
        "model_name": "XGBoost (Bayesian Tuned)",
        "algorithm_family": "Boosting",
        "tuning_method": "Bayesian Optimisation (Optuna TPESampler)",
        "objective_function": f"Maximize {n_splits}-fold GroupKFold (trip_id) CV Macro F1-score",
        "n_trials": n_trials,
        "random_seed": seed,
        "training_device": resolved_device,
        "inference_device": "cpu",
        "preprocessing": "SimpleImputer(strategy='median') inside sklearn Pipeline",
        "feature_cols": train_ds.feature_cols,
        "best_hyperparameters": best_params,
        "metrics_tuned": tuned_row,
        "confusion_matrix_student_test": test_metrics["confusion_matrix"],
        "usage": "pipe = joblib.load('artifacts/final_model.joblib'); probs = pipe.predict_proba(df[metadata['feature_cols']])[:, 1]",
    }
    with (art_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # Save structured Section 4 summary JSON
    summary = {
        "model": "XGBoost",
        "sampler": "Optuna TPESampler (Tree-structured Parzen Estimator)",
        "cv_strategy": f"{n_splits}-fold GroupKFold grouped by trip_id on 80 training trips",
        "objective_function": "Maximize mean 5-fold GroupKFold CV Macro F1-score",
        "n_trials": n_trials,
        "random_seed": seed,
        "device_used": resolved_device,
        "search_space": SEARCH_SPACE_SPEC,
        "optimisation_wall_time_sec": round(opt_wall_sec, 2),
        "optimisation_wall_time_min": round(opt_wall_sec / 60.0, 2),
        "best_trial_number": int(study.best_trial.number + 1),
        "best_cv_macro_f1": round(float(study.best_value), 5),
        "best_hyperparameters": best_params,
        "untuned_metrics": untuned_row,
        "tuned_metrics": tuned_row,
        "tuned_cv_full": tuned_cv,
        "tuned_student_test_confusion_matrix": test_metrics["confusion_matrix"],
    }
    with (out_dir / "bayesian_tuning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("UNTUNED VS TUNED COMPARISON (XGBoost)")
    print(
        cmp_df[
            [
                "Version",
                "CV Macro F1-score",
                "Student-Test Top-1 Accuracy",
                "Student-Test Macro F1-score",
                "Log Loss",
                "Route Continuity",
                "Inference Time / Trip (ms)",
            ]
        ].to_string(index=False)
    )
    print("=" * 78)
    print(f"Exported final deployment model -> {art_dir / 'final_model.joblib'}")
    print(f"Exported model metadata        -> {art_dir / 'model_metadata.json'}")
    print(f"Exported Member 3 tuned probs  -> {out_dir.parent / 'student_test_probs_tuned.parquet'}")
    return cmp_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bayesian Hyperparameter Optimisation (Optuna TPE) on XGBoost."
    )
    parser.add_argument("--train", default="data_geo/outputs/data/train_candidates.parquet")
    parser.add_argument("--test", default="data_geo/outputs/data/student_test_candidates.parquet")
    parser.add_argument("--metadata", default="data_geo/outputs/data/metadata.json")
    parser.add_argument("--raw-dir", default="Student folder")
    parser.add_argument("--out-dir", default="results/metrics")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna Bayesian trials")
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="XGBoost training device ('auto' uses CUDA GPU if available; final export is always CPU-portable)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    meta_file = Path(args.metadata) if args.metadata and Path(args.metadata).exists() else None
    run_bayesian_tuning(
        train_path=args.train,
        student_test_path=args.test,
        metadata_path=meta_file,
        raw_data_dir=args.raw_dir,
        output_dir=args.out_dir,
        artifacts_dir=args.artifacts_dir,
        n_trials=args.trials,
        n_splits=args.folds,
        device=args.device,
        seed=args.seed,
    )
