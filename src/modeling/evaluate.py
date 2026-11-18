"""Unified leakage-free evaluation engine for PathSnap candidate classifiers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def build_pipeline(estimator: Any, scale: bool = False) -> Pipeline:
    """Wrap an estimator in a leakage-free preprocessing Pipeline fit strictly per fold."""
    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def get_positive_proba(fitted_estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Extract P(label=1) cleanly from any classifier (or sigmoid of decision_function)."""
    if hasattr(fitted_estimator, "predict_proba"):
        probs = fitted_estimator.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.asarray(probs[:, 1], dtype=np.float64)
        return np.asarray(probs.ravel(), dtype=np.float64)
    if hasattr(fitted_estimator, "decision_function"):
        scores = np.asarray(fitted_estimator.decision_function(X), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
    raise AttributeError(
        f"Estimator {type(fitted_estimator).__name__} exposes neither predict_proba nor decision_function."
    )


def compute_candidate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute candidate-level classification metrics and confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob_clipped = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    y_pred = (y_prob_clipped >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob_clipped, labels=[0, 1])),
        "confusion_matrix": cm.tolist(),
    }


def _attach_arc_endpoints_from_raw(
    top1_df: pd.DataFrame,
    raw_data_dir: Optional[Path],
) -> pd.DataFrame:
    """Fallback: load from_node/to_node from Student folder .arcs files if missing in meta."""
    if raw_data_dir is None or not raw_data_dir.exists():
        return top1_df

    df = top1_df.copy()
    from_nodes = np.full(len(df), -1, dtype=np.int64)
    to_nodes = np.full(len(df), -1, dtype=np.int64)

    for trip_id, idx in df.groupby("trip_id").groups.items():
        arc_path = None
        for sub in ("train", "student_test", "evaluator_test", ""):
            candidate = raw_data_dir / sub / str(trip_id) / f"{trip_id}.arcs"
            if candidate.exists():
                arc_path = candidate
                break
        if arc_path is None:
            continue
        arcs = np.loadtxt(arc_path, dtype=np.int64)
        if arcs.ndim == 1:
            arcs = arcs.reshape(1, -1)
        pos = df.index.get_indexer(idx)
        arc_ids = df.loc[idx, "arc_id"].to_numpy(dtype=np.int64)
        valid = (arc_ids >= 0) & (arc_ids < len(arcs))
        from_nodes[pos[valid]] = arcs[arc_ids[valid], 0]
        to_nodes[pos[valid]] = arcs[arc_ids[valid], 1]

    if (from_nodes >= 0).any():
        df["from_node"] = from_nodes
        df["to_node"] = to_nodes
    return df


def compute_observation_metrics(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    total_gps_observations: Optional[int] = None,
    raw_data_dir: Optional[str | Path] = None,
) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:
    """Compute Observation-level Top-1 Accuracy, Candidate Miss Rate, and Route Continuity."""
    work = meta[["trip_id", "obs_idx", "arc_id"]].copy()
    if "from_node" in meta.columns and "to_node" in meta.columns:
        work["from_node"] = meta["from_node"].to_numpy()
        work["to_node"] = meta["to_node"].to_numpy()

    work["label"] = np.asarray(y_true, dtype=int)
    work["prob"] = np.asarray(y_prob, dtype=np.float64)

    # Pick highest-probability candidate per (trip_id, obs_idx) deterministically
    work_sorted = work.sort_values(
        by=["trip_id", "obs_idx", "prob", "arc_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top1_df = work_sorted.drop_duplicates(
        subset=["trip_id", "obs_idx"], keep="first"
    ).reset_index(drop=True)

    obs_pos_sum = work.groupby(["trip_id", "obs_idx"])["label"].sum()
    n_obs_in_table = int(len(obs_pos_sum))
    n_obs_with_pos = int((obs_pos_sum > 0).sum())
    n_total_obs = int(total_gps_observations) if total_gps_observations else n_obs_in_table

    n_top1_correct = int(top1_df["label"].sum())
    top1_covered = float(n_top1_correct / n_obs_in_table) if n_obs_in_table > 0 else 0.0
    top1_overall = float(n_top1_correct / n_total_obs) if n_total_obs > 0 else 0.0
    miss_rate = (
        float((n_total_obs - n_obs_with_pos) / n_total_obs) if n_total_obs > 0 else 0.0
    )

    # Route continuity (classifier-only)
    if "from_node" not in top1_df.columns or "to_node" not in top1_df.columns:
        top1_df = _attach_arc_endpoints_from_raw(
            top1_df, Path(raw_data_dir) if raw_data_dir else None
        )

    route_continuity: Optional[float] = None
    if "from_node" in top1_df.columns and "to_node" in top1_df.columns:
        continuous = 0
        total_transitions = 0
        for _, grp in top1_df.groupby("trip_id", sort=False):
            grp = grp.sort_values("obs_idx")
            if len(grp) < 2:
                continue
            arcs = grp["arc_id"].to_numpy()
            from_n = grp["from_node"].to_numpy()
            to_n = grp["to_node"].to_numpy()
            same_arc = arcs[:-1] == arcs[1:]
            direct_edge = (to_n[:-1] >= 0) & (to_n[:-1] == from_n[1:])
            continuous += int(np.sum(same_arc | direct_edge))
            total_transitions += int(len(grp) - 1)
        route_continuity = (
            float(continuous / total_transitions) if total_transitions > 0 else 0.0
        )

    metrics: Dict[str, Optional[float]] = {
        "top1_accuracy": top1_overall,
        "top1_accuracy_covered": top1_covered,
        "candidate_miss_rate": miss_rate,
        "route_continuity": route_continuity,
    }
    return metrics, top1_df


def evaluate_predictions(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    total_gps_observations: Optional[int] = None,
    raw_data_dir: Optional[str | Path] = None,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run both candidate-level and observation/sequence-level evaluation."""
    cand_metrics = compute_candidate_metrics(y_true, y_prob, threshold=threshold)
    obs_metrics, top1_df = compute_observation_metrics(
        meta=meta,
        y_true=y_true,
        y_prob=y_prob,
        total_gps_observations=total_gps_observations,
        raw_data_dir=raw_data_dir,
    )
    return {**cand_metrics, **obs_metrics}, top1_df


def cross_validate_model(
    estimator_factory: Callable[[np.ndarray], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    n_splits: int = 5,
    scale: bool = False,
    raw_data_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run trip-grouped GroupKFold CV on permitted training trips only.

    `estimator_factory(y_train_fold)` receives the fold's training labels so models
    like XGBoost can compute `scale_pos_weight` strictly from the training fold.
    """
    groups = meta["trip_id"].to_numpy()
    n_unique_trips = int(pd.Series(groups).nunique())
    actual_splits = min(n_splits, n_unique_trips)
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 unique trips for GroupKFold, got {n_unique_trips}.")

    gkf = GroupKFold(n_splits=actual_splits)
    fold_records: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        meta_va = meta.iloc[va_idx]

        estimator = estimator_factory(y_tr)
        pipe = build_pipeline(estimator, scale=scale)

        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        va_prob = get_positive_proba(pipe, X_va)
        infer_sec = time.perf_counter() - t1

        tr_prob = get_positive_proba(pipe, X_tr)
        tr_metrics = compute_candidate_metrics(y_tr, tr_prob)
        va_metrics, _ = evaluate_predictions(
            meta=meta_va,
            y_true=y_va,
            y_prob=va_prob,
            raw_data_dir=raw_data_dir,
        )

        fold_records.append(
            {
                "fold": fold_idx,
                "train_macro_f1": tr_metrics["macro_f1"],
                "val_macro_f1": va_metrics["macro_f1"],
                "val_accuracy": va_metrics["accuracy"],
                "val_precision": va_metrics["precision"],
                "val_recall": va_metrics["recall"],
                "val_log_loss": va_metrics["log_loss"],
                "val_top1_accuracy": va_metrics["top1_accuracy"],
                "val_route_continuity": va_metrics["route_continuity"],
                "train_time_sec": train_sec,
                "inference_time_sec": infer_sec,
            }
        )

    def _agg(key: str) -> Tuple[Optional[float], Optional[float]]:
        vals = [r[key] for r in fold_records if r[key] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    summary: Dict[str, Any] = {"n_splits": actual_splits, "folds": fold_records}
    for metric_key in (
        "train_macro_f1",
        "val_macro_f1",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_log_loss",
        "val_top1_accuracy",
        "val_route_continuity",
        "train_time_sec",
        "inference_time_sec",
    ):
        mean_v, std_v = _agg(metric_key)
        summary[f"{metric_key}_mean"] = mean_v
        summary[f"{metric_key}_std"] = std_v

    return summary
