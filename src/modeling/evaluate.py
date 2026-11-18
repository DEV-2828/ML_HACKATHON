"""Step 2: Unified leakage-free evaluation engine for PathSnap candidate classifiers.

Every model is judged identically across:
  1. Leakage-Free Cross-Validation:
     - GroupKFold(n_splits=5) or StratifiedGroupKFold(n_splits=5) grouped strictly by trip_id.
     - Preprocessing (SimpleImputer + optional StandardScaler) lives inside an sklearn
       Pipeline fit strictly on each training fold.
  2. Candidate-Level Metrics (Train, CV mean±std, Student-Test):
     - Accuracy, Precision, Recall, Macro F1-score, Binary F1-score, Log Loss, Confusion Matrix.
     - Train-vs-Val Macro F1 gap logged explicitly for overfitting/underfitting evidence.
  3. Observation-Level Top-1 Accuracy & Candidate Miss Rate:
     - Picks highest-probability candidate argmax(prob) per (trip_id, obs_idx).
     - Reports BOTH `top1_accuracy_covered` (over label_available observations) and
       `top1_accuracy_overall` (over all GPS observations, counting candidate misses as wrong),
       alongside `candidate_miss_rate`.
  4. Route Continuity (Classifier-only, aligned with Member 3):
     - For consecutive observations (arc_t, arc_{t+1}) within each trip sorted by obs_idx:
       continuous if `arc_t == arc_{t+1}` OR `to_node(arc_t) == from_node(arc_{t+1})`
       (optionally undirected if `allow_undirected=True`).
     - Route Continuity = continuous_transitions / total_consecutive_transitions.
  5. Timing:
     - `train_time_sec`, `inference_time_sec`, `inference_time_per_trip_sec`,
       and `inference_time_per_trip_ms` (for Laptop 1 FastAPI latency budgeting).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
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
    y_true_arr = np.asarray(y_true, dtype=np.int8)
    y_prob_clipped = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    y_pred = (y_prob_clipped >= threshold).astype(np.int8)

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true_arr, y_pred, average="binary", zero_division=0)),
        "log_loss": float(log_loss(y_true_arr, y_prob_clipped, labels=[0, 1])),
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

    for trip_id, idx in df.groupby("trip_id", sort=False).groups.items():
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


def compute_route_continuity(
    top1_df: pd.DataFrame,
    allow_undirected: bool = False,
) -> Dict[str, Optional[float | int]]:
    """Vectorized Route Continuity over classifier Top-1 predicted arcs.

    Within each trip (ordered by obs_idx), consecutive transition t -> t+1 is continuous if:
      - arc_id[t] == arc_id[t+1] (vehicle remains on same road segment), OR
      - to_node[t] == from_node[t+1] (directed graph transition), OR
      - (when allow_undirected=True) any shared endpoint between arc_t and arc_{t+1}.
    """
    if "from_node" not in top1_df.columns or "to_node" not in top1_df.columns:
        return {
            "route_continuity": None,
            "continuous_transitions": None,
            "total_transitions": None,
        }

    ordered = top1_df.sort_values(["trip_id", "obs_idx"], kind="mergesort")
    if len(ordered) < 2:
        return {
            "route_continuity": 0.0,
            "continuous_transitions": 0,
            "total_transitions": 0,
        }

    trips = ordered["trip_id"].to_numpy()
    arcs = ordered["arc_id"].to_numpy()
    from_n = ordered["from_node"].to_numpy()
    to_n = ordered["to_node"].to_numpy()

    same_trip = trips[:-1] == trips[1:]
    total_transitions = int(np.sum(same_trip))
    if total_transitions == 0:
        return {
            "route_continuity": 0.0,
            "continuous_transitions": 0,
            "total_transitions": 0,
        }

    same_arc = arcs[:-1] == arcs[1:]
    direct_edge = (to_n[:-1] >= 0) & (to_n[:-1] == from_n[1:])
    if allow_undirected:
        shared_endpoint = (
            (to_n[:-1] >= 0)
            & (
                (to_n[:-1] == from_n[1:])
                | (to_n[:-1] == to_n[1:])
                | (from_n[:-1] == from_n[1:])
                | (from_n[:-1] == to_n[1:])
            )
        )
        is_continuous = same_trip & (same_arc | shared_endpoint)
    else:
        is_continuous = same_trip & (same_arc | direct_edge)

    continuous_transitions = int(np.sum(is_continuous))
    return {
        "route_continuity": float(continuous_transitions / total_transitions),
        "continuous_transitions": continuous_transitions,
        "total_transitions": total_transitions,
    }


def compute_observation_metrics(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    total_gps_observations: Optional[int] = None,
    candidate_miss_rate: Optional[float] = None,
    raw_data_dir: Optional[str | Path] = None,
    allow_undirected_continuity: bool = False,
) -> Tuple[Dict[str, Optional[float | int]], pd.DataFrame]:
    """Compute Observation-level Top-1 Accuracy, Candidate Miss Rate, and Route Continuity.

    Vectorized for speed on 300k–3M candidate rows.
    """
    cols = ["trip_id", "obs_idx", "arc_id"]
    if "from_node" in meta.columns and "to_node" in meta.columns:
        cols.extend(["from_node", "to_node"])
    work = meta[cols].copy()
    work["label"] = np.asarray(y_true, dtype=np.int8)
    work["prob"] = np.asarray(y_prob, dtype=np.float64)

    # Deterministic argmax(prob) per (trip_id, obs_idx)
    work_sorted = work.sort_values(
        by=["trip_id", "obs_idx", "prob", "arc_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top1_df = work_sorted.drop_duplicates(
        subset=["trip_id", "obs_idx"], keep="first"
    ).reset_index(drop=True)

    n_obs_covered = int(len(top1_df))
    n_top1_correct = int(top1_df["label"].sum())

    if total_gps_observations is not None and total_gps_observations > 0:
        n_total_obs = int(total_gps_observations)
    else:
        n_total_obs = n_obs_covered

    top1_covered = float(n_top1_correct / n_obs_covered) if n_obs_covered > 0 else 0.0
    top1_overall = float(n_top1_correct / n_total_obs) if n_total_obs > 0 else 0.0

    if candidate_miss_rate is not None:
        miss_rate = float(candidate_miss_rate)
    else:
        obs_pos_sum = work.groupby(["trip_id", "obs_idx"], sort=False)["label"].sum()
        n_obs_with_pos = int((obs_pos_sum > 0).sum())
        miss_rate = (
            float(max(0, n_total_obs - n_obs_with_pos) / n_total_obs)
            if n_total_obs > 0
            else 0.0
        )

    if "from_node" not in top1_df.columns or "to_node" not in top1_df.columns:
        top1_df = _attach_arc_endpoints_from_raw(
            top1_df, Path(raw_data_dir) if raw_data_dir else None
        )

    cont_info = compute_route_continuity(
        top1_df, allow_undirected=allow_undirected_continuity
    )

    metrics: Dict[str, Optional[float | int]] = {
        "top1_accuracy": top1_overall,
        "top1_accuracy_covered": top1_covered,
        "top1_correct_observations": n_top1_correct,
        "covered_observations": n_obs_covered,
        "total_gps_observations": n_total_obs,
        "candidate_miss_rate": miss_rate,
        "route_continuity": cont_info["route_continuity"],
        "continuous_transitions": cont_info["continuous_transitions"],
        "total_transitions": cont_info["total_transitions"],
    }
    return metrics, top1_df


def evaluate_predictions(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    total_gps_observations: Optional[int] = None,
    candidate_miss_rate: Optional[float] = None,
    raw_data_dir: Optional[str | Path] = None,
    threshold: float = 0.5,
    allow_undirected_continuity: bool = False,
    inference_time_sec: Optional[float] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run candidate-level, observation-level, continuity, and per-trip timing evaluation."""
    cand_metrics = compute_candidate_metrics(y_true, y_prob, threshold=threshold)
    obs_metrics, top1_df = compute_observation_metrics(
        meta=meta,
        y_true=y_true,
        y_prob=y_prob,
        total_gps_observations=total_gps_observations,
        candidate_miss_rate=candidate_miss_rate,
        raw_data_dir=raw_data_dir,
        allow_undirected_continuity=allow_undirected_continuity,
    )
    out: Dict[str, Any] = {**cand_metrics, **obs_metrics}
    if inference_time_sec is not None:
        n_trips = max(int(meta["trip_id"].nunique()), 1)
        out["inference_time_sec"] = float(inference_time_sec)
        out["inference_time_per_trip_sec"] = float(inference_time_sec / n_trips)
        out["inference_time_per_trip_ms"] = float(1000.0 * inference_time_sec / n_trips)
    return out, top1_df


def cross_validate_model(
    estimator_factory: Callable[[np.ndarray], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    n_splits: int = 5,
    scale: bool = False,
    cv_strategy: Literal["group", "stratified_group"] = "group",
    raw_data_dir: Optional[str | Path] = None,
    allow_undirected_continuity: bool = False,
) -> Dict[str, Any]:
    """Run trip-grouped GroupKFold / StratifiedGroupKFold CV strictly on training trips.

    `estimator_factory(y_train_fold)` receives fold training labels so models like
    XGBoost compute `scale_pos_weight` strictly from the training fold (no leakage).
    """
    groups = meta["trip_id"].to_numpy()
    n_unique_trips = int(pd.Series(groups).nunique())
    actual_splits = min(n_splits, n_unique_trips)
    if actual_splits < 2:
        raise ValueError(
            f"Need at least 2 unique trips for grouped CV, got {n_unique_trips}."
        )

    if cv_strategy == "stratified_group":
        splitter = StratifiedGroupKFold(n_splits=actual_splits)
    elif cv_strategy == "group":
        splitter = GroupKFold(n_splits=actual_splits)
    else:
        raise ValueError(
            f"Unknown cv_strategy '{cv_strategy}'. Use 'group' or 'stratified_group'."
        )

    fold_records: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups), start=1):
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
            allow_undirected_continuity=allow_undirected_continuity,
            inference_time_sec=infer_sec,
        )

        fold_records.append(
            {
                "fold": fold_idx,
                "val_trips": int(meta_va["trip_id"].nunique()),
                "train_macro_f1": tr_metrics["macro_f1"],
                "val_macro_f1": va_metrics["macro_f1"],
                "train_minus_val_macro_f1": tr_metrics["macro_f1"] - va_metrics["macro_f1"],
                "val_accuracy": va_metrics["accuracy"],
                "val_precision": va_metrics["precision"],
                "val_recall": va_metrics["recall"],
                "val_log_loss": va_metrics["log_loss"],
                "val_top1_accuracy": va_metrics["top1_accuracy"],
                "val_route_continuity": va_metrics["route_continuity"],
                "train_time_sec": train_sec,
                "inference_time_sec": infer_sec,
                "inference_time_per_trip_ms": va_metrics["inference_time_per_trip_ms"],
            }
        )

    def _agg(key: str) -> Tuple[Optional[float], Optional[float]]:
        vals = [r[key] for r in fold_records if r[key] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    summary: Dict[str, Any] = {
        "cv_strategy": cv_strategy,
        "n_splits": actual_splits,
        "folds": fold_records,
    }
    for metric_key in (
        "train_macro_f1",
        "val_macro_f1",
        "train_minus_val_macro_f1",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_log_loss",
        "val_top1_accuracy",
        "val_route_continuity",
        "train_time_sec",
        "inference_time_sec",
        "inference_time_per_trip_ms",
    ):
        mean_v, std_v = _agg(metric_key)
        summary[f"{metric_key}_mean"] = mean_v
        summary[f"{metric_key}_std"] = std_v

    return summary
