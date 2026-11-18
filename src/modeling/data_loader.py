"""Data loading, schema normalization, and leakage validation for PathSnap candidate datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Canonical metadata columns excluded from model features
REQUIRED_META_COLS = ["trip_id", "obs_idx", "arc_id", "label"]
OPTIONAL_META_COLS = ["from_node", "to_node", "timestamp", "lon", "lat", "x", "y"]

# Common column aliases Member 1 might use, normalized to canonical names
COLUMN_ALIASES: Dict[str, str] = {
    "trip": "trip_id",
    "trajectory_id": "trip_id",
    "gps_idx": "obs_idx",
    "point_idx": "obs_idx",
    "observation_idx": "obs_idx",
    "segment_id": "arc_id",
    "edge_id": "arc_id",
    "candidate_arc_id": "arc_id",
    "from_node_id": "from_node",
    "to_node_id": "to_node",
    "u": "from_node",
    "v": "to_node",
    "target": "label",
    "is_match": "label",
    "y": "label",
}


@dataclass
class CandidateDataset:
    """Validated candidate-level dataset split ready for modeling."""

    split_name: str
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    feature_cols: List[str]
    n_trips: int
    n_observations_in_table: int
    total_gps_observations: int
    candidate_miss_rate: float
    positive_rate: float
    nan_count: int


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format '{suffix}'. Expected .parquet or .csv.")


def _normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rename_map: Dict[str, str] = {}
    existing = set(df.columns)
    for col in df.columns:
        lower = col.strip().lower()
        if lower in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[lower]
            if canonical not in existing and canonical not in rename_map.values():
                rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map


def _load_candidate_stats(stats_path: Optional[Path]) -> Dict[str, dict]:
    if stats_path is None or not stats_path.exists():
        return {}
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_candidate_split(
    path: str | Path,
    split_name: str,
    feature_cols: Optional[Sequence[str]] = None,
    stats_path: Optional[str | Path] = None,
    expected_trips: Optional[int] = None,
) -> CandidateDataset:
    """Load, normalize, and validate a single candidate split (train or student_test)."""
    path = Path(path)
    df = _read_table(path)
    df, renamed = _normalize_columns(df)
    if renamed:
        print(f"[{split_name}] Normalized column names: {renamed}")

    missing_req = [c for c in REQUIRED_META_COLS if c not in df.columns]
    if missing_req:
        raise ValueError(
            f"[{split_name}] Missing required columns {missing_req}. "
            f"Found columns: {list(df.columns)}"
        )

    # Ensure clean types for metadata
    df["trip_id"] = df["trip_id"].astype(str)
    df["obs_idx"] = df["obs_idx"].astype(int)
    df["arc_id"] = df["arc_id"].astype(int)

    unique_labels = set(pd.Series(df["label"]).dropna().unique().tolist())
    if not unique_labels.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(
            f"[{split_name}] 'label' must be binary {{0, 1}}, found: {sorted(unique_labels)}"
        )
    y = df["label"].astype(int).to_numpy()

    meta_cols = [c for c in REQUIRED_META_COLS + OPTIONAL_META_COLS if c in df.columns]
    meta = df[meta_cols].copy()

    if feature_cols is None:
        excluded = set(meta_cols)
        selected_features = [
            c
            for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        selected_features = list(feature_cols)
        missing_feat = [c for c in selected_features if c not in df.columns]
        if missing_feat:
            raise ValueError(
                f"[{split_name}] Missing expected feature columns: {missing_feat}"
            )

    if not selected_features:
        raise ValueError(f"[{split_name}] No numeric feature columns detected.")

    # Replace +/-inf with NaN so in-fold SimpleImputer handles them cleanly without leakage
    X = df[selected_features].replace([np.inf, -np.inf], np.nan).astype(np.float64)
    nan_count = int(X.isna().sum().sum())

    n_trips = int(meta["trip_id"].nunique())
    if expected_trips is not None and n_trips != expected_trips:
        print(
            f"[WARNING][{split_name}] Expected {expected_trips} trips, found {n_trips}."
        )

    # Observation-level coverage & Candidate Miss Rate
    obs_pos_sum = meta.groupby(["trip_id", "obs_idx"])["label"].sum()
    n_obs_in_table = int(len(obs_pos_sum))
    n_obs_with_pos = int((obs_pos_sum > 0).sum())

    stats = _load_candidate_stats(Path(stats_path) if stats_path else None)
    split_stats = stats.get(split_name, {}) if isinstance(stats, dict) else {}

    if "total_gps_observations" in split_stats:
        total_gps_obs = int(split_stats["total_gps_observations"])
    else:
        total_gps_obs = n_obs_in_table

    if "candidate_miss_rate" in split_stats:
        miss_rate = float(split_stats["candidate_miss_rate"])
    elif total_gps_obs > 0:
        miss_rate = float((total_gps_obs - n_obs_with_pos) / total_gps_obs)
    else:
        miss_rate = 0.0

    pos_rate = float(y.mean()) if len(y) > 0 else 0.0

    return CandidateDataset(
        split_name=split_name,
        X=X,
        y=y,
        meta=meta,
        feature_cols=selected_features,
        n_trips=n_trips,
        n_observations_in_table=n_obs_in_table,
        total_gps_observations=total_gps_obs,
        candidate_miss_rate=miss_rate,
        positive_rate=pos_rate,
        nan_count=nan_count,
    )


def load_train_and_student_test(
    train_path: str | Path,
    student_test_path: str | Path,
    stats_path: Optional[str | Path] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[CandidateDataset, CandidateDataset]:
    """Load both permitted splits and enforce zero trip-level data leakage."""
    train_ds = load_candidate_split(
        path=train_path,
        split_name="train",
        feature_cols=feature_cols,
        stats_path=stats_path,
        expected_trips=80,
    )
    test_ds = load_candidate_split(
        path=student_test_path,
        split_name="student_test",
        feature_cols=train_ds.feature_cols,
        stats_path=stats_path,
        expected_trips=10,
    )

    train_trips = set(train_ds.meta["trip_id"].unique())
    test_trips = set(test_ds.meta["trip_id"].unique())
    overlap = train_trips.intersection(test_trips)
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} trip(s) appear in both train and "
            f"student_test: {sorted(overlap)[:10]}"
        )

    print("=" * 72)
    print("DATASET VALIDATION PASSED (ZERO TRIP OVERLAP)")
    for ds in (train_ds, test_ds):
        print(
            f"  {ds.split_name:12s} | trips={ds.n_trips:2d} | rows={len(ds.y):8,d} | "
            f"obs={ds.n_observations_in_table:6,d}/{ds.total_gps_observations:6,d} | "
            f"pos_rate={ds.positive_rate:.4f} | miss_rate={ds.candidate_miss_rate:.4f} | "
            f"NaNs={ds.nan_count:,d}"
        )
    print(f"  Features ({len(train_ds.feature_cols)}): {train_ds.feature_cols}")
    print(
        f"  Graph node cols present: "
        f"{'from_node' in train_ds.meta.columns and 'to_node' in train_ds.meta.columns}"
    )
    print("=" * 72)

    return train_ds, test_ds
