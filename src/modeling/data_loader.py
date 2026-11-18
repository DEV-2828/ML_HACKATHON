"""Data loading, schema normalization, and leakage validation for PathSnap candidate datasets.

Compatible directly with Member 1's output in `data_geo/outputs/data/`:
  - train_candidates.parquet
  - student_test_candidates.parquet
  - metadata.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_META_COLS = ["trip_id", "obs_idx", "arc_id", "label"]
OPTIONAL_META_COLS = [
    "from_node",
    "to_node",
    "label_available",
    "crs_epsg",
    "timestamp",
    "lon",
    "lat",
    "x",
    "y",
]

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
    """Validated candidate-level dataset split ready for leakage-free modeling."""

    split_name: str
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    feature_cols: List[str]
    n_trips: int
    n_rows_raw: int
    n_rows_labeled: int
    n_observations_labeled: int
    total_gps_observations: int
    candidate_missed_observations: int
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


def _load_metadata(metadata_path: Optional[Path]) -> Dict[str, object]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_candidate_split(
    path: str | Path,
    split_name: str,
    feature_cols: Optional[Sequence[str]] = None,
    metadata_path: Optional[str | Path] = None,
    filter_label_available: bool = True,
    expected_trips: Optional[int] = None,
) -> CandidateDataset:
    """Load, filter (`label_available == True`), and validate a candidate split."""
    path = Path(path)
    df = _read_table(path)
    n_rows_raw = int(len(df))
    df, renamed = _normalize_columns(df)
    if renamed:
        print(f"[{split_name}] Normalized column names: {renamed}")

    missing_req = [c for c in REQUIRED_META_COLS if c not in df.columns]
    if missing_req:
        raise ValueError(
            f"[{split_name}] Missing required columns {missing_req}. "
            f"Found columns: {list(df.columns)}"
        )

    metadata = _load_metadata(Path(metadata_path) if metadata_path else None)
    splits_meta = metadata.get("splits", metadata) if isinstance(metadata, dict) else {}
    split_meta = splits_meta.get(split_name, {}) if isinstance(splits_meta, dict) else {}

    # Compute raw observation counts before filtering out label_available == False
    has_label_avail = "label_available" in df.columns
    if has_label_avail and filter_label_available:
        avail_mask = df["label_available"].astype(bool).to_numpy()
        df = df.loc[avail_mask].reset_index(drop=True)

    df["trip_id"] = df["trip_id"].astype(str)
    df["obs_idx"] = df["obs_idx"].astype(np.int32)
    df["arc_id"] = df["arc_id"].astype(np.int32)

    unique_labels = set(pd.Series(df["label"]).dropna().unique().tolist())
    if not unique_labels.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(
            f"[{split_name}] 'label' must be binary {{0, 1}}, found: {sorted(unique_labels)}"
        )
    y = df["label"].astype(np.int8).to_numpy()

    meta_cols = [c for c in REQUIRED_META_COLS + OPTIONAL_META_COLS if c in df.columns]
    if isinstance(metadata, dict) and isinstance(metadata.get("metadata_columns"), list):
        for mc in metadata["metadata_columns"]:
            if mc in df.columns and mc not in meta_cols:
                meta_cols.append(mc)
    meta = df[meta_cols].copy()

    # Determine feature columns strictly from explicit arg, metadata.json, or non-meta numerics
    if feature_cols is not None:
        selected_features = list(feature_cols)
    elif isinstance(metadata, dict) and isinstance(metadata.get("feature_columns"), list):
        selected_features = [c for c in metadata["feature_columns"] if c in df.columns]
    else:
        excluded = set(meta_cols)
        selected_features = [
            c
            for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    missing_feat = [c for c in selected_features if c not in df.columns]
    if missing_feat:
        raise ValueError(
            f"[{split_name}] Missing expected feature columns: {missing_feat}"
        )
    if not selected_features:
        raise ValueError(f"[{split_name}] No numeric feature columns detected.")

    # Guard against any metadata/target leaking into features
    leaked = set(selected_features).intersection(set(meta_cols))
    if leaked:
        raise ValueError(
            f"[{split_name}] Metadata/target columns leaked into features: {sorted(leaked)}"
        )

    # Keep features in float32 for memory & speed efficiency on ~3M candidate rows
    X = df[selected_features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    nan_count = int(X.isna().sum().sum())

    n_trips = int(meta["trip_id"].nunique())
    if expected_trips is not None and n_trips != expected_trips:
        print(
            f"[WARNING][{split_name}] Expected {expected_trips} trips, found {n_trips}."
        )

    obs_pos_sum = meta.groupby(["trip_id", "obs_idx"], sort=False)["label"].sum()
    n_obs_labeled = int(len(obs_pos_sum))
    n_obs_with_pos = int((obs_pos_sum > 0).sum())

    total_gps_obs = int(
        split_meta.get(
            "gps_observations",
            split_meta.get("total_gps_observations", n_obs_labeled),
        )
    )
    if "candidate_missed_observations" in split_meta:
        missed_obs = int(split_meta["candidate_missed_observations"])
    else:
        missed_obs = max(0, total_gps_obs - n_obs_with_pos)

    if "candidate_miss_rate" in split_meta:
        miss_rate = float(split_meta["candidate_miss_rate"])
    elif total_gps_obs > 0:
        miss_rate = float(missed_obs / total_gps_obs)
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
        n_rows_raw=n_rows_raw,
        n_rows_labeled=int(len(y)),
        n_observations_labeled=n_obs_labeled,
        total_gps_observations=total_gps_obs,
        candidate_missed_observations=missed_obs,
        candidate_miss_rate=miss_rate,
        positive_rate=pos_rate,
        nan_count=nan_count,
    )


def load_train_and_student_test(
    train_path: str | Path = "data_geo/outputs/data/train_candidates.parquet",
    student_test_path: str | Path = "data_geo/outputs/data/student_test_candidates.parquet",
    metadata_path: Optional[str | Path] = "data_geo/outputs/data/metadata.json",
    feature_cols: Optional[Sequence[str]] = None,
    filter_label_available: bool = True,
) -> Tuple[CandidateDataset, CandidateDataset]:
    """Load permitted train (80 trips) and student_test (10 trips) splits with leakage checks."""
    meta_file = Path(metadata_path) if metadata_path and Path(metadata_path).exists() else None
    train_ds = load_candidate_split(
        path=train_path,
        split_name="train",
        feature_cols=feature_cols,
        metadata_path=meta_file,
        filter_label_available=filter_label_available,
        expected_trips=80,
    )
    test_ds = load_candidate_split(
        path=student_test_path,
        split_name="student_test",
        feature_cols=train_ds.feature_cols,
        metadata_path=meta_file,
        filter_label_available=filter_label_available,
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

    print("=" * 78)
    print("DATASET VALIDATION PASSED (ZERO TRIP OVERLAP | FLOAT32 FEATURES)")
    for ds in (train_ds, test_ds):
        print(
            f"  {ds.split_name:12s} | trips={ds.n_trips:2d} | "
            f"rows={ds.n_rows_labeled:9,d}/{ds.n_rows_raw:9,d} | "
            f"obs={ds.n_observations_labeled:7,d}/{ds.total_gps_observations:7,d} | "
            f"pos_rate={ds.positive_rate:.4f} | miss_rate={ds.candidate_miss_rate:.4f} | "
            f"NaNs={ds.nan_count:,d}"
        )
    print(f"  Features ({len(train_ds.feature_cols)}): {train_ds.feature_cols}")
    print(
        f"  Graph endpoints present: "
        f"{'from_node' in train_ds.meta.columns and 'to_node' in train_ds.meta.columns}"
    )
    print("=" * 78)

    return train_ds, test_ds
