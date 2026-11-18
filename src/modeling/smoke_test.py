"""Self-contained synthetic smoke test for Steps 1-3 (data loader, evaluator, 6-model trainer).

Runs entirely inside a temporary directory and deletes it afterward so results/ and
artifacts/ stay clean for real data from Member 1.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.train_baselines import run_baseline_benchmark


def _make_synthetic_split(
    trip_ids: list[str],
    obs_per_trip: int,
    k_candidates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for trip_id in trip_ids:
        cur_node = int(rng.integers(0, 50))
        for obs_idx in range(obs_per_trip):
            true_arc = 1000 + obs_idx // 3
            next_node = cur_node if obs_idx % 3 != 0 else cur_node + 1
            for cand_i in range(k_candidates):
                is_pos = 1 if cand_i == 0 else 0
                arc_id = true_arc if is_pos else int(rng.integers(2000, 9000))
                f_node = cur_node if is_pos else int(rng.integers(500, 999))
                t_node = next_node if is_pos else int(rng.integers(500, 999))
                # Realistic noisy features
                dist_m = rng.normal(4.0, 3.0) if is_pos else rng.normal(22.0, 10.0)
                heading_diff = rng.normal(8.0, 7.0) if is_pos else rng.normal(55.0, 30.0)
                speed_mps = rng.normal(12.0, 3.0)
                seg_len = rng.normal(45.0, 15.0)
                rows.append(
                    {
                        "trip_id": trip_id,
                        "obs_idx": obs_idx,
                        "arc_id": arc_id,
                        "from_node": f_node,
                        "to_node": t_node,
                        "label": is_pos,
                        "perp_dist_m": max(0.0, dist_m),
                        "heading_diff_deg": np.clip(heading_diff, 0.0, 180.0),
                        "speed_mps": max(0.0, speed_mps),
                        "seg_len_m": max(5.0, seg_len),
                        "dist_rank": float(cand_i + 1) + rng.normal(0, 0.4),
                    }
                )
            cur_node = next_node
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="pathsnap_smoke_") as tmp:
        tmp_path = Path(tmp)
        train_df = _make_synthetic_split(
            [f"T{i:03d}" for i in range(10)], obs_per_trip=25, k_candidates=5, rng=rng
        )
        test_df = _make_synthetic_split(
            [f"S{i:03d}" for i in range(3)], obs_per_trip=25, k_candidates=5, rng=rng
        )
        # Inject a couple NaNs to verify pipeline imputation works
        train_df.loc[0, "heading_diff_deg"] = np.nan
        test_df.loc[0, "perp_dist_m"] = np.inf

        train_p = tmp_path / "train_candidates.parquet"
        test_p = tmp_path / "student_test_candidates.parquet"
        stats_p = tmp_path / "candidate_stats.json"
        train_df.to_parquet(train_p, index=False)
        test_df.to_parquet(test_p, index=False)
        with stats_p.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "train": {"total_gps_observations": 255},
                    "student_test": {"total_gps_observations": 78},
                },
                f,
            )

        out_dir = tmp_path / "metrics"
        art_dir = tmp_path / "artifacts"
        comp_df = run_baseline_benchmark(
            train_path=train_p,
            student_test_path=test_p,
            stats_path=stats_p,
            raw_data_dir=None,
            output_dir=out_dir,
            artifacts_dir=art_dir,
            n_splits=5,
        )

        # Verify all expected outputs exist
        expected_files = [
            out_dir / "model_comparison.csv",
            out_dir / "model_configs.json",
            out_dir / "confusion_matrices.json",
            out_dir / "experiment_summary.json",
            out_dir / "history_XGBoost.csv",
            out_dir / "history_MLP.csv",
            out_dir / "student_test_probs_XGBoost.parquet",
            art_dir / "baseline_best_model.joblib",
            art_dir / "baseline_best_metadata.json",
        ]
        for ef in expected_files:
            assert ef.exists(), f"Missing expected output: {ef}"

        print("\nSMOKE TEST PASSED — Comparison Table Preview:")
        print(
            comp_df[
                [
                    "Model",
                    "CV Macro F1-score",
                    "Top-1 Accuracy",
                    "Route Continuity",
                    "Log Loss",
                    "Inference Time (s)",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
