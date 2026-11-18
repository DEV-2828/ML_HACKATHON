"""Step 2 (Member 3): Tune HMM/Viterbi parameters on permitted dev data and benchmark
Classifier-Only vs Classifier + HMM/Viterbi route reconstruction across student_test.

Strict leakage rules:
  - Uses ONLY permitted development data (student_test_probs_tuned.parquet + Student folder/student_test).
  - Never accesses evaluator_test.

Outputs:
  - artifacts/hmm_config.json
  - results/metrics/classifier_vs_hmm.csv
  - results/metrics/hmm_per_trip_breakdown.csv
  - results/metrics/hmm_summary.json
  - PLOT/classifier_vs_hmm.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyproj import CRS, Transformer

from src.hmm_decoder import HMMConfig, HMMRouteDecoder, RoadNetworkGraph


def _choose_utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def load_trip_spatial_context(
    trip_dir: Path,
    trip_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RoadNetworkGraph, np.ndarray]:
    """Load projected track coords, timestamps, road graph, and reference .route arcs."""
    track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    route_path = trip_dir / f"{trip_id}.route"
    route = (
        np.loadtxt(route_path, dtype=np.int64).ravel()
        if route_path.exists()
        else np.array([], dtype=np.int64)
    )

    lon_med = float(np.nanmedian(track[:, 0]))
    lat_med = float(np.nanmedian(track[:, 1]))
    epsg = _choose_utm_epsg(lon_med, lat_med)
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)

    gx, gy = tf.transform(track[:, 0], track[:, 1])
    nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
    timestamps = track[:, 2] if track.shape[1] >= 3 else np.arange(len(track), dtype=np.float64)

    graph = RoadNetworkGraph(nx, ny, arcs[:, 0], arcs[:, 1])
    return np.asarray(gx), np.asarray(gy), np.asarray(timestamps), graph, route


def compute_sequence_metrics(
    pred_df: pd.DataFrame,
    reference_route_arcs: np.ndarray,
    total_gps_obs: int,
) -> Dict[str, float | int]:
    """Compute pointwise Top-1 accuracy, route continuity, and sequence-level arc F1."""
    ordered = pred_df.sort_values("obs_idx")
    arcs = ordered["arc_id"].to_numpy(dtype=np.int64)
    from_n = ordered["from_node"].to_numpy(dtype=np.int64)
    to_n = ordered["to_node"].to_numpy(dtype=np.int64)

    n_covered = len(ordered)
    n_correct = int(ordered["label"].sum()) if "label" in ordered.columns else 0
    top1_covered = float(n_correct / n_covered) if n_covered > 0 else 0.0
    top1_overall = float(n_correct / total_gps_obs) if total_gps_obs > 0 else top1_covered

    # Route continuity: same_arc OR to_node[t] == from_node[t+1]
    if len(arcs) >= 2:
        same_arc = arcs[:-1] == arcs[1:]
        direct_edge = (to_n[:-1] >= 0) & (to_n[:-1] == from_n[1:])
        cont_count = int(np.sum(same_arc | direct_edge))
        total_trans = int(len(arcs) - 1)
        continuity = float(cont_count / total_trans)
    else:
        cont_count, total_trans, continuity = 0, 0, 1.0

    # Sequence-level route accuracy: compare unique predicted traversed arc set vs .route set
    pred_unique = set(int(a) for a in arcs if a >= 0)
    ref_unique = set(int(a) for a in reference_route_arcs if a >= 0)
    if pred_unique and ref_unique:
        inter = len(pred_unique.intersection(ref_unique))
        seq_prec = inter / len(pred_unique)
        seq_rec = inter / len(ref_unique)
        seq_f1 = (2 * seq_prec * seq_rec / (seq_prec + seq_rec)) if (seq_prec + seq_rec) > 0 else 0.0
    else:
        seq_prec, seq_rec, seq_f1 = 0.0, 0.0, 0.0

    # Deduplicated consecutive arc transitions count (measures jitter / ping-ponging)
    n_arc_switches = int(np.sum(arcs[:-1] != arcs[1:])) if len(arcs) >= 2 else 0

    return {
        "n_covered_obs": n_covered,
        "n_total_obs": int(total_gps_obs),
        "n_correct_obs": n_correct,
        "top1_accuracy_overall": top1_overall,
        "top1_accuracy_covered": top1_covered,
        "route_continuity": continuity,
        "continuous_transitions": cont_count,
        "total_transitions": total_trans,
        "seq_arc_precision": float(seq_prec),
        "seq_arc_recall": float(seq_rec),
        "seq_arc_f1": float(seq_f1),
        "arc_switches": n_arc_switches,
    }


def _classifier_only_top1(trip_df: pd.DataFrame) -> pd.DataFrame:
    return (
        trip_df.sort_values(["obs_idx", "prob", "arc_id"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates(subset=["obs_idx"], keep="first")
        .reset_index(drop=True)
    )


def _evaluate_config_across_trips(
    cfg: HMMConfig,
    trip_data: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, RoadNetworkGraph, np.ndarray, int]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    decoder = HMMRouteDecoder(cfg)
    per_trip = []
    tot_correct = 0
    tot_covered = 0
    tot_obs = 0
    tot_cont = 0
    tot_trans = 0
    seq_f1s = []
    seq_precs = []
    seq_recs = []
    decode_secs = []

    for trip_id, (tdf, gx, gy, ts, graph, route, n_gps) in trip_data.items():
        t0 = time.perf_counter()
        decoded = decoder.decode_trip(tdf, gps_x=gx, gps_y=gy, timestamps_s=ts, graph=graph)
        dt = time.perf_counter() - t0
        decode_secs.append(dt)

        m = compute_sequence_metrics(decoded, route, total_gps_obs=n_gps)
        tot_correct += int(m["n_correct_obs"])
        tot_covered += int(m["n_covered_obs"])
        tot_obs += int(m["n_total_obs"])
        tot_cont += int(m["continuous_transitions"])
        tot_trans += int(m["total_transitions"])
        seq_f1s.append(float(m["seq_arc_f1"]))
        seq_precs.append(float(m["seq_arc_precision"]))
        seq_recs.append(float(m["seq_arc_recall"]))
        per_trip.append({"trip_id": trip_id, "decode_ms": dt * 1000.0, **m})

    agg = {
        "top1_accuracy_overall": tot_correct / max(tot_obs, 1),
        "top1_accuracy_covered": tot_correct / max(tot_covered, 1),
        "route_continuity": tot_cont / max(tot_trans, 1),
        "seq_arc_precision": float(np.mean(seq_precs)),
        "seq_arc_recall": float(np.mean(seq_recs)),
        "seq_arc_f1": float(np.mean(seq_f1s)),
        "decode_time_total_s": float(np.sum(decode_secs)),
        "decode_time_per_trip_ms": float(np.mean(decode_secs) * 1000.0),
    }
    return agg, per_trip


def _plot_classifier_vs_hmm(cmp_df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    metrics = [
        ("Top-1 Accuracy (Overall)", "Top-1 Accuracy\n(Overall)"),
        ("Top-1 Accuracy (Covered)", "Top-1 Accuracy\n(Covered)"),
        ("Route Continuity", "Route\nContinuity"),
        ("Sequence Arc F1", "Sequence-Level\nArc F1"),
        ("Sequence Arc Precision", "Sequence\nPrecision"),
    ]
    x = np.arange(len(metrics))
    w = 0.35
    v_clf = [float(cmp_df.iloc[0][k]) for k, _ in metrics]
    v_hmm = [float(cmp_df.iloc[1][k]) for k, _ in metrics]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - w / 2, v_clf, width=w, label="Classifier-Only (Tuned XGBoost)", color="#7f9fc4", edgecolor="#1b2a4a")
    b2 = ax.bar(x + w / 2, v_hmm, width=w, label="Classifier + HMM/Viterbi", color="#2ca02c", edgecolor="#124212")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.014,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("Score", fontsize=10.5, fontweight="bold")
    ax.set_title("PathSnap Route Reconstruction: Classifier-Only vs Classifier + HMM/Viterbi (10 Student-Test Trips)", fontsize=12, fontweight="bold", pad=16)
    ax.legend(frameon=True, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def run_hmm_evaluation(
    probs_path: str | Path = "results/student_test_probs_tuned.parquet",
    raw_test_dir: str | Path = "Student folder/student_test",
    output_dir: str | Path = "results/metrics",
    artifacts_dir: str | Path = "artifacts",
    plot_dir: str | Path = "PLOT",
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    art_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    probs_df = pd.read_parquet(probs_path)
    probs_df["trip_id"] = probs_df["trip_id"].astype(str)
    test_root = Path(raw_test_dir)
    if not test_root.exists():
        alt = Path("Student folder/Student folder/student_test")
        if alt.exists():
            test_root = alt

    print("=" * 78)
    print("LOADING ROAD NETWORKS & TRACKS FOR 10 STUDENT_TEST TRIPS...")
    trip_data = {}
    clf_per_trip = []
    tot_clf_correct = tot_clf_cov = tot_obs = tot_clf_cont = tot_clf_trans = 0
    clf_seq_f1s, clf_seq_precs, clf_seq_recs = [], [], []

    for trip_id, grp in probs_df.groupby("trip_id", sort=True):
        t_dir = test_root / trip_id
        gx, gy, ts, graph, route = load_trip_spatial_context(t_dir, trip_id)
        n_gps = len(gx)
        trip_data[trip_id] = (grp.copy(), gx, gy, ts, graph, route, n_gps)

        clf_top1 = _classifier_only_top1(grp)
        cm = compute_sequence_metrics(clf_top1, route, total_gps_obs=n_gps)
        tot_clf_correct += int(cm["n_correct_obs"])
        tot_clf_cov += int(cm["n_covered_obs"])
        tot_obs += int(cm["n_total_obs"])
        tot_clf_cont += int(cm["continuous_transitions"])
        tot_clf_trans += int(cm["total_transitions"])
        clf_seq_f1s.append(float(cm["seq_arc_f1"]))
        clf_seq_precs.append(float(cm["seq_arc_precision"]))
        clf_seq_recs.append(float(cm["seq_arc_recall"]))
        clf_per_trip.append({"trip_id": trip_id, **cm})

    clf_agg = {
        "top1_accuracy_overall": tot_clf_correct / max(tot_obs, 1),
        "top1_accuracy_covered": tot_clf_correct / max(tot_clf_cov, 1),
        "route_continuity": tot_clf_cont / max(tot_clf_trans, 1),
        "seq_arc_precision": float(np.mean(clf_seq_precs)),
        "seq_arc_recall": float(np.mean(clf_seq_recs)),
        "seq_arc_f1": float(np.mean(clf_seq_f1s)),
    }
    print(
        f"Classifier-Only Baseline | Top-1={clf_agg['top1_accuracy_overall']:.4f} "
        f"(Cov={clf_agg['top1_accuracy_covered']:.4f}) | Cont={clf_agg['route_continuity']:.4f} | "
        f"SeqF1={clf_agg['seq_arc_f1']:.4f}"
    )

    # Tune HMM decoder parameters strictly on permitted dev data
    param_grid = [
        HMMConfig(emission_weight=1.0, beta=10.0, self_transition_bonus=0.5, direct_edge_bonus=0.2, disconnect_penalty=-8.0),
        HMMConfig(emission_weight=1.0, beta=15.0, self_transition_bonus=1.0, direct_edge_bonus=0.4, disconnect_penalty=-10.0),
        HMMConfig(emission_weight=1.0, beta=20.0, self_transition_bonus=1.5, direct_edge_bonus=0.6, disconnect_penalty=-12.0),
        HMMConfig(emission_weight=1.0, beta=25.0, self_transition_bonus=2.0, direct_edge_bonus=1.0, disconnect_penalty=-15.0),
        HMMConfig(emission_weight=0.7, beta=20.0, self_transition_bonus=1.5, direct_edge_bonus=0.8, disconnect_penalty=-12.0),
        HMMConfig(emission_weight=1.5, beta=15.0, self_transition_bonus=1.0, direct_edge_bonus=0.5, disconnect_penalty=-10.0),
    ]

    print("\nTuning HMM/Viterbi hyperparameters on development split...")
    best_cfg = param_grid[0]
    best_score = -1.0
    best_agg: Dict[str, float] = {}
    best_per_trip: List[Dict[str, Any]] = []
    tuning_log = []

    for idx, cfg in enumerate(param_grid, start=1):
        agg, per_trip = _evaluate_config_across_trips(cfg, trip_data)
        # Joint objective balancing pointwise accuracy, continuity, and sequence arc F1
        composite = (
            0.45 * agg["top1_accuracy_overall"]
            + 0.30 * agg["route_continuity"]
            + 0.25 * agg["seq_arc_f1"]
        )
        tuning_log.append({"trial": idx, "composite_score": round(composite, 5), **cfg.to_dict(), **agg})
        print(
            f"  [HMM {idx}/{len(param_grid)}] Top-1={agg['top1_accuracy_overall']:.4f} | "
            f"Cont={agg['route_continuity']:.4f} | SeqF1={agg['seq_arc_f1']:.4f} | "
            f"SeqPrec={agg['seq_arc_precision']:.4f} | Decode={agg['decode_time_per_trip_ms']:.1f}ms/trip"
        )
        if composite > best_score:
            best_score = composite
            best_cfg = cfg
            best_agg = agg
            best_per_trip = per_trip

    # Save tuned HMM config for Laptop 1 FastAPI server
    with (art_dir / "hmm_config.json").open("w", encoding="utf-8") as f:
        json.dump(best_cfg.to_dict(), f, indent=2)

    # Save Classifier-Only vs Classifier+HMM comparison table
    cmp_rows = [
        {
            "Pipeline": "Classifier-Only (Bayesian-Tuned XGBoost)",
            "Top-1 Accuracy (Overall)": round(clf_agg["top1_accuracy_overall"], 5),
            "Top-1 Accuracy (Covered)": round(clf_agg["top1_accuracy_covered"], 5),
            "Route Continuity": round(clf_agg["route_continuity"], 5),
            "Sequence Arc Precision": round(clf_agg["seq_arc_precision"], 5),
            "Sequence Arc Recall": round(clf_agg["seq_arc_recall"], 5),
            "Sequence Arc F1": round(clf_agg["seq_arc_f1"], 5),
            "HMM Decode Time / Trip (ms)": 0.0,
        },
        {
            "Pipeline": "Classifier + HMM/Viterbi",
            "Top-1 Accuracy (Overall)": round(best_agg["top1_accuracy_overall"], 5),
            "Top-1 Accuracy (Covered)": round(best_agg["top1_accuracy_covered"], 5),
            "Route Continuity": round(best_agg["route_continuity"], 5),
            "Sequence Arc Precision": round(best_agg["seq_arc_precision"], 5),
            "Sequence Arc Recall": round(best_agg["seq_arc_recall"], 5),
            "Sequence Arc F1": round(best_agg["seq_arc_f1"], 5),
            "HMM Decode Time / Trip (ms)": round(best_agg["decode_time_per_trip_ms"], 2),
        },
        {
            "Pipeline": "Delta (HMM - Classifier-Only)",
            "Top-1 Accuracy (Overall)": round(best_agg["top1_accuracy_overall"] - clf_agg["top1_accuracy_overall"], 5),
            "Top-1 Accuracy (Covered)": round(best_agg["top1_accuracy_covered"] - clf_agg["top1_accuracy_covered"], 5),
            "Route Continuity": round(best_agg["route_continuity"] - clf_agg["route_continuity"], 5),
            "Sequence Arc Precision": round(best_agg["seq_arc_precision"] - clf_agg["seq_arc_precision"], 5),
            "Sequence Arc Recall": round(best_agg["seq_arc_recall"] - clf_agg["seq_arc_recall"], 5),
            "Sequence Arc F1": round(best_agg["seq_arc_f1"] - clf_agg["seq_arc_f1"], 5),
            "HMM Decode Time / Trip (ms)": round(best_agg["decode_time_per_trip_ms"], 2),
        },
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(out_dir / "classifier_vs_hmm.csv", index=False)

    # Save per-trip breakdown (highlights where HMM helped vs didn't help)
    clf_map = {r["trip_id"]: r for r in clf_per_trip}
    breakdown_rows = []
    for hr in best_per_trip:
        tid = hr["trip_id"]
        cr = clf_map[tid]
        breakdown_rows.append(
            {
                "trip_id": tid,
                "n_total_obs": hr["n_total_obs"],
                "clf_top1": round(cr["top1_accuracy_overall"], 5),
                "hmm_top1": round(hr["top1_accuracy_overall"], 5),
                "delta_top1": round(hr["top1_accuracy_overall"] - cr["top1_accuracy_overall"], 5),
                "clf_continuity": round(cr["route_continuity"], 5),
                "hmm_continuity": round(hr["route_continuity"], 5),
                "delta_continuity": round(hr["route_continuity"] - cr["route_continuity"], 5),
                "clf_seq_f1": round(cr["seq_arc_f1"], 5),
                "hmm_seq_f1": round(hr["seq_arc_f1"], 5),
                "delta_seq_f1": round(hr["seq_arc_f1"] - cr["seq_arc_f1"], 5),
                "clf_arc_switches": cr["arc_switches"],
                "hmm_arc_switches": hr["arc_switches"],
                "decode_ms": round(hr["decode_ms"], 2),
            }
        )
    bd_df = pd.DataFrame(breakdown_rows)
    bd_df.to_csv(out_dir / "hmm_per_trip_breakdown.csv", index=False)

    with (out_dir / "hmm_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_config": best_cfg.to_dict(),
                "classifier_only": clf_agg,
                "classifier_plus_hmm": best_agg,
                "tuning_trials": tuning_log,
                "per_trip_breakdown": breakdown_rows,
            },
            f,
            indent=2,
        )

    _plot_classifier_vs_hmm(cmp_df, Path(plot_dir) / "classifier_vs_hmm.png")

    print("\n" + "=" * 78)
    print("CLASSIFIER-ONLY VS CLASSIFIER + HMM/VITERBI (10 STUDENT_TEST TRIPS)")
    print(cmp_df.to_string(index=False))
    print("\nPER-TRIP BREAKDOWN:")
    print(bd_df[["trip_id", "clf_top1", "hmm_top1", "delta_top1", "clf_continuity", "hmm_continuity", "clf_seq_f1", "hmm_seq_f1", "decode_ms"]].to_string(index=False))
    print("=" * 78)
    return cmp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and evaluate HMM/Viterbi route reconstruction.")
    parser.add_argument("--probs", default="results/student_test_probs_tuned.parquet")
    parser.add_argument("--raw-test-dir", default="Student folder/student_test")
    parser.add_argument("--out-dir", default="results/metrics")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--plot-dir", default="PLOT")
    args = parser.parse_args()
    run_hmm_evaluation(
        probs_path=args.probs,
        raw_test_dir=args.raw_test_dir,
        output_dir=args.out_dir,
        artifacts_dir=args.artifacts_dir,
        plot_dir=args.plot_dir,
    )
