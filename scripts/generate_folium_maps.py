"""PathSnap Interactive Folium Map Generator (Real Production Pipeline — XGBoost + HMM/Viterbi).

Generates genuine multi-layer interactive Folium HTML maps (and a high-res static PNG
comparison figure for the PDF report) directly from:
  - `artifacts/final_model.joblib` (Bayesian-tuned XGBoost)
  - `artifacts/hmm_config.json` (Tuned log-space HMM/Viterbi decoder)
  - `Student folder/student_test/` (permitted development trips ONLY)

All 5 mandatory/diagnostic layers (Deliverables.pdf Section 6):
  1. Road Network Geometry (candidate street grid, gray)
  2. Original Noisy GPS Observations (blue circular markers)
  3. Reference Route from `.route` (amber/orange solid line)
  4. Classifier-Only Predicted Route (red dashed line)
  5. HMM + Viterbi Predicted Route (emerald green solid line + Start/End markers)
  + Clear legend and interactive LayerControl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.hmm_decoder import HMMConfig, HMMRouteDecoder
from src.inference_pipeline import (
    FEATURE_COLUMNS,
    arcs_to_latlon_polyline,
    evaluate_against_reference_server_side,
    extract_candidate_segments_latlon,
    extract_trip_candidates,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_student_test_dir() -> Path:
    for cand in (
        BASE_DIR / "Student folder" / "student_test",
        BASE_DIR / "Student folder" / "Student folder" / "student_test",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError("Could not locate Student folder/student_test/")


def _reference_route_latlon(
    route_file: Path, nodes_lonlat: np.ndarray, arcs_uv: np.ndarray
) -> list[list[float]]:
    if not route_file.exists():
        return []
    route_arcs = np.loadtxt(route_file, dtype=np.int64).ravel()
    pts: list[list[float]] = []
    n_nodes = len(nodes_lonlat)
    for ai in route_arcs:
        ai_int = int(ai)
        if 0 <= ai_int < len(arcs_uv):
            u, v = int(arcs_uv[ai_int, 0]), int(arcs_uv[ai_int, 1])
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                u_pt = [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])]
                v_pt = [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])]
                if not pts or pts[-1] != u_pt:
                    pts.append(u_pt)
                pts.append(v_pt)
    return pts


def _add_legend(f_map: folium.Map, trip_id: str, stats: dict) -> None:
    legend_html = f"""
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: rgba(255, 255, 255, 0.95); padding: 12px 14px;
        border: 1.5px solid #334155; border-radius: 8px;
        font-family: Arial, sans-serif; font-size: 12px; line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    ">
      <div style="font-weight: 700; margin-bottom: 4px;">PathSnap — Trip {trip_id} (student_test)</div>
      <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2563EB;margin-right:6px;"></span>Original Noisy GPS Observations</div>
      <div><span style="display:inline-block;width:18px;height:3px;background:#64748B;margin-right:6px;vertical-align:middle;"></span>Nearby Road Network Geometry</div>
      <div><span style="display:inline-block;width:18px;height:4px;background:#F59E0B;margin-right:6px;vertical-align:middle;"></span>Reference Route (.route)</div>
      <div><span style="display:inline-block;width:18px;height:3px;border-top:3px dashed #DC2626;margin-right:6px;vertical-align:middle;"></span>Classifier-Only Route (XGBoost)</div>
      <div><span style="display:inline-block;width:18px;height:4px;background:#059669;margin-right:6px;vertical-align:middle;"></span>Predicted Route (XGBoost + HMM/Viterbi)</div>
      <div style="margin-top:6px;padding-top:4px;border-top:1px solid #cbd5e1;font-size:11px;color:#334155;">
        Clf Top-1: <b>{stats['classifier_top_1_accuracy']*100:.2f}%</b> &rarr; HMM Top-1: <b>{stats['top_1_accuracy']*100:.2f}%</b><br>
        Clf Cont: <b>{stats['classifier_continuity']*100:.2f}%</b> &rarr; HMM Cont: <b>{stats['route_continuity']*100:.2f}%</b>
      </div>
    </div>
    """
    f_map.get_root().html.add_child(folium.Element(legend_html))


def generate_trip_folium_map(
    trip_id: str,
    test_root: Path,
    model,
    feature_cols: list[str],
    decoder: HMMRouteDecoder,
    out_html: Path,
    out_png: Path | None = None,
) -> dict:
    """Run real XGBoost + HMM inference on trip_id and save interactive HTML (+ optional static PNG)."""
    trip_dir = test_root / trip_id
    print(f"Generating REAL 5-layer Folium map for student_test Trip {trip_id}...")

    ctx = extract_trip_candidates(trip_dir, trip_id, radius_m=80.0, max_candidates=40)
    X_trip = ctx.candidates_df[feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    probs = model.predict_proba(X_trip)[:, 1].astype(np.float64)

    scored = ctx.candidates_df[["obs_idx", "arc_id", "from_node", "to_node"]].copy()
    scored["prob"] = probs

    clf_top1 = (
        scored.sort_values(["obs_idx", "prob", "arc_id"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates(subset=["obs_idx"], keep="first")
        .reset_index(drop=True)
    )
    hmm_decoded = decoder.decode_trip(
        scored, gps_x=ctx.gx, gps_y=ctx.gy, timestamps_s=ctx.timestamps_s, graph=ctx.graph
    )
    stats = evaluate_against_reference_server_side(
        ctx=ctx,
        clf_top1_df=clf_top1,
        hmm_decoded_df=hmm_decoded,
        route_file=trip_dir / f"{trip_id}.route",
    )

    raw_gps = [[float(lat), float(lon)] for lon, lat in ctx.track_lonlat]
    step_gps = max(1, len(raw_gps) // 450)
    sampled_gps = raw_gps[::step_gps]

    ref_route = _reference_route_latlon(trip_dir / f"{trip_id}.route", ctx.nodes_lonlat, ctx.arcs_uv)
    clf_route = arcs_to_latlon_polyline(clf_top1, ctx.nodes_lonlat, ctx.arcs_uv, max_points=1500)
    hmm_route = arcs_to_latlon_polyline(hmm_decoded, ctx.nodes_lonlat, ctx.arcs_uv, max_points=1500)
    cand_segs = extract_candidate_segments_latlon(ctx.candidates_df, ctx.nodes_lonlat, ctx.arcs_uv, max_segments=400)

    lats = [p[0] for p in raw_gps]
    lons = [p[1] for p in raw_gps]
    f_map = folium.Map(
        location=[float(np.mean(lats)), float(np.mean(lons))],
        zoom_start=14,
        tiles="OpenStreetMap",
    )
    f_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    if cand_segs:
        g_road = folium.FeatureGroup(name="1. Road Network Geometry (Candidate Arcs)", show=True).add_to(f_map)
        for seg in cand_segs:
            folium.PolyLine(seg, color="#64748B", weight=1.8, opacity=0.45, tooltip="Road Network Arc").add_to(g_road)

    g_gps = folium.FeatureGroup(name="2. Original GPS Observations (Noisy)", show=True).add_to(f_map)
    for idx, pt in enumerate(sampled_gps):
        folium.CircleMarker(
            location=pt,
            radius=3.0,
            color="#1D4ED8",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.75,
            tooltip=f"GPS Obs #{idx * step_gps}: ({pt[0]:.5f}, {pt[1]:.5f})",
        ).add_to(g_gps)

    if ref_route:
        g_ref = folium.FeatureGroup(name="3. Reference Route (.route)", show=True).add_to(f_map)
        folium.PolyLine(ref_route, color="#F59E0B", weight=6.0, opacity=0.75, tooltip="Reference Route (.route)").add_to(g_ref)

    if clf_route:
        g_clf = folium.FeatureGroup(name="4. Classifier-Only Route (XGBoost)", show=False).add_to(f_map)
        folium.PolyLine(clf_route, color="#DC2626", weight=3.0, opacity=0.85, dash_array="6, 6", tooltip="Classifier-Only Prediction").add_to(g_clf)

    if hmm_route:
        g_hmm = folium.FeatureGroup(name="5. Predicted Route (XGBoost + HMM/Viterbi)", show=True).add_to(f_map)
        folium.PolyLine(hmm_route, color="#059669", weight=4.0, opacity=0.95, tooltip="Predicted Route (HMM/Viterbi)").add_to(g_hmm)
        folium.Marker(hmm_route[0], popup=f"Trip {trip_id} Start", icon=folium.Icon(color="green", icon="play")).add_to(g_hmm)
        folium.Marker(hmm_route[-1], popup=f"Trip {trip_id} End", icon=folium.Icon(color="red", icon="stop")).add_to(g_hmm)

    _add_legend(f_map, trip_id, stats)
    folium.LayerControl(collapsed=False, position="topright").add_to(f_map)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    f_map.save(str(out_html))
    print(f"  Saved interactive Folium HTML -> {out_html}")

    # Optional static high-res route comparison PNG for PDF report
    if out_png is not None:
        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        for seg in cand_segs:
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="#cbd5e1", linewidth=0.9, alpha=0.7, zorder=1)
        if ref_route:
            rr = np.asarray(ref_route)
            ax.plot(rr[:, 1], rr[:, 0], color="#F59E0B", linewidth=4.8, alpha=0.75, label="Reference Route (.route)", zorder=2)
        if clf_route:
            cr = np.asarray(clf_route)
            ax.plot(cr[:, 1], cr[:, 0], color="#DC2626", linewidth=1.6, linestyle="--", alpha=0.75, label="Classifier-Only (XGBoost)", zorder=3)
        if hmm_route:
            hr = np.asarray(hmm_route)
            ax.plot(hr[:, 1], hr[:, 0], color="#059669", linewidth=2.6, alpha=0.95, label="Predicted Route (XGBoost + HMM/Viterbi)", zorder=4)
            ax.scatter([hr[0, 1]], [hr[0, 0]], color="#16a34a", s=70, marker="^", edgecolor="black", zorder=6, label="Start")
            ax.scatter([hr[-1, 1]], [hr[-1, 0]], color="#dc2626", s=70, marker="s", edgecolor="black", zorder=6, label="End")
        sg = np.asarray(sampled_gps)
        ax.scatter(sg[:, 1], sg[:, 0], color="#2563EB", s=10, alpha=0.65, label="Raw Noisy GPS Observations", zorder=5)
        ax.set_xlabel("Longitude (°)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Latitude (°)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"PathSnap Geographical Route Comparison — student_test Trip {trip_id}\n"
            f"Clf Top-1={stats['classifier_top_1_accuracy']*100:.2f}% -> HMM Top-1={stats['top_1_accuracy']*100:.2f}% | "
            f"Clf Cont={stats['classifier_continuity']*100:.2f}% -> HMM Cont={stats['route_continuity']*100:.2f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved static route map PNG   -> {out_png}")

    return stats


def main() -> None:
    test_root = _resolve_student_test_dir()
    model = joblib.load(BASE_DIR / "artifacts" / "final_model.joblib")
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "set_params") and "device" in inner.get_params():
            inner.set_params(device="cpu", n_jobs=-1)

    meta_path = BASE_DIR / "artifacts" / "model_metadata.json"
    feature_cols = list(FEATURE_COLUMNS)
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            feature_cols = list(json.load(f).get("feature_cols", FEATURE_COLUMNS))

    hmm_cfg_path = BASE_DIR / "artifacts" / "hmm_config.json"
    hmm_cfg = HMMConfig()
    if hmm_cfg_path.exists():
        with hmm_cfg_path.open("r", encoding="utf-8") as f:
            hmm_cfg = HMMConfig.from_dict(json.load(f))
    decoder = HMMRouteDecoder(hmm_cfg)

    out_dir = BASE_DIR / "results" / "plots"
    plot_dir = BASE_DIR / "PLOT"
    for tid in ("00000000", "00000007"):
        if (test_root / tid).exists():
            generate_trip_folium_map(
                trip_id=tid,
                test_root=test_root,
                model=model,
                feature_cols=feature_cols,
                decoder=decoder,
                out_html=out_dir / f"folium_map_trip_{tid}.html",
                out_png=plot_dir / f"route_map_trip_{tid}.png",
            )


if __name__ == "__main__":
    main()
