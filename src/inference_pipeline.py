"""Fast, zero-leakage single-trip inference feature extractor & server-side evaluator (Member 3).

Strict separation of concerns:
  1. `extract_trip_candidates()` reads ONLY `.track`, `.nodes`, `.arcs` — NEVER `.route`.
     Produces the exact 12 feature columns trained into `artifacts/final_model.joblib`.
  2. `evaluate_against_reference_server_side()` runs ONLY after classifier & HMM predictions
     are complete, keeping ground-truth `.route` labels strictly server-side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from src.hmm_decoder import RoadNetworkGraph

FEATURE_COLUMNS: List[str] = [
    "distance_to_arc_m",
    "signed_lateral_m",
    "distance_to_arc_midpoint_m",
    "distance_to_nearest_endpoint_m",
    "arc_length_m",
    "gps_heading_sin_delta",
    "gps_heading_cos_delta",
    "gps_speed_mps",
    "gps_step_distance_m",
    "gps_time_delta_s",
    "gps_turn_angle_rad",
    "point_to_arc_along_fraction",
]


@dataclass
class TripInferenceContext:
    trip_id: str
    crs_epsg: int
    track_lonlat: np.ndarray        # (T, 2) [lon, lat]
    timestamps_s: np.ndarray        # (T,)
    gx: np.ndarray                  # (T,) projected UTM x
    gy: np.ndarray                  # (T,) projected UTM y
    nodes_lonlat: np.ndarray        # (N, 2) [lon, lat]
    px: np.ndarray                  # (N,) projected UTM x
    py: np.ndarray                  # (N,) projected UTM y
    arcs_uv: np.ndarray             # (E, 2) [from_node, to_node]
    candidates_df: pd.DataFrame     # candidate rows + 12 features (NO route / NO label)
    graph: RoadNetworkGraph


def _choose_utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def _point_segment_features(
    px: float,
    py: float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    frac = np.divide(
        (px - ax) * dx + (py - ay) * dy,
        length2,
        out=np.zeros_like(dx),
        where=length2 > 0,
    )
    frac = np.clip(frac, 0.0, 1.0)
    qx = ax + frac * dx
    qy = ay + frac * dy
    offx = px - qx
    offy = py - qy
    distance = np.hypot(offx, offy)
    length = np.sqrt(length2)
    signed = np.divide(
        dx * offy - dy * offx,
        length,
        out=np.zeros_like(dx),
        where=length > 0,
    )
    return distance, frac, length, signed


def _make_track_context(
    timestamps_s: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> Dict[str, np.ndarray]:
    n = len(gx)
    dt = np.r_[np.nan, np.diff(timestamps_s)] if n > 1 else np.array([np.nan])
    step = np.r_[np.nan, np.hypot(np.diff(gx), np.diff(gy))] if n > 1 else np.array([np.nan])
    heading = (
        np.arctan2(np.r_[np.nan, np.diff(gy)], np.r_[np.nan, np.diff(gx)])
        if n > 1
        else np.array([np.nan])
    )
    speed = np.divide(
        step, dt, out=np.full(n, np.nan), where=(dt > 0) & np.isfinite(dt)
    )
    turn = np.full(n, np.nan)
    if n > 2:
        turn[2:] = (heading[2:] - heading[1:-1] + np.pi) % (2.0 * np.pi) - np.pi
    return {"dt": dt, "step": step, "heading": heading, "speed": speed, "turn": turn}


def extract_trip_candidates(
    trip_dir: Path,
    trip_id: str,
    radius_m: float = 80.0,
    max_candidates: int = 40,
) -> TripInferenceContext:
    """Load `.track`, `.nodes`, `.arcs` ONLY and extract the 12 features per candidate arc."""
    track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    track_lonlat = track[:, :2]
    timestamps_s = track[:, 2] if track.shape[1] >= 3 else np.arange(len(track), dtype=np.float64)
    nodes_lonlat = nodes[:, :2]
    arcs_uv = arcs[:, :2]

    lon_med = float(np.nanmedian(track_lonlat[:, 0]))
    lat_med = float(np.nanmedian(track_lonlat[:, 1]))
    crs_epsg = _choose_utm_epsg(lon_med, lat_med)
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(crs_epsg), always_xy=True)

    gx, gy = tf.transform(track_lonlat[:, 0], track_lonlat[:, 1])
    px, py = tf.transform(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    src_all = arcs_uv[:, 0]
    dst_all = arcs_uv[:, 1]
    valid_arc = (
        (src_all >= 0) & (dst_all >= 0) & (src_all < len(px)) & (dst_all < len(px))
    )
    arc_ids_valid = np.flatnonzero(valid_arc)
    src = src_all[valid_arc]
    dst = dst_all[valid_arc]

    ax = px[src]
    ay = py[src]
    bx = px[dst]
    by = py[dst]
    mids_x = 0.5 * (ax + bx)
    mids_y = 0.5 * (ay + by)
    mids = np.column_stack((mids_x, mids_y))
    road_heading_all = np.arctan2(by - ay, bx - ax)

    tree = cKDTree(mids)
    context = _make_track_context(timestamps_s, gx, gy)

    k = min(max_candidates, len(mids))
    # Batch cKDTree query across ALL GPS points simultaneously for blazing speed
    _, near_all = tree.query(np.column_stack((gx, gy)), k=k)
    if near_all.ndim == 1:
        near_all = near_all[:, None]

    obs_idx_chunks: List[np.ndarray] = []
    arc_id_chunks: List[np.ndarray] = []
    from_chunks: List[np.ndarray] = []
    to_chunks: List[np.ndarray] = []
    feat_chunks: Dict[str, List[np.ndarray]] = {col: [] for col in FEATURE_COLUMNS}

    for i in range(len(gx)):
        x = gx[i]
        y = gy[i]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        near = near_all[i]
        d, frac, length, signed = _point_segment_features(
            x, y, ax[near], ay[near], bx[near], by[near]
        )
        keep = d <= radius_m
        if not np.any(keep):
            keep[int(np.argmin(d))] = True

        # Preserve exact column mapping from Member 1's training data_pipeline.py (line 204)
        # so inference features match artifacts/final_model.joblib bit-for-bit.
        near_k = near[keep]
        d_k = d[keep]
        frac_col_k = length[keep]
        length_col_k = frac[keep]
        signed_k = signed[keep]

        m = len(near_k)
        obs_h = context["heading"][i]
        if math.isfinite(obs_h):
            delta = (obs_h - road_heading_all[near_k] + np.pi) % (2.0 * np.pi) - np.pi
            sin_d = np.sin(delta)
            cos_d = np.cos(delta)
        else:
            sin_d = np.full(m, np.nan)
            cos_d = np.full(m, np.nan)

        dist_mid = np.hypot(x - mids_x[near_k], y - mids_y[near_k])
        dist_ep = np.minimum(
            np.hypot(x - ax[near_k], y - ay[near_k]),
            np.hypot(x - bx[near_k], y - by[near_k]),
        )

        obs_idx_chunks.append(np.full(m, i, dtype=np.int32))
        arc_id_chunks.append(arc_ids_valid[near_k].astype(np.int32))
        from_chunks.append(src[near_k].astype(np.int32))
        to_chunks.append(dst[near_k].astype(np.int32))

        feat_chunks["distance_to_arc_m"].append(d_k.astype(np.float32))
        feat_chunks["signed_lateral_m"].append(signed_k.astype(np.float32))
        feat_chunks["distance_to_arc_midpoint_m"].append(dist_mid.astype(np.float32))
        feat_chunks["distance_to_nearest_endpoint_m"].append(dist_ep.astype(np.float32))
        feat_chunks["arc_length_m"].append(length_col_k.astype(np.float32))
        feat_chunks["gps_heading_sin_delta"].append(sin_d.astype(np.float32))
        feat_chunks["gps_heading_cos_delta"].append(cos_d.astype(np.float32))
        feat_chunks["gps_speed_mps"].append(np.full(m, context["speed"][i], dtype=np.float32))
        feat_chunks["gps_step_distance_m"].append(np.full(m, context["step"][i], dtype=np.float32))
        feat_chunks["gps_time_delta_s"].append(np.full(m, context["dt"][i], dtype=np.float32))
        feat_chunks["gps_turn_angle_rad"].append(np.full(m, context["turn"][i], dtype=np.float32))
        feat_chunks["point_to_arc_along_fraction"].append(frac_col_k.astype(np.float32))

    data_dict: Dict[str, Any] = {
        "trip_id": trip_id,
        "obs_idx": np.concatenate(obs_idx_chunks) if obs_idx_chunks else np.array([], dtype=np.int32),
        "arc_id": np.concatenate(arc_id_chunks) if arc_id_chunks else np.array([], dtype=np.int32),
        "from_node": np.concatenate(from_chunks) if from_chunks else np.array([], dtype=np.int32),
        "to_node": np.concatenate(to_chunks) if to_chunks else np.array([], dtype=np.int32),
    }
    for col in FEATURE_COLUMNS:
        data_dict[col] = (
            np.concatenate(feat_chunks[col])
            if feat_chunks[col]
            else np.array([], dtype=np.float32)
        )

    candidates_df = pd.DataFrame(data_dict)
    graph = RoadNetworkGraph(px, py, arcs_uv[:, 0], arcs_uv[:, 1])

    return TripInferenceContext(
        trip_id=trip_id,
        crs_epsg=crs_epsg,
        track_lonlat=track_lonlat,
        timestamps_s=timestamps_s,
        gx=gx,
        gy=gy,
        nodes_lonlat=nodes_lonlat,
        px=px,
        py=py,
        arcs_uv=arcs_uv,
        candidates_df=candidates_df,
        graph=graph,
    )


def _continuity_rate(decoded_df: pd.DataFrame) -> float:
    if len(decoded_df) < 2:
        return 1.0
    ordered = decoded_df.sort_values("obs_idx")
    arcs = ordered["arc_id"].to_numpy(dtype=np.int64)
    from_n = ordered["from_node"].to_numpy(dtype=np.int64)
    to_n = ordered["to_node"].to_numpy(dtype=np.int64)
    same_arc = arcs[:-1] == arcs[1:]
    direct_edge = (to_n[:-1] >= 0) & (to_n[:-1] == from_n[1:])
    return float(np.mean(same_arc | direct_edge))


def evaluate_against_reference_server_side(
    ctx: TripInferenceContext,
    clf_top1_df: pd.DataFrame,
    hmm_decoded_df: pd.DataFrame,
    route_file: Path,
) -> Dict[str, Optional[float | int]]:
    """Compute real accuracy & continuity strictly server-side using internal `.route`."""
    hmm_cont = _continuity_rate(hmm_decoded_df)
    clf_cont = _continuity_rate(clf_top1_df)
    total_obs = int(len(ctx.gx))

    if not route_file.exists():
        return {
            "top_1_accuracy": 0.0,
            "classifier_top_1_accuracy": 0.0,
            "route_continuity": hmm_cont,
            "classifier_continuity": clf_cont,
            "candidate_miss_rate": 0.0,
            "total_observations": total_obs,
            "reference_available": False,
        }

    route_arcs = np.loadtxt(route_file, dtype=np.int64).ravel()
    valid_r = route_arcs[(route_arcs >= 0) & (route_arcs < len(ctx.arcs_uv))]
    route_unique = np.unique(valid_r)

    if len(route_unique) == 0:
        return {
            "top_1_accuracy": 0.0,
            "classifier_top_1_accuracy": 0.0,
            "route_continuity": hmm_cont,
            "classifier_continuity": clf_cont,
            "candidate_miss_rate": 0.0,
            "total_observations": total_obs,
            "reference_available": False,
        }

    r_src = ctx.arcs_uv[route_unique, 0]
    r_dst = ctx.arcs_uv[route_unique, 1]
    r_ax, r_ay = ctx.px[r_src], ctx.py[r_src]
    r_bx, r_by = ctx.px[r_dst], ctx.py[r_dst]
    r_mids = np.column_stack((0.5 * (r_ax + r_bx), 0.5 * (r_ay + r_by)))
    r_tree = cKDTree(r_mids)

    k_r = min(16, len(route_unique))
    _, r_near_all = r_tree.query(np.column_stack((ctx.gx, ctx.gy)), k=k_r)
    if r_near_all.ndim == 1:
        r_near_all = r_near_all[:, None]

    # Geometric reference arc per GPS observation (same protocol as Member 1)
    ref_arc_per_obs = np.full(total_obs, -1, dtype=np.int64)
    for i in range(total_obs):
        if not (math.isfinite(ctx.gx[i]) and math.isfinite(ctx.gy[i])):
            continue
        ridx = r_near_all[i]
        rd, _, _, _ = _point_segment_features(
            ctx.gx[i], ctx.gy[i], r_ax[ridx], r_ay[ridx], r_bx[ridx], r_by[ridx]
        )
        ref_arc_per_obs[i] = int(route_unique[ridx[int(np.argmin(rd))]])

    # Candidate miss rate
    cand_arcs_by_obs = ctx.candidates_df.groupby("obs_idx")["arc_id"].apply(set).to_dict()
    misses = 0
    for i in range(total_obs):
        if ref_arc_per_obs[i] >= 0 and int(ref_arc_per_obs[i]) not in cand_arcs_by_obs.get(i, set()):
            misses += 1
    miss_rate = float(misses / max(total_obs, 1))

    # Observation-level Top-1 accuracy (Overall)
    clf_map = dict(zip(clf_top1_df["obs_idx"].to_numpy(int), clf_top1_df["arc_id"].to_numpy(int)))
    hmm_map = dict(zip(hmm_decoded_df["obs_idx"].to_numpy(int), hmm_decoded_df["arc_id"].to_numpy(int)))

    clf_correct = sum(
        1 for i in range(total_obs) if ref_arc_per_obs[i] >= 0 and clf_map.get(i, -2) == ref_arc_per_obs[i]
    )
    hmm_correct = sum(
        1 for i in range(total_obs) if ref_arc_per_obs[i] >= 0 and hmm_map.get(i, -2) == ref_arc_per_obs[i]
    )

    # Sequence-level unique arc F1
    ref_set = set(int(a) for a in route_unique)
    hmm_set = set(int(a) for a in hmm_decoded_df["arc_id"].to_numpy(int) if a >= 0)
    inter = len(ref_set.intersection(hmm_set))
    seq_prec = inter / len(hmm_set) if hmm_set else 0.0
    seq_rec = inter / len(ref_set) if ref_set else 0.0
    seq_f1 = (2 * seq_prec * seq_rec / (seq_prec + seq_rec)) if (seq_prec + seq_rec) > 0 else 0.0

    return {
        "top_1_accuracy": float(hmm_correct / max(total_obs, 1)),
        "classifier_top_1_accuracy": float(clf_correct / max(total_obs, 1)),
        "route_continuity": hmm_cont,
        "classifier_continuity": clf_cont,
        "candidate_miss_rate": miss_rate,
        "total_observations": total_obs,
        "sequence_arc_f1": float(seq_f1),
        "reference_available": True,
    }


def arcs_to_latlon_polyline(
    decoded_df: pd.DataFrame,
    nodes_lonlat: np.ndarray,
    arcs_uv: np.ndarray,
    max_points: int = 1200,
) -> List[List[float]]:
    """Convert ordered predicted arc_id sequence into [lat, lon] polyline for Folium/Leaflet."""
    if decoded_df.empty:
        return []
    arcs = decoded_df.sort_values("obs_idx")["arc_id"].to_numpy(dtype=np.int64)
    # Deduplicate consecutive repeated arcs so Leaflet draws crisp road paths
    dedup = [int(arcs[0])]
    for a in arcs[1:]:
        ai = int(a)
        if ai != dedup[-1]:
            dedup.append(ai)

    pts: List[List[float]] = []
    n_nodes = len(nodes_lonlat)
    for ai in dedup:
        if 0 <= ai < len(arcs_uv):
            u, v = int(arcs_uv[ai, 0]), int(arcs_uv[ai, 1])
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                u_pt = [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])]
                v_pt = [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])]
                if not pts or pts[-1] != u_pt:
                    pts.append(u_pt)
                pts.append(v_pt)

    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    return pts


def extract_candidate_segments_latlon(
    candidates_df: pd.DataFrame,
    nodes_lonlat: np.ndarray,
    arcs_uv: np.ndarray,
    max_segments: int = 250,
) -> List[List[List[float]]]:
    """Return representative nearby candidate road segments [[lat1, lon1], [lat2, lon2]]."""
    if candidates_df.empty:
        return []
    unique_arcs = candidates_df["arc_id"].drop_duplicates().to_numpy(dtype=np.int64)
    if len(unique_arcs) > max_segments:
        step = max(1, len(unique_arcs) // max_segments)
        unique_arcs = unique_arcs[::step][:max_segments]

    segs: List[List[List[float]]] = []
    n_nodes = len(nodes_lonlat)
    for ai in unique_arcs:
        if 0 <= ai < len(arcs_uv):
            u, v = int(arcs_uv[ai, 0]), int(arcs_uv[ai, 1])
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                segs.append(
                    [
                        [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])],
                        [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])],
                    ]
                )
    return segs
