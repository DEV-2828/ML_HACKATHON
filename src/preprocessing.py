"""
PathSnap Preprocessing & Candidate Generation Pipeline (Member 1)
Reads raw .track, .nodes, .arcs, .route files.
Extracts candidate road segments using KD-Tree spatial indexing and computes rich ML features:
- dist_m: perpendicular distance from GPS observation to candidate road segment
- heading_diff: angular difference between GPS velocity heading and road segment direction
- speed_mps: vehicle speed computed between consecutive GPS fixes
- seg_len: candidate road segment physical length in meters
- proj_ratio: normalized projection scalar [0, 1] of GPS point along segment
- label: 1 if candidate arc matches ground-truth route arc, 0 otherwise
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes compass bearing in degrees [0, 360) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def point_to_segment_projection(
    p_lat: float, p_lon: float,
    u_lat: float, u_lon: float,
    v_lat: float, v_lon: float
) -> Tuple[float, float, float]:
    """
    Projects GPS point P onto road segment UV (approximated via flat local projection).
    Returns (perpendicular_dist_m, proj_ratio, seg_len_m).
    """
    # Convert lat/lon degrees to local meters around segment midpoint
    mid_lat = (u_lat + v_lat) / 2.0
    meters_per_lat = 111139.0
    meters_per_lon = 111139.0 * math.cos(math.radians(mid_lat))

    px = (p_lon - u_lon) * meters_per_lon
    py = (p_lat - u_lat) * meters_per_lat

    vx = (v_lon - u_lon) * meters_per_lon
    vy = (v_lat - u_lat) * meters_per_lat

    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq < 1e-6:
        dist = math.sqrt(px * px + py * py)
        return dist, 0.0, 0.0

    seg_len = math.sqrt(seg_len_sq)
    # Dot product projection
    t = max(0.0, min(1.0, (px * vx + py * vy) / seg_len_sq))
    proj_x = t * vx
    proj_y = t * vy

    dx = px - proj_x
    dy = py - proj_y
    dist_m = math.sqrt(dx * dx + dy * dy)
    return dist_m, t, seg_len


def process_trip_candidates(
    trip_dir: Path,
    trip_id: str,
    k_candidates: int = 5,
    search_radius_m: float = 65.0,
    subsample_obs: int = 1
) -> Tuple[List[dict], int, int]:
    """
    Extracts candidate road segments and features for a single trip.
    Returns (candidate_rows, total_observations, true_arcs_found).
    """
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    if not (track_path.exists() and nodes_path.exists() and arcs_path.exists()):
        return [], 0, 0

    # 1. Parse tracks: lon, lat, timestamp
    tracks_raw = np.loadtxt(track_path)
    if tracks_raw.ndim == 1:
        tracks_raw = tracks_raw.reshape(1, -1)
    total_obs = len(tracks_raw)

    # 2. Parse nodes: lon, lat
    nodes_raw = np.loadtxt(nodes_path)
    if nodes_raw.ndim == 1:
        nodes_raw = nodes_raw.reshape(1, -1)

    # 3. Parse arcs: from_node, to_node
    arcs_raw = np.loadtxt(arcs_path, dtype=int)
    if arcs_raw.ndim == 1:
        arcs_raw = arcs_raw.reshape(1, -1)

    # 4. Parse route if available
    ground_truth_arcs = set()
    if route_path.exists():
        r_raw = np.loadtxt(route_path, dtype=int)
        ground_truth_arcs = set(r_raw.tolist() if r_raw.ndim > 0 else [int(r_raw)])

    # Map node -> connected arcs
    node_to_arcs: Dict[int, List[int]] = {}
    for arc_id, (u, v) in enumerate(arcs_raw):
        node_to_arcs.setdefault(u, []).append(arc_id)
        node_to_arcs.setdefault(v, []).append(arc_id)

    # Build KD-Tree on projected nodes (meters)
    ref_lat = np.mean(nodes_raw[:, 1])
    m_per_lat = 111139.0
    m_per_lon = 111139.0 * math.cos(math.radians(ref_lat))
    nodes_xy = np.column_stack([
        nodes_raw[:, 0] * m_per_lon,
        nodes_raw[:, 1] * m_per_lat
    ])
    kdtree = KDTree(nodes_xy)

    rows = []
    found_positive_count = 0

    # Subsample if requested for faster training/validation
    obs_indices = list(range(0, total_obs, subsample_obs))

    for idx, obs_i in enumerate(obs_indices):
        p_lon, p_lat, p_t = tracks_raw[obs_i]

        # Calculate speed and heading from consecutive GPS points
        if obs_i < total_obs - 1:
            next_lon, next_lat, next_t = tracks_raw[obs_i + 1]
            dt = max(0.1, next_t - p_t)
            step_dist = haversine_distance(p_lat, p_lon, next_lat, next_lon)
            speed_mps = step_dist / dt
            gps_heading = compute_bearing(p_lat, p_lon, next_lat, next_lon)
        elif obs_i > 0:
            prev_lon, prev_lat, prev_t = tracks_raw[obs_i - 1]
            dt = max(0.1, p_t - prev_t)
            step_dist = haversine_distance(prev_lat, p_lon, p_lat, p_lon)
            speed_mps = step_dist / dt
            gps_heading = compute_bearing(prev_lat, prev_lon, p_lat, p_lon)
        else:
            speed_mps = 0.0
            gps_heading = 0.0

        # Query KD-Tree for nearby nodes
        p_xy = np.array([p_lon * m_per_lon, p_lat * m_per_lat])
        nearby_node_indices = kdtree.query_ball_point(p_xy, r=search_radius_m)

        # Collect candidate arcs
        cand_arcs = set()
        for n_idx in nearby_node_indices:
            cand_arcs.update(node_to_arcs.get(n_idx, []))

        if not cand_arcs:
            # Fallback to 3 nearest nodes if radius was too small
            _, nearest_nodes = kdtree.query(p_xy, k=min(3, len(nodes_raw)))
            if np.isscalar(nearest_nodes):
                nearest_nodes = [nearest_nodes]
            for n_idx in nearest_nodes:
                cand_arcs.update(node_to_arcs.get(int(n_idx), []))

        # Score candidate arcs by perpendicular distance
        candidates_scored = []
        for arc_id in cand_arcs:
            u, v = arcs_raw[arc_id]
            u_lon, u_lat = nodes_raw[u]
            v_lon, v_lat = nodes_raw[v]

            dist_m, proj_t, seg_len = point_to_segment_projection(
                p_lat, p_lon, u_lat, u_lon, v_lat, v_lon
            )
            arc_heading = compute_bearing(u_lat, u_lon, v_lat, v_lon)
            h_diff = abs(gps_heading - arc_heading)
            h_diff = min(h_diff, 360.0 - h_diff)

            is_true = 1 if arc_id in ground_truth_arcs else 0
            candidates_scored.append({
                "trip_id": trip_id,
                "obs_idx": obs_i,
                "arc_id": int(arc_id),
                "from_node": int(u),
                "to_node": int(v),
                "dist_m": dist_m,
                "heading_diff": h_diff,
                "speed_mps": speed_mps,
                "seg_len": seg_len,
                "proj_ratio": proj_t,
                "label": is_true
            })

        # Sort by distance and pick top K
        candidates_scored.sort(key=lambda x: x["dist_m"])
        top_k = candidates_scored[:k_candidates]

        has_pos = any(c["label"] == 1 for c in top_k)
        if has_pos:
            found_positive_count += 1

        rows.extend(top_k)

    return rows, len(obs_indices), found_positive_count


def generate_candidate_dataset(
    data_dir: Path,
    out_parquet: Path,
    max_trips: Optional[int] = None,
    k_candidates: int = 5,
    subsample_obs: int = 1
) -> Dict[str, float]:
    """Generates candidate parquet file across all trips in data_dir."""
    trip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if max_trips:
        trip_dirs = trip_dirs[:max_trips]

    all_rows = []
    total_obs_all = 0
    total_pos_all = 0

    print(f"Generating candidates for {len(trip_dirs)} trips from {data_dir.name}...")
    t0 = time.perf_counter()

    for idx, t_dir in enumerate(trip_dirs):
        trip_id = t_dir.name
        rows, n_obs, n_pos = process_trip_candidates(
            t_dir, trip_id, k_candidates=k_candidates, subsample_obs=subsample_obs
        )
        all_rows.extend(rows)
        total_obs_all += n_obs
        total_pos_all += n_pos
        if (idx + 1) % 15 == 0 or (idx + 1) == len(trip_dirs):
            print(f"  Processed {idx + 1}/{len(trip_dirs)} trips ({len(all_rows):,} candidates)...")

    df = pd.DataFrame(all_rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    miss_rate = 1.0 - (total_pos_all / max(1, total_obs_all))
    elapsed = time.perf_counter() - t0

    stats = {
        "n_trips": len(trip_dirs),
        "total_gps_observations": total_obs_all,
        "total_candidates": len(df),
        "candidate_miss_rate": round(miss_rate, 4),
        "positive_rate": round(float(df["label"].mean()), 4),
        "elapsed_seconds": round(elapsed, 2)
    }
    print(f"Done! Saved {out_parquet.name} ({len(df):,} rows) in {elapsed:.1f}s. Miss Rate: {miss_rate*100:.2f}%")
    return stats
