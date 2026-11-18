"""PathSnap data and geo pipeline: inspect trips, create candidates/features, and EDA.

Run from the project folder:
  python data_geo/data_pipeline.py --data-root "C:/.../Student folder/Student folder"

Reference-route supervision is an explicit geometric proxy: for each GPS point,
the closest road arc appearing in that trip's supplied .route sequence is used
as its reference arc. Candidate misses are recorded and are not silently
converted into successful labels. The route files should be treated as supplied
reference data, not independently verified ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS, Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


RAW_COLUMNS = {
    ".track": ["longitude", "latitude", "timestamp_s"],
    ".nodes": ["longitude", "latitude"],
    ".arcs": ["from_node_id", "to_node_id"],
    ".route": ["arc_id"],
}
FEATURE_COLUMNS = [
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
class Trip:
    trip_id: str
    partition: str
    directory: Path
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray | None
    crs_epsg: int


def read_numeric_file(path: Path, suffix: str) -> pd.DataFrame:
    names = RAW_COLUMNS[suffix]
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=names, comment="#", engine="python")
    if frame.empty:
        return frame.astype(float)
    for col in names:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def choose_utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def discover_trips(data_root: Path) -> list[tuple[str, str, Path]]:
    found = []
    for partition in ("train", "student_test"):
        part_dir = data_root / partition
        if not part_dir.is_dir():
            continue
        for directory in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            if re.fullmatch(r"\d+", directory.name):
                found.append((partition, directory.name, directory))
    return found


def load_trip(partition: str, trip_id: str, directory: Path) -> Trip:
    track = read_numeric_file(directory / f"{trip_id}.track", ".track")
    nodes = read_numeric_file(directory / f"{trip_id}.nodes", ".nodes")
    arcs = read_numeric_file(directory / f"{trip_id}.arcs", ".arcs")
    route_path = directory / f"{trip_id}.route"
    route = read_numeric_file(route_path, ".route")["arc_id"].dropna().astype(int).to_numpy() if route_path.exists() else None
    valid = track[["longitude", "latitude"]].dropna()
    lon = float(valid["longitude"].median()) if len(valid) else float(nodes["longitude"].median())
    lat = float(valid["latitude"].median()) if len(valid) else float(nodes["latitude"].median())
    return Trip(trip_id, partition, directory, track, nodes, arcs, route, choose_utm_epsg(lon, lat))


def project_trip(trip: Trip) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(trip.crs_epsg), always_xy=True)
    tx = trip.track["longitude"].to_numpy(float)
    ty = trip.track["latitude"].to_numpy(float)
    nx = trip.nodes["longitude"].to_numpy(float)
    ny = trip.nodes["latitude"].to_numpy(float)
    gx, gy = transformer.transform(tx, ty)
    px, py = transformer.transform(nx, ny)
    return np.asarray(gx), np.asarray(gy), np.asarray(px), np.asarray(py)


def graph_stats(node_count: int, from_ids: np.ndarray, to_ids: np.ndarray) -> dict[str, int | float]:
    valid = (from_ids >= 0) & (to_ids >= 0) & (from_ids < node_count) & (to_ids < node_count)
    src, dst = from_ids[valid].astype(int), to_ids[valid].astype(int)
    degree = np.bincount(np.r_[src, dst], minlength=node_count) if node_count else np.array([], dtype=int)
    if len(src) and node_count:
        graph = coo_matrix((np.ones(2 * len(src)), (np.r_[src, dst], np.r_[dst, src])), shape=(node_count, node_count)).tocsr()
        component_count, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels, minlength=component_count)
        largest = int(sizes.max()) if len(sizes) else 0
    else:
        component_count, largest = node_count, 0
    return {
        "invalid_arc_endpoints": int((~valid).sum()),
        "isolated_node_count": int((degree == 0).sum()),
        "weak_component_count": int(component_count),
        "largest_component_nodes": largest,
    }


def point_segment_features(px: float, py: float, ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    frac = np.divide((px - ax) * dx + (py - ay) * dy, length2, out=np.zeros_like(dx), where=length2 > 0)
    frac = np.clip(frac, 0.0, 1.0)
    qx, qy = ax + frac * dx, ay + frac * dy
    offx, offy = px - qx, py - qy
    distance = np.hypot(offx, offy)
    length = np.sqrt(length2)
    signed = np.divide(dx * offy - dy * offx, length, out=np.zeros_like(dx), where=length > 0)
    return distance, frac, length, signed


def make_track_context(track: pd.DataFrame, gx: np.ndarray, gy: np.ndarray) -> dict[str, np.ndarray]:
    n = len(track)
    t = track["timestamp_s"].to_numpy(float)
    dt = np.r_[np.nan, np.diff(t)]
    step = np.r_[np.nan, np.hypot(np.diff(gx), np.diff(gy))]
    heading = np.arctan2(np.r_[np.nan, np.diff(gy)], np.r_[np.nan, np.diff(gx)])
    speed = np.divide(step, dt, out=np.full(n, np.nan), where=(dt > 0) & np.isfinite(dt))
    turn = np.full(n, np.nan)
    if n > 2:
        d = (heading[2:] - heading[1:-1] + np.pi) % (2 * np.pi) - np.pi
        turn[2:] = d
    return {"dt": dt, "step": step, "heading": heading, "speed": speed, "turn": turn}


def candidate_features(trip: Trip, gx: np.ndarray, gy: np.ndarray, px: np.ndarray, py: np.ndarray, radius_m: float, max_candidates: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    arc_rows = trip.arcs.dropna().copy()
    src = arc_rows["from_node_id"].astype(int).to_numpy()
    dst = arc_rows["to_node_id"].astype(int).to_numpy()
    valid_arc = (src >= 0) & (dst >= 0) & (src < len(px)) & (dst < len(px))
    arc_ids = arc_rows.index.to_numpy(dtype=int)[valid_arc]
    src, dst = src[valid_arc], dst[valid_arc]
    ax, ay, bx, by = px[src], py[src], px[dst], py[dst]
    mids = np.column_stack(((ax + bx) / 2, (ay + by) / 2))
    tree = cKDTree(mids) if len(mids) else None
    route = trip.route
    route_unique = np.unique(route[(route >= 0) & (route < len(trip.arcs))]) if route is not None else np.array([], dtype=int)
    route_positions = np.searchsorted(arc_ids, route_unique)
    route_valid = (route_positions < len(arc_ids))
    if len(arc_ids):
        route_valid &= arc_ids[np.minimum(route_positions, len(arc_ids) - 1)] == route_unique
    route_pos = route_positions[route_valid]
    route_ids = route_unique[route_valid]
    route_tree = cKDTree(mids[route_pos]) if len(route_pos) else None
    context = make_track_context(trip.track, gx, gy)
    rows: list[dict[str, Any]] = []
    misses = 0
    ref_distances = []
    candidate_counts = []
    for i, (x, y) in enumerate(zip(gx, gy)):
        if i and i % 100 == 0:
            print(f"    {trip.partition}/{trip.trip_id}: candidates for GPS point {i}/{len(gx)}", flush=True)
        if tree is None or not len(mids):
            continue
        k = min(max_candidates, len(mids))
        _, near = tree.query([x, y], k=k)
        near = np.atleast_1d(near).astype(int)
        d, frac, length, signed = point_segment_features(x, y, ax[near], ay[near], bx[near], by[near])
        keep = d <= radius_m
        if not np.any(keep):
            keep[np.argmin(d)] = True  # Ensure every observation has at least one candidate.
        near, d, frac, length, signed = near[keep], d[keep], length[keep], frac[keep], signed[keep]
        obs_heading = context["heading"][i]
        road_heading = np.arctan2(by[near] - ay[near], bx[near] - ax[near])
        delta = (obs_heading - road_heading + np.pi) % (2 * np.pi) - np.pi if np.isfinite(obs_heading) else np.full(len(near), np.nan)
        arc_ids_here = arc_ids[near]
        reference_arc = None
        if route_tree is not None:
            _, ridx = route_tree.query([x, y], k=min(16, len(route_pos)))
            ridx = np.atleast_1d(ridx).astype(int)
            rd, _, _, _ = point_segment_features(x, y, ax[route_pos[ridx]], ay[route_pos[ridx]], bx[route_pos[ridx]], by[route_pos[ridx]])
            best = int(np.argmin(rd))
            reference_arc = int(route_ids[ridx[best]])
            ref_distances.append(float(rd[best]))
        has_reference_candidate = reference_arc is not None and reference_arc in arc_ids_here
        if reference_arc is not None and not has_reference_candidate:
            misses += 1
        candidate_counts.append(len(near))
        for j, arc_id in enumerate(arc_ids_here):
            delta_j = delta[j]
            rows.append({
                "trip_id": trip.trip_id,
                "partition": trip.partition,
                "obs_idx": i,
                "arc_id": int(arc_id),
                "reference_arc_id": reference_arc,
                "label_available": bool(has_reference_candidate),
                "is_reference_candidate": bool(has_reference_candidate and int(arc_id) == reference_arc),
                "crs_epsg": trip.crs_epsg,
                "distance_to_arc_m": float(d[j]),
                "signed_lateral_m": float(signed[j]),
                "distance_to_arc_midpoint_m": float(math.hypot(x - mids[near[j], 0], y - mids[near[j], 1])),
                "distance_to_nearest_endpoint_m": float(min(math.hypot(x - ax[near[j]], y - ay[near[j]]), math.hypot(x - bx[near[j]], y - by[near[j]]))),
                "arc_length_m": float(length[j]),
                "gps_heading_sin_delta": float(math.sin(delta_j)) if np.isfinite(delta_j) else np.nan,
                "gps_heading_cos_delta": float(math.cos(delta_j)) if np.isfinite(delta_j) else np.nan,
                "gps_speed_mps": float(context["speed"][i]) if np.isfinite(context["speed"][i]) else np.nan,
                "gps_step_distance_m": float(context["step"][i]) if np.isfinite(context["step"][i]) else np.nan,
                "gps_time_delta_s": float(context["dt"][i]) if np.isfinite(context["dt"][i]) else np.nan,
                "gps_turn_angle_rad": float(context["turn"][i]) if np.isfinite(context["turn"][i]) else np.nan,
                "point_to_arc_along_fraction": float(frac[j]),
            })
    meta = {
        "reference_points": len(ref_distances),
        "reference_arc_mean_distance_m": float(np.mean(ref_distances)) if ref_distances else None,
        "candidate_misses": misses,
        "candidate_miss_rate": float(misses / len(ref_distances)) if ref_distances else None,
        "mean_candidates_per_observation": float(np.mean(candidate_counts)) if candidate_counts else None,
        "max_candidates_per_observation": int(max(candidate_counts)) if candidate_counts else 0,
        "feature_columns": FEATURE_COLUMNS,
    }
    return pd.DataFrame(rows), meta


def trip_quality(trip: Trip, gx: np.ndarray, gy: np.ndarray, graph: dict[str, Any]) -> dict[str, Any]:
    track = trip.track
    t = track["timestamp_s"].to_numpy(float)
    dt = np.diff(t)
    dxy = np.hypot(np.diff(gx), np.diff(gy))
    valid = np.isfinite(track[["longitude", "latitude", "timestamp_s"]].to_numpy(float)).all(axis=1)
    return {
        "trip_id": trip.trip_id, "partition": trip.partition,
        "gps_rows": int(len(track)), "gps_invalid_rows": int((~valid).sum()),
        "gps_duplicate_coordinates": int(track.duplicated(["longitude", "latitude"]).sum()),
        "timestamp_nonincreasing_steps": int((dt <= 0).sum()),
        "timestamp_gap_over_2s": int((dt > 2).sum()),
        "timestamp_min_step_s": float(np.nanmin(dt)) if len(dt) else None,
        "timestamp_median_step_s": float(np.nanmedian(dt)) if len(dt) else None,
        "timestamp_max_step_s": float(np.nanmax(dt)) if len(dt) else None,
        "track_duration_s": float(t[-1] - t[0]) if len(t) >= 2 else 0.0,
        "track_length_m": float(np.nansum(dxy)),
        "nodes": int(len(trip.nodes)), "arcs": int(len(trip.arcs)),
        "route_arc_rows": int(len(trip.route)) if trip.route is not None else None,
        "route_invalid_arc_ids": int(np.sum((trip.route < 0) | (trip.route >= len(trip.arcs)))) if trip.route is not None else None,
        "crs_epsg": trip.crs_epsg,
        **graph,
    }


def draw_eda(quality: pd.DataFrame, inventory: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for partition, group in quality.groupby("partition"):
        axes[0, 0].hist(group["gps_rows"], bins=12, alpha=0.6, label=partition)
        axes[0, 1].hist(group["track_length_m"], bins=12, alpha=0.6, label=partition)
        axes[1, 0].scatter(group["arcs"], group["weak_component_count"], alpha=0.75, label=partition)
        axes[1, 1].scatter(group["timestamp_max_step_s"], group["timestamp_gap_over_2s"], alpha=0.75, label=partition)
    axes[0, 0].set(title="GPS observations per trip", xlabel="observations", ylabel="trip count")
    axes[0, 1].set(title="Approximate GPS path length", xlabel="metres", ylabel="trip count")
    axes[1, 0].set(title="Road graph connectivity", xlabel="arc rows", ylabel="weak components")
    axes[1, 1].set(title="Timestamp gap check", xlabel="largest step (s)", ylabel="steps > 2 s")
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "dataset_eda.png", dpi=160)
    plt.close(fig)


def create_folium_map(trip: Trip, gx: np.ndarray, gy: np.ndarray, out_file: Path, max_map_edges: int = 3000) -> None:
    track = trip.track
    center = [float(track["latitude"].median()), float(track["longitude"].median())]
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    gps_group = folium.FeatureGroup(name="Observed GPS", show=True)
    gps_coords = track[["latitude", "longitude"]].dropna().to_numpy().tolist()
    folium.PolyLine(gps_coords, color="#2677d8", weight=3, opacity=0.8, tooltip="Observed GPS path").add_to(gps_group)
    for idx in np.linspace(0, max(0, len(gps_coords) - 1), min(30, len(gps_coords)), dtype=int) if gps_coords else []:
        folium.CircleMarker(gps_coords[idx], radius=2, color="#2677d8", fill=True, tooltip=f"GPS point {idx}").add_to(gps_group)
    gps_group.add_to(fmap)

    arcs = trip.arcs.dropna().copy()
    src, dst = arcs["from_node_id"].astype(int).to_numpy(), arcs["to_node_id"].astype(int).to_numpy()
    valid = (src >= 0) & (dst >= 0) & (src < len(trip.nodes)) & (dst < len(trip.nodes))
    all_ids = arcs.index.to_numpy(dtype=int)[valid]
    src, dst = src[valid], dst[valid]
    node_xy = trip.nodes[["longitude", "latitude"]].to_numpy(float)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(trip.crs_epsg), always_xy=True)
    mids_lon = (node_xy[src, 0] + node_xy[dst, 0]) / 2
    mids_lat = (node_xy[src, 1] + node_xy[dst, 1]) / 2
    mx, my = transformer.transform(mids_lon, mids_lat)
    tree = cKDTree(np.column_stack((mx, my))) if len(all_ids) else None
    gx0, gy0 = transformer.transform(track["longitude"].to_numpy(float), track["latitude"].to_numpy(float))
    selected: set[int] = set()
    if tree is not None:
        for x, y in zip(gx0, gy0):
            selected.update(tree.query_ball_point([x, y], r=140.0))
    if len(selected) > max_map_edges:
        selected = set(sorted(selected)[:max_map_edges])
    road_group = folium.FeatureGroup(name="Nearby road network", show=True)
    for j in selected:
        a, b = node_xy[src[j]], node_xy[dst[j]]
        folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="#6a9f58", weight=1, opacity=0.5).add_to(road_group)
    road_group.add_to(fmap)

    if trip.route is not None:
        route_group = folium.FeatureGroup(name="Supplied reference route (approximate reference)", show=True)
        for arc_id in trip.route:
            if arc_id < 0 or arc_id >= len(trip.arcs) or arc_id not in set(all_ids.tolist()):
                continue
            a_id = int(trip.arcs.iloc[int(arc_id), 0]); b_id = int(trip.arcs.iloc[int(arc_id), 1])
            if 0 <= a_id < len(node_xy) and 0 <= b_id < len(node_xy):
                a, b = node_xy[a_id], node_xy[b_id]
                folium.PolyLine([[a[1], a[0]], [b[1], b[0]]], color="#d94b4b", weight=4, opacity=0.8).add_to(route_group)
        route_group.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(track["latitude"].min()), float(track["longitude"].min())], [float(track["latitude"].max()), float(track["longitude"].max())]])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_file))


def create_static_map(trip: Trip, gx: np.ndarray, gy: np.ndarray, out_file: Path) -> None:
    """Export a report-ready local projection view of GPS, nearby roads, and route."""
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(trip.crs_epsg), always_xy=True)
    track = trip.track
    gx, gy = np.asarray(gx), np.asarray(gy)
    node_lon = trip.nodes["longitude"].to_numpy(float)
    node_lat = trip.nodes["latitude"].to_numpy(float)
    nx, ny = transformer.transform(node_lon, node_lat)
    arcs = trip.arcs.dropna()
    src = arcs["from_node_id"].astype(int).to_numpy()
    dst = arcs["to_node_id"].astype(int).to_numpy()
    valid = (src >= 0) & (dst >= 0) & (src < len(nx)) & (dst < len(nx))
    src, dst = src[valid], dst[valid]
    mids = np.column_stack(((nx[src] + nx[dst]) / 2, (ny[src] + ny[dst]) / 2))
    tree = cKDTree(mids) if len(mids) else None
    nearby: set[int] = set()
    if tree is not None:
        for x, y in zip(gx, gy):
            nearby.update(tree.query_ball_point([x, y], 140.0))
    fig, ax = plt.subplots(figsize=(9, 8))
    for j in nearby:
        ax.plot([nx[src[j]], nx[dst[j]]], [ny[src[j]], ny[dst[j]]], color="#74a96a", linewidth=0.7, alpha=0.55)
    ax.plot(gx, gy, color="#2878c8", linewidth=1.1, alpha=0.7, label="Observed GPS path", zorder=3)
    ax.scatter(gx, gy, s=5, color="#2878c8", alpha=0.7, label="GPS observations", zorder=4)
    if trip.route is not None:
        route_valid = trip.route[(trip.route >= 0) & (trip.route < len(trip.arcs))]
        plotted = False
        for arc_id in route_valid:
            a, b = trip.arcs.iloc[int(arc_id)].to_numpy(int)
            if 0 <= a < len(nx) and 0 <= b < len(nx):
                ax.plot([nx[a], nx[b]], [ny[a], ny[b]], color="#dc554f", linewidth=2.2, alpha=0.8,
                        label="Supplied reference route" if not plotted else None, zorder=2)
                plotted = True
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"Easting (m), EPSG:{trip.crs_epsg}")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Trip {trip.trip_id}: GPS, nearby roads, and supplied route")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def create_model_contract(data_root: Path, output_dir: Path, trips_found: list[tuple[str, str, Path]], quality: pd.DataFrame,
                          candidate_summaries: list[dict[str, Any]], radius_m: float, max_candidates: int) -> dict[str, Any]:
    """Write the partition-level feature tables agreed for model/HMM handoff."""
    contract_dir = output_dir / "data"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_paths: dict[str, str] = {}
    split_metadata: dict[str, Any] = {}
    by_trip_summary = {(str(x["partition"]), str(x["trip_id"])): x for x in candidate_summaries}
    for partition in ("train", "student_test"):
        target = contract_dir / f"{partition}_candidates.parquet"
        writer = None
        for part, trip_id, _ in trips_found:
            if part != partition:
                continue
            source = output_dir / "candidate_features" / f"{part}_{trip_id}.parquet"
            if not source.exists():
                continue
            frame = pd.read_parquet(source)
            arcs = read_numeric_file(data_root / part / trip_id / f"{trip_id}.arcs", ".arcs")
            arc_ids = frame["arc_id"].to_numpy(int)
            frame["from_node"] = arcs.iloc[arc_ids]["from_node_id"].to_numpy(int)
            frame["to_node"] = arcs.iloc[arc_ids]["to_node_id"].to_numpy(int)
            frame["label"] = frame["is_reference_candidate"].astype("int8")
            for col in FEATURE_COLUMNS:
                frame[col] = frame[col].astype("float32")
            for col in ("obs_idx", "arc_id", "from_node", "to_node", "crs_epsg"):
                frame[col] = frame[col].astype("int32")
            # Keep IDs, target availability and route reference for evaluation;
            # model code should select FEATURE_COLUMNS explicitly.
            # The split is encoded in the filename; omit raw route labels to
            # avoid leaking reference arc IDs beyond the binary target.
            columns = ["trip_id", "obs_idx", "arc_id", "from_node", "to_node", "label",
                       "label_available", "crs_epsg", *FEATURE_COLUMNS]
            table = pa.Table.from_pandas(frame[columns], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd", compression_level=15)
            writer.write_table(table)
        if writer is not None:
            writer.close()
        else:
            pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id", "from_node", "to_node", "label",
                                  "label_available", "crs_epsg", *FEATURE_COLUMNS]).to_parquet(target, index=False)
        summaries = [v for (p, _), v in by_trip_summary.items() if p == partition]
        gps_observations = int(quality.loc[quality["partition"] == partition, "gps_rows"].sum())
        reference_observations = sum(int(v["reference_points"]) for v in summaries)
        missed_observations = reference_observations - sum(int(v["labelled_points"]) for v in summaries)
        split_metadata[partition] = {
            "trip_count": int((quality["partition"] == partition).sum()),
            "gps_observations": gps_observations,
            "reference_observations": reference_observations,
            "candidate_missed_observations": missed_observations,
            "candidate_miss_rate": float(missed_observations / max(reference_observations, 1)),
            "candidate_rows": sum(int(v["candidate_rows"]) for v in summaries),
            "artifact": str(target.relative_to(output_dir)),
        }
        contract_paths[partition] = str(target.relative_to(output_dir))
    metadata = {
        "schema_version": 1,
        "artifacts": contract_paths,
        "splits": split_metadata,
        "target": "label (1 when candidate arc is the nearest supplied-route arc for that GPS point; otherwise 0)",
        "filter_rule": "For supervised classifier fitting and pointwise scoring, retain only observations where label_available is true. Candidate misses are counted in split metadata and excluded as unlabeled observations.",
        "feature_columns": FEATURE_COLUMNS,
        "metadata_columns": ["trip_id", "obs_idx", "arc_id", "from_node", "to_node", "label", "label_available", "crs_epsg"],
        "candidate_radius_m": radius_m,
        "max_candidates_per_observation": max_candidates,
        "reference_label_method": "For each GPS point, select the closest segment among arc IDs listed in that trip's supplied .route sequence, using projected point-to-segment distance.",
        "limitations": ["Supplied routes are reference information and may be imperfect.", "Route files have no explicit GPS observation-to-arc alignment; labels are a geometric proxy.", "Evaluator-test data and labels were not included in this pipeline run."],
    }
    (contract_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run(data_root: Path, output_dir: Path, radius_m: float, max_candidates: int, map_trip_id: str | None) -> None:
    trips_found = discover_trips(data_root)
    if not trips_found:
        raise FileNotFoundError(f"No train/student_test trip folders found under {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    qualities, inventories, candidate_summaries = [], [], []
    map_trip = None
    map_coords = None
    for partition, trip_id, directory in trips_found:
        print(f"Processing {partition}/{trip_id}...", flush=True)
        trip = load_trip(partition, trip_id, directory)
        gx, gy, px, py = project_trip(trip)
        src = trip.arcs["from_node_id"].fillna(-1).to_numpy(int)
        dst = trip.arcs["to_node_id"].fillna(-1).to_numpy(int)
        graph = graph_stats(len(trip.nodes), src, dst)
        qualities.append(trip_quality(trip, gx, gy, graph))
        inv = {"trip_id": trip_id, "partition": partition, "track_exists": True, "nodes_exists": True, "arcs_exists": True,
               "route_exists": trip.route is not None, "track_bytes": (directory / f"{trip_id}.track").stat().st_size,
               "nodes_bytes": (directory / f"{trip_id}.nodes").stat().st_size, "arcs_bytes": (directory / f"{trip_id}.arcs").stat().st_size,
               "route_bytes": (directory / f"{trip_id}.route").stat().st_size if trip.route is not None else None}
        inventories.append(inv)
        candidates, _ = candidate_features(trip, gx, gy, px, py, radius_m, max_candidates)
        # Persist each trip immediately so feature generation does not keep all
        # candidate rows in memory. These files form a Parquet dataset.
        feature_dir = output_dir / "candidate_features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(feature_dir / f"{partition}_{trip_id}.parquet", index=False, compression="zstd")
        print(f"  wrote {len(candidates):,} candidate rows", flush=True)
        if not candidates.empty:
            per_observation = candidates.groupby("obs_idx", as_index=False).agg(
                reference_arc_id=("reference_arc_id", "first"), label_available=("label_available", "first"),
                candidate_count=("arc_id", "size"), positive_candidates=("is_reference_candidate", "sum"))
            candidate_summaries.append({
                "partition": partition, "trip_id": trip_id,
                "observations": int(per_observation["obs_idx"].nunique()),
                "candidate_rows": int(len(candidates)),
                "reference_points": int(per_observation["reference_arc_id"].notna().sum()),
                "labelled_points": int(per_observation["label_available"].sum()),
                "positive_candidates": int(per_observation["positive_candidates"].sum()),
                "mean_candidates_per_observation": float(per_observation["candidate_count"].mean()),
            })
        if map_trip_id == trip_id or (map_trip_id is None and trip_id == trips_found[0][1]):
            map_trip, map_coords = trip, (gx, gy)
    quality = pd.DataFrame(qualities)
    inventory = pd.DataFrame(inventories)
    quality.to_csv(output_dir / "trip_quality.csv", index=False)
    inventory.to_csv(output_dir / "trip_inventory.csv", index=False)
    draw_eda(quality, inventory, output_dir)
    if map_trip and map_coords:
        create_folium_map(map_trip, *map_coords, output_dir / "folium" / f"trip_{map_trip.trip_id}.html")
        create_static_map(map_trip, *map_coords, output_dir / "folium" / f"trip_{map_trip.trip_id}_report.png")
    cand_summary = pd.DataFrame(candidate_summaries)
    cand_summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    contract = create_model_contract(data_root, output_dir, trips_found, quality, candidate_summaries, radius_m, max_candidates)
    total_reference_points = int(cand_summary["reference_points"].sum()) if not cand_summary.empty else 0
    total_labelled_points = int(cand_summary["labelled_points"].sum()) if not cand_summary.empty else 0
    summary = {
        "data_root": str(data_root.resolve()), "trip_count_by_partition": quality.groupby("partition").size().to_dict(),
        "total_trips": int(len(quality)), "gps_observations": int(quality["gps_rows"].sum()),
        "total_candidate_rows": int(cand_summary["candidate_rows"].sum()) if not cand_summary.empty else 0,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_radius_m": radius_m, "candidate_limit_per_observation": max_candidates,
        "mean_candidate_miss_rate": None,
        "partitions_used": sorted(quality["partition"].unique().tolist()),
        "model_handoff": contract["splits"],
        "reference_label_method": "closest supplied-route arc to each GPS point, by projected point-to-segment distance; observations where that arc is absent from generated candidates are marked label_available=false",
        "warnings": ["Supplied .route files are reference labels and may have limitations.",
                     "student_test is development data; do not use evaluator-test data to tune the model.",
                     "Candidate generation uses a nearest-midpoint shortlist before exact point-to-segment filtering."],
    }
    summary["mean_candidate_miss_rate"] = float(1 - total_labelled_points / max(total_reference_points, 1))
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"trips": summary["trip_count_by_partition"], "observations": summary["gps_observations"],
                      "candidate_rows": summary["total_candidate_rows"], "mean_candidate_miss_rate": summary["mean_candidate_miss_rate"],
                      "output_dir": str(output_dir.resolve())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing train/ and student_test/ folders")
    parser.add_argument("--output-dir", type=Path, default=Path("data_geo/outputs"))
    parser.add_argument("--candidate-radius-m", type=float, default=80.0)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--map-trip-id", type=str, default=None, help="Trip ID to map; default is first discovered trip")
    args = parser.parse_args()
    run(args.data_root.resolve(), args.output_dir.resolve(), args.candidate_radius_m, args.max_candidates, args.map_trip_id)


if __name__ == "__main__":
    main()
