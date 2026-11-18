"""
PathSnap Preprocessing & Feature Extraction
Module for Member 1: Raw data ingestion, geospatial transformations, and candidate extraction.
"""

import math
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def load_trip_data(trip_dir: Path, trip_id: str):
    """
    Loads raw trip data files: .track, .nodes, .arcs, .route
    Returns:
        track: (N, 3) array of [lon, lat, timestamp]
        nodes: (M, 2) array of [lon, lat]
        arcs: (K, 2) array of [from_node, to_node]
        route: List[int] of ground-truth arc_ids (if available)
    """
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    track = np.loadtxt(track_path) if track_path.exists() else np.array([])
    nodes = np.loadtxt(nodes_path) if nodes_path.exists() else np.array([])
    arcs = np.loadtxt(arcs_path, dtype=int) if arcs_path.exists() else np.array([])
    route = np.loadtxt(route_path, dtype=int).tolist() if route_path.exists() else []

    if isinstance(route, int):
        route = [route]

    return track, nodes, arcs, route
