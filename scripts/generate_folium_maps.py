"""
PathSnap Folium Map Generator (Member 4 & Member 1)
Generates high-resolution interactive Folium HTML maps for evaluation trips.
Visualizes:
1. Original Noisy GPS Observations (Blue CircleMarkers with popups)
2. Road Network Street Grid (.nodes & .arcs in gray)
3. Classifier-Only Prediction (Red dashed polyline showing jumping)
4. HMM Reconstructed Route (Solid Emerald-Green polyline with start/stop markers)
5. LayerControl widget for interactive comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import numpy as np

from backend.mock_api import evaluate_trip
from backend.schemas import TripRequest


def generate_trip_folium_map(trip_id: str, out_html: Path) -> folium.Map:
    """Generates and saves a complete multi-layer Folium map for trip_id."""
    print(f"Generating interactive Folium map for Trip {trip_id}...")
    res = evaluate_trip(TripRequest(trip_id=trip_id))

    raw_gps = res.raw_gps
    hmm_route = res.hmm_reconstructed_route
    clf_route = res.classifier_only_route
    cand_segments = res.candidate_segments or []

    center_lat = np.mean([p[0] for p in raw_gps])
    center_lon = np.mean([p[1] for p in raw_gps])

    f_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    # 1. Road Network Arcs (Candidate Street Grid)
    if cand_segments:
        cand_group = folium.FeatureGroup(name="1. Road Network Geometry (Candidate Arcs)", show=False).add_to(f_map)
        for seg in cand_segments:
            folium.PolyLine(
                seg,
                color="#64748B",
                weight=2,
                opacity=0.6,
                tooltip="Road Network Arc"
            ).add_to(cand_group)

    # 2. Raw Noisy GPS Observations
    gps_group = folium.FeatureGroup(name="2. Raw GPS Observations (Noisy)", show=True).add_to(f_map)
    for i, pt in enumerate(raw_gps):
        folium.CircleMarker(
            location=pt,
            radius=3.5,
            color="#1D4ED8",
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=0.8,
            tooltip=f"GPS Fix #{i*10}: Lat {pt[0]:.5f}, Lon {pt[1]:.5f}"
        ).add_to(gps_group)

    # 3. Classifier-Only Prediction (Independent, discontinuous)
    if clf_route:
        clf_group = folium.FeatureGroup(name="3. Classifier-Only (Independent Predictions)", show=True).add_to(f_map)
        folium.PolyLine(
            clf_route,
            color="#DC2626",
            weight=3,
            opacity=0.8,
            dash_array="6, 6",
            tooltip="Classifier Prediction (Shows Disconnected Jumps)"
        ).add_to(clf_group)

    # 4. HMM Reconstructed Route (Final Continuous Trajectory)
    if hmm_route:
        hmm_group = folium.FeatureGroup(name="4. HMM + Viterbi Reconstructed Route (Snapping)", show=True).add_to(f_map)
        folium.PolyLine(
            hmm_route,
            color="#059669",
            weight=5.5,
            opacity=0.95,
            tooltip="Final Topologically Snapped Route"
        ).add_to(hmm_group)

        # Route Start & End Markers
        folium.Marker(
            location=hmm_route[0],
            popup="<b>Trajectory Origin</b>",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(hmm_group)

        folium.Marker(
            location=hmm_route[-1],
            popup="<b>Trajectory Destination</b>",
            icon=folium.Icon(color="red", icon="flag", prefix="fa")
        ).add_to(hmm_group)

    # Add Layer Control
    folium.LayerControl(collapsed=False, position="topright").add_to(f_map)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    f_map.save(str(out_html))
    print(f" Saved interactive map to: {out_html}")
    return f_map


if __name__ == "__main__":
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_trip_folium_map("00000000", out_dir / "folium_map_trip_00000000.html")
    generate_trip_folium_map("00000007", out_dir / "folium_map_trip_00000007.html")
