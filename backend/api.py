"""PathSnap Production FastAPI Backend (Laptop 1 — Member 3).

Runs the complete, live end-to-end map-matching pipeline:
  1. Loads `artifacts/final_model.joblib` (Bayesian-tuned XGBoost Pipeline) +
     `artifacts/model_metadata.json` + `artifacts/hmm_config.json` once on startup.
  2. Discovers all 10 evaluation trips from `PATHSNAP_EVAL_DIR` (or `evaluator_test/`
     -> `Student folder/evaluator_test/` -> `Student folder/student_test/`).
  3. On `POST /predict` or `POST /evaluate`:
       - Extracts candidate road segments & 12 metric features from `.track`, `.nodes`, `.arcs` ONLY.
       - Scores candidates with the trained XGBoost classifier (Classifier-Only predictions).
       - Reconstructs coherent road sequence via log-space HMM + Viterbi (`HMMRouteDecoder`).
       - Computes accuracy strictly server-side against internal `.route` (never exposed to model/UI).
       - Logs clear timestamped pipeline telemetry to CMD/terminal for live evaluation & screenshots.

Run on Laptop 1:
  uvicorn backend.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    EvaluationMetrics,
    HealthResponse,
    TripEvaluationResponse,
    TripListResponse,
    TripRequest,
)
from src.hmm_decoder import HMMConfig, HMMRouteDecoder
from src.inference_pipeline import (
    FEATURE_COLUMNS,
    arcs_to_latlon_polyline,
    evaluate_against_reference_server_side,
    extract_candidate_segments_latlon,
    extract_trip_candidates,
)

# Configure crisp terminal logging for live CMD demonstration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pathsnap.api")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

app = FastAPI(
    title="PathSnap Production Map-Matching API",
    description="Laptop 1 FastAPI server hosting Bayesian-tuned XGBoost + HMM/Viterbi route reconstruction.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ServerState:
    def __init__(self) -> None:
        self.model: Any = None
        self.feature_cols: List[str] = list(FEATURE_COLUMNS)
        self.model_metadata: Dict[str, Any] = {}
        self.hmm_config: HMMConfig = HMMConfig()
        self.decoder: HMMRouteDecoder = HMMRouteDecoder(self.hmm_config)
        self.eval_dir: Optional[Path] = None

    def resolve_eval_dir(self) -> Path:
        env_dir = os.environ.get("PATHSNAP_EVAL_DIR")
        candidates = []
        if env_dir:
            candidates.append(Path(env_dir))
        candidates.extend(
            [
                BASE_DIR / "evaluator_test",
                BASE_DIR / "Student folder" / "evaluator_test",
                BASE_DIR / "Student folder" / "Student folder" / "evaluator_test",
                BASE_DIR / "Student folder" / "student_test",
                BASE_DIR / "Student folder" / "Student folder" / "student_test",
            ]
        )
        for p in candidates:
            if p.exists() and any(d.is_dir() for d in p.iterdir()):
                return p
        raise RuntimeError(
            "No evaluation trips directory found. Set PATHSNAP_EVAL_DIR or place trips in evaluator_test/ or Student folder/student_test/."
        )

    def load_assets(self) -> None:
        model_path = ARTIFACTS_DIR / "final_model.joblib"
        if not model_path.exists():
            model_path = ARTIFACTS_DIR / "baseline_best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found in {ARTIFACTS_DIR}. Expected final_model.joblib."
            )

        self.model = joblib.load(model_path)
        # Ensure CPU inference portability on Laptop 1
        if hasattr(self.model, "named_steps") and "model" in self.model.named_steps:
            inner = self.model.named_steps["model"]
            if hasattr(inner, "set_params") and "device" in inner.get_params():
                inner.set_params(device="cpu", n_jobs=-1)

        meta_path = ARTIFACTS_DIR / "model_metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                self.model_metadata = json.load(f)
            if "feature_cols" in self.model_metadata:
                self.feature_cols = list(self.model_metadata["feature_cols"])

        hmm_cfg_path = ARTIFACTS_DIR / "hmm_config.json"
        if hmm_cfg_path.exists():
            with hmm_cfg_path.open("r", encoding="utf-8") as f:
                self.hmm_config = HMMConfig.from_dict(json.load(f))
            self.decoder = HMMRouteDecoder(self.hmm_config)

        self.eval_dir = self.resolve_eval_dir()
        logger.info("=" * 72)
        logger.info("PATHSNAP PRODUCTION API INITIALISED (LAPTOP 1)")
        logger.info("  Model artifact : %s (%s)", model_path.name, self.model_metadata.get("model_name", type(self.model).__name__))
        logger.info("  Features (%d)  : %s", len(self.feature_cols), self.feature_cols)
        logger.info("  HMM config     : %s", self.hmm_config.to_dict())
        logger.info("  Eval trips dir : %s", self.eval_dir)
        logger.info("  Trips found    : %s", self.list_trip_ids())
        logger.info("=" * 72)

    def list_trip_ids(self) -> List[str]:
        if self.eval_dir is None:
            self.eval_dir = self.resolve_eval_dir()
        return sorted([d.name for d in self.eval_dir.iterdir() if d.is_dir()])

    def locate_trip_dir(self, trip_id: str) -> Path:
        if self.eval_dir is None:
            self.eval_dir = self.resolve_eval_dir()
        for base in (
            self.eval_dir,
            BASE_DIR / "Student folder" / "student_test",
            BASE_DIR / "Student folder" / "train",
        ):
            cand = base / trip_id
            if cand.exists() and cand.is_dir():
                return cand
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found on server.")


state = ServerState()


@app.on_event("startup")
def _startup() -> None:
    state.load_assets()


@app.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    client_ip = request.client.host if request.client else "unknown"
    logger.info("[GET /health] Client=%s -> 200 OK (model loaded, %d trips ready)", client_ip, len(state.list_trip_ids()))
    return HealthResponse(
        status="healthy",
        service="PathSnap Production Route Reconstruction API (XGBoost + HMM/Viterbi)",
        host="Laptop 1 (Model & Decoder Host)",
    )


@app.get("/trips", response_model=TripListResponse)
def list_trips(request: Request) -> TripListResponse:
    client_ip = request.client.host if request.client else "unknown"
    trips = state.list_trip_ids()
    logger.info("[GET /trips] Client=%s -> returning %d evaluation trips: %s", client_ip, len(trips), trips)
    return TripListResponse(total_trips=len(trips), trips=trips)


@app.post("/predict", response_model=TripEvaluationResponse)
@app.post("/evaluate", response_model=TripEvaluationResponse)
def evaluate_trip(req: TripRequest, request: Request) -> TripEvaluationResponse:
    if state.model is None:
        state.load_assets()

    t_total = time.perf_counter()
    trip_id = req.trip_id.strip()
    client_ip = request.client.host if request.client else "unknown"
    trip_dir = state.locate_trip_dir(trip_id)

    logger.info("-" * 72)
    logger.info("[POST %s] Client=%s | trip_id=%s | Starting live pipeline...", request.url.path, client_ip, trip_id)

    # 1. Extract candidates & 12 metric features (strictly NO access to .route)
    t0 = time.perf_counter()
    ctx = extract_trip_candidates(trip_dir, trip_id, radius_m=80.0, max_candidates=40)
    feat_ms = (time.perf_counter() - t0) * 1000.0
    if ctx.candidates_df.empty:
        raise HTTPException(status_code=422, detail=f"No valid GPS observations or candidates for trip {trip_id}.")

    # 2. Score candidates with trained Bayesian-tuned XGBoost classifier
    t1 = time.perf_counter()
    X_trip = ctx.candidates_df[state.feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    probs = state.model.predict_proba(X_trip)[:, 1].astype(np.float64)
    clf_ms = (time.perf_counter() - t1) * 1000.0

    scored_df = ctx.candidates_df[["obs_idx", "arc_id", "from_node", "to_node"]].copy()
    scored_df["prob"] = probs

    # Classifier-only Top-1 route (independent pointwise argmax)
    clf_top1_df = (
        scored_df.sort_values(["obs_idx", "prob", "arc_id"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates(subset=["obs_idx"], keep="first")
        .reset_index(drop=True)
    )

    # 3. Sequential HMM + Viterbi route reconstruction
    t2 = time.perf_counter()
    hmm_decoded_df = state.decoder.decode_trip(
        scored_df,
        gps_x=ctx.gx,
        gps_y=ctx.gy,
        timestamps_s=ctx.timestamps_s,
        graph=ctx.graph,
    )
    hmm_ms = (time.perf_counter() - t2) * 1000.0

    # 4. Server-side evaluation against internal .route (ground truth stays strictly server-side)
    t3 = time.perf_counter()
    eval_stats = evaluate_against_reference_server_side(
        ctx=ctx,
        clf_top1_df=clf_top1_df,
        hmm_decoded_df=hmm_decoded_df,
        route_file=trip_dir / f"{trip_id}.route",
    )
    eval_ms = (time.perf_counter() - t3) * 1000.0

    # 5. Build UI geometry payloads ([lat, lon])
    raw_gps_latlon = [[float(lat), float(lon)] for lon, lat in ctx.track_lonlat]
    step_gps = max(1, len(raw_gps_latlon) // 400)
    sampled_gps = raw_gps_latlon[::step_gps]

    clf_route_latlon = arcs_to_latlon_polyline(clf_top1_df, ctx.nodes_lonlat, ctx.arcs_uv)
    hmm_route_latlon = arcs_to_latlon_polyline(hmm_decoded_df, ctx.nodes_lonlat, ctx.arcs_uv)
    cand_segs_latlon = extract_candidate_segments_latlon(ctx.candidates_df, ctx.nodes_lonlat, ctx.arcs_uv)

    total_ms = (time.perf_counter() - t_total) * 1000.0

    metrics = EvaluationMetrics(
        top_1_accuracy=round(float(eval_stats["top_1_accuracy"]), 4),
        route_continuity=round(float(eval_stats["route_continuity"]), 4),
        candidate_miss_rate=round(float(eval_stats["candidate_miss_rate"]), 4),
        inference_time_ms=round(total_ms, 2),
        total_observations=int(eval_stats["total_observations"]),
        classifier_top_1_accuracy=round(float(eval_stats["classifier_top_1_accuracy"]), 4),
        classifier_continuity=round(float(eval_stats["classifier_continuity"]), 4),
    )

    logger.info(
        "[200 OK] trip_id=%s | obs=%d | cands=%d | feat=%.1fms | xgb=%.1fms | hmm=%.1fms | eval=%.1fms | total=%.1fms",
        trip_id,
        metrics.total_observations,
        len(scored_df),
        feat_ms,
        clf_ms,
        hmm_ms,
        eval_ms,
        total_ms,
    )
    logger.info(
        "         Clf-Only: Top1=%.4f, Cont=%.4f  ==>  Clf+HMM: Top1=%.4f, Cont=%.4f, MissRate=%.4f",
        metrics.classifier_top_1_accuracy,
        metrics.classifier_continuity,
        metrics.top_1_accuracy,
        metrics.route_continuity,
        metrics.candidate_miss_rate,
    )
    logger.info("-" * 72)

    return TripEvaluationResponse(
        trip_id=trip_id,
        status="success",
        metrics=metrics,
        raw_gps=sampled_gps,
        classifier_only_route=clf_route_latlon,
        hmm_reconstructed_route=hmm_route_latlon,
        candidate_segments=cand_segs_latlon,
        details={
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "crs_epsg": ctx.crs_epsg,
            "nodes_count": int(len(ctx.nodes_lonlat)),
            "arcs_count": int(len(ctx.arcs_uv)),
            "candidates_scored": int(len(scored_df)),
            "sequence_arc_f1": round(float(eval_stats.get("sequence_arc_f1", 0.0)), 4),
            "model_name": state.model_metadata.get("model_name", "XGBoost (Bayesian Tuned)"),
            "model_family": "Boosting — XGBoost (Bayesian Tuned) + Log-Space HMM/Viterbi",
            "decoder": "Log-Space HMM + Viterbi (Classifier Emissions + Bounded Dijkstra Road Transitions)",
            "hmm_config": state.hmm_config.to_dict(),
            "timings_ms": {
                "feature_extraction": round(feat_ms, 2),
                "classifier_scoring": round(clf_ms, 2),
                "hmm_viterbi_decoding": round(hmm_ms, 2),
                "server_side_eval": round(eval_ms, 2),
                "total": round(total_ms, 2),
            },
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
