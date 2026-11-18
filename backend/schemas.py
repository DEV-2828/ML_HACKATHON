"""
PathSnap API Schemas & Data Contract
Defines the JSON contract between Laptop 1 (Model/API) and Laptop 2 (UI).
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="PathSnap Route Reconstruction API")
    host: str = Field(..., example="Laptop 1 (Model & Decoder Host)")


class TripListResponse(BaseModel):
    total_trips: int
    trips: List[str]


class TripRequest(BaseModel):
    trip_id: str = Field(..., description="ID of the evaluation trip, e.g. '00000000'")


class EvaluationMetrics(BaseModel):
    top_1_accuracy: float = Field(..., description="Observation-level Top-1 Accuracy of final route")
    route_continuity: float = Field(..., description="Proportion of consecutive segments that form valid transitions")
    candidate_miss_rate: float = Field(..., description="Proportion of observations where ground truth was missing from candidates")
    inference_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    total_observations: int = Field(..., description="Total number of GPS points in the trajectory")
    classifier_top_1_accuracy: Optional[float] = Field(None, description="Classifier-only Top-1 Accuracy before HMM")
    classifier_continuity: Optional[float] = Field(None, description="Classifier-only route continuity before HMM")


class TripEvaluationResponse(BaseModel):
    trip_id: str
    status: str
    metrics: EvaluationMetrics
    raw_gps: List[List[float]] = Field(..., description="List of [latitude, longitude] pairs for GPS observations")
    classifier_only_route: List[List[float]] = Field(..., description="List of [latitude, longitude] pairs for classifier predictions")
    hmm_reconstructed_route: List[List[float]] = Field(..., description="List of [latitude, longitude] pairs for HMM/Viterbi route")
    candidate_segments: Optional[List[List[List[float]]]] = Field(
        default=[], description="List of [[lat1, lon1], [lat2, lon2]] for nearby candidate road segments"
    )
    details: Dict[str, Any] = Field(default_factory=dict)
