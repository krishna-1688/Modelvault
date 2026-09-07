"""Request/response models for the FastAPI gateway. Validation lives here,
not in api.py -- Pydantic rejects malformed requests before any handler code
(rate limiting, threat scoring, model inference) ever runs, which is the
correct place for a public-facing API to draw its trust boundary."""
from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, field_validator

MAX_QUERIES_PER_VERIFY_REQUEST = 5000


def _validate_finite_features(features: list[float]) -> list[float]:
    if not features:
        raise ValueError("features must not be empty")
    if not all(math.isfinite(v) for v in features):
        raise ValueError("features must all be finite numbers (no NaN or Infinity)")
    return features


class PredictRequest(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=256, description="Caller-supplied identifier. Not trusted for detection -- see Layer 2's design rationale.")
    features: list[float] = Field(..., description="Raw (unscaled) feature vector, in the order the model was trained on.")

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: list[float]) -> list[float]:
        return _validate_finite_features(v)


class PredictResponse(BaseModel):
    tier: str
    label: int
    probabilities: Optional[list[float]]
    threat_index: float
    watermarked: bool


class VerifyOwnershipRequest(BaseModel):
    """A batch of (client_id, query, suspect_model_label) triples: for each
    query, which client_id it was originally sent under and what label the
    suspect model under investigation produced. The gateway looks each
    (client_id, query) pair up in its own watermark event log to see whether
    it was ever watermarked, and if so what label it should reproduce if
    trained on our output. client_ids is per-query because a real extraction
    attack is often split across many sybil accounts."""
    client_ids: list[str] = Field(..., max_length=MAX_QUERIES_PER_VERIFY_REQUEST)
    queries: list[list[float]] = Field(..., max_length=MAX_QUERIES_PER_VERIFY_REQUEST)
    suspect_labels: list[int] = Field(..., max_length=MAX_QUERIES_PER_VERIFY_REQUEST)
    confidence_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class VerifyOwnershipResponse(BaseModel):
    verified: bool
    matches: int
    total_triggers: int
    confidence: float
    chance_rate: float


class AdminStatsResponse(BaseModel):
    total_requests: int
    total_clients: int
    watermark_triggers: int
    tier_counts: dict[str, int]
    defense_enabled: bool
    recent_threat_indices: list[float]
    recent_events: list[dict] = Field(default_factory=list)


class ToggleDefenseRequest(BaseModel):
    enabled: bool


class ReadyResponse(BaseModel):
    ready: bool
    detail: str
