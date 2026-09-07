"""Request/response models for the FastAPI gateway."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    client_id: str = Field(..., description="Caller-supplied identifier. Not trusted for detection -- see Layer 2's design rationale.")
    features: list[float]


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
    client_ids: list[str]
    queries: list[list[float]]
    suspect_labels: list[int]
    confidence_threshold: float = 0.95


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


class ToggleDefenseRequest(BaseModel):
    enabled: bool
