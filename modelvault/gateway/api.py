"""FastAPI gateway wiring Layers 1-4 together in sequence:

  Layer 1 (velocity_governor) -> reject if rate-limited
  Layer 2 (threat_index)      -> score the query against the pooled reservoir
  Layer 3 (throttling)        -> degrade the response based on the tier
  Layer 4 (watermark)         -> possibly flip the label, log the event

Also exposes /verify-ownership (statistical proof of theft) and
/admin/stats + /admin/toggle-defense for the dashboard and demo script.
"""
from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI, HTTPException

from modelvault.gateway.schemas import (
    AdminStatsResponse,
    PredictRequest,
    PredictResponse,
    ToggleDefenseRequest,
    VerifyOwnershipRequest,
    VerifyOwnershipResponse,
)
from modelvault.gateway.state import get_state
from modelvault.layer2_detection.threat_index import compute_threat_index
from modelvault.layer3_response.boundary_perturbation import apply_boundary_perturbation
from modelvault.layer3_response.throttling import ResponseTier, apply_throttling, classify_tier
from modelvault.layer4_watermark.verification import WatermarkEvent, verify_ownership
from modelvault.layer4_watermark.watermark import decide_and_apply_watermark
from modelvault.model.predict import load_target_model, predict_proba
from modelvault.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="ModelVault Gateway", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    state = get_state()
    features = np.asarray(request.features, dtype=float)

    # Layer 1 -- ingress velocity governor.
    if not state.velocity_governor.allow(request.client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if not state.defense_enabled:
        probabilities = predict_proba(features)[0]
        label = int(np.argmax(probabilities))
        return PredictResponse(
            tier=ResponseTier.NORMAL.value,
            label=label,
            probabilities=probabilities.round(6).tolist(),
            threat_index=0.0,
            watermarked=False,
        )

    # Layer 2 -- content-based anomaly detection (scored before this query
    # joins the reservoir, then added so subsequent queries see it).
    threat_result = compute_threat_index(features, state.reservoir)
    threat_index = threat_result["threat_index"]
    state.reservoir.add(request.client_id, features, timestamp=time.monotonic())

    # Layer 3 -- graduated response degradation.
    tier = classify_tier(threat_index)
    probabilities = predict_proba(features)[0]
    n_classes = len(probabilities)
    response = apply_throttling(probabilities, tier)

    if tier == ResponseTier.CRITICAL:
        response["label"] = apply_boundary_perturbation(
            response["label"], probabilities, request.client_id, features
        )

    # Layer 4 -- deterministic watermark decision + injection.
    watermark_decision = decide_and_apply_watermark(
        request.client_id, features, response["label"], n_classes, tier
    )
    watermarked = watermark_decision.triggered
    if watermarked:
        response["label"] = watermark_decision.final_label
        state.record_watermark_event(request.client_id, features, watermark_decision.final_label)

    state.record_request(request.client_id, tier, threat_index)

    return PredictResponse(
        tier=tier.value,
        label=response["label"],
        probabilities=response["probabilities"],
        threat_index=threat_index,
        watermarked=watermarked,
    )


@app.post("/verify-ownership", response_model=VerifyOwnershipResponse)
def verify_ownership_endpoint(request: VerifyOwnershipRequest):
    state = get_state()
    if not (len(request.queries) == len(request.suspect_labels) == len(request.client_ids)):
        raise HTTPException(status_code=400, detail="client_ids, queries and suspect_labels must be the same length")

    events: list[WatermarkEvent] = []
    predictions_by_index: dict[int, int] = {}
    for i, (client_id, query, suspect_label) in enumerate(zip(request.client_ids, request.queries, request.suspect_labels)):
        features = np.asarray(query, dtype=float)
        logged = state.lookup_watermark_event(client_id, features)
        if logged is not None:
            events.append(WatermarkEvent(client_id=client_id, query_features=features, expected_label=logged.expected_label))
            predictions_by_index[len(events) - 1] = suspect_label

    if not events:
        return VerifyOwnershipResponse(verified=False, matches=0, total_triggers=0, confidence=0.0, chance_rate=0.0)

    def suspect_predict_fn(query_features: np.ndarray) -> int:
        for idx, event in enumerate(events):
            if np.array_equal(event.query_features, query_features):
                return predictions_by_index[idx]
        raise ValueError("query not found")

    n_classes = len(load_target_model().classes_)
    result = verify_ownership(events, suspect_predict_fn, n_classes=n_classes, confidence_threshold=request.confidence_threshold)

    return VerifyOwnershipResponse(
        verified=result.verified,
        matches=result.matches,
        total_triggers=result.total_triggers,
        confidence=result.confidence,
        chance_rate=result.chance_rate,
    )


@app.get("/admin/stats", response_model=AdminStatsResponse)
def admin_stats():
    return get_state().stats()


@app.post("/admin/toggle-defense")
def admin_toggle_defense(request: ToggleDefenseRequest):
    state = get_state()
    state.defense_enabled = request.enabled
    return {"defense_enabled": state.defense_enabled}
