"""FastAPI gateway wiring Layers 1-4 together in sequence:

  Layer 1 (velocity_governor) -> reject if rate-limited
  Layer 2 (threat_index)      -> score the query against the pooled reservoir
  Layer 3 (throttling)        -> degrade the response based on the tier
  Layer 4 (watermark)         -> possibly flip the label, log the event

Raw features are scaled once at the boundary (transform_features) and every
downstream layer -- reservoir storage, threat scoring, model inference,
watermark hashing -- operates on that same scaled representation, matching
what the artifacts were fit on in train.py.

Also exposes /verify-ownership (statistical proof of theft), /ready (startup
health), and /admin/stats + /admin/toggle-defense for the dashboard and demo
script. Admin routes and /verify-ownership require an X-API-Key header when
ADMIN_API_KEY is configured -- see gateway/auth.py.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from modelvault.gateway.auth import require_admin_key
from modelvault.gateway.schemas import (
    AdminStatsResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
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
from modelvault.model.predict import load_reference_model, load_scaler, load_target_model, predict_proba, transform_features
from modelvault.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLE_DIR = PROJECT_ROOT / "console"

app = FastAPI(title="ModelVault Gateway", version="1.0.0")

# Serves the live HTML/JS/CSS console at /console/ (open with
# /console/?key=<ADMIN_API_KEY> if one is configured -- the page reads the
# key from the URL and attaches it as X-API-Key on its own polling requests).
# Mounted on the gateway itself so there's exactly one process to run for
# the whole demo -- no separate Streamlit server needed.
if CONSOLE_DIR.exists():
    app.mount("/console", StaticFiles(directory=str(CONSOLE_DIR), html=True), name="console")


def _sanitize_non_finite(value):
    """Starlette's JSONResponse refuses to encode NaN/Infinity (correctly,
    per the JSON spec) -- but FastAPI's default validation-error response
    echoes the REJECTED input value back in the error detail, so a client
    sending literal NaN (which Python's own json.loads accepts as a
    non-standard extension on parse) would otherwise crash this handler with
    an unhandled 500 while trying to report that the input was invalid.
    Replacing non-finite floats with their string form before encoding keeps
    the error response itself always serializable."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_non_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_finite(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    sanitized = _sanitize_non_finite(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": sanitized})


@app.get("/health")
def health():
    """Liveness: is the process up. Does not check model artifacts -- see /ready for that."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness: are the trained artifacts actually loadable. A real
    orchestrator (k8s, etc.) should gate traffic on this, not /health --
    a process can be alive with no model loaded."""
    try:
        load_target_model()
        load_reference_model()
        load_scaler()
    except Exception as exc:
        return ReadyResponse(ready=False, detail=f"Artifacts not loadable: {exc}")
    return ReadyResponse(ready=True, detail="Model, reference density, and scaler all loaded.")


def _validate_feature_count(features: list[float]) -> None:
    expected = load_target_model().n_features_in_
    if len(features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(features)}.",
        )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    state = get_state()
    _validate_feature_count(request.features)

    # Layer 1 -- ingress velocity governor. Checked before any scaling/model
    # work so a rate-limited caller doesn't burn compute on rejected requests.
    if not state.velocity_governor.allow(request.client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    features = transform_features(np.asarray(request.features, dtype=float))[0]

    if not state.is_defense_enabled():
        probabilities = predict_proba(features)[0]
        label = int(np.argmax(probabilities))
        # Still logged (unlike Layers 2-4, which are genuinely skipped) so the
        # dashboard visibly shows "traffic flowing, unscored" rather than
        # freezing entirely -- a flat, zero-threat line is a clearer contrast
        # against defended traffic than a stats panel that stops moving.
        state.record_request(request.client_id, ResponseTier.NORMAL, 0.0, label=label, watermarked=False)
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

    state.record_request(request.client_id, tier, threat_index, label=response["label"], watermarked=watermarked)

    return PredictResponse(
        tier=tier.value,
        label=response["label"],
        probabilities=response["probabilities"],
        threat_index=threat_index,
        watermarked=watermarked,
    )


@app.post("/verify-ownership", response_model=VerifyOwnershipResponse, dependencies=[Depends(require_admin_key)])
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

    predictions_by_bytes = {event.query_features.tobytes(): predictions_by_index[idx] for idx, event in enumerate(events)}

    def suspect_predict_fn(query_features: np.ndarray) -> int:
        try:
            return predictions_by_bytes[query_features.tobytes()]
        except KeyError:
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


@app.get("/admin/stats", response_model=AdminStatsResponse, dependencies=[Depends(require_admin_key)])
def admin_stats():
    return get_state().stats()


@app.post("/admin/toggle-defense", dependencies=[Depends(require_admin_key)])
def admin_toggle_defense(request: ToggleDefenseRequest):
    state = get_state()
    state.set_defense_enabled(request.enabled)
    return {"defense_enabled": state.is_defense_enabled()}
