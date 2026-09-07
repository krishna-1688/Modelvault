import numpy as np
import pytest
from fastapi.testclient import TestClient

from modelvault.gateway.api import app
from modelvault.gateway import state as state_module
from modelvault.model.data_prep import generate_dataset
from modelvault.model.predict import load_target_model

TEST_ADMIN_KEY = "test-admin-key"
ADMIN_HEADERS = {"X-API-Key": TEST_ADMIN_KEY}


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", TEST_ADMIN_KEY)
    state_module.reset_state()
    yield
    state_module.reset_state()


@pytest.fixture
def client():
    return TestClient(app)


def _require_artifacts():
    try:
        load_target_model()
    except FileNotFoundError:
        pytest.skip("Run `python -m modelvault.model.train` before this test.")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_when_artifacts_present(client):
    _require_artifacts()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_predict_returns_full_response_for_normal_traffic(client):
    """Uses a REAL raw transaction row (not a sample from the reference
    density model, which lives in the gateway's internal scaled feature
    space -- sending that as if it were raw API input would get scaled
    twice and no longer look like real traffic)."""
    _require_artifacts()
    X_train, _, _, _ = generate_dataset()
    real_row = X_train[0]
    response = client.post("/predict", json={"client_id": "user-1", "features": real_row.tolist()})
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == "normal"
    assert body["label"] in (0, 1)
    assert body["probabilities"] is not None


def test_predict_rejects_wrong_feature_count(client):
    _require_artifacts()
    response = client.post("/predict", json={"client_id": "user-1", "features": [0.1, 0.2]})
    assert response.status_code == 422


def test_predict_rejects_non_finite_features(client):
    """A strictly-compliant JSON client (like httpx's `json=` helper) refuses
    to even encode NaN/Infinity, so this sends a raw body with a literal NaN
    token instead -- Python's json module accepts that non-standard literal
    on parse, which is exactly the gap our own validator needs to close."""
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    features_json = "[" + ", ".join(["0.1"] * (n_features - 1) + ["NaN"]) + "]"
    body = f'{{"client_id": "user-1", "features": {features_json}}}'
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_predict_rejects_empty_client_id(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    response = client.post("/predict", json={"client_id": "", "features": [0.1] * n_features})
    assert response.status_code == 422


def test_predict_same_query_twice_is_identical(client):
    """Proves the discrepancy leak is closed: resending the same query must
    not let a client detect the defense by diffing two responses."""
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    rng = np.random.default_rng(7)
    features = rng.normal(scale=50, size=n_features).tolist()  # push toward critical tier

    r1 = client.post("/predict", json={"client_id": "attacker-1", "features": features})
    r2 = client.post("/predict", json={"client_id": "attacker-1", "features": features})
    assert r1.json()["label"] == r2.json()["label"]
    assert r1.json()["probabilities"] == r2.json()["probabilities"]
    assert r1.json()["tier"] == r2.json()["tier"]


def test_rate_limiting_returns_429(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    for _ in range(45):
        response = client.post("/predict", json={"client_id": "spammer", "features": [0.1] * n_features})
    assert response.status_code == 429


def test_admin_endpoints_require_api_key(client):
    _require_artifacts()
    response = client.get("/admin/stats")
    assert response.status_code == 401


def test_admin_endpoints_reject_wrong_api_key(client):
    _require_artifacts()
    response = client.get("/admin/stats", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_admin_stats_reflects_requests(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    client.post("/predict", json={"client_id": "user-2", "features": [0.1] * n_features})
    stats = client.get("/admin/stats", headers=ADMIN_HEADERS).json()
    assert stats["total_requests"] == 1
    assert stats["total_clients"] == 1


def test_toggle_defense_bypasses_layers(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    client.post("/admin/toggle-defense", json={"enabled": False}, headers=ADMIN_HEADERS)
    response = client.post("/predict", json={"client_id": "user-3", "features": [0.1] * n_features})
    body = response.json()
    assert body["tier"] == "normal"
    assert body["threat_index"] == 0.0
    client.post("/admin/toggle-defense", json={"enabled": True}, headers=ADMIN_HEADERS)


def test_verify_ownership_requires_api_key(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    response = client.post(
        "/verify-ownership",
        json={"client_ids": ["x"], "queries": [[0.1] * n_features], "suspect_labels": [0]},
    )
    assert response.status_code == 401


def test_verify_ownership_with_no_watermark_history_returns_unverified(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    response = client.post(
        "/verify-ownership",
        json={
            "client_ids": ["never-seen-client"],
            "queries": [[0.1] * n_features],
            "suspect_labels": [0],
        },
        headers=ADMIN_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is False
    assert body["total_triggers"] == 0
