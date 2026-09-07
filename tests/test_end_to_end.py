import numpy as np
import pytest
from fastapi.testclient import TestClient

from modelvault.gateway.api import app
from modelvault.gateway import state as state_module
from modelvault.model.predict import load_reference_model, load_target_model


@pytest.fixture(autouse=True)
def fresh_state():
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


def test_predict_returns_full_response_for_normal_traffic(client):
    _require_artifacts()
    reference = load_reference_model()
    sample, _ = reference.sample(1)
    response = client.post("/predict", json={"client_id": "user-1", "features": sample[0].tolist()})
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == "normal"
    assert body["label"] in (0, 1)
    assert body["probabilities"] is not None


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


def test_admin_stats_reflects_requests(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    client.post("/predict", json={"client_id": "user-2", "features": [0.1] * n_features})
    stats = client.get("/admin/stats").json()
    assert stats["total_requests"] == 1
    assert stats["total_clients"] == 1


def test_toggle_defense_bypasses_layers(client):
    _require_artifacts()
    n_features = load_target_model().n_features_in_
    client.post("/admin/toggle-defense", json={"enabled": False})
    response = client.post("/predict", json={"client_id": "user-3", "features": [0.1] * n_features})
    body = response.json()
    assert body["tier"] == "normal"
    assert body["threat_index"] == 0.0
    client.post("/admin/toggle-defense", json={"enabled": True})


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
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is False
    assert body["total_triggers"] == 0
