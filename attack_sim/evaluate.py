"""Runs all three attacks (random-query, in-distribution, boundary) against
both the raw undefended target model and the full ModelVault gateway, trains
a surrogate on each attack's collected responses, and prints the resulting
undefended-vs-defended agreement comparison. This IS the demo number: lower
defended agreement means the gateway measurably degraded the quality of the
stolen clone.

Also runs the ownership-verification demo at the end: using the defended
boundary attack's watermark event log, it asks /verify-ownership whether the
resulting surrogate reproduces our watermark, and prints the statistical
confidence.
"""
from __future__ import annotations

import logging

import numpy as np
from fastapi.testclient import TestClient

logging.getLogger("httpx").setLevel(logging.WARNING)

from attack_sim import boundary_attack, in_distribution_attack, random_query_attack
from attack_sim.surrogate_trainer import train_surrogate
from modelvault.gateway import state as state_module
from modelvault.gateway.api import app
from modelvault.model.data_prep import generate_dataset
from modelvault.model.predict import load_target_model, predict_label

N_ATTACK_QUERIES = 800
N_SYBIL_CLIENTS = 25


def query_undefended(queries: np.ndarray) -> np.ndarray:
    return predict_label(queries)


def query_through_gateway(client: TestClient, queries: np.ndarray, attack_name: str) -> tuple[np.ndarray, list[dict]]:
    """Rotates through a pool of fake client ids -- an attacker splitting
    queries across sybil accounts to dodge Layer 1's per-client rate limit.
    Layer 2 still catches this because its detection is content-based and
    pools all clients into one shared reservoir, not per-identity."""
    labels = []
    raw_responses = []
    for i, query in enumerate(queries):
        client_id = f"{attack_name}-sybil-{i % N_SYBIL_CLIENTS}"
        response = client.post("/predict", json={"client_id": client_id, "features": query.tolist()})
        if response.status_code == 429:
            continue
        body = response.json()
        labels.append(body["label"])
        raw_responses.append({**body, "client_id": client_id, "features": query.tolist()})
    return np.array(labels), raw_responses


def agreement_with_target(surrogate, X_test: np.ndarray, target_labels: np.ndarray) -> float:
    surrogate_labels = surrogate.predict(X_test)
    return float(np.mean(surrogate_labels == target_labels))


def run_attack(name: str, query_generator_fn, client: TestClient, X_test: np.ndarray, target_test_labels: np.ndarray) -> dict:
    state_module.reset_state()

    undefended_queries = query_generator_fn()
    undefended_labels = query_undefended(undefended_queries)
    undefended_surrogate = train_surrogate(undefended_queries, undefended_labels)
    undefended_agreement = agreement_with_target(undefended_surrogate, X_test, target_test_labels)

    defended_queries = query_generator_fn()
    defended_labels, raw_responses = query_through_gateway(client, defended_queries, name)
    matched_queries = defended_queries[: len(defended_labels)]
    defended_surrogate = train_surrogate(matched_queries, defended_labels)
    defended_agreement = agreement_with_target(defended_surrogate, X_test, target_test_labels)

    return {
        "name": name,
        "undefended_agreement": undefended_agreement,
        "defended_agreement": defended_agreement,
        "raw_responses": raw_responses,
        "defended_surrogate": defended_surrogate,
    }


def run_ownership_verification_demo(client: TestClient, attack_result: dict) -> None:
    state = state_module.get_state()
    events = state.all_watermark_events()
    if not events:
        print("\nOwnership verification: no watermark triggers recorded for this attack, skipping.")
        return

    surrogate = attack_result["defended_surrogate"]
    client_ids = [event.client_id for event in events]
    queries = [event.query_features.tolist() for event in events]
    suspect_labels = surrogate.predict(np.array([event.query_features for event in events])).tolist()

    response = client.post(
        "/verify-ownership",
        json={
            "client_ids": client_ids,
            "queries": queries,
            "suspect_labels": suspect_labels,
        },
    )
    result = response.json()
    print(f"\nOwnership verification ({attack_result['name']} surrogate):")
    print(f"  verified={result['verified']}, {result['matches']}/{result['total_triggers']} watermark triggers matched, "
          f"confidence={result['confidence']:.2f}")


def main() -> None:
    load_target_model()  # fail fast with a clear error if training hasn't run yet
    _, X_test, _, y_test = generate_dataset()
    target_test_labels = predict_label(X_test)
    n_features = X_test.shape[1]

    client = TestClient(app)

    attacks = [
        ("random_query", lambda: random_query_attack.generate_queries(N_ATTACK_QUERIES, n_features, seed=1)),
        ("in_distribution", lambda: in_distribution_attack.generate_queries(N_ATTACK_QUERIES, seed=2)),
        ("boundary", lambda: boundary_attack.generate_queries(N_ATTACK_QUERIES, predict_fn=lambda x: int(predict_label(x)[0]), seed=3)),
    ]

    results = []
    print(f"{'Attack':<18} {'Undefended agreement':>22} {'Defended agreement':>20}")
    print("-" * 62)
    for name, generator_fn in attacks:
        result = run_attack(name, generator_fn, client, X_test, target_test_labels)
        results.append(result)
        print(f"{name:<18} {result['undefended_agreement']*100:>21.1f}% {result['defended_agreement']*100:>19.1f}%")

    boundary_result = next(r for r in results if r["name"] == "boundary")
    run_ownership_verification_demo(client, boundary_result)


if __name__ == "__main__":
    main()
