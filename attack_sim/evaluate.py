"""Runs all three attacks (random-query, in-distribution, boundary) against
both the raw undefended target model and the full ModelVault gateway, trains
a surrogate on each attack's collected responses, and prints the resulting
undefended-vs-defended agreement comparison. This IS the demo number: lower
defended agreement means the gateway measurably degraded the quality of the
stolen clone.

Agreement is measured on a CLASS-BALANCED evaluation set, not a random
held-out slice. At 0.17% fraud prevalence, a trivial "always predict
not-fraud" surrogate would already agree with the target model on >99.8% of
a random slice -- that number would be technically true and completely
uninformative about clone quality. Balancing the evaluation set (all fraud
rows + an equal sample of legitimate rows) is the standard fix for measuring
model fidelity under severe class imbalance, the same reason precision/
recall/F1/PR-AUC are used instead of accuracy in train.py.

Each attack is run across multiple independent random seeds and the
agreement numbers are averaged. A single run's surrogate is a KNN fit on a
few thousand points evaluated against a ~200-row balanced set -- individual
runs can and do shift by several points either direction on pure query-
sampling luck, occasionally even flipping which side "wins" for attacks with
a small underlying effect size. Reporting a mean over independent seeds
(with the spread) is the standard fix, and is a more honest number than
citing whichever single run happened to look best.

Also runs the ownership-verification demo at the end: using the defended
boundary attack's watermark event log from its LAST seed, it asks
/verify-ownership whether the resulting surrogate reproduces our watermark,
and prints the statistical confidence.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi.testclient import TestClient

logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
ADMIN_HEADERS = {"X-API-Key": os.environ["ADMIN_API_KEY"]} if os.environ.get("ADMIN_API_KEY") else {}

from attack_sim import boundary_attack, in_distribution_attack, random_query_attack
from attack_sim.surrogate_trainer import train_surrogate
from modelvault.gateway import state as state_module
from modelvault.gateway.api import app
from modelvault.model.data_prep import generate_dataset
from modelvault.model.predict import load_target_model, predict_raw_label

N_ATTACK_QUERIES = 800
N_SEEDS_PER_ATTACK = 2
N_SYBIL_CLIENTS = 25


def build_balanced_eval_set(X_test: np.ndarray, y_test: np.ndarray, seed: int = 0) -> np.ndarray:
    """All fraud rows plus an equal-sized random sample of legitimate rows,
    so agreement isn't dominated by the majority class."""
    rng = np.random.default_rng(seed)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    sampled_legit_idx = rng.choice(legit_idx, size=len(fraud_idx), replace=False)
    balanced_idx = np.concatenate([fraud_idx, sampled_legit_idx])
    rng.shuffle(balanced_idx)
    return X_test[balanced_idx]


def query_undefended(queries: np.ndarray) -> np.ndarray:
    return predict_raw_label(queries)


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


def agreement_with_target(surrogate, X_eval: np.ndarray, target_labels: np.ndarray) -> float:
    surrogate_labels = surrogate.predict(X_eval)
    return float(np.mean(surrogate_labels == target_labels))


def run_attack(name: str, query_generator_fn, client: TestClient, X_eval: np.ndarray, target_eval_labels: np.ndarray) -> dict:
    """Both passes use the SAME generated query set -- only the labels differ
    (the gateway may corrupt some via throttling/watermarking). Generating
    two independent random samples here would confound "effect of the
    defense" with plain query-sampling variance, which at these sample sizes
    and effect sizes can be larger than the effect itself and even flip the
    sign of the measured gap."""
    state_module.reset_state()

    queries = query_generator_fn()

    undefended_labels = query_undefended(queries)
    undefended_surrogate = train_surrogate(queries, undefended_labels)
    undefended_agreement = agreement_with_target(undefended_surrogate, X_eval, target_eval_labels)

    defended_labels, raw_responses = query_through_gateway(client, queries, name)
    matched_queries = queries[: len(defended_labels)]
    defended_surrogate = train_surrogate(matched_queries, defended_labels)
    defended_agreement = agreement_with_target(defended_surrogate, X_eval, target_eval_labels)

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
        headers=ADMIN_HEADERS,
    )
    result = response.json()
    print(f"\nOwnership verification ({attack_result['name']} surrogate):")
    print(f"  verified={result['verified']}, {result['matches']}/{result['total_triggers']} watermark triggers matched, "
          f"confidence={result['confidence']:.2f}")


def main() -> None:
    load_target_model()  # fail fast with a clear error if training hasn't run yet
    _, X_test, _, y_test = generate_dataset()
    X_eval = build_balanced_eval_set(X_test, y_test)
    target_eval_labels = predict_raw_label(X_eval)
    n_features = X_test.shape[1]

    print(f"Evaluating on a class-balanced set of {len(X_eval)} transactions "
          f"({(target_eval_labels == 1).sum()} predicted fraud, {(target_eval_labels == 0).sum()} predicted legitimate).\n")
    print(f"Each attack averaged over {N_SEEDS_PER_ATTACK} independent seeds ({N_ATTACK_QUERIES} queries each).\n")

    client = TestClient(app)

    attack_generators = {
        "random_query": lambda seed: random_query_attack.generate_queries(N_ATTACK_QUERIES, n_features, seed=seed),
        "in_distribution": lambda seed: in_distribution_attack.generate_queries(N_ATTACK_QUERIES, seed=seed),
        "boundary": lambda seed: boundary_attack.generate_queries(N_ATTACK_QUERIES, predict_fn=predict_raw_label, seed=seed),
    }

    print(f"{'Attack':<18} {'Undefended agreement':>26} {'Defended agreement':>26}")
    print("-" * 72)

    last_boundary_result = None
    for attack_idx, (name, generator_fn) in enumerate(attack_generators.items()):
        undefended_runs, defended_runs = [], []
        for seed_idx in range(N_SEEDS_PER_ATTACK):
            seed = attack_idx * 100 + seed_idx  # distinct, deterministic seed per (attack, run)
            result = run_attack(name, lambda s=seed: generator_fn(s), client, X_eval, target_eval_labels)
            undefended_runs.append(result["undefended_agreement"])
            defended_runs.append(result["defended_agreement"])
            if name == "boundary":
                last_boundary_result = result

        u_mean, u_std = np.mean(undefended_runs) * 100, np.std(undefended_runs) * 100
        d_mean, d_std = np.mean(defended_runs) * 100, np.std(defended_runs) * 100
        print(f"{name:<18} {u_mean:>18.1f}% (+/-{u_std:>4.1f}) {d_mean:>18.1f}% (+/-{d_std:>4.1f})")

    run_ownership_verification_demo(client, last_boundary_result)


if __name__ == "__main__":
    main()
