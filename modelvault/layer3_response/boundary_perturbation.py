"""Stochastic-looking noise applied near the decision boundary at the critical
response tier, aimed specifically at boundary-search attacks (e.g.
HopSkipJump-style) that query densely around the decision boundary to map its
exact shape.

Critical-tier responses are label-only (see throttling.py), so there are no
probabilities left to perturb -- instead, when a query's true prediction sits
within `boundary_margin` of the decision boundary, we deterministically flip
the returned label with a query-dependent probability. "Deterministic" here
matters for the same reason as Layer 4's watermark: the perturbation is seeded
from a hash of (client_id, query, salt), so it looks random to an attacker but
a client resending the identical query gets the identical (possibly flipped)
answer -- no discrepancy for an attacker to detect the defense by.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_BOUNDARY_MARGIN = 0.1
DEFAULT_FLIP_PROBABILITY = 0.3


def _get_salt() -> str:
    return os.environ.get("SECRET_SALT", "dev-only-insecure-default-salt")


def _deterministic_unit_interval(client_id: str, query_features: np.ndarray) -> float:
    """Maps (client_id, query, salt) to a deterministic value in [0, 1)."""
    salt = _get_salt()
    payload = f"{salt}:{client_id}:".encode() + np.asarray(query_features).tobytes()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def near_boundary(probabilities: np.ndarray, margin: float = DEFAULT_BOUNDARY_MARGIN) -> bool:
    probabilities = np.asarray(probabilities).flatten()
    top_two = np.sort(probabilities)[-2:]
    return bool((top_two[-1] - top_two[-2]) < margin)


def apply_boundary_perturbation(
    predicted_label: int,
    probabilities: np.ndarray,
    client_id: str,
    query_features: np.ndarray,
    margin: float = DEFAULT_BOUNDARY_MARGIN,
    flip_probability: float = DEFAULT_FLIP_PROBABILITY,
) -> int:
    """Returns the label the client should see: unchanged unless the query
    sits near the decision boundary, in which case it is deterministically
    flipped with probability `flip_probability`."""
    if not near_boundary(probabilities, margin=margin):
        return predicted_label

    draw = _deterministic_unit_interval(client_id, query_features)
    if draw < flip_probability:
        n_classes = np.asarray(probabilities).flatten().shape[0]
        return (predicted_label + 1) % n_classes
    return predicted_label
