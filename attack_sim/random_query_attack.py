"""Data-free extraction: the attacker has no seed data at all and probes
feature space with wide-range uniform random queries. This is the crudest and
most detectable attack -- Layer 2's macro (feature distortion) signal should
catch most of it on its own, since these queries routinely land far outside
the training manifold.
"""
from __future__ import annotations

import numpy as np


def generate_queries(n: int, n_features: int, low: float = -6.0, high: float = 6.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(n, n_features))
