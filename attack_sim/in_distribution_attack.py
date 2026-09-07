"""In-distribution extraction: the attacker has some knowledge of what real
inputs look like (e.g. a public dataset in the same domain) and samples
queries that stay on the training manifold. Layer 2's macro signal alone is
mostly blind to this -- these queries score as normal by design. This is
exactly the gap the earlier ModelVault prototype had no answer for, and is
what the micro (coverage/redundancy) signal exists to catch: even though each
individual query looks legitimate, the attacker sweeps feature space far more
efficiently and with far less redundancy than real usage ever does.
"""
from __future__ import annotations

import numpy as np

from modelvault.model.predict import load_reference_model


def generate_queries(n: int, seed: int | None = None) -> np.ndarray:
    reference = load_reference_model()
    # Draw broadly across the reference manifold rather than clustering
    # tightly, to simulate an attacker deliberately maximizing coverage.
    samples, _ = reference.sample(n)
    rng = np.random.default_rng(seed)
    # Reference.sample() reuses its own internal seeded RNG regardless of our
    # seed, so perturb slightly and shuffle to get seed-dependent variety
    # while staying on-manifold.
    jitter = rng.normal(scale=0.05, size=samples.shape)
    queries = samples + jitter
    rng.shuffle(queries, axis=0)
    return queries
