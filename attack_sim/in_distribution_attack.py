"""In-distribution extraction: the attacker has some knowledge of what real
inputs look like (e.g. a public dataset in the same domain, or a handful of
leaked/observed real transactions) and samples queries that stay on the
training manifold. Layer 2's macro signal alone is mostly blind to this --
these queries score as normal by design. This is exactly the gap the earlier
ModelVault prototype had no answer for, and is what the micro (coverage/
redundancy) signal exists to catch: even though each individual query looks
legitimate, the attacker sweeps feature space far more efficiently and with
far less redundancy than real usage ever does.

Queries are drawn from REAL raw rows of the training data (not the fitted
reference density model, which lives in the gateway's internal SCALED
feature space) -- this both matches the API's raw-features contract and is a
more realistic attacker model: someone attempting this attack has real or
real-like transaction records, not access to our internal density model.
"""
from __future__ import annotations

import numpy as np

from modelvault.model.data_prep import generate_dataset


def generate_queries(n: int, seed: int | None = None) -> np.ndarray:
    X_train, _, _, _ = generate_dataset()
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, len(X_train), size=n)
    base = X_train[idx]

    # Small relative jitter per-column so the attacker isn't replaying exact
    # rows verbatim, while staying on-manifold.
    column_scale = np.std(X_train, axis=0, keepdims=True)
    jitter = rng.normal(scale=0.02, size=base.shape) * column_scale
    return base + jitter
