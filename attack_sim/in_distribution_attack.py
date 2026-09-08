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
    """Distinct real transactions, sampled WITHOUT replacement -- an attacker
    maximizing information per query has no reason to ask the same thing
    twice. That non-repetition is the whole signature: individually every
    query is a genuine transaction (so the macro signal correctly stays
    quiet), but collectively they sweep the space far more broadly than any
    real consumer's traffic does, which is what the micro signal reads.

    Deliberately almost no synthetic jitter, for the same reason as
    legit_client: independent noise across 29 dimensions pushes points off
    the manifold and would let the macro signal separate the populations for
    the wrong reason -- flattering the defense with a simulator artifact.
    """
    X_train, _, _, _ = generate_dataset()
    rng = np.random.default_rng(seed)

    take = min(n, len(X_train))
    idx = rng.choice(len(X_train), size=take, replace=False)
    base = X_train[idx]
    if take < n:  # only if someone asks for more queries than rows available
        extra = X_train[rng.integers(0, len(X_train), size=n - take)]
        base = np.vstack([base, extra])
    return base * (1.0 + rng.normal(scale=0.002, size=base.shape))
