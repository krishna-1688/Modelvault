"""Simulates LEGITIMATE API consumers -- the paying customers whose traffic
must not be disturbed while the gateway hunts for extraction attempts.

The whole value proposition of a defense like this rests on telling these
two populations apart from CONTENT alone, so it matters that the simulated
legitimate traffic differs from an attack in the ways real traffic actually
does, not in ways that would make the job artificially easy:

  - Real consumers query a NARROW, REPETITIVE slice of the space. A payments
    processor scores the same kinds of customer profiles over and over; it
    does not sweep the feature space. That produces high redundancy between
    nearby queries, which is exactly what Layer 2's micro (coverage/
    redundancy) signal reads as normal.
  - An extraction attack does the opposite: it maximizes information per
    query, covering as much distinct ground as possible, avoiding repeats.

  - Real consumers are also stable, named identities with modest volume --
    not 25 freshly-minted ids splitting one burst between them.

Note the simulated consumers still send REAL transactions drawn from the same
dataset the attacker draws from. They are not made easy to classify by using
different data -- only by using it the way a real customer would.
"""
from __future__ import annotations

import numpy as np

from modelvault.model.data_prep import generate_dataset

# A few stable, named consumers, as a real deployment would have.
CONSUMER_IDS = [
    "acme-payments-prod",
    "northwind-checkout",
    "globex-risk-api",
    "initech-billing",
]

PROFILE_POOL_SIZE = 10   # each consumer sees a small, recurring set of customer profiles


def build_consumer_profiles(seed: int | None = None) -> dict[str, np.ndarray]:
    """Gives each consumer its own small pool of recurring transaction
    profiles -- the 'regulars' that account for most of its traffic."""
    X_train, _, _, _ = generate_dataset()
    rng = np.random.default_rng(seed)
    profiles = {}
    for consumer in CONSUMER_IDS:
        idx = rng.choice(len(X_train), size=PROFILE_POOL_SIZE, replace=False)
        profiles[consumer] = X_train[idx]
    return profiles


def next_query(profiles: dict[str, np.ndarray], consumer: str, rng: np.random.Generator) -> np.ndarray:
    """One realistic request: a recurring customer profile, sent essentially
    as-is.

    Deliberately almost no synthetic noise. Adding independent Gaussian
    jitter across all 29 dimensions sounds harmless but in high-dimensional
    space it pushes a point far off the data manifold -- enough that the
    macro signal flags it as distorted. That would be an artifact of the
    simulator, not of real consumer behaviour: a payments API receives
    genuine transactions, not real transactions plus noise. Keeping these as
    real rows means macro correctly stays quiet for BOTH populations, and
    the two are separated by query PATTERN (Layer 2's micro signal) --
    which is the actual thesis of this project.
    """
    pool = profiles[consumer]
    base = pool[rng.integers(0, len(pool))]
    # A whisper of variation so repeat requests aren't byte-identical, far
    # below the scale that would move the point off-manifold.
    return base * (1.0 + rng.normal(scale=0.002, size=base.shape))
