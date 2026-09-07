"""HopSkipJump-style boundary attack: repeatedly binary-searches along the
line between a point of each class to converge on points sitting exactly on
the decision boundary, then queries densely in that neighborhood. This is the
attack Layer 3's boundary_perturbation.py exists to defend against -- an
attacker who can precisely map the boundary can reconstruct a near-exact
surrogate with very few queries, and the deterministic label flip near the
boundary is designed to corrupt exactly this kind of query pattern.

This uses the TARGET model's raw predictions to steer the binary search
(as a real attacker would use whatever label the API returns) and then
records the AT-boundary queries as the attack's collected dataset. Candidate
seed points are drawn from real raw rows of the training data (not the
gateway's internal scaled reference density) so the generated queries match
the API's raw-features contract.

Two things matter for performance at real-world scale, and both mirror what
an actual attacker would also have to do:

  - Candidates are drawn stratified by PREDICTED label, not uniformly at
    random. At 0.17% fraud prevalence, two uniformly random rows would
    almost never disagree on predicted class -- a naive attacker doing pure
    random pairing would need tens of thousands of draws per boundary point.
  - The binary search runs on ALL boundary points simultaneously (one
    batched model call per step) rather than one point at a time. A real
    attacker querying a remote API can't usefully parallelize like this
    (each query is a network round trip), but they WOULD batch multiple
    independent binary searches into concurrent requests -- the model call
    count is the same either way, only the wall-clock shape differs. We
    batch here because we're calling the model in-process; it does not
    change how many queries the "attacker" makes, which is what the
    detection layers actually see and react to.
"""
from __future__ import annotations

import numpy as np

from modelvault.model.data_prep import generate_dataset


def generate_queries(n: int, predict_fn, seed: int | None = None, jitter_scale: float = 0.02, steps: int = 8) -> np.ndarray:
    """predict_fn: callable(raw_features_batch) -> array of predicted labels,
    one per row. Must accept RAW (unscaled) features -- the same contract
    /predict exposes over HTTP -- and must be batch-capable (shape
    (n_samples, n_features) in, shape (n_samples,) out)."""
    X_train, _, _, _ = generate_dataset()
    rng = np.random.default_rng(seed)
    column_scale = np.std(X_train, axis=0)

    all_labels = np.asarray(predict_fn(X_train))
    label_buckets = {label: np.where(all_labels == label)[0] for label in np.unique(all_labels)}

    if len(label_buckets) < 2:
        # Degenerate case: the model predicts only one class across all of
        # X_train. Nothing to binary-search toward -- fall back to plain sampling.
        idx = rng.integers(0, len(X_train), size=n)
        return X_train[idx]

    labels = list(label_buckets.keys())
    label_a_choices = rng.choice(labels, size=n)
    # label_b must differ from label_a on each row; with only 2 labels this
    # is just "the other one", generalizes to >2 classes via rejection.
    label_b_choices = np.array([rng.choice([l for l in labels if l != la]) for la in label_a_choices])

    a_points = np.vstack([X_train[rng.choice(label_buckets[la])] for la in label_a_choices])
    b_points = np.vstack([X_train[rng.choice(label_buckets[lb])] for lb in label_b_choices])

    low, high = a_points.copy(), b_points.copy()
    for _ in range(steps):
        mid = (low + high) / 2.0
        mid_labels = np.asarray(predict_fn(mid))
        matches_a = mid_labels == label_a_choices
        low[matches_a] = mid[matches_a]
        high[~matches_a] = mid[~matches_a]

    boundary_points = (low + high) / 2.0
    jitter = rng.normal(scale=jitter_scale, size=boundary_points.shape) * column_scale
    return boundary_points + jitter
