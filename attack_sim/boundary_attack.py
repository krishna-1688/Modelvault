"""HopSkipJump-style boundary attack: repeatedly binary-searches along the
line between a point of each class to converge on points sitting exactly on
the decision boundary, then queries densely in that neighborhood. This is the
attack Layer 3's boundary_perturbation.py exists to defend against -- an
attacker who can precisely map the boundary can reconstruct a near-exact
surrogate with very few queries, and the deterministic label flip near the
boundary is designed to corrupt exactly this kind of query pattern.

This uses the TARGET model's raw predictions to steer the binary search
(as a real attacker would use whatever label the API returns) and then
records the AT-boundary queries as the attack's collected dataset.
"""
from __future__ import annotations

import numpy as np

from modelvault.model.predict import load_reference_model


def _binary_search_to_boundary(
    predict_fn,
    point_a: np.ndarray,
    point_b: np.ndarray,
    label_a: int,
    steps: int = 8,
) -> np.ndarray:
    """point_a and point_b must have different predicted labels under predict_fn."""
    low, high = point_a.copy(), point_b.copy()
    for _ in range(steps):
        mid = (low + high) / 2.0
        mid_label = predict_fn(mid)
        if mid_label == label_a:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate_queries(n: int, predict_fn, seed: int | None = None, jitter_scale: float = 0.05) -> np.ndarray:
    """predict_fn: callable(features) -> single predicted label, used to steer
    the boundary search exactly as an attacker would use the API's own label."""
    reference = load_reference_model()
    rng = np.random.default_rng(seed)

    boundary_points: list[np.ndarray] = []
    attempts = 0
    while len(boundary_points) < n and attempts < n * 20:
        attempts += 1
        candidates, _ = reference.sample(2)
        a, b = candidates[0], candidates[1]
        label_a, label_b = predict_fn(a), predict_fn(b)
        if label_a == label_b:
            continue
        boundary_point = _binary_search_to_boundary(predict_fn, a, b, label_a)
        # Query densely around the found boundary point, not just the point itself.
        boundary_points.append(boundary_point + rng.normal(scale=jitter_scale, size=boundary_point.shape))

    if not boundary_points:
        # Degenerate fallback if the model never disagreed across sampled pairs.
        samples, _ = reference.sample(n)
        return samples

    result = np.vstack(boundary_points)
    if len(result) < n:
        # Pad by resampling near existing boundary points if we hit the attempt cap.
        extra_idx = rng.integers(0, len(result), size=n - len(result))
        extra = result[extra_idx] + rng.normal(scale=jitter_scale, size=(n - len(result), result.shape[1]))
        result = np.vstack([result, extra])
    return result[:n]
