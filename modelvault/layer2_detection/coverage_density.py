"""Micro signal: query coverage/redundancy within the reservoir window.

Real usage tends to cluster around a handful of "interesting" regions of
feature space and naturally repeats similar queries. An in-distribution
extraction attack instead sweeps feature space efficiently and deliberately
avoids redundancy, to maximize information gained per query. We approximate
that by looking at each query's distance to its nearest neighbors within the
current window: sparse, evenly-spread neighborhoods (low redundancy, high
coverage) score as more suspicious than tight, repeat-heavy clusters.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def coverage_density_score(feature_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """Returns a 0-100 suspicion score per row in feature_matrix, based on how
    sparsely each query's local neighborhood is populated relative to the rest
    of the window. Requires at least k+1 rows to compute meaningful neighbor
    distances; returns zeros otherwise."""
    n = len(feature_matrix)
    if n <= k:
        return np.zeros(n)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(feature_matrix)
    distances, _ = nn.kneighbors(feature_matrix)
    # Exclude the zero-distance self-match in column 0.
    mean_neighbor_dist = distances[:, 1:].mean(axis=1)

    low = float(mean_neighbor_dist.min())
    high = float(mean_neighbor_dist.max())
    if high == low:
        return np.zeros(n)

    normalized = (mean_neighbor_dist - low) / (high - low)
    return normalized * 100.0


def coverage_density_score_single(query_features: np.ndarray, feature_matrix: np.ndarray, k: int = 5) -> float:
    """Scores a single new query against the existing window contents (window
    should NOT yet include this query). The query's distance to its k nearest
    window neighbors is normalized against the window's own internal
    neighbor-distance spread (each window point vs. its neighbors), so the
    scale stays consistent with coverage_density_score."""
    n = len(feature_matrix)
    if n < k:
        return 0.0

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(feature_matrix)
    distances, _ = nn.kneighbors(np.atleast_2d(query_features))
    mean_dist = float(distances[0].mean())

    internal_distances, _ = nn.kneighbors(feature_matrix)
    # feature_matrix rows include themselves as a neighbor (distance 0); drop it.
    reference_dists = internal_distances[:, 1:].mean(axis=1)

    low = float(reference_dists.min())
    high = float(reference_dists.max())
    if high == low:
        return 0.0

    normalized = np.clip((mean_dist - low) / (high - low), 0.0, 1.0)
    return float(normalized * 100.0)
