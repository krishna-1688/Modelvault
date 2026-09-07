"""Fuses the macro (feature distortion) and micro (coverage/redundancy) signals
into a single 0-100 Threat Index that drives Layer 3's graduated response.

Fusion is a simple weighted average: both signals are already scaled 0-100
independently, so no further calibration is needed here. Macro is weighted
slightly higher because it directly measures distance from real usage, while
micro is a corroborating signal that catches attacks macro alone would miss
(e.g. in-distribution sweeps that never leave the training manifold).
"""
from __future__ import annotations

import numpy as np

from modelvault.layer2_detection.coverage_density import coverage_density_score_single
from modelvault.layer2_detection.feature_distortion import feature_distortion_score
from modelvault.layer2_detection.reservoir import QueryReservoir

MACRO_WEIGHT = 0.6
MICRO_WEIGHT = 0.4


def compute_threat_index(
    query_features: np.ndarray,
    reservoir: QueryReservoir,
) -> dict:
    """Computes the fused threat index for a new query, given the current
    (pre-insertion) state of the reservoir. Returns a dict with the fused
    score plus each component, so callers/tests/dashboard can inspect why."""
    macro = feature_distortion_score(query_features)

    window_matrix = reservoir.get_feature_matrix()
    micro = coverage_density_score_single(query_features, window_matrix) if len(reservoir) > 0 else 0.0

    fused = MACRO_WEIGHT * macro + MICRO_WEIGHT * micro
    fused = float(np.clip(fused, 0.0, 100.0))

    return {
        "threat_index": fused,
        "macro_feature_distortion": macro,
        "micro_coverage_density": micro,
    }
