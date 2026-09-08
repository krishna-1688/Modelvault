"""Fuses the macro (feature distortion) and micro (coverage/redundancy)
signals into a single 0-100 Threat Index that drives Layer 3's response.

The two signals detect DIFFERENT attacks, and each is silent for the attack
the other catches:

  - A data-free attacker probing with synthetic inputs trips macro (their
    queries do not look like real records) while micro stays low.
  - An attacker replaying real records they already hold trips micro (they
    sweep the space without repeating) while macro stays low -- correctly
    so, because every individual query genuinely IS a real record.

That makes a weighted average the wrong fusion. Averaging lets a quiet
signal cancel a screaming one: an in-distribution sweep scoring macro 20 /
micro 80 would land at 44 under a 0.6/0.4 average and never cross the
elevated threshold, even though micro is certain. This was measurable --
under mixed legitimate + attacker traffic the averaged index put both
populations in the same band and the detector was effectively blind.

Instead the signals combine as a probabilistic OR (noisy-or):

    threat = 1 - (1 - macro)(1 - micro)

Either signal alone can raise suspicion, both together raise it further, and
the result still saturates cleanly at 100. Legitimate traffic, which scores
low on both, stays low: 20 and 10 fuse to 28, comfortably normal.
"""
from __future__ import annotations

import numpy as np

from modelvault.layer2_detection.coverage_density import coverage_density_score_single
from modelvault.layer2_detection.feature_distortion import feature_distortion_score
from modelvault.layer2_detection.reservoir import QueryReservoir


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

    m, u = macro / 100.0, micro / 100.0
    fused = float(np.clip((1.0 - (1.0 - m) * (1.0 - u)) * 100.0, 0.0, 100.0))

    return {
        "threat_index": fused,
        "macro_feature_distortion": macro,
        "micro_coverage_density": micro,
    }
