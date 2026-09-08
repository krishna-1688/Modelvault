"""Macro signal: how far incoming queries sit from the training data manifold.

Extraction attacks (random-query and boundary-search variants especially) tend
to probe feature space broadly, including regions the real training
distribution never covers. We use the fitted reference density model
(GaussianMixture, from modelvault/model/train.py) to score that distance via
negative log-likelihood.

Log-likelihood under a multi-dimensional Gaussian is heavily left-skewed (the
"curse of dimensionality": most probability mass sits away from the mode, so
even a typical in-distribution sample usually has much lower likelihood than
the density's peak). Raw likelihoods are therefore meaningless as an absolute
scale. Instead a query's likelihood is ranked against REAL training traffic,
and that percentile is mapped to a 0-100 score.

Two calibration choices matter, and both were wrong in an earlier version in
ways that produced a 33% false-positive rate on legitimate customers:

  - Rank against REAL data, not against samples drawn from the fitted
    GaussianMixture. The mixture is an approximation; its own samples sit in
    a slightly different region than genuine records, so ranking real
    traffic against synthetic traffic mislabels ordinary customers.

  - The percentile -> score curve must be steep enough that only genuinely
    rare inputs score high. A cubic curve scored a transaction in the bottom
    10% of REAL traffic at 73/100 -- flagged critical for the crime of being
    slightly unusual, which is not an attack. DECAY_EXPONENT is set so the
    5th percentile of genuine traffic lands near the elevated threshold (35),
    which caps the expected false-positive rate at roughly 5% by
    construction while still driving genuinely off-manifold inputs to ~100.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from modelvault.model.predict import load_reference_model

N_CALIBRATION_SAMPLES = 8000
# (1 - 0.05) ** 20 ~= 0.36, so the 5th percentile of genuine traffic scores
# ~36 -- just into the elevated tier. See module docstring.
DECAY_EXPONENT = 20


@lru_cache(maxsize=1)
def _calibration_scores() -> np.ndarray:
    """Sorted log-likelihoods of REAL training traffic under the reference
    density -- the empirical distribution a live query is ranked against."""
    # Imported lazily: this module is imported during gateway startup, and
    # data_prep pulls in the full dataset.
    from modelvault.model.data_prep import generate_dataset
    from modelvault.model.predict import transform_features

    reference = load_reference_model()
    X_train, _, _, _ = generate_dataset()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_train), size=min(N_CALIBRATION_SAMPLES, len(X_train)), replace=False)
    scores = reference.score_samples(transform_features(X_train[idx]))
    return np.sort(scores)


def _percentile_rank(log_likelihood: float) -> float:
    """Fraction of REAL traffic that is at least as unlikely as this query.
    Low percentile = rarer than almost all genuine records."""
    calibration = _calibration_scores()
    rank = np.searchsorted(calibration, log_likelihood, side="right")
    return rank / len(calibration)


def _distortion_from_percentile(percentile: float) -> float:
    return float(((1.0 - percentile) ** DECAY_EXPONENT) * 100.0)


def feature_distortion_score(features: np.ndarray) -> float:
    """Returns a 0-100 distortion score for a single query. Higher = further from
    the training manifold = more suspicious."""
    reference = load_reference_model()
    X = np.atleast_2d(features)
    log_likelihood = reference.score_samples(X)[0]
    percentile = _percentile_rank(log_likelihood)
    return _distortion_from_percentile(percentile)


def batch_feature_distortion_score(feature_matrix: np.ndarray) -> np.ndarray:
    if feature_matrix.size == 0:
        return np.empty((0,))
    reference = load_reference_model()
    log_likelihoods = reference.score_samples(feature_matrix)
    calibration = _calibration_scores()
    ranks = np.searchsorted(calibration, log_likelihoods, side="right") / len(calibration)
    return ((1.0 - ranks) ** DECAY_EXPONENT) * 100.0
