"""Macro signal: how far incoming queries sit from the training data manifold.

Extraction attacks (random-query and boundary-search variants especially) tend
to probe feature space broadly, including regions the real training
distribution never covers. We use the fitted reference density model
(GaussianMixture, from modelvault/model/train.py) to score that distance via
negative log-likelihood.

Log-likelihood under a multi-dimensional Gaussian is heavily left-skewed (the
"curse of dimensionality": most probability mass sits away from the mode, so
even a typical in-distribution sample usually has much lower likelihood than
the density's peak). Linear min-max scaling against that distribution makes
ordinary traffic look artificially suspicious. Instead we rank a query's
likelihood against a calibration sample drawn from the reference model itself,
then apply a cubic decay so only the tail of genuinely rare/out-of-distribution
likelihoods pushes the score toward 100 -- the bulk of normal traffic
(everything above roughly the calibration median) stays comfortably in the
normal tier.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from modelvault.model.predict import load_reference_model

N_CALIBRATION_SAMPLES = 5000


@lru_cache(maxsize=1)
def _calibration_scores() -> np.ndarray:
    """Sorted log-likelihoods of a large sample drawn from the reference
    model, used as the empirical reference distribution for percentile ranks."""
    reference = load_reference_model()
    samples, _ = reference.sample(N_CALIBRATION_SAMPLES)
    scores = reference.score_samples(samples)
    return np.sort(scores)


def _percentile_rank(log_likelihood: float) -> float:
    """Fraction of the calibration distribution with likelihood <= this
    value. Low percentile = rarer/more distorted than most in-distribution traffic."""
    calibration = _calibration_scores()
    rank = np.searchsorted(calibration, log_likelihood, side="right")
    return rank / len(calibration)


def _distortion_from_percentile(percentile: float) -> float:
    # Cubic decay: only the bottom tail of the calibration distribution
    # (rare, low-likelihood queries) scores high; the median and above stay low.
    return float(((1.0 - percentile) ** 3) * 100.0)


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
    return ((1.0 - ranks) ** 3) * 100.0
