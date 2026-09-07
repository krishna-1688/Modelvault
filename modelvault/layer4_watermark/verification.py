"""/verify-ownership logic: proves a suspect model was trained on ModelVault's
watermarked responses, with a statistical confidence score.

The idea: for every historical query that was a watermark trigger (decided in
watermark.py, deterministic from (client_id, query)), we know exactly what
label our gateway returned instead of the true prediction. If a suspect model
was trained by scraping our API, a surrogate trained on those responses will
have learned to reproduce that same "wrong" watermark label on those specific
queries far more often than chance would predict. A surrogate that was NOT
trained on our output has no reason to agree with an arbitrary watermark
label above the random-chance rate.

Confidence is computed as 1 - p_value of a one-sided exact binomial test:
"what is the probability of seeing at least this many matches by chance if
the suspect model's agreement with our watermark labels were pure luck?"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WatermarkEvent:
    client_id: str
    query_features: np.ndarray
    expected_label: int  # the watermark-flipped label our gateway returned


@dataclass
class VerificationResult:
    verified: bool
    matches: int
    total_triggers: int
    confidence: float
    chance_rate: float


def _binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return min(1.0, total)


def verify_ownership(
    watermark_events: list[WatermarkEvent],
    suspect_predict_fn,
    n_classes: int,
    confidence_threshold: float = 0.95,
) -> VerificationResult:
    """suspect_predict_fn: callable(query_features) -> predicted label (int),
    from the model under suspicion of being a surrogate clone."""
    total = len(watermark_events)
    if total == 0:
        return VerificationResult(verified=False, matches=0, total_triggers=0, confidence=0.0, chance_rate=0.0)

    matches = 0
    for event in watermark_events:
        suspect_label = suspect_predict_fn(event.query_features)
        if int(suspect_label) == int(event.expected_label):
            matches += 1

    # An unrelated model has no reason to favor the specific (wrong) watermark
    # label over any other class, so its chance of matching it is the uniform
    # base rate 1 / n_classes. (Conditioning on "not the true label" would
    # degenerate to 1.0 for binary classifiers, which is wrong here: an
    # unrelated model isn't guessing among non-true classes, it's guessing
    # among all classes.)
    chance_rate = 1.0 / n_classes
    p_value = _binomial_sf(matches, total, chance_rate)
    confidence = 1.0 - p_value

    return VerificationResult(
        verified=confidence >= confidence_threshold,
        matches=matches,
        total_triggers=total,
        confidence=confidence,
        chance_rate=chance_rate,
    )
