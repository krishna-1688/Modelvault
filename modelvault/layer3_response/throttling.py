"""Layer 3 -- graduated response degradation based on the Layer 2 threat index.

Tiers (from config/settings.yaml):
  0-35   (normal):   full response -- exact probabilities, no degradation
  36-70  (elevated):  rounded confidence -- probabilities rounded to reduce
                      the precision an attacker can extract per query
  71-100 (critical): label-only -- only the predicted class is returned,
                      plus boundary noise (see boundary_perturbation.py)

This is graduated rather than a hard block specifically because a binary
block either locks out real users hitting a false positive, or hands the
attacker a clean signal of exactly where the detection threshold sits.
Degrading smoothly avoids both failure modes.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from modelvault.utils.config_loader import get_settings


class ResponseTier(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


def classify_tier(threat_index: float) -> ResponseTier:
    settings = get_settings()
    if threat_index <= settings.threat_index.tier_normal_max:
        return ResponseTier.NORMAL
    if threat_index <= settings.threat_index.tier_elevated_max:
        return ResponseTier.ELEVATED
    return ResponseTier.CRITICAL


def apply_throttling(probabilities: np.ndarray, tier: ResponseTier) -> dict:
    """Takes the raw predict_proba output (shape (n_classes,)) and returns a
    dict describing what the client actually receives at this tier."""
    probabilities = np.asarray(probabilities).flatten()
    predicted_label = int(np.argmax(probabilities))

    if tier == ResponseTier.NORMAL:
        return {
            "tier": tier.value,
            "label": predicted_label,
            "probabilities": probabilities.round(6).tolist(),
        }

    if tier == ResponseTier.ELEVATED:
        # Round to 1 decimal place -- still useful for legitimate confidence
        # thresholds, but far too coarse to reconstruct decision-boundary shape.
        rounded = np.round(probabilities, 1)
        return {
            "tier": tier.value,
            "label": predicted_label,
            "probabilities": rounded.tolist(),
        }

    # CRITICAL: label only, no probabilities at all.
    return {
        "tier": tier.value,
        "label": predicted_label,
        "probabilities": None,
    }
