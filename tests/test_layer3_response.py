import numpy as np

from modelvault.layer3_response.boundary_perturbation import (
    apply_boundary_perturbation,
    near_boundary,
)
from modelvault.layer3_response.throttling import ResponseTier, apply_throttling, classify_tier


def test_classify_tier_boundaries():
    assert classify_tier(0) == ResponseTier.NORMAL
    assert classify_tier(35) == ResponseTier.NORMAL
    assert classify_tier(36) == ResponseTier.ELEVATED
    assert classify_tier(70) == ResponseTier.ELEVATED
    assert classify_tier(71) == ResponseTier.CRITICAL
    assert classify_tier(100) == ResponseTier.CRITICAL


def test_normal_tier_returns_full_precision_probabilities():
    probs = np.array([0.123456, 0.876544])
    result = apply_throttling(probs, ResponseTier.NORMAL)
    assert result["probabilities"] == probs.round(6).tolist()
    assert result["label"] == 1


def test_elevated_tier_rounds_probabilities():
    probs = np.array([0.123456, 0.876544])
    result = apply_throttling(probs, ResponseTier.ELEVATED)
    assert result["probabilities"] == [0.1, 0.9]


def test_critical_tier_returns_label_only():
    probs = np.array([0.123456, 0.876544])
    result = apply_throttling(probs, ResponseTier.CRITICAL)
    assert result["probabilities"] is None
    assert result["label"] == 1


def test_near_boundary_detection():
    assert near_boundary(np.array([0.5, 0.5]), margin=0.1) is True
    assert near_boundary(np.array([0.05, 0.95]), margin=0.1) is False


def test_boundary_perturbation_is_deterministic():
    query = np.array([1.0, 2.0, 3.0])
    probs = np.array([0.5, 0.5])
    result_a = apply_boundary_perturbation(0, probs, "client-x", query)
    result_b = apply_boundary_perturbation(0, probs, "client-x", query)
    assert result_a == result_b


def test_boundary_perturbation_leaves_confident_predictions_alone():
    query = np.array([1.0, 2.0, 3.0])
    probs = np.array([0.02, 0.98])
    result = apply_boundary_perturbation(1, probs, "client-x", query)
    assert result == 1
