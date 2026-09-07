import numpy as np

from modelvault.layer3_response.throttling import ResponseTier
from modelvault.layer4_watermark.verification import WatermarkEvent, verify_ownership
from modelvault.layer4_watermark.watermark import (
    decide_and_apply_watermark,
    is_watermark_trigger,
    watermark_target_label,
)


def test_normal_tier_never_watermarked():
    query = np.array([1.0, 2.0])
    decision = decide_and_apply_watermark("client-a", query, predicted_label=0, n_classes=2, tier=ResponseTier.NORMAL)
    assert decision.triggered is False
    assert decision.final_label == 0


def test_watermark_decision_is_deterministic():
    query = np.array([1.0, 2.0, 3.0])
    d1 = decide_and_apply_watermark("client-a", query, predicted_label=0, n_classes=2, tier=ResponseTier.CRITICAL)
    d2 = decide_and_apply_watermark("client-a", query, predicted_label=0, n_classes=2, tier=ResponseTier.CRITICAL)
    assert d1 == d2


def test_watermark_trigger_rate_roughly_matches_config_over_many_queries():
    rng = np.random.default_rng(0)
    triggers = 0
    n = 2000
    for i in range(n):
        query = rng.normal(size=5)
        if is_watermark_trigger(f"client-{i}", query, ResponseTier.CRITICAL):
            triggers += 1
    observed_rate = triggers / n
    # flip_rate_critical is 0.45 in default settings.yaml; allow generous tolerance
    # since this is a hash-based deterministic draw, not a true RNG.
    assert 0.35 < observed_rate < 0.55


def test_watermark_target_label_differs_from_true_label():
    query = np.array([1.0, 2.0])
    target = watermark_target_label("client-a", query, true_label=0, n_classes=3)
    assert target != 0
    assert target in (1, 2)


def test_verify_ownership_detects_trained_surrogate():
    n_classes = 2
    events = []
    rng = np.random.default_rng(1)
    for i in range(60):
        query = rng.normal(size=4)
        expected = watermark_target_label(f"client-{i}", query, true_label=0, n_classes=n_classes)
        events.append(WatermarkEvent(client_id=f"client-{i}", query_features=query, expected_label=expected))

    def surrogate_predict(query_features):
        for event in events:
            if np.array_equal(event.query_features, query_features):
                return event.expected_label
        return 0

    result = verify_ownership(events, surrogate_predict, n_classes=n_classes)
    assert result.verified is True
    assert result.matches == 60
    assert result.confidence > 0.99


def test_verify_ownership_rejects_unrelated_model():
    n_classes = 2
    events = []
    rng = np.random.default_rng(2)
    for i in range(60):
        query = rng.normal(size=4)
        expected = watermark_target_label(f"client-{i}", query, true_label=0, n_classes=n_classes)
        events.append(WatermarkEvent(client_id=f"client-{i}", query_features=query, expected_label=expected))

    def unrelated_predict(query_features):
        return 0  # always predicts the true label, never the watermark flip

    result = verify_ownership(events, unrelated_predict, n_classes=n_classes)
    assert result.verified is False
