import numpy as np
import pytest

from modelvault.layer2_detection.reservoir import QueryReservoir
from modelvault.layer2_detection.threat_index import compute_threat_index
from modelvault.model.predict import load_reference_model

pytestmark = pytest.mark.filterwarnings("ignore")


def _require_artifacts():
    try:
        load_reference_model()
    except FileNotFoundError:
        pytest.skip("Run `python -m modelvault.model.train` before this test.")


def test_reservoir_respects_window_size():
    reservoir = QueryReservoir(window_size=5)
    for i in range(10):
        reservoir.add(f"client-{i}", np.zeros(10), timestamp=float(i))
    assert len(reservoir) == 5
    assert reservoir.is_full()


def test_out_of_distribution_query_scores_higher_than_in_distribution():
    _require_artifacts()
    reference = load_reference_model()
    n_features = reference.means_.shape[1]

    rng = np.random.default_rng(42)
    reservoir = QueryReservoir(window_size=100)

    # Populate the reservoir with "normal" queries clustered near a reference mean.
    center = reference.means_[0]
    for i in range(50):
        normal_query = center + rng.normal(scale=0.3, size=n_features)
        reservoir.add(f"client-{i % 5}", normal_query, timestamp=float(i))

    in_distribution_query = center + rng.normal(scale=0.3, size=n_features)
    out_of_distribution_query = center + rng.normal(scale=0.0, size=n_features) + 50.0

    normal_result = compute_threat_index(in_distribution_query, reservoir)
    attack_result = compute_threat_index(out_of_distribution_query, reservoir)

    assert attack_result["threat_index"] > normal_result["threat_index"]
    assert attack_result["macro_feature_distortion"] >= normal_result["macro_feature_distortion"]


def test_threat_index_bounded_0_to_100():
    _require_artifacts()
    reference = load_reference_model()
    n_features = reference.means_.shape[1]
    reservoir = QueryReservoir(window_size=20)

    rng = np.random.default_rng(0)
    for i in range(10):
        reservoir.add(f"client-{i}", rng.normal(size=n_features), timestamp=float(i))

    result = compute_threat_index(rng.normal(scale=100, size=n_features), reservoir)
    assert 0.0 <= result["threat_index"] <= 100.0
