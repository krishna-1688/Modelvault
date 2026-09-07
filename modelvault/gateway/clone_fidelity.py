"""Estimates how much of the target model an attacker has actually
reconstructed from the answers this gateway has given them so far.

This is the number the whole project exists to move, so the console shows it
directly rather than a proxy. It is computed honestly, from real data:

  - We keep a rolling buffer of every answered query: the features, the label
    we ACTUALLY disclosed, and the label the undefended model WOULD have
    returned (both already computed during /predict).
  - We split that buffer into train/test. Two surrogates are fit on the
    training split -- one on the labels the attacker really received, one on
    the undefended labels -- and both are then scored on the held-out split
    against the TARGET MODEL'S TRUE answers.
  - The gap between them is the defense's effect: what the attacker would
    have had by now vs. what they actually have.

Two design points that matter for this number to mean anything:

  - Both surrogates train on identical queries, so the ONLY difference
    between them is the label degradation this gateway applied. That
    isolates the defense's contribution from query-sampling luck.
  - Evaluation happens on the attacker's OWN query distribution, not on a
    generic dataset holdout. An attacker probing near the decision boundary
    is trying to reconstruct the model THERE; scoring their clone on a
    random sample of ordinary transactions would mostly measure how badly a
    boundary-trained model extrapolates, which is noise, not defense effect.

Results are cached with a short TTL because fitting two models per dashboard
poll (1s) would burn CPU for no benefit -- the number moves on the scale of
seconds, not milliseconds.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelvault.utils.logging import get_logger

logger = get_logger(__name__)

MIN_SAMPLES = 40          # below this a surrogate estimate is noise, not signal
CACHE_TTL_SECONDS = 3.0
TEST_FRACTION = 0.25

_cache: dict | None = None
_cache_time = 0.0
_lock = threading.Lock()


def _surrogate_agreement(X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, target_labels: np.ndarray) -> float:
    """Fits the attacker's clone on the data they hold and scores how often it
    matches the real model. The clone scales its own stolen data first -- a
    competent adversary would, and assuming otherwise understates the threat.

    If the attacker only ever received ONE class (very common here: real
    transaction traffic is 99.8% legitimate), no decision boundary can be
    learned at all and their best possible clone is a constant classifier.
    That's a real, meaningful outcome -- scored as such rather than reported
    as 'unavailable', since 'their clone can only ever say not-fraud' is
    precisely the situation the defense is trying to produce."""
    classes = np.unique(y)
    if len(classes) < 2:
        return float(np.mean(target_labels == classes[0]))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=min(5, len(X)))),
    ])
    model.fit(X, y)
    return float(np.mean(model.predict(X_eval) == target_labels))


def compute_clone_fidelity(samples: list[dict]) -> dict:
    """Returns the defended vs. undefended reconstruction estimate. Cached
    for CACHE_TTL_SECONDS -- callers can poll this every second safely."""
    global _cache, _cache_time

    with _lock:
        if _cache is not None and (time.time() - _cache_time) < CACHE_TTL_SECONDS:
            return _cache

    if len(samples) < MIN_SAMPLES:
        result = {
            "ready": False,
            "samples": len(samples),
            "min_samples": MIN_SAMPLES,
            "defended_fidelity": None,
            "undefended_fidelity": None,
            "prevented_points": None,
        }
    else:
        try:
            X = np.vstack([s["features"] for s in samples])
            disclosed = np.array([s["disclosed_label"] for s in samples])
            true = np.array([s["true_label"] for s in samples])

            # Held-out slice of the attacker's own queries, scored against the
            # target model's true answers on those same queries.
            rng = np.random.default_rng(0)
            order = rng.permutation(len(X))
            split = int(len(X) * (1 - TEST_FRACTION))
            train_idx, test_idx = order[:split], order[split:]

            X_train, X_eval = X[train_idx], X[test_idx]
            target_labels = true[test_idx]

            defended = _surrogate_agreement(X_train, disclosed[train_idx], X_eval, target_labels)
            undefended = _surrogate_agreement(X_train, true[train_idx], X_eval, target_labels)
            prevented = undefended - defended

            result = {
                "ready": True,
                "samples": len(samples),
                "min_samples": MIN_SAMPLES,
                "defended_fidelity": defended,
                "undefended_fidelity": undefended,
                "prevented_points": prevented,
            }
        except Exception as exc:
            logger.warning("Clone fidelity estimation failed: %s", exc)
            result = {
                "ready": False,
                "samples": len(samples),
                "min_samples": MIN_SAMPLES,
                "defended_fidelity": None,
                "undefended_fidelity": None,
                "prevented_points": None,
            }

    with _lock:
        _cache = result
        _cache_time = time.time()
    return result


def reset_cache() -> None:
    global _cache, _cache_time
    with _lock:
        _cache = None
        _cache_time = 0.0
