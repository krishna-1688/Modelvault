"""Loads the trained target classifier and reference density model for inference."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = PROJECT_ROOT / "artifacts" / "target" / "target_classifier.joblib"
REFERENCE_PATH = PROJECT_ROOT / "artifacts" / "reference" / "reference_density.joblib"


@lru_cache(maxsize=1)
def load_target_model():
    return joblib.load(TARGET_PATH)


@lru_cache(maxsize=1)
def load_reference_model():
    return joblib.load(REFERENCE_PATH)


def predict_proba(features: np.ndarray) -> np.ndarray:
    """features: shape (n_features,) or (n_samples, n_features)."""
    model = load_target_model()
    X = np.atleast_2d(features)
    return model.predict_proba(X)


def predict_label(features: np.ndarray) -> np.ndarray:
    model = load_target_model()
    X = np.atleast_2d(features)
    return model.predict(X)
