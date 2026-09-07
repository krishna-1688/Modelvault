"""Loads the trained target classifier, reference density model, and feature
scaler for inference. The scaler MUST be applied before either the classifier
or the reference density model sees a feature vector -- both were fit on
scaled features (see train.py), and Amount's raw scale (0 to tens of
thousands) would otherwise dominate every distance/split computation."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = PROJECT_ROOT / "artifacts" / "target" / "target_classifier.joblib"
REFERENCE_PATH = PROJECT_ROOT / "artifacts" / "reference" / "reference_density.joblib"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "target" / "scaler.joblib"


@lru_cache(maxsize=1)
def load_target_model():
    return joblib.load(TARGET_PATH)


@lru_cache(maxsize=1)
def load_reference_model():
    return joblib.load(REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_scaler():
    return joblib.load(SCALER_PATH)


def transform_features(features: np.ndarray) -> np.ndarray:
    """Applies the training-time StandardScaler to raw incoming features.
    Call this once at the gateway boundary -- everything downstream
    (reservoir, threat scoring, model inference) then operates in the same
    scaled feature space the artifacts were fit on."""
    scaler = load_scaler()
    X = np.atleast_2d(features)
    return scaler.transform(X)


def predict_proba(features: np.ndarray) -> np.ndarray:
    """features: already-scaled, shape (n_features,) or (n_samples, n_features)."""
    model = load_target_model()
    X = np.atleast_2d(features)
    return model.predict_proba(X)


def predict_label(features: np.ndarray) -> np.ndarray:
    model = load_target_model()
    X = np.atleast_2d(features)
    return model.predict(X)


def predict_raw_proba(raw_features: np.ndarray) -> np.ndarray:
    """Convenience for callers outside the gateway (attack simulations, ad-hoc
    scripts) that have RAW, unscaled features -- the same contract /predict
    exposes over HTTP. Applies the training-time scaler before predicting.
    Batch-friendly: pass shape (n_features,) for one query or
    (n_samples, n_features) for many; the result shape matches accordingly."""
    return predict_proba(transform_features(raw_features))


def predict_raw_label(raw_features: np.ndarray) -> np.ndarray:
    return predict_label(transform_features(raw_features))
