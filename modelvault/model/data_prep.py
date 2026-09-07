"""Loads the real-world Kaggle/ULB Credit Card Fraud Detection dataset
(284,807 transactions, 492 confirmed frauds -- a 0.17% positive rate), the
industry-standard benchmark for fraud/anomaly detection demos. Features
V1-V28 are PCA components of the original (never publicly released, for
confidentiality) transaction fields; Amount is the raw transaction amount.

This is a genuinely high-stakes target for extraction: a production fraud
model is expensive to build (years of labeled transaction history, adversarial
feedback loops, compliance review) and a cloned surrogate handed to a fraud
ring would let them probe for exactly which transaction patterns evade
detection.

Data path, in priority order:
  1. A local cache at data/raw/creditcard.csv (fastest, fully offline).
  2. Fetched once from OpenML (mirrors the original Kaggle release, data_id
     1597) and cached to (1) for every run after.
  3. A synthetic fallback matching the same schema and class imbalance, used
     ONLY if neither of the above is available (e.g. no internet on first
     run) -- so the system never hard-fails, but this path is loud about
     the fact that it isn't real data.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from modelvault.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREDITCARD_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
OPENML_DATA_ID = 1597  # "creditcard" -- OpenML mirror of the Kaggle/ULB Credit Card Fraud dataset
N_FEATURES = 29  # V1-V28 + Amount
TARGET_COLUMN = "Class"


def _load_from_local_cache() -> pd.DataFrame | None:
    if not CREDITCARD_CSV_PATH.exists():
        return None
    logger.info("Loading cached credit card fraud dataset from %s", CREDITCARD_CSV_PATH)
    return pd.read_csv(CREDITCARD_CSV_PATH)


def _fetch_and_cache_from_openml() -> pd.DataFrame | None:
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        return None

    logger.info("Fetching credit card fraud dataset from OpenML (data_id=%s) -- this runs once and is cached locally.", OPENML_DATA_ID)
    try:
        bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    except Exception as exc:  # network failure, OpenML outage, etc.
        logger.warning("Could not fetch dataset from OpenML (%s). Falling back to synthetic data.", exc)
        return None

    df = bunch.frame
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    CREDITCARD_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CREDITCARD_CSV_PATH, index=False)
    logger.info("Cached dataset to %s for future offline runs.", CREDITCARD_CSV_PATH)
    return df


def _synthetic_fallback(n_samples: int = 50_000, random_state: int = 42) -> pd.DataFrame:
    """Only used if the real dataset is unavailable and there is no internet
    access to fetch it. Matches the real dataset's shape and severe class
    imbalance (~0.17% positive) so the rest of the pipeline behaves
    consistently, but this is NOT real transaction data -- callers should
    treat any results produced from this path as a smoke test, not a
    benchmark."""
    logger.warning(
        "Using SYNTHETIC fallback data -- the real credit card fraud dataset "
        "was not available locally or over the network. Results from this "
        "run are not representative of the real benchmark."
    )
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        weights=[0.9983, 0.0017],
        flip_y=0.001,
        class_sep=1.5,
        random_state=random_state,
    )
    columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame(X, columns=columns)
    df["Amount"] = np.abs(df["Amount"]) * 50  # loosely mimic a real amount scale
    df[TARGET_COLUMN] = y
    return df


@lru_cache(maxsize=1)
def load_raw_dataframe() -> pd.DataFrame:
    if os.environ.get("MODELVAULT_FORCE_SYNTHETIC_DATA") == "1":
        # Explicit escape hatch for CI and fast local smoke tests -- downloading
        # and training on the full 284k-row real dataset on every CI run would
        # be slow and network-dependent for what's meant to be a quick import
        # and unit-test check, not a benchmark reproduction.
        return _synthetic_fallback()

    df = _load_from_local_cache()
    if df is None:
        df = _fetch_and_cache_from_openml()
    if df is None:
        df = _synthetic_fallback()
    return df


def generate_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, X_test, y_train, y_test) as raw (unscaled) feature
    arrays -- scaling is fit and applied downstream in train.py / predict.py
    so that the exact same transform is saved and reused at serving time."""
    df = load_raw_dataframe()
    X = df.drop(columns=[TARGET_COLUMN]).values.astype(float)
    y = df[TARGET_COLUMN].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
