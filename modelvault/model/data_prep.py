"""Synthetic data generation for the target model.

Default path is sklearn.make_classification: deterministic, zero network dependency.
External dataset fetch is intentionally not implemented here — demo day has no
internet guarantee, so synthetic data is the only supported path in this build.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from modelvault.utils.config_loader import get_settings


def generate_dataset(
    n_samples: int = 4000,
    n_features: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    settings = get_settings()
    n_features = n_features if n_features is not None else settings.model.n_features
    random_state = random_state if random_state is not None else settings.model.random_state

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(4, n_features // 2),
        n_redundant=max(1, n_features // 5),
        n_classes=2,
        class_sep=2.5,
        flip_y=0.01,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
