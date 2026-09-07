"""Trains a surrogate model on (query, response_label) pairs collected by an
attack -- this is the "clone" the attacker walks away with. A KNeighborsClassifier
is enough to demonstrate agreement/disagreement with the real target model;
the point of this project isn't to build a sophisticated extraction attack,
it's to show that a defended API yields a measurably worse clone than an
undefended one.

The surrogate fits its OWN StandardScaler on the stolen queries before
training KNN. This isn't generosity toward the attacker -- it's the realistic
threat model: raw Amount ranges from 0 to tens of thousands while the V1-V28
PCA features sit roughly in [-50, 50], so an unscaled, distance-based
classifier like KNN would have its neighborhoods dominated almost entirely by
Amount and effectively ignore the other 28 features. A competent attacker
trying to build a working clone would obviously normalize their own stolen
data first; assuming otherwise would understate the real threat and make the
defended-vs-undefended comparison meaningless (both would look equally bad
for the wrong reason).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_surrogate(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> Pipeline:
    n_neighbors = min(n_neighbors, len(X))
    surrogate = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])
    surrogate.fit(X, y)
    return surrogate
