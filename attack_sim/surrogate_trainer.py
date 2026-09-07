"""Trains a surrogate model on (query, response_label) pairs collected by an
attack -- this is the "clone" the attacker walks away with. A simple
KNeighborsClassifier is enough to demonstrate agreement/disagreement with the
real target model; the point of this project isn't to build a sophisticated
extraction attack, it's to show that a defended API yields a measurably worse
clone than an undefended one.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def train_surrogate(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> KNeighborsClassifier:
    n_neighbors = min(n_neighbors, len(X))
    surrogate = KNeighborsClassifier(n_neighbors=n_neighbors)
    surrogate.fit(X, y)
    return surrogate
