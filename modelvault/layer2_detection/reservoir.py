"""Cross-client windowed query buffer.

This is what makes Sybil evasion fail: the reservoir pools queries across ALL
clients into one sliding window, so scoring (in feature_distortion.py and
coverage_density.py) operates on the shape of recent traffic as a whole, not
on any single client's identity. Splitting an attack across many fake
accounts doesn't shrink the window's view of what's actually being asked.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from modelvault.utils.config_loader import get_settings


class QueryReservoir:
    def __init__(self, window_size: int | None = None):
        settings = get_settings()
        self.window_size = window_size if window_size is not None else settings.reservoir.window_size
        
        self._matrix: Optional[np.ndarray] = None
        self._ptr: int = 0
        self._count: int = 0
        
        # Pooled across all clients by design (see module docstring), which
        # means concurrent requests from DIFFERENT clients touch the same
        # buffer -- this must be locked, not just per-client state.
        self._lock = threading.Lock()

    def add(self, client_id: str, features: np.ndarray, timestamp: float) -> None:
        with self._lock:
            features = np.asarray(features)
            if self._matrix is None:
                n_features = len(features)
                self._matrix = np.zeros((self.window_size, n_features), dtype=features.dtype)
            
            self._matrix[self._ptr] = features
            self._ptr = (self._ptr + 1) % self.window_size
            if self._count < self.window_size:
                self._count += 1

    def get_feature_matrix(self) -> np.ndarray:
        with self._lock:
            if self._count == 0 or self._matrix is None:
                return np.empty((0, 0))
            # Must return a copy because caller executes NearestNeighbors.fit()
            # on this array outside of our lock, and another thread calling add()
            # could overwrite rows while the tree is being built!
            if self._count < self.window_size:
                return self._matrix[:self._count].copy()
            return self._matrix.copy()

    def __len__(self) -> int:
        with self._lock:
            return self._count

    def is_full(self) -> bool:
        with self._lock:
            return self._count == self.window_size
