"""Cross-client windowed query buffer.

This is what makes Sybil evasion fail: the reservoir pools queries across ALL
clients into one sliding window, so scoring (in feature_distortion.py and
coverage_density.py) operates on the shape of recent traffic as a whole, not
on any single client's identity. Splitting an attack across many fake
accounts doesn't shrink the window's view of what's actually being asked.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from modelvault.utils.config_loader import get_settings


@dataclass
class QueryRecord:
    client_id: str
    features: np.ndarray
    timestamp: float


class QueryReservoir:
    def __init__(self, window_size: int | None = None):
        settings = get_settings()
        self.window_size = window_size if window_size is not None else settings.reservoir.window_size
        self._buffer: deque[QueryRecord] = deque(maxlen=self.window_size)
        # Pooled across all clients by design (see module docstring), which
        # means concurrent requests from DIFFERENT clients touch the same
        # deque -- this must be locked, not just per-client state.
        self._lock = threading.Lock()

    def add(self, client_id: str, features: np.ndarray, timestamp: float) -> None:
        with self._lock:
            self._buffer.append(QueryRecord(client_id=client_id, features=np.asarray(features), timestamp=timestamp))

    def get_feature_matrix(self) -> np.ndarray:
        with self._lock:
            if not self._buffer:
                return np.empty((0, 0))
            return np.vstack([record.features for record in self._buffer])

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) == self.window_size
