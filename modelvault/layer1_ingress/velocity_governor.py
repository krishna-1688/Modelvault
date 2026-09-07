"""Layer 1 -- coarse rate limiting via a per-client token bucket.

This is deliberately coarse: it just caps raw request velocity. Anything that
looks at the CONTENT of queries (is this actually an extraction pattern?) is
Layer 2's job, not this one's.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from modelvault.utils.config_loader import get_settings


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class VelocityGovernor:
    def __init__(self, requests_per_window: int | None = None, window_seconds: int | None = None):
        settings = get_settings()
        self.capacity = requests_per_window if requests_per_window is not None else settings.rate_limit.requests_per_window
        self.window_seconds = window_seconds if window_seconds is not None else settings.rate_limit.window_seconds
        self.refill_rate = self.capacity / self.window_seconds  # tokens per second
        self._buckets: dict[str, _Bucket] = {}
        # FastAPI runs sync `def` endpoints in a threadpool, so concurrent
        # requests can race on the same client's bucket without this --
        # a token-bucket read-modify-write is not atomic on its own.
        self._lock = threading.Lock()

    def _get_bucket(self, client_id: str, now: float) -> _Bucket:
        bucket = self._buckets.get(client_id)
        if bucket is None:
            bucket = _Bucket(tokens=float(self.capacity), last_refill=now)
            self._buckets[client_id] = bucket
        return bucket

    def allow(self, client_id: str, now: float | None = None) -> bool:
        """Returns True if the request is allowed, consuming one token. Never raises."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            bucket = self._get_bucket(client_id, now)

            elapsed = max(0.0, now - bucket.last_refill)
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
            bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def reset(self, client_id: str) -> None:
        with self._lock:
            self._buckets.pop(client_id, None)
