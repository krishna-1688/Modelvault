"""Sole owner of gateway state. In-memory only, by design -- see the project
brief's explicit call-out that a SQLite telemetry layer here would duplicate
this module's job. State does not need to survive a restart for this build.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from modelvault.layer1_ingress.velocity_governor import VelocityGovernor
from modelvault.layer2_detection.reservoir import QueryReservoir
from modelvault.layer3_response.throttling import ResponseTier


@dataclass
class WatermarkEvent:
    client_id: str
    query_features: np.ndarray
    expected_label: int


class GatewayState:
    def __init__(self):
        self.velocity_governor = VelocityGovernor()
        self.reservoir = QueryReservoir()
        self.defense_enabled = True

        self.total_requests = 0
        self.tier_counts: dict[str, int] = {tier.value: 0 for tier in ResponseTier}
        self.watermark_trigger_count = 0
        self.seen_clients: set[str] = set()
        self.recent_threat_indices: deque[float] = deque(maxlen=200)
        # Per-request log for the dashboard's live activity table -- separate
        # from recent_threat_indices (which only tracks the score) since the
        # UI wants tier/label/watermark context per row too.
        self.recent_events: deque[dict] = deque(maxlen=100)

        # key: (client_id, query bytes) -> WatermarkEvent, used by /verify-ownership
        # to look up what label a suspect model SHOULD reproduce for a given query
        # if it was trained on our watermarked responses.
        self._watermark_events: dict[tuple[str, bytes], WatermarkEvent] = {}

        # velocity_governor and reservoir lock themselves internally; this
        # lock covers the counters/sets/dict above, which FastAPI's
        # threadpool-executed sync endpoints can otherwise mutate concurrently.
        self._lock = threading.Lock()

    def is_defense_enabled(self) -> bool:
        with self._lock:
            return self.defense_enabled

    def set_defense_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.defense_enabled = enabled

    def record_request(
        self,
        client_id: str,
        tier: ResponseTier,
        threat_index: float,
        label: int | None = None,
        watermarked: bool = False,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.seen_clients.add(client_id)
            self.tier_counts[tier.value] += 1
            self.recent_threat_indices.append(threat_index)
            self.recent_events.append({
                "timestamp": time.time(),
                "client_id": client_id,
                "tier": tier.value,
                "threat_index": threat_index,
                "label": label,
                "watermarked": watermarked,
            })

    def record_watermark_event(self, client_id: str, query_features: np.ndarray, expected_label: int) -> None:
        with self._lock:
            self.watermark_trigger_count += 1
            key = (client_id, np.asarray(query_features).tobytes())
            self._watermark_events[key] = WatermarkEvent(
                client_id=client_id, query_features=np.asarray(query_features), expected_label=expected_label
            )

    def lookup_watermark_event(self, client_id: str, query_features: np.ndarray) -> WatermarkEvent | None:
        key = (client_id, np.asarray(query_features).tobytes())
        with self._lock:
            return self._watermark_events.get(key)

    def all_watermark_events(self) -> list[WatermarkEvent]:
        with self._lock:
            return list(self._watermark_events.values())

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_clients": len(self.seen_clients),
                "watermark_triggers": self.watermark_trigger_count,
                "tier_counts": dict(self.tier_counts),
                "defense_enabled": self.defense_enabled,
                "recent_threat_indices": list(self.recent_threat_indices),
                "recent_events": list(self.recent_events)[::-1],  # most recent first
            }


_state: GatewayState | None = None


def get_state() -> GatewayState:
    global _state
    if _state is None:
        _state = GatewayState()
    return _state


def reset_state() -> None:
    global _state
    _state = GatewayState()
