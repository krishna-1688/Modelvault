"""Sole owner of gateway state. In-memory only, by design -- see the project
brief's explicit call-out that a SQLite telemetry layer here would duplicate
this module's job. State does not need to survive a restart for this build.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

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

        # key: (client_id, query bytes) -> WatermarkEvent, used by /verify-ownership
        # to look up what label a suspect model SHOULD reproduce for a given query
        # if it was trained on our watermarked responses.
        self._watermark_events: dict[tuple[str, bytes], WatermarkEvent] = {}

    def record_request(self, client_id: str, tier: ResponseTier, threat_index: float) -> None:
        self.total_requests += 1
        self.seen_clients.add(client_id)
        self.tier_counts[tier.value] += 1
        self.recent_threat_indices.append(threat_index)

    def record_watermark_event(self, client_id: str, query_features: np.ndarray, expected_label: int) -> None:
        self.watermark_trigger_count += 1
        key = (client_id, np.asarray(query_features).tobytes())
        self._watermark_events[key] = WatermarkEvent(
            client_id=client_id, query_features=np.asarray(query_features), expected_label=expected_label
        )

    def lookup_watermark_event(self, client_id: str, query_features: np.ndarray) -> WatermarkEvent | None:
        key = (client_id, np.asarray(query_features).tobytes())
        return self._watermark_events.get(key)

    def all_watermark_events(self) -> list[WatermarkEvent]:
        return list(self._watermark_events.values())

    def stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_clients": len(self.seen_clients),
            "watermark_triggers": self.watermark_trigger_count,
            "tier_counts": dict(self.tier_counts),
            "defense_enabled": self.defense_enabled,
            "recent_threat_indices": list(self.recent_threat_indices),
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
