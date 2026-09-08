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
        self.blocked_count = 0
        # Per-request log for the dashboard's live trace inspector -- each
        # entry carries the FULL per-layer decision trail (not just the final
        # tier), so the UI can show exactly what Layer 1-4 each did for a
        # given request, not just its outcome.
        self.recent_events: deque[dict] = deque(maxlen=150)

        # key: (client_id, query bytes) -> WatermarkEvent, used by /verify-ownership
        # to look up what label a suspect model SHOULD reproduce for a given query
        # if it was trained on our watermarked responses.
        self._watermark_events: dict[tuple[str, bytes], WatermarkEvent] = {}

        # source_ip -> {client_ids, requests, suspicious, first_seen, last_seen}.
        # A single attacker splitting traffic across many sybil client_ids
        # still shares one network origin -- surfacing that makes the
        # adversary concrete on the console instead of an anonymous stream.
        self.origins: dict[str, dict] = {}

        # Rolling record of what each answered query actually taught the
        # caller: the scaled features, the label we DISCLOSED, and the label
        # the undefended model WOULD have given. Feeding a surrogate on each
        # of those two label sets and comparing is what lets the console show
        # how much of the model an attacker has actually reconstructed --
        # the real "how much have they stolen" number, not a proxy.
        self.extraction_buffer: deque[dict] = deque(maxlen=1500)

        # Incident tracking for the attack-phase timeline.
        self.incident_started_at: float | None = None
        self.first_degraded_at: float | None = None
        self.first_watermark_at: float | None = None

        # Per-consumer service quality: how each caller is actually being
        # treated. A defense that catches attackers by degrading everyone is
        # worthless, so "are paying customers still getting full fidelity"
        # is a first-class metric, not an afterthought.
        self.consumer_quality: dict[str, dict] = {}

        # DEMO-ONLY ground truth from the X-Demo-Label header. Recorded so the
        # console can display a genuinely MEASURED false-positive rate rather
        # than asserting one. Never read by any detection layer -- detection
        # stays purely content-based, which is the entire point of Layer 2.
        self.demo_truth = {
            "legitimate": {"total": 0, "degraded": 0, "blocked": 0},
            "attacker": {"total": 0, "degraded": 0, "blocked": 0},
        }

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

    def record_service_quality(self, client_id: str, tier: str, degraded: bool, blocked: bool, demo_label: str | None) -> None:
        """Tracks how each caller is actually being served, plus (when the
        demo header is present) the ground truth needed to measure the
        false-positive rate. demo_label NEVER influences detection."""
        with self._lock:
            entry = self.consumer_quality.get(client_id)
            if entry is None:
                entry = {"total": 0, "degraded": 0, "blocked": 0, "tier_counts": {}, "demo_label": demo_label}
                self.consumer_quality[client_id] = entry
            entry["total"] += 1
            entry["tier_counts"][tier] = entry["tier_counts"].get(tier, 0) + 1
            if degraded:
                entry["degraded"] += 1
            if blocked:
                entry["blocked"] += 1
            if demo_label and entry.get("demo_label") is None:
                entry["demo_label"] = demo_label

            if demo_label in self.demo_truth:
                bucket = self.demo_truth[demo_label]
                bucket["total"] += 1
                if degraded:
                    bucket["degraded"] += 1
                if blocked:
                    bucket["blocked"] += 1

    def service_quality_summary(self) -> dict:
        with self._lock:
            consumers = []
            for client_id, v in self.consumer_quality.items():
                consumers.append({
                    "client_id": client_id,
                    "total": v["total"],
                    "degraded": v["degraded"],
                    "blocked": v["blocked"],
                    "demo_label": v.get("demo_label"),
                    "full_fidelity_rate": (v["total"] - v["degraded"]) / v["total"] if v["total"] else 0.0,
                })
            consumers.sort(key=lambda c: c["total"], reverse=True)
            return {
                "consumers": consumers[:12],
                "demo_truth": {k: dict(v) for k, v in self.demo_truth.items()},
            }

    def record_origin(self, source_ip: str, client_id: str, suspicious: bool) -> None:
        """Tracks the network origin behind a client_id. Sybil identities are
        cheap to mint; a network origin is not, so grouping by it is what
        turns 25 anonymous 'clients' into one visible adversary."""
        now = time.time()
        with self._lock:
            origin = self.origins.get(source_ip)
            if origin is None:
                origin = {"client_ids": set(), "requests": 0, "suspicious": 0, "first_seen": now, "last_seen": now}
                self.origins[source_ip] = origin
            origin["client_ids"].add(client_id)
            origin["requests"] += 1
            origin["last_seen"] = now
            if suspicious:
                origin["suspicious"] += 1

    def record_extraction_sample(self, features: np.ndarray, disclosed_label: int, true_label: int) -> None:
        with self._lock:
            self.extraction_buffer.append({
                "features": np.asarray(features, dtype=float),
                "disclosed_label": int(disclosed_label),
                "true_label": int(true_label),
            })

    def extraction_samples(self) -> list[dict]:
        with self._lock:
            return list(self.extraction_buffer)

    def origins_summary(self) -> list[dict]:
        with self._lock:
            return sorted(
                (
                    {
                        "source_ip": ip,
                        "identities": len(v["client_ids"]),
                        "requests": v["requests"],
                        "suspicious": v["suspicious"],
                        "first_seen": v["first_seen"],
                        "last_seen": v["last_seen"],
                    }
                    for ip, v in self.origins.items()
                ),
                key=lambda o: o["suspicious"],
                reverse=True,
            )

    def incident_markers(self) -> dict:
        with self._lock:
            return {
                "incident_started_at": self.incident_started_at,
                "first_degraded_at": self.first_degraded_at,
                "first_watermark_at": self.first_watermark_at,
            }

    def record_request(
        self,
        client_id: str,
        tier: ResponseTier,
        threat_index: float,
        label: int | None = None,
        watermarked: bool = False,
        trace: dict | None = None,
    ) -> None:
        """trace: the full per-layer decision trail for this request (rate
        limit status, macro/micro signal breakdown, response degradation
        applied, watermark decision) -- everything the dashboard's request
        inspector needs to explain WHY a request got the verdict it did,
        not just what the verdict was."""
        now = time.time()
        with self._lock:
            self.total_requests += 1
            self.seen_clients.add(client_id)
            self.tier_counts[tier.value] += 1
            self.recent_threat_indices.append(threat_index)

            # Incident phase markers, for the console's attack timeline.
            if tier != ResponseTier.NORMAL:
                if self.incident_started_at is None:
                    self.incident_started_at = now
                if self.first_degraded_at is None:
                    self.first_degraded_at = now
            if watermarked and self.first_watermark_at is None:
                self.first_watermark_at = now

            self.recent_events.append({
                "timestamp": now,
                "client_id": client_id,
                "tier": tier.value,
                "threat_index": threat_index,
                "label": label,
                "watermarked": watermarked,
                "blocked": False,
                "trace": trace or {},
            })

    def record_blocked_request(self, client_id: str) -> None:
        """Layer 1 rejected this request before it ever reached scoring --
        logged separately so the dashboard can show Layer 1 rate-limit
        rejections, which previously vanished with no trace at all."""
        with self._lock:
            self.total_requests += 1
            self.blocked_count += 1
            self.seen_clients.add(client_id)
            self.recent_events.append({
                "timestamp": time.time(),
                "client_id": client_id,
                "tier": "blocked",
                "threat_index": None,
                "label": None,
                "watermarked": False,
                "blocked": True,
                "trace": {"layer1": {"allowed": False, "reason": "rate limit exceeded"}},
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
                "blocked_count": self.blocked_count,
                "tier_counts": dict(self.tier_counts),
                "defense_enabled": self.defense_enabled,
                "recent_threat_indices": list(self.recent_threat_indices),
                "recent_events": list(self.recent_events)[::-1],  # most recent first
                "incident_started_at": self.incident_started_at,
                "first_degraded_at": self.first_degraded_at,
                "first_watermark_at": self.first_watermark_at,
                "origins": sorted(
                    (
                        {
                            "source_ip": ip,
                            "identities": len(v["client_ids"]),
                            "requests": v["requests"],
                            "suspicious": v["suspicious"],
                            "first_seen": v["first_seen"],
                            "last_seen": v["last_seen"],
                        }
                        for ip, v in self.origins.items()
                    ),
                    key=lambda o: o["suspicious"],
                    reverse=True,
                )[:6],
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
