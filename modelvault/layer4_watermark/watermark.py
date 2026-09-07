"""Layer 4 -- deterministic watermark trigger-selection + injection, merged
into one atomic decide-then-inject operation (an earlier draft split these
into two files; that just meant every caller had to keep two decisions in
sync for no benefit).

The watermark is planted into a fraction of responses to already-suspicious
clients (elevated/critical tiers only -- normal-tier traffic is never
watermarked). Whether a given (client_id, query) gets watermarked is decided
by HMAC(secret, client_id + query) compared against the tier's configured
flip rate from config/settings.yaml. This makes the decision:

  - Deterministic: the same (client_id, query) always resolves the same way,
    so a client can't detect the defense by resending an identical query and
    diffing the two responses.
  - Unpredictable to the attacker: without the secret salt, the HMAC output
    is indistinguishable from random, so the attacker can't tell which of
    their queries were watermarked.

When triggered, the returned label is flipped to a fixed alternate class
(deterministic function of the true label), which is exactly what
verification.py later checks for when proving ownership of a suspect model.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from modelvault.layer3_response.throttling import ResponseTier
from modelvault.utils.config_loader import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class WatermarkDecision:
    triggered: bool
    final_label: int
    original_label: int


def _get_secret() -> bytes:
    settings = get_settings()
    salt = os.environ.get(settings.watermark.secret_salt_env_var, "dev-only-insecure-default-salt")
    return salt.encode()


def _flip_rate_for_tier(tier: ResponseTier) -> float:
    settings = get_settings()
    if tier == ResponseTier.ELEVATED:
        return settings.watermark.flip_rate_elevated
    if tier == ResponseTier.CRITICAL:
        return settings.watermark.flip_rate_critical
    return 0.0


def _deterministic_draw(client_id: str, query_features: np.ndarray) -> float:
    secret = _get_secret()
    message = f"{client_id}:".encode() + np.asarray(query_features).tobytes()
    digest = hmac.new(secret, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def watermark_target_label(client_id: str, query_features: np.ndarray, true_label: int, n_classes: int) -> int:
    """The label that would be returned IF this query is chosen as a watermark
    trigger. Deterministic and independent of the flip-rate check, so
    verification can recompute it without needing to know whether the
    original response was actually watermarked."""
    secret = _get_secret()
    message = f"target:{client_id}:".encode() + np.asarray(query_features).tobytes()
    digest = hmac.new(secret, message, hashlib.sha256).digest()
    offset = 1 + (int.from_bytes(digest[:4], "big") % max(1, n_classes - 1))
    return (true_label + offset) % n_classes


def decide_and_apply_watermark(
    client_id: str,
    query_features: np.ndarray,
    predicted_label: int,
    n_classes: int,
    tier: ResponseTier,
) -> WatermarkDecision:
    flip_rate = _flip_rate_for_tier(tier)
    if flip_rate <= 0.0:
        return WatermarkDecision(triggered=False, final_label=predicted_label, original_label=predicted_label)

    draw = _deterministic_draw(client_id, query_features)
    if draw >= flip_rate:
        return WatermarkDecision(triggered=False, final_label=predicted_label, original_label=predicted_label)

    final_label = watermark_target_label(client_id, query_features, predicted_label, n_classes)
    return WatermarkDecision(triggered=True, final_label=final_label, original_label=predicted_label)


def is_watermark_trigger(client_id: str, query_features: np.ndarray, tier: ResponseTier) -> bool:
    """Recomputes only the trigger decision, without needing the true label --
    used by verification to know which historical queries should have been
    watermarked."""
    flip_rate = _flip_rate_for_tier(tier)
    if flip_rate <= 0.0:
        return False
    return _deterministic_draw(client_id, query_features) < flip_rate
