"""Loads and validates config/settings.yaml. This is the sole entry point for tunable thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class ReservoirConfig:
    window_size: int


@dataclass(frozen=True)
class ThreatIndexConfig:
    tier_normal_max: int
    tier_elevated_max: int


@dataclass(frozen=True)
class RateLimitConfig:
    requests_per_window: int
    window_seconds: int


@dataclass(frozen=True)
class WatermarkConfig:
    flip_rate_elevated: float
    flip_rate_critical: float
    secret_salt_env_var: str


@dataclass(frozen=True)
class ModelConfig:
    n_features: int
    random_state: int


@dataclass(frozen=True)
class Settings:
    reservoir: ReservoirConfig
    threat_index: ThreatIndexConfig
    rate_limit: RateLimitConfig
    watermark: WatermarkConfig
    model: ModelConfig


def load_settings(path: Path | str = DEFAULT_SETTINGS_PATH) -> Settings:
    raw = yaml.safe_load(Path(path).read_text())

    required_sections = {"reservoir", "threat_index", "rate_limit", "watermark", "model"}
    missing = required_sections - raw.keys()
    if missing:
        raise ValueError(f"settings.yaml is missing required sections: {missing}")

    threat = raw["threat_index"]
    if not (0 < threat["tier_normal_max"] < threat["tier_elevated_max"] <= 100):
        raise ValueError(
            "threat_index tiers must satisfy 0 < tier_normal_max < tier_elevated_max <= 100"
        )

    return Settings(
        reservoir=ReservoirConfig(**raw["reservoir"]),
        threat_index=ThreatIndexConfig(**threat),
        rate_limit=RateLimitConfig(**raw["rate_limit"]),
        watermark=WatermarkConfig(**raw["watermark"]),
        model=ModelConfig(**raw["model"]),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
