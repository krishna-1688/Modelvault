"""Streamlit dashboard: polls the gateway's /admin/stats over HTTP and renders
live-updating threat monitoring while attack_sim (or real traffic) runs
against the gateway in another process."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Streamlit adds this script's own directory to sys.path, not the project
# root, so `dashboard` and `modelvault` aren't importable as packages without
# this -- add the root explicitly before importing either.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components import alert_feed, query_scatter, threat_timeline
from modelvault.utils.config_loader import get_settings

GATEWAY_PORT = os.environ.get("GATEWAY_PORT", "8000")
GATEWAY_URL = f"http://localhost:{GATEWAY_PORT}"

st.set_page_config(page_title="ModelVault Dashboard", layout="wide")
st.title("ModelVault -- Live Threat Monitor")
st.caption(f"Polling {GATEWAY_URL}/admin/stats")

settings = get_settings()

placeholder = st.empty()


def fetch_stats() -> dict | None:
    try:
        response = requests.get(f"{GATEWAY_URL}/admin/stats", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def toggle_defense(enabled: bool) -> None:
    try:
        requests.post(f"{GATEWAY_URL}/admin/toggle-defense", json={"enabled": enabled}, timeout=2)
    except requests.RequestException:
        pass


with st.sidebar:
    st.header("Controls")
    defense_on = st.toggle("Defense enabled", value=True)
    toggle_defense(defense_on)
    refresh_seconds = st.slider("Refresh interval (seconds)", 1, 10, 2)

while True:
    stats = fetch_stats()
    with placeholder.container():
        if stats is None:
            st.error(f"Could not reach the gateway at {GATEWAY_URL}. Is `scripts/run_gateway.sh` running?")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total requests", stats["total_requests"])
            col2.metric("Distinct clients", stats["total_clients"])
            col3.metric("Watermark triggers", stats["watermark_triggers"])

            left, right = st.columns(2)
            with left:
                threat_timeline.render(
                    stats["recent_threat_indices"],
                    settings.threat_index.tier_normal_max,
                    settings.threat_index.tier_elevated_max,
                )
            with right:
                query_scatter.render(stats["tier_counts"])

            alert_feed.render(stats)

    time.sleep(refresh_seconds)
    st.rerun()
