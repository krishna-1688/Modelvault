"""Streamlit dashboard: polls the gateway's /admin/stats over HTTP and renders
live-updating threat monitoring while attack_sim.live_demo (or real traffic)
runs against the gateway in another process."""
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

GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "localhost")  # set to the compose service name ("gateway") in Docker
GATEWAY_PORT = os.environ.get("GATEWAY_PORT", "8000")
GATEWAY_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"
ADMIN_HEADERS = {"X-API-Key": os.environ["ADMIN_API_KEY"]} if os.environ.get("ADMIN_API_KEY") else {}

st.set_page_config(page_title="ModelVault Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.75; }
    .mv-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.25rem; }
    .mv-status-live { color: #22c55e; font-weight: 600; }
    .mv-status-down { color: #ef4444; font-weight: 600; }
    .mv-pulse {
        display: inline-block; width: 9px; height: 9px; border-radius: 50%;
        background: #22c55e; margin-right: 6px; animation: mv-pulse 1.6s infinite;
    }
    @keyframes mv-pulse {
        0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.6); }
        70% { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }
</style>
""", unsafe_allow_html=True)

settings = get_settings()
placeholder = st.empty()


def fetch_stats() -> dict | None:
    try:
        response = requests.get(f"{GATEWAY_URL}/admin/stats", headers=ADMIN_HEADERS, timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def toggle_defense(enabled: bool) -> None:
    try:
        requests.post(f"{GATEWAY_URL}/admin/toggle-defense", json={"enabled": enabled}, headers=ADMIN_HEADERS, timeout=2)
    except requests.RequestException:
        pass


with st.sidebar:
    st.markdown("### ⚙️ Controls")
    defense_on = st.toggle("Defense enabled", value=st.session_state.get("defense_on", True))
    if defense_on != st.session_state.get("defense_on"):
        toggle_defense(defense_on)
        st.session_state["defense_on"] = defense_on

    refresh_seconds = st.slider("Refresh interval (seconds)", 1, 10, 2)

    st.markdown("---")
    st.markdown("### 🎯 Target")
    st.caption("Credit card fraud classifier (RandomForest)")
    st.caption(f"Gateway: `{GATEWAY_URL}`")

while True:
    stats = fetch_stats()
    with placeholder.container():
        gateway_up = stats is not None

        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown("## 🛡️ ModelVault — Live Threat Monitor")
        with header_right:
            if gateway_up:
                st.markdown(
                    '<div style="text-align:right; padding-top: 18px;">'
                    '<span class="mv-pulse"></span><span class="mv-status-live">LIVE</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="text-align:right; padding-top: 18px;">'
                    '<span class="mv-status-down">● OFFLINE</span></div>',
                    unsafe_allow_html=True,
                )

        if not gateway_up:
            st.error(
                f"Could not reach the gateway at {GATEWAY_URL} (or `ADMIN_API_KEY` in `.env` doesn't match the "
                "gateway's). Is the gateway running, and do both `.env` files agree?"
            )
        else:
            tier_counts = stats["tier_counts"]
            total = sum(tier_counts.values())
            suspicious_pct = ((tier_counts.get("elevated", 0) + tier_counts.get("critical", 0)) / total * 100) if total else 0.0

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Requests", stats["total_requests"])
            k2.metric("Distinct Clients", stats["total_clients"])
            k3.metric("Watermark Triggers", stats["watermark_triggers"])
            k4.metric("Suspicious Traffic", f"{suspicious_pct:.0f}%")
            k5.metric("Defense", "ON" if stats["defense_enabled"] else "OFF")

            st.markdown("")
            alert_feed.render_banner(stats)

            st.markdown("---")
            left, right = st.columns([3, 2])
            with left:
                threat_timeline.render(
                    stats["recent_threat_indices"],
                    settings.threat_index.tier_normal_max,
                    settings.threat_index.tier_elevated_max,
                )
            with right:
                query_scatter.render_donut(stats["tier_counts"])

            query_scatter.render_scatter(stats.get("recent_events", []))

            st.markdown("---")
            alert_feed.render_activity_log(stats.get("recent_events", []))

    time.sleep(refresh_seconds)
    st.rerun()
