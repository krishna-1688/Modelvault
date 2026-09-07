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

from dashboard.components import alert_feed, gauge, kpi_cards, query_scatter, threat_timeline
from modelvault.utils.config_loader import get_settings

GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "localhost")  # set to the compose service name ("gateway") in Docker
GATEWAY_PORT = os.environ.get("GATEWAY_PORT", "8000")
GATEWAY_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"
ADMIN_HEADERS = {"X-API-Key": os.environ["ADMIN_API_KEY"]} if os.environ.get("ADMIN_API_KEY") else {}

st.set_page_config(page_title="ModelVault Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .mv-mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #cbd5e1; }

    .block-container { padding-top: 1.2rem; max-width: 1400px; }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    /* ---- Hero header ---- */
    .mv-hero {
        display: flex; align-items: center; justify-content: space-between;
        padding: 18px 26px; border-radius: 16px; margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(34,211,238,0.10) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .mv-hero-title {
        font-size: 1.65rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #22d3ee, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .mv-hero-sub { color: #94a3b8; font-size: 0.85rem; margin-top: 2px; }
    .mv-status-live {
        color: #22c55e; font-weight: 700; font-size: 0.95rem;
        display: flex; align-items: center; gap: 6px;
    }
    .mv-status-down { color: #ef4444; font-weight: 700; font-size: 0.95rem; }
    .mv-pulse {
        display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        background: #22c55e; animation: mv-pulse 1.6s infinite;
    }
    @keyframes mv-pulse {
        0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.6); }
        70% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }

    /* ---- KPI glass cards ---- */
    .mv-kpi {
        background: rgba(148, 163, 184, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-top: 3px solid var(--accent, #6366f1);
        border-radius: 12px; padding: 16px 18px 14px 18px;
        backdrop-filter: blur(6px);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .mv-kpi:hover { transform: translateY(-2px); border-color: rgba(148,163,184,0.3); }
    .mv-kpi-icon { font-size: 1.3rem; opacity: 0.85; margin-bottom: 6px; }
    .mv-kpi-value { font-size: 1.9rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
    .mv-kpi-label { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }

    /* ---- Section cards ---- */
    .mv-section {
        background: rgba(148, 163, 184, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 14px; padding: 18px 20px; margin-bottom: 16px;
    }

    /* ---- Alert banners ---- */
    .mv-banner {
        padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;
        font-size: 0.92rem; color: #f1f5f9;
    }
    .mv-banner-critical { background: rgba(239, 68, 68, 0.14); border: 1px solid rgba(239, 68, 68, 0.35); }
    .mv-banner-warning { background: rgba(245, 158, 11, 0.14); border: 1px solid rgba(245, 158, 11, 0.35); }
    .mv-banner-ok { background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.3); }
    .mv-banner-info { background: rgba(99, 102, 241, 0.14); border: 1px solid rgba(99, 102, 241, 0.35); }

    /* ---- Tier pills ---- */
    .mv-pill {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
    }
    .mv-pill-normal { background: rgba(34,197,94,0.15); color: #4ade80; }
    .mv-pill-elevated { background: rgba(245,158,11,0.15); color: #fbbf24; }
    .mv-pill-critical { background: rgba(239,68,68,0.15); color: #f87171; }
    .mv-star { color: #facc15; font-weight: 600; font-size: 0.82rem; }
    .mv-dim { color: #475569; }

    /* ---- Activity table ---- */
    .mv-table-wrap { border-radius: 10px; overflow: hidden; border: 1px solid rgba(148,163,184,0.12); }
    .mv-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .mv-table thead { background: rgba(148,163,184,0.08); }
    .mv-table th {
        text-align: left; padding: 9px 14px; color: #94a3b8;
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;
    }
    .mv-table td { padding: 8px 14px; border-top: 1px solid rgba(148,163,184,0.08); color: #e2e8f0; }
    .mv-table tbody tr:hover { background: rgba(148,163,184,0.05); }

    section[data-testid="stSidebar"] { background: rgba(10, 14, 26, 0.6); }
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
    st.markdown("### 🎯 Protected Target")
    st.caption("Credit card fraud classifier")
    st.caption("RandomForest · 284K real transactions · 0.17% fraud rate")
    st.markdown("---")
    st.caption(f"Gateway: `{GATEWAY_URL}`")

while True:
    stats = fetch_stats()
    with placeholder.container():
        gateway_up = stats is not None

        status_html = (
            '<div class="mv-status-live"><span class="mv-pulse"></span>LIVE</div>' if gateway_up
            else '<div class="mv-status-down">● OFFLINE</div>'
        )
        st.markdown(f"""
        <div class="mv-hero">
            <div>
                <p class="mv-hero-title">🛡️ ModelVault</p>
                <p class="mv-hero-sub">Live Model Extraction Defense Monitor</p>
            </div>
            {status_html}
        </div>
        """, unsafe_allow_html=True)

        if not gateway_up:
            st.error(
                f"Could not reach the gateway at {GATEWAY_URL} (or `ADMIN_API_KEY` in `.env` doesn't match the "
                "gateway's). Is the gateway running, and do both `.env` files agree?"
            )
        else:
            tier_counts = stats["tier_counts"]
            total = sum(tier_counts.values())
            suspicious_pct = ((tier_counts.get("elevated", 0) + tier_counts.get("critical", 0)) / total * 100) if total else 0.0

            gauge_col, kpi_col = st.columns([1, 3])
            with gauge_col:
                st.markdown('<div class="mv-section">', unsafe_allow_html=True)
                st.markdown("###### 🎚️ Current Threat Level")
                gauge.render(
                    stats["recent_threat_indices"],
                    settings.threat_index.tier_normal_max,
                    settings.threat_index.tier_elevated_max,
                )
                st.markdown('</div>', unsafe_allow_html=True)
            with kpi_col:
                defense_label = "ACTIVE" if stats["defense_enabled"] else "DISABLED"
                kpi_cards.render([
                    {"icon": "📡", "value": stats["total_requests"], "label": "Total Requests", "accent": "#6366f1"},
                    {"icon": "👥", "value": stats["total_clients"], "label": "Distinct Clients", "accent": "#22d3ee"},
                    {"icon": "💧", "value": stats["watermark_triggers"], "label": "Watermark Triggers", "accent": "#a78bfa"},
                    {"icon": "⚠️", "value": f"{suspicious_pct:.0f}%", "label": "Suspicious Traffic", "accent": "#f59e0b"},
                    {"icon": "🛡️", "value": defense_label, "label": "Defense Status",
                     "accent": "#22c55e" if stats["defense_enabled"] else "#ef4444"},
                ])

            st.markdown("")
            alert_feed.render_banner(stats)

            st.markdown("---")
            left, right = st.columns([3, 2])
            with left:
                st.markdown('<div class="mv-section">', unsafe_allow_html=True)
                threat_timeline.render(
                    stats["recent_threat_indices"],
                    settings.threat_index.tier_normal_max,
                    settings.threat_index.tier_elevated_max,
                )
                st.markdown('</div>', unsafe_allow_html=True)
            with right:
                st.markdown('<div class="mv-section">', unsafe_allow_html=True)
                query_scatter.render_donut(stats["tier_counts"])
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="mv-section">', unsafe_allow_html=True)
            query_scatter.render_scatter(stats.get("recent_events", []))
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="mv-section">', unsafe_allow_html=True)
            alert_feed.render_activity_log(stats.get("recent_events", []))
            st.markdown('</div>', unsafe_allow_html=True)

    time.sleep(refresh_seconds)
    st.rerun()
