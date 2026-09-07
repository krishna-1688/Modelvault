"""Alert banner (is traffic currently suspicious?) plus a live activity log
rendered as a custom HTML table with colored pill badges -- the two things a
SOC-style monitoring view always has: a headline verdict, and the raw feed
to verify it against."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

TIER_PILL = {
    "normal": '<span class="mv-pill mv-pill-normal">● normal</span>',
    "elevated": '<span class="mv-pill mv-pill-elevated">● elevated</span>',
    "critical": '<span class="mv-pill mv-pill-critical">● critical</span>',
}


def render_banner(stats: dict) -> None:
    tier_counts = stats["tier_counts"]
    total = sum(tier_counts.values())
    if total == 0:
        st.info("⏳ No requests yet -- waiting for traffic.")
        return

    suspicious_share = (tier_counts.get("elevated", 0) + tier_counts.get("critical", 0)) / total

    if suspicious_share > 0.5:
        st.markdown(
            f'<div class="mv-banner mv-banner-critical">🚨 <strong>HIGH ALERT</strong> — '
            f'{suspicious_share*100:.0f}% of recent traffic is elevated/critical tier. Likely active extraction attempt.</div>',
            unsafe_allow_html=True,
        )
    elif suspicious_share > 0.2:
        st.markdown(
            f'<div class="mv-banner mv-banner-warning">⚠️ <strong>Elevated suspicion</strong> — '
            f'{suspicious_share*100:.0f}% of recent traffic is elevated/critical tier.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="mv-banner mv-banner-ok">✅ <strong>Traffic looks normal</strong> '
            f'({suspicious_share*100:.0f}% elevated/critical).</div>',
            unsafe_allow_html=True,
        )

    if stats["watermark_triggers"] > 0:
        st.markdown(
            f'<div class="mv-banner mv-banner-info">💧 {stats["watermark_triggers"]} watermark trigger(s) '
            f'planted so far — available for ownership verification.</div>',
            unsafe_allow_html=True,
        )


def render_activity_log(recent_events: list[dict], limit: int = 12) -> None:
    st.markdown("#### 📋 Live Activity Log")
    if not recent_events:
        st.caption("No requests logged yet.")
        return

    rows_html = []
    for event in recent_events[:limit]:
        ts = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
        pill = TIER_PILL.get(event["tier"], event["tier"])
        watermark = '<span class="mv-star">⭐ watermarked</span>' if event.get("watermarked") else '<span class="mv-dim">—</span>'
        rows_html.append(f"""
        <tr>
            <td class="mv-mono">{ts}</td>
            <td class="mv-mono">{event['client_id']}</td>
            <td>{pill}</td>
            <td class="mv-mono">{event['threat_index']:.1f}</td>
            <td class="mv-mono">{event.get('label')}</td>
            <td>{watermark}</td>
        </tr>
        """)

    table_html = f"""
    <div class="mv-table-wrap">
    <table class="mv-table">
        <thead>
            <tr><th>Time</th><th>Client</th><th>Tier</th><th>Threat</th><th>Label</th><th>Watermark</th></tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
