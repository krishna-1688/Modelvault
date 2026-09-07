"""Alert banner (is traffic currently suspicious?) plus a live activity log
table of the most recent individual requests -- the two things a SOC-style
monitoring view always has: a headline verdict, and the raw feed to verify it against."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

TIER_BADGE = {"normal": "🟢 normal", "elevated": "🟠 elevated", "critical": "🔴 critical"}


def render_banner(stats: dict) -> None:
    tier_counts = stats["tier_counts"]
    total = sum(tier_counts.values())
    if total == 0:
        st.info("⏳ No requests yet -- waiting for traffic.")
        return

    suspicious_share = (tier_counts.get("elevated", 0) + tier_counts.get("critical", 0)) / total

    if suspicious_share > 0.5:
        st.error(f"🚨 **HIGH ALERT** -- {suspicious_share*100:.0f}% of recent traffic is elevated/critical tier. Likely active extraction attempt.")
    elif suspicious_share > 0.2:
        st.warning(f"⚠️ **Elevated suspicion** -- {suspicious_share*100:.0f}% of recent traffic is elevated/critical tier.")
    else:
        st.success(f"✅ **Traffic looks normal** ({suspicious_share*100:.0f}% elevated/critical).")

    if stats["watermark_triggers"] > 0:
        st.info(f"💧 {stats['watermark_triggers']} watermark trigger(s) planted so far -- available for ownership verification.")


def render_activity_log(recent_events: list[dict], limit: int = 15) -> None:
    st.markdown("#### 📋 Live Activity Log")
    if not recent_events:
        st.caption("No requests logged yet.")
        return

    rows = []
    for event in recent_events[:limit]:
        rows.append({
            "Time": datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S"),
            "Client": event["client_id"],
            "Tier": TIER_BADGE.get(event["tier"], event["tier"]),
            "Threat Index": round(event["threat_index"], 1),
            "Label": event.get("label"),
            "Watermarked": "⭐ yes" if event.get("watermarked") else "-",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True, height=min(400, 40 + 35 * len(rows)))
