"""Simple alert feed: flags when the recent traffic mix looks like an active
extraction attempt (a rising share of elevated/critical tier traffic)."""
from __future__ import annotations

import streamlit as st


def render(stats: dict) -> None:
    st.subheader("Alerts")
    tier_counts = stats["tier_counts"]
    total = sum(tier_counts.values())
    if total == 0:
        st.info("No requests yet.")
        return

    suspicious_share = (tier_counts.get("elevated", 0) + tier_counts.get("critical", 0)) / total

    if suspicious_share > 0.5:
        st.error(f"HIGH ALERT: {suspicious_share*100:.0f}% of recent traffic is elevated/critical tier -- likely active extraction attempt.")
    elif suspicious_share > 0.2:
        st.warning(f"Elevated suspicion: {suspicious_share*100:.0f}% of recent traffic is elevated/critical tier.")
    else:
        st.success(f"Traffic looks normal ({suspicious_share*100:.0f}% elevated/critical).")

    if stats["watermark_triggers"] > 0:
        st.info(f"{stats['watermark_triggers']} watermark trigger(s) planted so far -- available for ownership verification.")
