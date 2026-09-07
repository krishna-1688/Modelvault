"""Line chart of recent threat index values plus a tier breakdown, so a viewer
can see at a glance how much traffic is sitting in each Layer 3 response tier."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render(recent_threat_indices: list[float], tier_normal_max: int, tier_elevated_max: int) -> None:
    st.subheader("Threat Index Timeline")
    if not recent_threat_indices:
        st.info("No requests yet.")
        return

    df = pd.DataFrame({"threat_index": recent_threat_indices})
    st.line_chart(df, height=280)
    st.caption(f"Tiers: normal 0-{tier_normal_max} | elevated {tier_normal_max + 1}-{tier_elevated_max} | critical {tier_elevated_max + 1}-100")
