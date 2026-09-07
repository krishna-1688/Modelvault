"""Bar chart of how many requests landed in each Layer 3 response tier."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render(tier_counts: dict[str, int]) -> None:
    st.subheader("Response Tier Distribution")
    if not any(tier_counts.values()):
        st.info("No requests yet.")
        return

    df = pd.DataFrame({"tier": list(tier_counts.keys()), "count": list(tier_counts.values())}).set_index("tier")
    st.bar_chart(df, height=280)
