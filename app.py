import streamlit as st
import matplotlib.pyplot as plt

from drawing.court import draw_half_court

st.set_page_config(
    page_title="Basketball Possession Simulator",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 Basketball Possession Simulator")
st.markdown("---")

# ── Layout: 2×2 grid ──────────────────────────────────────────────────────────
row1_left, row1_right = st.columns(2)
row2_left, row2_right = st.columns(2)

# ── Top-left: Basketball Court ────────────────────────────────────────────────
with row1_left:
    st.subheader("Court")
    fig = draw_half_court()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Top-right: Placeholder ────────────────────────────────────────────────────
with row1_right:
    st.subheader("Pane 2")
    st.info("Top-right panel — coming soon.")

# ── Bottom-left: Placeholder ──────────────────────────────────────────────────
with row2_left:
    st.subheader("Pane 3")
    st.info("Bottom-left panel — coming soon.")

# ── Bottom-right: Placeholder ─────────────────────────────────────────────────
with row2_right:
    st.subheader("Pane 4")
    st.info("Bottom-right panel — coming soon.")
