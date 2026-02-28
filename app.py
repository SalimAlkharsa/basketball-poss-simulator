import streamlit as st
import matplotlib.pyplot as plt

from drawing.court import draw_half_court
from data.loader import load_teams

st.set_page_config(
    page_title="Basketball Possession Simulator",
    page_icon="🏀",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Controls")
debug_mode = st.sidebar.checkbox("Debug: Show Court Zones", value=False)

# ── Load teams and place players ───────────────────────────────────────────────
blue_team, red_team = load_teams()
blue_team.place_at_defaults()
red_team.place_at_defaults()

all_players = [*blue_team.players, *red_team.players]

# ── Sidebar: player zone info ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Player Zones")
for player in all_players:
    zone = player.zone
    zone_label = zone.value if zone else "Off Court"
    team_icon = "🔵" if player.team == "Blue Team" else "🔴"
    st.sidebar.text(f"{team_icon} {player.name}: {zone_label}")

st.title("🏀 Basketball Possession Simulator")
st.markdown("---")

# ── Layout: 2×2 grid ──────────────────────────────────────────────────────────
row1_left, row1_right = st.columns(2)
row2_left, row2_right = st.columns(2)

# ── Top-left: Basketball Court ────────────────────────────────────────────────
with row1_left:
    st.subheader("Court")
    fig = draw_half_court(debug=debug_mode, players=all_players)
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
