import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from drawing.court import draw_half_court
from data.loader import load_teams
from simulation.engine import new_possession, step_possession
from simulation.utils import player_dist

st.set_page_config(
    page_title="Basketball Possession Simulator",
    page_icon="🏀",
    layout="wide",
)

# ── Animation constants ────────────────────────────────────────────────────────
_N_FRAMES    = 24
_FRAME_DELAY = 1 / 24  # seconds per frame → 24 FPS, ~1 s total

_BASKET_X, _BASKET_Y = 25.0, 5.25


def _animation_frames(annotation: dict) -> list[tuple[float, float]]:
    """Return (ball_x, ball_y) for each animation frame given a last_annotation.

    For DRIVE, use _run_drive_animation instead — this path handles PASS & SHOT only.
    """
    if annotation is None:
        return []
    t = np.linspace(0.0, 1.0, _N_FRAMES)
    atype = annotation["type"]
    if atype == "PASS":
        fx, fy = annotation["from_x"], annotation["from_y"]
        tx, ty = annotation["to_x"], annotation["to_y"]
        xs = fx + (tx - fx) * t
        ys = fy + (ty - fy) * t
        return list(zip(xs.tolist(), ys.tolist()))
    if atype == "SHOT":
        fx, fy = annotation["from_x"], annotation["from_y"]
        xs = fx + (_BASKET_X - fx) * t
        ys = fy + (_BASKET_Y - fy) * t + np.sin(t * np.pi) * 8.0
        return list(zip(xs.tolist(), ys.tolist()))
    return []


def _run_drive_animation(
    annotation: dict,
    all_players: list,
    debug_mode: bool,
    court_placeholder,
) -> None:
    """Animate a drive: the driver dot tracks the ball; the defender chases.

    Player positions are temporarily mutated each frame and then restored to
    their correct post-step_possession values so the model remains consistent.
    """
    bfx, bfy = annotation["from_x"], annotation["from_y"]
    btx, bty = annotation["to_x"],   annotation["to_y"]
    driver_name   = annotation.get("driver_name")
    defender_name = annotation.get("defender_name")

    driver   = next((p for p in all_players if p.name == driver_name),   None)
    defender = next((p for p in all_players if p.name == defender_name), None)

    # Snapshot the correct final positions (already set by step_possession).
    driver_final   = (driver.x,   driver.y)   if driver   else None
    defender_final = (defender.x, defender.y) if defender else None

    dfx = annotation.get("defender_from_x")
    dfy = annotation.get("defender_from_y")
    dtx = annotation.get("defender_to_x", dfx)
    dty = annotation.get("defender_to_y", dfy)
    has_defender_anim = (
        defender is not None
        and dfx is not None
        and dfy is not None
        and (dtx, dty) != (dfx, dfy)
    )

    t_vals = np.linspace(0.0, 1.0, _N_FRAMES)
    for ti in t_vals:
        bx = bfx + (btx - bfx) * ti
        by = bfy + (bty - bfy) * ti

        # Driver dot tracks the ball.
        if driver:
            driver.x = bx
            driver.y = by

        # Defender chases (only when visible movement exists).
        if has_defender_anim:
            defender.x = dfx + (dtx - dfx) * ti
            defender.y = dfy + (dty - dfy) * ti

        fig = draw_half_court(
            debug=debug_mode, players=all_players, ball_pos=(bx, by)
        )
        court_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        time.sleep(_FRAME_DELAY)

    # Restore to the post-step model positions so Streamlit rerenders correctly.
    if driver and driver_final:
        driver.x, driver.y = driver_final
    if defender and defender_final:
        defender.x, defender.y = defender_final

    # ── Phase 2: layup arc (if the drive reached the rim) ─────────────────
    layup = annotation.get("layup")
    if layup:
        fx, fy = layup["from_x"], layup["from_y"]
        t2 = np.linspace(0.0, 1.0, _N_FRAMES)
        for ti in t2:
            bx = fx + (_BASKET_X - fx) * ti
            by = fy + (_BASKET_Y - fy) * ti + np.sin(ti * np.pi) * 4.0  # lower arc for a layup
            fig = draw_half_court(
                debug=debug_mode, players=all_players, ball_pos=(bx, by)
            )
            court_placeholder.pyplot(fig, use_container_width=True)
            plt.close(fig)
            time.sleep(_FRAME_DELAY)

# ── Session state bootstrap (runs once per session) ───────────────────────────
if "blue_team" not in st.session_state:
    blue_team, red_team = load_teams()
    st.session_state.blue_team = blue_team
    st.session_state.red_team = red_team
    st.session_state.possession = new_possession(blue_team, red_team)
    st.session_state.auto_play = False

if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

blue_team = st.session_state.blue_team
red_team = st.session_state.red_team
possession = st.session_state.possession

# ── Sidebar: debug + player zones ─────────────────────────────────────────────
st.sidebar.title("Debug")
debug_mode = st.sidebar.checkbox("Show Zones & Radii", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Player Zones")
all_players = [*blue_team.players, *red_team.players]
for player in all_players:
    zone = player.zone
    zone_label = zone.value if zone else "Off Court"
    team_icon = "🔵" if player.team == "Blue Team" else "🔴"
    ball_marker = " ●" if player.name == possession.ball_handler.name else ""
    st.sidebar.text(f"{team_icon} {player.name}{ball_marker}: {zone_label}")

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🏀 Basketball Possession Simulator")
st.markdown("---")

# ── Layout: 2×2 grid ──────────────────────────────────────────────────────────
row1_left, row1_right = st.columns(2)
row2_left, row2_right = st.columns(2)

# ── Top-left: Court (placeholder for animation) ────────────────────────────────
with row1_left:
    st.subheader("Court")
    court_placeholder = st.empty()

# ── Top-right: Possession (controls + status + log) ───────────────────────────
with row1_right:
    st.subheader("Possession")

    # Controls
    col_step, col_new, col_play = st.columns(3)
    with col_step:
        step_clicked = st.button(
            "▶ Step",
            disabled=possession.is_over or st.session_state.auto_play,
            use_container_width=True,
        )
    with col_new:
        new_clicked = st.button(
            "↺ New",
            disabled=st.session_state.auto_play,
            use_container_width=True,
        )
    with col_play:
        play_label = "⏸ Pause" if st.session_state.auto_play else "▶ Play"
        if st.button(play_label, use_container_width=True):
            st.session_state.auto_play = not st.session_state.auto_play
            st.rerun()

    st.markdown("---")

    # Status
    if possession.is_over:
        outcome_labels = {
            "MADE_2": ("✅", "Made — 2 pts"),
            "MADE_3": ("✅", "Made — 3 pts"),
            "MISSED": ("❌", "Shot missed"),
            "TURNOVER": ("🔄", "Turnover"),
            "INTERCEPTED": ("🔄", "Pass intercepted"),
        }
        icon, label = outcome_labels.get(possession.outcome, ("❓", possession.outcome))
        st.metric("Result", f"{icon} {label}")
        st.metric("Score", f"+{possession.score} pts")
    else:
        bh = possession.ball_handler
        defender = possession.matchups.get(bh)
        dist_desc = (
            f"{player_dist(bh, defender):.1f} ft away"
            if defender and defender.is_on_court()
            else "unguarded"
        )
        st.metric("Ball Handler", bh.name)
        st.metric("Zone", bh.zone.value if bh.zone else "—")
        st.metric("Defender", f"{defender.name} ({dist_desc})" if defender else "None")
        st.metric("Steps taken", len(possession.action_log))

    st.markdown("---")

    # Action log — small font via HTML
    st.markdown("**Action Log**")
    if possession.action_log:
        lines = "".join(
            f"<div style='font-size:11px; padding:1px 0; color:#ccc;'>"
            f"<span style='color:#888; margin-right:6px;'>{i}.</span>{entry}</div>"
            for i, entry in enumerate(possession.action_log, start=1)
        )
        st.markdown(lines, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-size:11px; color:#666;'>No actions yet — press ▶ Step.</div>",
            unsafe_allow_html=True,
        )

# ── Step: animate then commit ──────────────────────────────────────────────────
if step_clicked and not possession.is_over:
    new_state = step_possession(possession, blue_team, red_team)
    ann = new_state.last_annotation
    all_players_anim = [*blue_team.players, *red_team.players]
    if ann and ann["type"] == "DRIVE":
        _run_drive_animation(ann, all_players_anim, debug_mode, court_placeholder)
    else:
        for bx, by in _animation_frames(ann):
            fig = draw_half_court(debug=debug_mode, players=all_players_anim, ball_pos=(bx, by))
            court_placeholder.pyplot(fig, use_container_width=True)
            plt.close(fig)
            time.sleep(_FRAME_DELAY)
    st.session_state.possession = new_state
    st.rerun()

if new_clicked:
    st.session_state.possession = new_possession(blue_team, red_team)
    st.rerun()

# ── Auto-play loop ─────────────────────────────────────────────────────────────
if st.session_state.auto_play:
    if possession.is_over:
        time.sleep(0.4)  # brief pause so the result is readable before next possession
        st.session_state.possession = new_possession(blue_team, red_team)
        st.rerun()
    else:
        new_state = step_possession(possession, blue_team, red_team)
        ann = new_state.last_annotation
        all_players_anim = [*blue_team.players, *red_team.players]
        if ann and ann["type"] == "DRIVE":
            _run_drive_animation(ann, all_players_anim, debug_mode, court_placeholder)
        else:
            for bx, by in _animation_frames(ann):
                fig = draw_half_court(debug=debug_mode, players=all_players_anim, ball_pos=(bx, by))
                court_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
                time.sleep(_FRAME_DELAY)
        st.session_state.possession = new_state
        st.rerun()

# ── Static court render (all non-animated frames) ─────────────────────────────
if not step_clicked and not st.session_state.auto_play:
    bh = possession.ball_handler
    ball_pos = (bh.x, bh.y) if bh.is_on_court() else None
    fig = draw_half_court(debug=debug_mode, players=all_players, ball_pos=ball_pos)
    court_placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Bottom-left: empty ────────────────────────────────────────────────────────
with row2_left:
    pass

# ── Bottom-right: Ball handler attributes ─────────────────────────────────────
with row2_right:
    st.subheader("Ball Handler Attributes")
    bh = possession.ball_handler
    off = bh.offense
    defs = bh.defense
    tend = bh.tendencies

    st.markdown(f"**{bh.name}** ({bh.position.value})")
    st.markdown("**Offense**")
    cols = st.columns(3)
    cols[0].metric("3PT", f"{off.three_pt_shooting:.0%}")
    cols[1].metric("Mid", f"{off.mid_range_shooting:.0%}")
    cols[2].metric("Drive", f"{off.drive_effectiveness:.0%}")
    cols2 = st.columns(3)
    cols2[0].metric("Layup", f"{off.layup:.0%}")
    cols2[1].metric("Pass", f"{off.passing:.0%}")
    cols2[2].metric("Strength", f"{off.strength:.0%}")

    st.markdown("**Tendencies**")
    labels = ["3PT", "MID", "DRV", "PASS", "LAY"]
    weights = tend.as_weights()
    tcols = st.columns(5)
    for col, lbl, w in zip(tcols, labels, weights):
        col.metric(lbl, f"{w:.0%}")
