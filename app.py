import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from drawing.court import draw_half_court
from data.loader import load_teams
from simulation.engine import new_possession, step_possession, effective_weights
from simulation.off_ball import OffBallTendencies, TENDENCIES
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

def _run_off_ball_animation(
    off_ball_annotations: list,
    all_players: list,
    debug_mode: bool,
    court_placeholder,
    ball_pos: tuple,
) -> None:
    """Animate cuts and screens *before* the ball-handler action.

    All off-ball player movements are played simultaneously over a short clip.
    Player positions are temporarily set back to their 'from' coords at the
    start of the animation, interpolated to their 'to' coords, then left at
    the 'to' values (which matches the model state after step_possession).
    """
    if not off_ball_annotations:
        return

    player_map = {p.name: p for p in all_players}

    # Collect (player, from_x, from_y, to_x, to_y) for every mover
    movers: list[tuple] = []
    for ann in off_ball_annotations:
        if ann["type"] == "CUT":
            p = player_map.get(ann["player_name"])
            if p:
                movers.append((p, ann["from_x"], ann["from_y"], ann["to_x"], ann["to_y"]))
        elif ann["type"] == "SCREEN":
            p = player_map.get(ann["screener_name"])
            if p:
                movers.append((
                    p,
                    ann["screener_from_x"], ann["screener_from_y"],
                    ann["final_x"],         ann["final_y"],
                ))

    if not movers:
        return

    # Temporarily set all movers back to their 'from' positions
    final_positions = {p.name: (p.x, p.y) for p, *_ in movers}
    for p, fx, fy, tx, ty in movers:
        p.x, p.y = fx, fy

    # Shorter clip for off-ball (~0.5 s)
    t_vals = np.linspace(0.0, 1.0, _N_FRAMES // 2)
    for ti in t_vals:
        for p, fx, fy, tx, ty in movers:
            p.x = fx + (tx - fx) * ti
            p.y = fy + (ty - fy) * ti
        fig = draw_half_court(debug=debug_mode, players=all_players, ball_pos=ball_pos)
        court_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        time.sleep(_FRAME_DELAY)

    # Restore to final model positions (already at 'to' thanks to step_possession)
    for p, *_ in movers:
        p.x, p.y = final_positions[p.name]



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

# ── Layout: top 2-col row + full-width bottom row ─────────────────────────────
row1_left, row1_right = st.columns([1.1, 0.9])

# ── Top-left: Court (placeholder for animation) ────────────────────────────────
with row1_left:
    st.subheader("Court")
    court_placeholder = st.empty()

# ── Top-right: Possession — compact dashboard ─────────────────────────────────
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

    # ── Compact status strip ───────────────────────────────────────────────────
    if possession.is_over:
        outcome_labels = {
            "MADE_2": ("✅", "Made — 2 pts"),
            "MADE_3": ("✅", "Made — 3 pts"),
            "MISSED": ("❌", "Shot missed"),
            "TURNOVER": ("🔄", "Turnover"),
            "INTERCEPTED": ("🔄", "Pass intercepted"),
        }
        icon, label = outcome_labels.get(possession.outcome, ("❓", possession.outcome))
        status_html = f"""
        <div style='display:flex; gap:16px; flex-wrap:wrap; padding:6px 0 8px 0;'>
          <div style='background:#1a1a2e; border-radius:6px; padding:6px 12px; min-width:90px;'>
            <div style='font-size:9px; color:#666; text-transform:uppercase; letter-spacing:.06em;'>Result</div>
            <div style='font-size:13px; color:#eee; font-weight:600;'>{icon} {label}</div>
          </div>
          <div style='background:#1a1a2e; border-radius:6px; padding:6px 12px; min-width:60px;'>
            <div style='font-size:9px; color:#666; text-transform:uppercase; letter-spacing:.06em;'>Score</div>
            <div style='font-size:13px; color:#4caf50; font-weight:600;'>+{possession.score} pts</div>
          </div>
        </div>
        """
    else:
        bh = possession.ball_handler
        defender = possession.matchups.get(bh)
        dist_desc = (
            f"{player_dist(bh, defender):.1f} ft"
            if defender and defender.is_on_court()
            else "open"
        )
        zone_str  = bh.zone.value if bh.zone else "—"
        def_str   = f"{defender.name} ({dist_desc})" if defender else "None"
        steps_str = str(len(possession.action_log))
        status_html = f"""
        <div style='display:flex; gap:10px; flex-wrap:wrap; padding:6px 0 8px 0;'>
          <div style='background:#1a1a2e; border-radius:6px; padding:5px 10px; min-width:80px;'>
            <div style='font-size:9px; color:#666; text-transform:uppercase; letter-spacing:.06em;'>Ball</div>
            <div style='font-size:12px; color:#64b5f6; font-weight:600; white-space:nowrap;'>{bh.name}</div>
          </div>
          <div style='background:#1a1a2e; border-radius:6px; padding:5px 10px; min-width:80px;'>
            <div style='font-size:9px; color:#666; text-transform:uppercase; letter-spacing:.06em;'>Zone</div>
            <div style='font-size:12px; color:#eee; font-weight:600; white-space:nowrap;'>{zone_str}</div>
          </div>
          <div style='background:#1a1a2e; border-radius:6px; padding:5px 10px; min-width:110px;'>
            <div style='font-size:9px; color:#666; text-transform:uppercase; letter-spacing:.06em;'>Defender</div>
            <div style='font-size:12px; color:#ef9a9a; font-weight:600; white-space:nowrap;'>{def_str}</div>
          </div>
          <div style='background:#1a1a2e; border-radius:6px; padding:5px 10px; min-width:50px;'>
            <div style='font-size:9px; color:#666; text-transform:uppercase; letter-spacing:.06em;'>Steps</div>
            <div style='font-size:12px; color:#eee; font-weight:600;'>{steps_str}</div>
          </div>
        </div>
        """
    st.markdown(status_html, unsafe_allow_html=True)

    # ── Action log in scrollable container ────────────────────────────────────
    st.markdown("**Action Log**")
    if possession.action_log:
        parts = []
        for i, entry in enumerate(possession.action_log, start=1):
            if isinstance(entry, dict):
                text    = entry.get("text", "")
                details = entry.get("details", [])
                style   = entry.get("style", "normal")
            else:
                text    = entry
                details = []
                style   = "normal"
            color = "#8a8a5c" if style == "offball" else "#bbb"
            parts.append(
                f"<div style='font-size:10px; padding:1px 0; color:{color};'>"
                f"<span style='color:#555; margin-right:5px;'>{i}.</span>{text}</div>"
            )
            for detail in details:
                parts.append(
                    f"<div style='font-size:9px; padding:0 0 1px 14px; color:#555;'>"
                    f"<span style='color:#3a3a3a; margin-right:4px;'>&#x21B3;</span>{detail}</div>"
                )
        log_html = "".join(parts)
        st.markdown(
            f"<div style='height:260px; overflow-y:auto; border:1px solid #2a2a2a; "
            f"border-radius:4px; padding:6px 8px; background:#111;'>{log_html}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:10px; color:#555;'>No actions yet — press ▶ Step.</div>",
            unsafe_allow_html=True,
        )

# ── Step: animate then commit ──────────────────────────────────────────────────
if step_clicked and not possession.is_over:
    new_state = step_possession(possession, blue_team, red_team)
    ann = new_state.last_annotation
    all_players_anim = [*blue_team.players, *red_team.players]

    # Phase 0: off-ball cuts / screens (ball stays put)
    if new_state.off_ball_annotations:
        off_ball_ball_pos = (
            (ann["from_x"], ann["from_y"]) if ann
            else (possession.ball_handler.x, possession.ball_handler.y)
        )
        _run_off_ball_animation(
            new_state.off_ball_annotations, all_players_anim,
            debug_mode, court_placeholder, off_ball_ball_pos,
        )

    # Phase 1: ball-handler action animation
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

# ── Auto-play loop
if st.session_state.auto_play:
    if possession.is_over:
        time.sleep(0.4)  # brief pause so the result is readable before next possession
        st.session_state.possession = new_possession(blue_team, red_team)
        st.rerun()
    else:
        new_state = step_possession(possession, blue_team, red_team)
        ann = new_state.last_annotation
        all_players_anim = [*blue_team.players, *red_team.players]

        # Phase 0: off-ball cuts / screens
        if new_state.off_ball_annotations:
            off_ball_ball_pos = (
                (ann["from_x"], ann["from_y"]) if ann
                else (possession.ball_handler.x, possession.ball_handler.y)
            )
            _run_off_ball_animation(
                new_state.off_ball_annotations, all_players_anim,
                debug_mode, court_placeholder, off_ball_ball_pos,
            )

        # Phase 1: ball-handler action animation
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

# ── Static court render (all non-animated frames) ────────────────────────────
if not step_clicked and not st.session_state.auto_play:
    bh = possession.ball_handler
    ball_pos = (bh.x, bh.y) if bh.is_on_court() else None
    fig = draw_half_court(debug=debug_mode, players=all_players, ball_pos=ball_pos)
    court_placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Sync any edited off-ball tendency values → TENDENCIES singleton ───────────
TENDENCIES.base_stay = st.session_state["tend_base_stay"]
for _pos in _POSITIONS:
    TENDENCIES.cut_factors[_pos]       = st.session_state[f"tend_cut_{_pos}"]
    TENDENCIES.screen_factors[_pos]    = st.session_state[f"tend_screen_{_pos}"]
    TENDENCIES.pop_probabilities[_pos] = st.session_state[f"tend_pop_{_pos}"]

# ── Bottom row: full-width — player tendencies + off-ball tendency editor ──────
st.markdown("---")
b_left, b_right = st.columns(2)

# ── Bottom-left: per-player on-ball tendencies table ──────────────────────────
with b_left:
    st.subheader("On-Ball Tendencies")
    bh_name = possession.ball_handler.name
    labels   = ["3PT", "MID", "DRV", "PASS", "LAY"]

    # Header
    header_cells = "".join(
        f"<th style='padding:3px 6px; font-size:9px; color:#666; text-align:right; "
        f"font-weight:normal; text-transform:uppercase; letter-spacing:.05em;'>{l}</th>"
        for l in labels
    )
    rows_html = (
        f"<table style='border-collapse:collapse; width:100%; font-size:10px;'>"
        f"<thead><tr>"
        f"<th style='padding:3px 6px; font-size:9px; color:#666; text-align:left; "
        f"font-weight:normal;'>Player</th>{header_cells}"
        f"</tr></thead><tbody>"
    )
    for player in blue_team.players:
        raw   = player.tendencies.as_weights()
        eff   = effective_weights(player, possession.matchups.get(player))
        is_bh = player.name == bh_name
        row_bg    = "#1a2540" if is_bh else "transparent"
        name_color = "#64b5f6" if is_bh else "#999"
        rows_html += (
            f"<tr style='background:{row_bg}; border-bottom:1px solid #1e1e1e;'>"
            f"<td style='padding:3px 6px; color:{name_color}; white-space:nowrap;'>"
            f"{'● ' if is_bh else ''}{player.name} ({player.position.value})</td>"
        )
        for rw, ew in zip(raw, eff):
            diff  = ew - rw
            if abs(diff) > 0.005:
                val_color = "#81c784" if diff > 0 else "#e57373"
            else:
                val_color = "#aaa"
            rows_html += (
                f"<td style='padding:3px 6px; text-align:right; color:{val_color}; "
                f"font-weight:{'600' if is_bh else '400'};'>{ew:.0%}</td>"
            )
        rows_html += "</tr>"
    rows_html += "</tbody></table>"
    st.markdown(rows_html, unsafe_allow_html=True)

# ── Bottom-right: off-ball tendency editor ─────────────────────────────────────
with b_right:
    st.subheader("Off-Ball Tendencies")
    st.caption("Edit values to adjust agent behaviour in real-time.")

    # base_stay
    st.markdown("<span style='font-size:11px; color:#888;'>Base Stay weight</span>",
                unsafe_allow_html=True)
    st.number_input(
        "base_stay", min_value=0.0, max_value=2.0, step=0.05,
        key="tend_base_stay", label_visibility="collapsed",
    )

    # Position tables: cut / screen / pop
    col_labels = ["Position", "Cut", "Screen", "Pop"]
    hdr = "".join(
        f"<th style='padding:2px 6px; font-size:9px; color:#666; font-weight:normal; "
        f"text-transform:uppercase; letter-spacing:.05em; text-align:{'left' if i==0 else 'center'};'>"
        f"{lbl}</th>"
        for i, lbl in enumerate(col_labels)
    )
    tb_html = (
        f"<table style='border-collapse:collapse; width:100%; font-size:10px; margin-top:6px;'>"
        f"<thead><tr>{hdr}</tr></thead><tbody>"
    )
    for _pos in _POSITIONS:
        tb_html += (
            f"<tr style='border-bottom:1px solid #1e1e1e;'>"
            f"<td style='padding:2px 6px; color:#aaa; width:40px;'>{_pos}</td>"
            f"<td style='padding:2px 6px; color:#4caf50; text-align:center; width:46px;'>"
            f"{st.session_state[f'tend_cut_{_pos}']:.2f}</td>"
            f"<td style='padding:2px 6px; color:#ff9800; text-align:center; width:46px;'>"
            f"{st.session_state[f'tend_screen_{_pos}']:.2f}</td>"
            f"<td style='padding:2px 6px; color:#64b5f6; text-align:center; width:46px;'>"
            f"{st.session_state[f'tend_pop_{_pos}']:.2f}</td>"
            f"</tr>"
        )
    tb_html += "</tbody></table>"
    st.markdown(tb_html, unsafe_allow_html=True)

    # Number input controls below the table
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    st.caption("Adjust by position:")
    for _pos in _POSITIONS:
        c1, c2, c3, c4 = st.columns([1, 3, 3, 3])
        c1.markdown(
            f"<div style='font-size:11px; color:#aaa; padding-top:6px;'>{_pos}</div>",
            unsafe_allow_html=True,
        )
        c2.number_input(
            f"Cut {_pos}", min_value=0.0, max_value=1.5, step=0.02,
            key=f"tend_cut_{_pos}", label_visibility="collapsed",
        )
        c3.number_input(
            f"Screen {_pos}", min_value=0.0, max_value=1.5, step=0.02,
            key=f"tend_screen_{_pos}", label_visibility="collapsed",
        )
        c4.number_input(
            f"Pop {_pos}", min_value=0.0, max_value=1.0, step=0.05,
            key=f"tend_pop_{_pos}", label_visibility="collapsed",
        )
