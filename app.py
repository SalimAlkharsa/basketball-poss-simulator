import os
import time
import traceback

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from drawing.court import draw_half_court
from data.loader import load_teams
from simulation.engine import new_possession, step_possession, effective_weights
from simulation.off_ball import OffBallTendencies, TENDENCIES
from simulation.utils import player_dist
from coaching.analytics import build_narrative_delta, record_possession, build_action_logs_text
from coaching.agent import CoachingAgent
from coaching.controller import NormalizationController

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
    Also animates any defenders who need to reposition.
    Player positions are temporarily set back to their 'from' coords at the
    start of the animation, interpolated to their 'to' coords, then left at
    the 'to' values (which matches the model state after step_possession).
    """
    if not off_ball_annotations:
        return

    player_map = {p.name: p for p in all_players}

    # Collect (player, from_x, from_y, to_x, to_y) for every off-ball mover
    # and separate list of (defender, from_x, from_y, to_x, to_y) for defenders
    movers: list[tuple] = []
    defender_movers: list[tuple] = []

    for ann in off_ball_annotations:
        if ann["type"] == "CUT":
            p = player_map.get(ann["player_name"])
            if p:
                movers.append((p, ann["from_x"], ann["from_y"], ann["to_x"], ann["to_y"]))
            # Add defender movement for cuts
            if ann.get("defender_name"):
                defender = player_map.get(ann["defender_name"])
                if defender:
                    defender_movers.append((
                        defender,
                        ann.get("defender_from_x", defender.x),
                        ann.get("defender_from_y", defender.y),
                        ann.get("defender_to_x", defender.x),
                        ann.get("defender_to_y", defender.y),
                    ))
        elif ann["type"] == "SCREEN":
            p = player_map.get(ann["screener_name"])
            if p:
                movers.append((
                    p,
                    ann["screener_from_x"], ann["screener_from_y"],
                    ann["final_x"],         ann["final_y"],
                ))
            # Add defender movement for screens
            if ann.get("defender_name"):
                defender = player_map.get(ann["defender_name"])
                if defender:
                    defender_movers.append((
                        defender,
                        ann.get("defender_from_x", defender.x),
                        ann.get("defender_from_y", defender.y),
                        ann.get("defender_to_x", defender.x),
                        ann.get("defender_to_y", defender.y),
                    ))

    if not movers:
        return

    # Temporarily set all movers and defender-movers back to their 'from' positions
    final_positions = {p.name: (p.x, p.y) for p, *_ in movers}
    for defender, *_ in defender_movers:
        if defender.name not in final_positions:
            final_positions[defender.name] = (defender.x, defender.y)

    for p, fx, fy, tx, ty in movers:
        p.x, p.y = fx, fy
    for d, fx, fy, tx, ty in defender_movers:
        d.x, d.y = fx, fy

    # Shorter clip for off-ball (~0.5 s)
    t_vals = np.linspace(0.0, 1.0, _N_FRAMES // 2)
    for ti in t_vals:
        for p, fx, fy, tx, ty in movers:
            p.x = fx + (tx - fx) * ti
            p.y = fy + (ty - fy) * ti
        for d, fx, fy, tx, ty in defender_movers:
            d.x = fx + (tx - fx) * ti
            d.y = fy + (ty - fy) * ti
        fig = draw_half_court(debug=debug_mode, players=all_players, ball_pos=ball_pos)
        court_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        time.sleep(_FRAME_DELAY)

    # Restore to final model positions (already at 'to' thanks to step_possession)
    for p, *_ in movers:
        p.x, p.y = final_positions[p.name]
    for d, *_ in defender_movers:
        d.x, d.y = final_positions[d.name]



if "blue_team" not in st.session_state:
    blue_team, red_team = load_teams()
    st.session_state.blue_team = blue_team
    st.session_state.red_team = red_team
    st.session_state.possession = new_possession(blue_team, red_team)
    st.session_state.auto_play = False

if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

st.session_state.setdefault("possession_history", [])
st.session_state.setdefault("coaching_cot", None)
st.session_state.setdefault("coached_positions", {})
st.session_state.setdefault("coaching_record", None)


@st.cache_resource
def get_coach() -> CoachingAgent:
    api_key = os.environ.get("API_MISTRAL", "")
    return CoachingAgent(api_key=api_key)


def _snap_tendencies(players) -> dict:
    return {
        p.name: {
            "3PT": p.tendencies.tendency_three,
            "MID": p.tendencies.tendency_mid,
            "DRV": p.tendencies.tendency_drive,
            "PAS": p.tendencies.tendency_pass,
            "LAY": p.tendencies.tendency_layup,
        }
        for p in players
    }


def _snap_off_ball() -> dict:
    return {
        "cut_factors":       dict(TENDENCIES.cut_factors),
        "screen_factors":    dict(TENDENCIES.screen_factors),
        "pop_probabilities": dict(TENDENCIES.pop_probabilities),
        "base_stay":         TENDENCIES.base_stay,
    }


def handle_timeout(blue_team, red_team) -> None:
    records   = st.session_state.possession_history
    narrative = build_action_logs_text(records)
    agent     = get_coach()

    before_tendencies = _snap_tendencies(blue_team.players)
    before_off_ball   = _snap_off_ball()

    cot_text          = None
    decision          = None
    prompt_data       = None
    logs              = []
    coached_positions = {}
    error_trace       = None

    with st.spinner("Head Coach calling timeout..."):
        for attempt in range(3):
            try:
                cot_text, decision, prompt_data = agent.call(
                    narrative, 
                    blue_team.players,
                    opponent_players=red_team.players
                )
                controller = NormalizationController(blue_team.players)
                logs, coached_positions = controller.apply(decision)
                
                st.session_state.coaching_cot = cot_text
                error_trace = None # Success
                break
            except Exception as e:
                error_trace = traceback.format_exc()
                if attempt == 2:
                    st.error(f"Coach agent error after 3 attempts: {e}")

    after_tendencies = _snap_tendencies(blue_team.players)
    after_off_ball   = _snap_off_ball()

    st.session_state.coaching_record = {
        "narrative":         narrative,
        "cot_text":          cot_text or "",
        "prompt_data":       prompt_data,
        "logs":              logs,
        "coached_positions": coached_positions,
        "before_tendencies": before_tendencies,
        "after_tendencies":  after_tendencies,
        "before_off_ball":   before_off_ball,
        "after_off_ball":    after_off_ball,
        "error_trace":       error_trace,
    }

    st.session_state.coached_positions = coached_positions

    for log in logs:
        st.session_state.possession.action_log.append(log)

    if decision is not None:
        st.success(f'Coach: "{decision.timeout_message}"')

# ── Off-ball tendency session-state init (runs once; also safe on hot-reload) ────
_POSITIONS = ["PG", "SG", "SF", "PF", "C"]
if "tend_base_stay" not in st.session_state:
    _t = TENDENCIES
    st.session_state["tend_base_stay"] = _t.base_stay
    for _pos in _POSITIONS:
        st.session_state[f"tend_cut_{_pos}"]    = _t.cut_factors[_pos]
        st.session_state[f"tend_screen_{_pos}"] = _t.screen_factors[_pos]
        st.session_state[f"tend_pop_{_pos}"]    = _t.pop_probabilities[_pos]

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
    col_step, col_new, col_play, col_timeout = st.columns(4)
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
    with col_timeout:
        timeout_clicked = st.button(
            "📋 Timeout",
            disabled=not st.session_state.possession_history or st.session_state.auto_play,
            use_container_width=True,
        )

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
        
        c_res, c_score = st.columns(2)
        c_res.metric("Result", f"{icon} {label}")
        c_score.metric("Score", f"+{possession.score} pts")
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
        
        c_ball, c_zone, c_def, c_steps = st.columns(4)
        c_ball.metric("Ball", bh.name)
        c_zone.metric("Zone", zone_str)
        c_def.metric("Defender", def_str)
        c_steps.metric("Steps", steps_str)

    # ── Action log in scrollable container ────────────────────────────────────
    st.markdown("**Action Log**")
    if possession.action_log:
        with st.container(height=260):
            for i, entry in enumerate(possession.action_log, start=1):
                if isinstance(entry, dict):
                    text    = entry.get("text", "")
                    details = entry.get("details", [])
                    style   = entry.get("style", "normal")
                else:
                    text    = entry
                    details = []
                    style   = "normal"
                
                # We can visually distinguish off-ball actions softly with italics
                if style == "offball":
                    st.markdown(f"*{i}. {text}*")
                else:
                    st.markdown(f"**{i}.** {text}")
                    
                for detail in details:
                    st.caption(f"↳ {detail}")
    else:
        st.caption("No actions yet — press ▶ Step.")

    # ── Coach's Tactical Analysis (CoT expander) ───────────────────────────────
    _cr = st.session_state.coaching_record
    if _cr and _cr.get("cot_text"):
        with st.expander("Coach's Tactical Analysis"):
            cot_display = _cr["cot_text"].split("## DATA_START")[0]
            st.markdown(cot_display)

# ── Timeout handler ────────────────────────────────────────────────────────────
if timeout_clicked:
    handle_timeout(blue_team, red_team)
    st.rerun()

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
    if st.session_state.possession.is_over:
        rec = record_possession(st.session_state.possession)
        st.session_state.possession_history.append(rec)
        st.session_state.possession_history = st.session_state.possession_history[-10:]
    st.session_state.possession = new_possession(blue_team, red_team)
    # Re-apply coached positions over the defaults set by new_possession()
    for name, (x, y) in st.session_state.get("coached_positions", {}).items():
        try:
            blue_team.player_by_name(name).place(x, y)
        except (ValueError, AttributeError):
            pass
    st.rerun()

# ── Auto-play loop
if st.session_state.auto_play:
    if possession.is_over:
        time.sleep(0.4)  # brief pause so the result is readable before next possession
        rec = record_possession(possession)
        st.session_state.possession_history.append(rec)
        st.session_state.possession_history = st.session_state.possession_history[-10:]
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

# ── Command Control Pane ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader(" Control Pane")

tab_matchups, tab_on_ball, tab_off_ball, tab_coach = st.tabs([
    "⚔️ Matchups (Cards)",
    "🎯 On-Ball Tendencies",
    "🏃‍♂️ Off-Ball System",
    "🧠 Coach Intel",
])

# ── Matchups (Cards) ───────────────────────────────────────────────────────────
with tab_matchups:
    st.caption("Attributes for current on-court matchups (Blue Offense vs Red Defense).")
    cols = st.columns(5)
    for i, player in enumerate(blue_team.players):
        defender = possession.matchups.get(player)
        is_bh = player.name == possession.ball_handler.name
        
        with cols[i]:
            with st.container(border=True):
                # Offensive Header
                bh_icon = "🏀 " if is_bh else ""
                st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:700; color:#64b5f6;'>{bh_icon}{player.name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center; font-size:11px; color:#aaa; margin-bottom:8px;'>{player.position.value} • Offense</div>", unsafe_allow_html=True)
                
                # Offensive Stats
                st.markdown(f"""
                <div style='font-size:11px; color:#eee;'>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding-bottom:2px;'><span>3PT</span><span style='color:#81c784;'>{player.offense.three_pt_shooting:.2f}</span></div>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding-bottom:2px;'><span>MID</span><span style='color:#81c784;'>{player.offense.mid_range_shooting:.2f}</span></div>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding-bottom:2px;'><span>DRV</span><span style='color:#81c784;'>{player.offense.drive_effectiveness:.2f}</span></div>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding-bottom:2px;'><span>PAS</span><span style='color:#81c784;'>{player.offense.passing:.2f}</span></div>
                    <div style='display:flex; justify-content:space-between; padding-top:2px;'><span>LAY</span><span style='color:#81c784;'>{player.offense.layup:.2f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='text-align:center; font-size:14px; font-weight:900; color:#555; margin:12px 0;'>VS</div>", unsafe_allow_html=True)
                
                # Defensive Header
                if defender:
                    st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:700; color:#ef9a9a;'>{defender.name}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center; font-size:11px; color:#aaa; margin-bottom:8px;'>{defender.position.value} • Defense</div>", unsafe_allow_html=True)
                    
                    # Defensive Stats
                    st.markdown(f"""
                    <div style='font-size:11px; color:#eee;'>
                        <div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding-bottom:2px;'><span>PERIM</span><span style='color:#e57373;'>{defender.defense.outside_defense:.2f}</span></div>
                        <div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding-bottom:2px;'><span>RIM PROT</span><span style='color:#e57373;'>{defender.defense.rim_protection:.2f}</span></div>
                        <div style='display:flex; justify-content:space-between; border-bottom:1px solid #333; padding-bottom:2px;'><span>SPEED</span><span style='color:#e57373;'>{defender.defense.speed:.2f}</span></div>
                        <div style='display:flex; justify-content:space-between; padding-top:2px;'><span>DEFLECT</span><span style='color:#e57373;'>{defender.defense.deflections:.2f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='text-align:center; color:#888; font-size:12px; padding:20px 0;'>None</div>", unsafe_allow_html=True)

# ── On-Ball Tendencies ──────────────────────────────────────────────────────────
with tab_on_ball:
    st.caption("Adjust player action probabilities directly. Values automatically normalize to 1.0. The table on the right shows the final effective weights against current defender.")
    col_edit, col_view = st.columns([1.2, 1])
    
    with col_edit:
        st.markdown("**Control Panel (Editable)**")
        # List of dicts for editor
        tendencies_data = []
        for p in blue_team.players:
            tendencies_data.append({
                "Player": p.name,
                "3PT": float(p.tendencies.tendency_three),
                "MID": float(p.tendencies.tendency_mid),
                "DRV": float(p.tendencies.tendency_drive),
                "PAS": float(p.tendencies.tendency_pass),
                "LAY": float(p.tendencies.tendency_layup),
            })
        
        edited_tendencies = st.data_editor(
            tendencies_data,
            hide_index=True,
            disabled=["Player"],
            use_container_width=True,
            key="on_ball_editor"
        )
        
        # Apply normalization back to players
        for i, row in enumerate(edited_tendencies):
            p = next(pl for pl in blue_team.players if pl.name == row["Player"])
            v3 = max(0.0, float(row["3PT"]))
            vm = max(0.0, float(row["MID"]))
            vd = max(0.0, float(row["DRV"]))
            vp = max(0.0, float(row["PAS"]))
            vl = max(0.0, float(row["LAY"]))
            
            tot = v3 + vm + vd + vp + vl
            if tot > 0:
                p.tendencies.tendency_three = v3 / tot
                p.tendencies.tendency_mid   = vm / tot
                p.tendencies.tendency_drive = vd / tot
                p.tendencies.tendency_pass  = vp / tot
                p.tendencies.tendency_layup = vl / tot
            elif tot == 0:
                p.tendencies.tendency_pass  = 1.0
                p.tendencies.tendency_three = 0.0
                p.tendencies.tendency_mid   = 0.0
                p.tendencies.tendency_drive = 0.0
                p.tendencies.tendency_layup = 0.0
                
    with col_view:
        st.markdown("**Live Effective Weights**")
        bh_name = possession.ball_handler.name
        labels   = ["3PT", "MID", "DRV", "PASS", "LAY"]
        header_cells = "".join(f"<th style='padding:3px 6px; font-size:9px; color:#666; text-align:right; font-weight:normal; text-transform:uppercase; letter-spacing:.05em;'>{l}</th>" for l in labels)
        
        rows_html = (
            f"<table style='border-collapse:collapse; width:100%; font-size:10px;'>"
            f"<thead><tr><th style='padding:3px 6px; font-size:9px; color:#666; text-align:left; font-weight:normal;'>Player</th>{header_cells}</tr></thead><tbody>"
        )
        for player in blue_team.players:
            raw   = player.tendencies.as_weights()
            eff   = effective_weights(player, possession.matchups.get(player))
            is_bh = player.name == bh_name
            row_bg     = "#1a2540" if is_bh else "transparent"
            name_color = "#64b5f6" if is_bh else "#999"
            rows_html += f"<tr style='background:{row_bg}; border-bottom:1px solid #1e1e1e;'>"
            rows_html += f"<td style='padding:3px 6px; color:{name_color}; white-space:nowrap;'>{'● ' if is_bh else ''}{player.name} ({player.position.value})</td>"
            
            for rw, ew in zip(raw, eff):
                diff  = ew - rw
                val_color = "#81c784" if diff > 0.005 else ("#e57373" if diff < -0.005 else "#aaa")
                rows_html += f"<td style='padding:3px 6px; text-align:right; color:{val_color}; font-weight:{'600' if is_bh else '400'};'>{ew:.0%}</td>"
            rows_html += "</tr>"
        rows_html += "</tbody></table>"
        st.markdown(rows_html, unsafe_allow_html=True)


# ── Off-Ball Tendencies ─────────────────────────────────────────────────────────
with tab_off_ball:
    st.caption("Edit system-wide off-ball movement probabilities.")
    col_base, col_table = st.columns([1, 2.2])
    
    with col_base:
        st.markdown("**Base Weights**")
        st.number_input("Base Stay weight", min_value=0.0, max_value=2.0, step=0.05, key="tend_base_stay", help="Higher value means players are more likely to stay in place.")
        
    with col_table:
        st.markdown("**Position Specifics (Editable)**")
        
        off_ball_data = []
        for _pos in _POSITIONS:
            off_ball_data.append({
                "Position": _pos,
                "Cut": float(st.session_state.get(f"tend_cut_{_pos}", TENDENCIES.cut_factors[_pos])),
                "Screen": float(st.session_state.get(f"tend_screen_{_pos}", TENDENCIES.screen_factors[_pos])),
                "Pop": float(st.session_state.get(f"tend_pop_{_pos}", TENDENCIES.pop_probabilities[_pos])),
            })
            
        edited_off_ball = st.data_editor(
            off_ball_data,
            hide_index=True,
            disabled=["Position"],
            use_container_width=True,
            key="off_ball_editor"
        )
        
        for row in edited_off_ball:
            _pos = row["Position"]
            st.session_state[f"tend_cut_{_pos}"]    = max(0.0, float(row["Cut"]))
            st.session_state[f"tend_screen_{_pos}"] = max(0.0, float(row["Screen"]))
            st.session_state[f"tend_pop_{_pos}"]    = max(0.0, float(row["Pop"]))

# ── Sync any edited off-ball tendency values → TENDENCIES singleton ───────────
TENDENCIES.base_stay = st.session_state.get("tend_base_stay", TENDENCIES.base_stay)
for _pos in _POSITIONS:
    TENDENCIES.cut_factors[_pos]       = st.session_state.get(f"tend_cut_{_pos}",    TENDENCIES.cut_factors[_pos])
    TENDENCIES.screen_factors[_pos]    = st.session_state.get(f"tend_screen_{_pos}", TENDENCIES.screen_factors[_pos])
    TENDENCIES.pop_probabilities[_pos] = st.session_state.get(f"tend_pop_{_pos}",    TENDENCIES.pop_probabilities[_pos])


# ── Coach Intel Tab helpers ────────────────────────────────────────────────────
def _render_tendency_deltas(before: dict, after: dict, players: list) -> None:
    """Render a color-coded HTML table of on-ball tendency deltas."""
    THRESHOLD = 0.005
    rows = []
    for p in players:
        name = p.name
        b, a = before[name], after[name]
        deltas = {k: a[k] - b[k] for k in b}
        if all(abs(d) < THRESHOLD for d in deltas.values()):
            continue
        rows.append((name, deltas))

    if not rows:
        st.caption("No on-ball tendency changes.")
        return

    cols_order = ["3PT", "MID", "DRV", "PAS", "LAY"]
    header = "".join(
        f"<th style='padding:3px 8px; font-size:9px; color:#666; text-align:right; font-weight:normal; text-transform:uppercase;'>{c}</th>"
        for c in cols_order
    )
    html = (
        "<table style='border-collapse:collapse; width:100%; font-size:11px;'>"
        f"<thead><tr>"
        f"<th style='padding:3px 8px; font-size:9px; color:#666; text-align:left; font-weight:normal;'>Player</th>"
        f"{header}"
        f"</tr></thead><tbody>"
    )
    for name, deltas in rows:
        html += f"<tr style='border-bottom:1px solid #1e1e1e;'>"
        html += f"<td style='padding:3px 8px; color:#ccc; white-space:nowrap;'>{name}</td>"
        for col in cols_order:
            d = deltas[col]
            if d > THRESHOLD:
                color = "#81c784"
                sign  = "+"
            elif d < -THRESHOLD:
                color = "#e57373"
                sign  = ""
            else:
                color = "#555"
                sign  = ""
            html += f"<td style='padding:3px 8px; text-align:right; color:{color}; font-weight:600;'>{sign}{d:+.3f}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def _render_off_ball_deltas(before: dict, after: dict) -> None:
    """Render a table of changed off-ball parameters."""
    THRESHOLD = 0.005
    rows = []
    for dict_key in ("cut_factors", "screen_factors", "pop_probabilities"):
        for pos in _POSITIONS:
            b_val = before[dict_key][pos]
            a_val = after[dict_key][pos]
            delta = a_val - b_val
            if abs(delta) >= THRESHOLD:
                rows.append((f"{dict_key} / {pos}", b_val, a_val, delta))
    # base_stay
    b_bs = before["base_stay"]
    a_bs = after["base_stay"]
    d_bs = a_bs - b_bs
    if abs(d_bs) >= THRESHOLD:
        rows.append(("base_stay", b_bs, a_bs, d_bs))

    if not rows:
        st.caption("No off-ball system changes.")
        return

    html = (
        "<table style='border-collapse:collapse; width:100%; font-size:11px;'>"
        "<thead><tr>"
        "<th style='padding:3px 8px; font-size:9px; color:#666; text-align:left; font-weight:normal;'>Parameter</th>"
        "<th style='padding:3px 8px; font-size:9px; color:#666; text-align:right; font-weight:normal;'>Before</th>"
        "<th style='padding:3px 8px; font-size:9px; color:#666; text-align:right; font-weight:normal;'>After</th>"
        "<th style='padding:3px 8px; font-size:9px; color:#666; text-align:right; font-weight:normal;'>Δ</th>"
        "</tr></thead><tbody>"
    )
    for param, bv, av, dv in rows:
        d_color = "#81c784" if dv > 0 else "#e57373"
        sign    = "+" if dv > 0 else ""
        html += (
            f"<tr style='border-bottom:1px solid #1e1e1e;'>"
            f"<td style='padding:3px 8px; color:#ccc;'>{param}</td>"
            f"<td style='padding:3px 8px; text-align:right; color:#aaa;'>{bv:.3f}</td>"
            f"<td style='padding:3px 8px; text-align:right; color:#aaa;'>{av:.3f}</td>"
            f"<td style='padding:3px 8px; text-align:right; color:{d_color}; font-weight:600;'>{sign}{dv:+.3f}</td>"
            f"</tr>"
        )
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


# ── Coach Intel Tab ────────────────────────────────────────────────────────────
with tab_coach:
    _rec = st.session_state.coaching_record
    if _rec is None:
        st.info("No timeout called yet. Complete a possession and click 📋 Timeout.")
    else:
        # Surface any errors that occurred during the coaching pipeline
        if _rec.get("error_trace"):
            st.error("An error occurred during the coaching pipeline — see details below.")
            with st.expander("⚠️ Error Traceback", expanded=True):
                st.code(_rec["error_trace"])

        # ── Section A: What the coach saw ──────────────────────────────────────
        with st.expander("📥 Inputs to Coach", expanded=True):
            if _rec.get("prompt_data"):
                st.subheader("System Prompt")
                st.code(_rec["prompt_data"]["system"], language="markdown")
                st.subheader("User Prompt (Game Narrative & State)")
                st.code(_rec["prompt_data"]["user"], language="markdown")
            else:
                st.subheader("Game Logs")
                st.markdown(_rec["narrative"] or "_No logic generated._")

            st.subheader("On-Ball Tendencies (before timeout)")
            _bt = _rec["before_tendencies"]
            _before_df = pd.DataFrame(_bt).T
            _before_df.index.name = "Player"
            st.dataframe(_before_df.style.format("{:.3f}"), use_container_width=True)

            st.subheader("Off-Ball State (before timeout)")
            _bob = _rec["before_off_ball"]
            _ob_html = (
                "<table style='border-collapse:collapse; width:100%; font-size:11px;'>"
                "<thead><tr>"
                + "".join(
                    f"<th style='padding:3px 8px; font-size:9px; color:#666; text-align:right; font-weight:normal;'>{p}</th>"
                    for p in [""] + _POSITIONS
                )
                + "</tr></thead><tbody>"
            )
            for dict_key, label in (
                ("cut_factors", "Cut"),
                ("screen_factors", "Screen"),
                ("pop_probabilities", "Pop"),
            ):
                _ob_html += f"<tr style='border-bottom:1px solid #1e1e1e;'><td style='padding:3px 8px; color:#aaa;'>{label}</td>"
                for pos in _POSITIONS:
                    _ob_html += f"<td style='padding:3px 8px; text-align:right; color:#ccc;'>{_bob[dict_key][pos]:.3f}</td>"
                _ob_html += "</tr>"
            _ob_html += (
                f"<tr><td style='padding:3px 8px; color:#aaa;'>Base Stay</td>"
                f"<td colspan='{len(_POSITIONS)}' style='padding:3px 8px; color:#ccc;'>{_bob['base_stay']:.3f}</td></tr>"
                "</tbody></table>"
            )
            st.markdown(_ob_html, unsafe_allow_html=True)

        # ── Section B: What the coach said ─────────────────────────────────────
        with st.expander("💬 Coach's Analysis", expanded=True):
            _cot = _rec.get("cot_text") or ""
            if not _cot:
                st.caption("No LLM response captured.")
            else:
                if "## DATA_START" in _cot:
                    _prose, _rest = _cot.split("## DATA_START", 1)
                    _json_block   = _rest.split("## DATA_END")[0].strip()
                else:
                    _prose      = _cot
                    _json_block = ""
                st.markdown(_prose.strip())
                if _json_block:
                    st.code(_json_block, language="json")

        # ── Section C: What materially changed ─────────────────────────────────
        with st.expander("📊 Strategy Changes", expanded=True):
            st.markdown("**On-Ball Tendency Deltas**")
            _render_tendency_deltas(
                _rec["before_tendencies"],
                _rec["after_tendencies"],
                blue_team.players,
            )

            st.markdown("**Off-Ball System Changes**")
            _render_off_ball_deltas(_rec["before_off_ball"], _rec["after_off_ball"])

            if _rec["coached_positions"]:
                st.markdown("**Positioning Changes**")
                for _pname, (_px, _py) in _rec["coached_positions"].items():
                    try:
                        _pp = blue_team.player_by_name(_pname)
                        _zone_label = _pp.zone.value if _pp.zone else "unknown zone"
                    except Exception:
                        _zone_label = "unknown zone"
                    st.markdown(
                        f"- **{_pname}** → {_zone_label} ({_px:.1f}, {_py:.1f})"
                    )

            with st.expander("Full Apply Log"):
                st.code("\n".join(_rec["logs"]))
