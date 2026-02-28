"""
simulation/engine.py
--------------------
Possession state machine: matchups, effective tendency weights, step
resolution, and man-to-man defensive repositioning.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from models.player import Player, Role, Tendencies
from models.team import Team
from models.court import CourtZone

from simulation.utils import (
    player_dist,
    contest_factor,
    distance_decay,
    CONTEST_RADIUS,
    DRIVE_CLOSE_THRESHOLD,
    DEFENDER_SNAP_OFFSET,
)
from simulation.actions import resolve_shot, resolve_drive, resolve_pass

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_STEPS = 12   # Hard cap; possession ends as a turnover if exceeded

_THREE_PT_ZONES = frozenset({
    CourtZone.CORNER_3_LEFT,
    CourtZone.CORNER_3_RIGHT,
    CourtZone.WING_3_LEFT,
    CourtZone.WING_3_RIGHT,
    CourtZone.TOP_OF_KEY_3,
})

# ── Matchup type ───────────────────────────────────────────────────────────────

# Maps each offensive player → their matched defender (paired by Position).
Matchups = dict   # dict[Player, Player]


def build_matchups(offense_team: Team, defense_team: Team) -> Matchups:
    """Pair each offensive player with the defender matching their Position."""
    matchups: Matchups = {}
    for off_player in offense_team:
        for def_player in defense_team:
            if def_player.position == off_player.position:
                matchups[off_player] = def_player
                break
    return matchups


# ── Possession state ───────────────────────────────────────────────────────────

@dataclass
class PossessionState:
    """Full mutable state for one possession."""
    ball_handler: Player
    matchups: Matchups
    action_log: list = field(default_factory=list)
    score: int = 0
    is_over: bool = False
    outcome: Optional[str] = None
    # outcome values: "MADE_2" | "MADE_3" | "MISSED" | "TURNOVER" | "INTERCEPTED"
    last_annotation: Optional[dict] = None
    # Annotation keys:
    #   type PASS  → from_x, from_y, to_x, to_y, success: bool
    #   type DRIVE → from_x, from_y, to_x, to_y
    #   type SHOT  → from_x, from_y, shot_type: str, made: bool


# ── Effective tendency weights ──────────────────────────────────────────────────

def _effective_weights(ball_handler: Player, defender: Optional[Player]) -> list[float]:
    """Compute per-action weights with shoot/drive multiplied by expected success.

    Pass tendency is left unmodified.  All weights are renormalised to sum 1.
    Falls back to raw tendencies if a multiplier cannot be computed.
    """
    raw = ball_handler.tendencies.as_weights()  # [3PT, MID, DRIVE, PASS, LAYUP]
    zone = ball_handler.zone

    if defender is None or not defender.is_on_court() or zone is None:
        return raw

    d = player_dist(ball_handler, defender)

    p_3pt = (
        ball_handler.offense.three_pt_shooting
        * contest_factor(d, defender.defense.outside_defense, CONTEST_RADIUS)
        * distance_decay(ball_handler.x, ball_handler.y)
    )
    p_mid = (
        ball_handler.offense.mid_range_shooting
        * contest_factor(d, defender.defense.outside_defense, CONTEST_RADIUS)
    )
    p_drive = (
        ball_handler.offense.drive_effectiveness
        * contest_factor(d, defender.defense.speed, DRIVE_CLOSE_THRESHOLD)
    )
    p_layup = (
        ball_handler.offense.layup
        * contest_factor(d, defender.defense.rim_protection, CONTEST_RADIUS)
    )

    weights = [
        raw[0] * p_3pt,    # 3PT
        raw[1] * p_mid,    # MID
        raw[2] * p_drive,  # DRIVE
        raw[3],            # PASS — unmodified
        raw[4] * p_layup,  # LAYUP
    ]

    # Zero out shot types that are impossible from the player's current zone.
    # Remaining weight is absorbed by valid actions through re-normalisation.
    if zone in _THREE_PT_ZONES:
        # In 3PT land: only 3PT shot, drive, or pass are valid.
        weights[1] = 0.0   # no MID
        weights[4] = 0.0   # no LAYUP
    elif zone == CourtZone.RESTRICTED_AREA:
        # At the rim: only layup, drive, or pass.
        weights[0] = 0.0   # no 3PT
        weights[1] = 0.0   # no MID
    elif zone in (CourtZone.PAINT, CourtZone.MID_RANGE):
        # Inside the arc but not at the rim: mid-range, drive, or pass.
        weights[0] = 0.0   # no 3PT
        weights[4] = 0.0   # no LAYUP

    total = sum(weights)
    if total == 0.0:
        return raw
    return [w / total for w in weights]


# ── Man-to-man defensive repositioning ────────────────────────────────────────

def update_defense(matchups: Matchups) -> None:
    """Probabilistically close each defender toward their matched offensive player.

    Chance of closing is proportional to the defender's speed attribute.
    On success, the defender snaps to (CONTEST_RADIUS - DEFENDER_SNAP_OFFSET) ft
    away from their man, approaching from their current angle.
    """
    for off_player, defender in matchups.items():
        if not off_player.is_on_court() or not defender.is_on_court():
            continue
        if random.random() > defender.defense.speed:
            continue    # too slow to close on this step

        dx = off_player.x - defender.x
        dy = off_player.y - defender.y
        current_d = math.sqrt(dx * dx + dy * dy)
        target_dist = max(0.0, CONTEST_RADIUS - DEFENDER_SNAP_OFFSET)

        if current_d < 0.01 or current_d <= target_dist:
            continue    # already close enough

        ratio = target_dist / current_d
        new_x = min(50.0, max(0.0, off_player.x - dx * ratio))
        new_y = min(47.0, max(0.0, off_player.y - dy * ratio))
        defender.place(new_x, new_y)


# ── Step resolver ──────────────────────────────────────────────────────────────

def _shot_type_for_zone(zone: Optional[CourtZone]) -> str:
    """Best natural shot type for a player's current zone."""
    if zone == CourtZone.RESTRICTED_AREA:
        return "LAYUP"
    if zone in _THREE_PT_ZONES:
        return "3PT"
    return "MID"


def step_possession(
    state: PossessionState,
    blue_team: Team,
    red_team: Team,
) -> PossessionState:
    """Advance the possession by one player action.

    Order:
        1. Defenders close in (man-to-man).
        2. Ball handler selects action via effective weights.
        3. Action is resolved; state is updated.
    """
    if state.is_over:
        return state

    # Hard cap guard
    if len(state.action_log) >= MAX_STEPS:
        state.action_log.append("Shot clock: possession ends — turnover.")
        state.outcome = "TURNOVER"
        state.is_over = True
        return state

    # 1. Defense closes in
    update_defense(state.matchups)

    bh = state.ball_handler
    defender = state.matchups.get(bh)
    zone = bh.zone

    # 2. Pick action
    action = random.choices(
        Tendencies.ACTION_LABELS,
        weights=_effective_weights(bh, defender),
    )[0]

    # Convert DRIVE to LAYUP when already in restricted area (can't drive further in)
    if action == "DRIVE" and zone == CourtZone.RESTRICTED_AREA:
        action = "LAYUP"

    offense_players = list(blue_team)
    all_defenders = list(red_team)

    # 3. Resolve action
    if action in ("3PT", "MID", "LAYUP"):
        result = resolve_shot(bh, defender, action, zone)
        state.action_log.append(result.description)
        state.last_annotation = {
            "type": "SHOT",
            "from_x": bh.x, "from_y": bh.y,
            "shot_type": action,
            "made": result.made,
        }
        if result.made:
            state.score = 3 if action == "3PT" else 2
            state.outcome = "MADE_3" if action == "3PT" else "MADE_2"
        else:
            state.outcome = "MISSED"
        state.is_over = True

    elif action == "DRIVE":
        result = resolve_drive(bh, defender)
        state.action_log.append(result.description)
        state.last_annotation = {
            "type": "DRIVE",
            "from_x": bh.x, "from_y": bh.y,
            "to_x": result.new_x, "to_y": result.new_y,
        }
        if not result.success:
            state.outcome = "TURNOVER"
            state.is_over = True
        else:
            bh.place(result.new_x, result.new_y)

    elif action == "PASS":
        from_x, from_y = bh.x, bh.y
        teammates = [p for p in offense_players if p is not bh]
        result = resolve_pass(bh, teammates, all_defenders)
        state.action_log.append(result.description)
        to_x = result.recipient.x if result.recipient else from_x
        to_y = result.recipient.y if result.recipient else from_y
        state.last_annotation = {
            "type": "PASS",
            "from_x": from_x, "from_y": from_y,
            "to_x": to_x, "to_y": to_y,
            "success": result.success,
        }
        if not result.success:
            state.outcome = "INTERCEPTED" if result.interceptor else "TURNOVER"
            state.is_over = True
        elif result.recipient is not None:
            state.ball_handler = result.recipient

    return state


# ── Possession factory ─────────────────────────────────────────────────────────

def new_possession(blue_team: Team, red_team: Team) -> PossessionState:
    """Create a fresh possession.

    Resets both teams to default positions.
    Ball starts with the PG on the Blue (offense) team.
    """
    blue_team.place_at_defaults()
    red_team.place_at_defaults()

    ball_handler = next(
        (p for p in blue_team if p.position.value == "PG"),
        blue_team.players[0],
    )
    matchups = build_matchups(blue_team, red_team)
    return PossessionState(ball_handler=ball_handler, matchups=matchups)
