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
from models.court import CourtZone, get_zone, THREE_PT_RADIUS

from simulation.utils import (
    player_dist,
    contest_factor,
    distance_decay,
    CONTEST_RADIUS,
    DRIVE_CLOSE_THRESHOLD,
    DEFENDER_SNAP_OFFSET,
)
from simulation.actions import resolve_shot, resolve_drive, resolve_pass
from simulation.off_ball import (
    OffBallTendencies,
    TENDENCIES as _DEFAULT_TENDENCIES,
    off_ball_decision_weights,
    resolve_cut,
    resolve_off_ball_screen,
    resolve_on_ball_screen,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_STEPS = 12   # Hard cap; possession ends as a turnover if exceeded

_BASKET_X = 25.0
_BASKET_Y = 5.25

# Drive candidates: (label, target_dist_from_basket, shot_type_if_reached, landing_zone_or_None)
# landing_zone=None means compute it from the landing coord (used for the 3PT candidate
# because the exact zone varies with the player's angle).
# Drive is the universal movement mechanism: any candidate whose landing zone differs
# from the player's current zone is fair game — inward or outward.
_DRIVE_CANDIDATES = [
    ("rim",            2.4,             "LAYUP", CourtZone.RESTRICTED_AREA),
    ("paint",          8.5,             "MID",   CourtZone.PAINT),
    ("mid-range",     15.0,             "MID",   CourtZone.MID_RANGE),
    ("three-pt line", THREE_PT_RADIUS + 0.5, "3PT", None),
]

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
    screened_defenders: set = field(default_factory=set)
    # Names of defenders pinned by a screen this step.  Cleared and
    # repopulated by _step_off_ball_actions at the start of each step.
    # A screened defender is treated as absent when the ball handler
    # selects and resolves their on-ball action.
    off_ball_annotations: list = field(default_factory=list)
    # Per-step CUT / SCREEN animation dicts for app.py to play before the
    # ball-handler action animation.
    tendencies: OffBallTendencies = field(default_factory=lambda: _DEFAULT_TENDENCIES)
    # Active off-ball tendency model.  Defaults to the module-level
    # TENDENCIES singleton so global changes are visible automatically.
    # Assign a fresh OffBallTendencies() to override for a single possession
    # without affecting other possessions.


# ── Drive target selection ─────────────────────────────────────────────────────

def _drive_landing(ball_handler, target_dist: float):
    """Return (lx, ly) for a landing spot target_dist ft from the basket,
    along the straight line from the ball handler toward the basket."""
    dx = ball_handler.x - _BASKET_X
    dy = ball_handler.y - _BASKET_Y
    current_dist = math.sqrt(dx * dx + dy * dy)
    if current_dist < 0.01:
        return _BASKET_X, _BASKET_Y
    scale = target_dist / current_dist
    lx = min(50.0, max(0.0, _BASKET_X + dx * scale))
    ly = min(47.0, max(0.0, _BASKET_Y + dy * scale))
    return lx, ly


def _best_drive_target(
    ball_handler,
    all_defenders: list,
) -> tuple:
    """Pick the most valuable drive target zone.

    For every candidate landing zone that is *closer* to the basket than the
    ball handler's current position, we compute:

        score = p_reach × shot_EV_at_landing

    where:
        p_reach            = drive_effectiveness × contest_factor(nearest_def_to_landing, speed)
        shot_EV_at_landing = shot_attr × contest_factor(nearest_def_to_landing, def_shot_attr)

    The candidate with the highest score is returned as
        (land_x, land_y, landing_zone, shot_type, label, score)
    """
    dx = ball_handler.x - _BASKET_X
    dy = ball_handler.y - _BASKET_Y
    current_dist = math.sqrt(dx * dx + dy * dy)

    on_court_defs = [d for d in all_defenders if d.is_on_court()]

    best_score = -1.0
    best = None

    for label, target_dist, shot_type, landing_zone_hint in _DRIVE_CANDIDATES:
        lx, ly = _drive_landing(ball_handler, target_dist)
        landing_zone = landing_zone_hint if landing_zone_hint is not None else get_zone(lx, ly)

        # Skip if this would leave the player in the same zone they're already in.
        if landing_zone == ball_handler.zone:
            continue

        if on_court_defs:
            nearest = min(
                on_court_defs,
                key=lambda d: math.sqrt((d.x - lx) ** 2 + (d.y - ly) ** 2),
            )
            def_dist = math.sqrt((nearest.x - lx) ** 2 + (nearest.y - ly) ** 2)

            # Probability of breaking through to this spot.
            p_reach = ball_handler.offense.drive_effectiveness * contest_factor(
                def_dist, nearest.defense.speed, DRIVE_CLOSE_THRESHOLD
            )

            # Expected shot quality once there.
            if shot_type == "LAYUP":
                shot_attr     = ball_handler.offense.layup
                def_shot_attr = nearest.defense.rim_protection
            elif shot_type == "3PT":
                shot_attr     = ball_handler.offense.three_pt_shooting
                def_shot_attr = nearest.defense.outside_defense
            else:
                shot_attr     = ball_handler.offense.mid_range_shooting
                def_shot_attr = nearest.defense.outside_defense
            shot_cf = contest_factor(def_dist, def_shot_attr, CONTEST_RADIUS)
        else:
            p_reach = ball_handler.offense.drive_effectiveness
            if shot_type == "LAYUP":
                shot_attr = ball_handler.offense.layup
            elif shot_type == "3PT":
                shot_attr = ball_handler.offense.three_pt_shooting
            else:
                shot_attr = ball_handler.offense.mid_range_shooting
            shot_cf = 1.0

        score = p_reach * shot_attr * shot_cf

        if score > best_score:
            best_score = score
            best = (lx, ly, landing_zone, shot_type, label, score)

    if best is None:
        # Fallback: always allow driving to the rim.
        lx, ly = _drive_landing(ball_handler, 2.4)
        best = (lx, ly, CourtZone.RESTRICTED_AREA, "LAYUP", "rim", 0.0)

    return best


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

    # ── Step 1: mask invalid actions for the current zone ──────────────────────
    # Build a validity mask first so the raw proportions are preserved when we
    # renormalise.  Drive is always valid — it moves the player to a new zone.
    valid = [1.0, 1.0, 1.0, 1.0, 1.0]  # [3PT, MID, DRIVE, PASS, LAYUP]
    if zone in _THREE_PT_ZONES:
        valid[1] = 0.0   # no MID
        valid[4] = 0.0   # no LAYUP
    elif zone == CourtZone.RESTRICTED_AREA:
        valid[0] = 0.0   # no 3PT
        valid[1] = 0.0   # no MID
    elif zone in (CourtZone.PAINT, CourtZone.MID_RANGE):
        valid[0] = 0.0   # no 3PT
        valid[4] = 0.0   # no LAYUP

    # Redistribute raw tendency mass from invalid actions proportionally to
    # valid ones, preserving the relative intent of the CSV tendencies.
    masked = [r * v for r, v in zip(raw, valid)]
    masked_total = sum(masked)
    if masked_total < 1e-9:
        # Fallback: everything is valid (shouldn't happen with well-formed data).
        masked = raw[:]
        masked_total = sum(masked)
    rebased = [m / masked_total for m in masked]

    # ── Step 2: scale each valid action by expected success ────────────────────
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
    dx = ball_handler.x - _BASKET_X
    dy = ball_handler.y - _BASKET_Y
    current_dist = math.sqrt(dx * dx + dy * dy)
    p_drive = 0.0
    for label, target_dist, shot_type, _lz_hint in _DRIVE_CANDIDATES:
        lx_tmp, ly_tmp = _drive_landing(ball_handler, target_dist) if current_dist > 0.01 else (_BASKET_X, _BASKET_Y)
        landing_zone_tmp = _lz_hint if _lz_hint is not None else get_zone(lx_tmp, ly_tmp)
        if landing_zone_tmp == zone:
            continue
        if shot_type == "LAYUP":
            shot_attr = ball_handler.offense.layup
            def_attr  = defender.defense.rim_protection
        elif shot_type == "3PT":
            shot_attr = ball_handler.offense.three_pt_shooting
            def_attr  = defender.defense.outside_defense
        else:
            shot_attr = ball_handler.offense.mid_range_shooting
            def_attr  = defender.defense.outside_defense
        ev = (
            ball_handler.offense.drive_effectiveness
            * contest_factor(d, defender.defense.speed, DRIVE_CLOSE_THRESHOLD)
            * shot_attr
            * contest_factor(d, def_attr, CONTEST_RADIUS)
        )
        if ev > p_drive:
            p_drive = ev
    p_layup = (
        ball_handler.offense.layup
        * contest_factor(d, defender.defense.rim_protection, CONTEST_RADIUS)
    )

    weights = [
        rebased[0] * p_3pt,    # 3PT
        rebased[1] * p_mid,    # MID
        rebased[2] * p_drive,  # DRIVE
        rebased[3],            # PASS — unmodified
        rebased[4] * p_layup,  # LAYUP
    ]

    total = sum(weights)
    if total == 0.0:
        return rebased
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


# ── Off-ball step ─────────────────────────────────────────────────────────────

def _step_off_ball_actions(
    state: PossessionState,
    offense_players: list,
    all_defenders: list,
) -> None:
    """Process off-ball decisions for every non-ball-handler offensive player.

    Called once per step *before* the ball handler picks their action.

    Mutates:
        * Player positions (cuts move the cutter; screens move the screener).
        * state.screened_defenders  — cleared then repopulated each step.
        * state.off_ball_annotations — overwritten with this step's events.
        * state.action_log          — off-ball descriptions appended.
    """
    state.screened_defenders.clear()
    state.off_ball_annotations = []

    bh          = state.ball_handler
    bh_defender = state.matchups.get(bh)

    for player in offense_players:
        if player is bh or not player.is_on_court():
            continue

        w_cut, w_screen, w_stay = off_ball_decision_weights(player, state.tendencies)
        action = random.choices(
            ["CUT", "SCREEN", "STAY"],
            weights=[w_cut, w_screen, w_stay],
        )[0]

        if action == "CUT":
            cutter_defender = state.matchups.get(player)
            result = resolve_cut(player, cutter_defender, all_defenders)

            # Move the cutter to the new spot unconditionally
            player.place(result.to_x, result.to_y)

            # Defender recovery: if cut was successful, determine if defender can still contest
            cut_outcome = "CONTESTED"  # default: defender stays contested
            defender_from_x = cutter_defender.x if cutter_defender else None
            defender_from_y = cutter_defender.y if cutter_defender else None
            defender_to_x = defender_from_x
            defender_to_y = defender_from_y

            if result.success and cutter_defender and cutter_defender.is_on_court():
                # Calculate distance from defender's current position to cut destination
                ddx = result.to_x - cutter_defender.x
                ddy = result.to_y - cutter_defender.y
                def_distance_to_cut = math.sqrt(ddx * ddx + ddy * ddy)

                # Recovery ability: defender's speed vs cutter's drive_effectiveness
                # Higher speed = better recovery; higher drive_effectiveness = harder to catch
                recovery_prob = cutter_defender.defense.speed * (1.0 - player.offense.drive_effectiveness * 0.5)

                if def_distance_to_cut > CONTEST_RADIUS:
                    # Defender is too far to contest; try to recover
                    if random.random() < recovery_prob:
                        # Defender recovers fast enough — snap to contest radius
                        ratio = CONTEST_RADIUS / def_distance_to_cut
                        defender_to_x = min(50.0, max(0.0, result.to_x - ddx * ratio))
                        defender_to_y = min(47.0, max(0.0, result.to_y - ddy * ratio))
                        cutter_defender.place(defender_to_x, defender_to_y)
                        cut_outcome = "CONTESTED"
                    else:
                        # Defender can't recover in time — cutter gets separation
                        cut_outcome = "OPEN"
                else:
                    # Already within contest radius
                    cut_outcome = "CONTESTED"

            # Build detailed log entry
            description = result.description
            if result.success:
                description += f" → {cut_outcome}."

            state.action_log.append(
                {"text": description, "details": [], "style": "offball"}
            )
            state.off_ball_annotations.append({
                "type":        "CUT",
                "player_name": result.player_name,
                "from_x":      result.from_x,
                "from_y":      result.from_y,
                "to_x":        result.to_x,
                "to_y":        result.to_y,
                "success":     result.success,
                "outcome":     cut_outcome if result.success else "COVERED",
                "defender_name": cutter_defender.name if cutter_defender else None,
                "defender_from_x": defender_from_x,
                "defender_from_y": defender_from_y,
                "defender_to_x": defender_to_x,
                "defender_to_y": defender_to_y,
            })

        elif action == "SCREEN":
            dist_to_bh  = player_dist(player, bh)
            bh_def_dist = (
                player_dist(bh, bh_defender)
                if bh_defender and bh_defender.is_on_court()
                else 99.0
            )
            # Prefer on-ball screen when screener is close to BH and the BH's
            # defender is tight (within 2.5× contest radius)
            prefer_on_ball = (dist_to_bh < 15.0 and bh_def_dist < CONTEST_RADIUS * 2.5)

            if prefer_on_ball:
                result = resolve_on_ball_screen(player, bh, bh_defender, all_defenders, state.tendencies)
                if result.success and bh_defender is not None:
                    state.screened_defenders.add(bh_defender.name)

                state.action_log.append(
                    {"text": result.description, "details": [], "style": "offball"}
                )
                player.place(result.final_x, result.final_y)
                state.off_ball_annotations.append({
                    "type":            "SCREEN",
                    "screener_name":   result.screener_name,
                    "screener_from_x": result.screener_from_x,
                    "screener_from_y": result.screener_from_y,
                    "target_name":     result.ball_handler_name,
                    "screen_x":        result.screen_x,
                    "screen_y":        result.screen_y,
                    "final_x":         result.final_x,
                    "final_y":         result.final_y,
                    "roll_or_pop":     result.roll_or_pop,
                    "success":         result.success,
                    "defender_name": bh_defender.name if bh_defender else None,
                    "defender_from_x": bh_defender.x if bh_defender else None,
                    "defender_from_y": bh_defender.y if bh_defender else None,
                    "defender_to_x": bh_defender.x if bh_defender else None,
                    "defender_to_y": bh_defender.y if bh_defender else None,
                })

            else:
                # Off-ball screen: free up the most tightly covered teammate
                off_ball_mates = [
                    p for p in offense_players
                    if p is not bh and p is not player and p.is_on_court()
                ]
                if not off_ball_mates:
                    continue

                def _coverage(t):
                    td = state.matchups.get(t)
                    return player_dist(t, td) if (td and td.is_on_court()) else 99.0

                target          = min(off_ball_mates, key=_coverage)
                target_defender = state.matchups.get(target)
                result = resolve_off_ball_screen(player, target, target_defender)

                if result.success and target_defender is not None:
                    state.screened_defenders.add(target_defender.name)

                state.action_log.append(
                    {"text": result.description, "details": [], "style": "offball"}
                )
                player.place(result.screen_x, result.screen_y)
                state.off_ball_annotations.append({
                    "type":            "SCREEN",
                    "screener_name":   result.screener_name,
                    "screener_from_x": result.screener_from_x,
                    "screener_from_y": result.screener_from_y,
                    "target_name":     result.target_name,
                    "screen_x":        result.screen_x,
                    "screen_y":        result.screen_y,
                    "final_x":         result.screen_x,
                    "final_y":         result.screen_y,
                    "roll_or_pop":     None,
                    "success":         result.success,
                    "defender_name": target_defender.name if target_defender else None,
                    "defender_from_x": target_defender.x if target_defender else None,
                    "defender_from_y": target_defender.y if target_defender else None,
                    "defender_to_x": target_defender.x if target_defender else None,
                    "defender_to_y": target_defender.y if target_defender else None,
                })


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

    offense_players = list(blue_team)
    all_defenders   = list(red_team)

    # 1.5  Off-ball actions (cuts and screens) happen before the ball handler
    _step_off_ball_actions(state, offense_players, all_defenders)

    bh       = state.ball_handler
    defender = state.matchups.get(bh)
    # When the ball handler's defender was pinned by a screen this step,
    # treat them as absent so the BH effectively acts uncontested.
    effective_defender = (
        None
        if (defender is not None and defender.name in state.screened_defenders)
        else defender
    )
    zone = bh.zone

    # 2. Pick action (screened defender → uncontested effective weights)
    action = random.choices(
        Tendencies.ACTION_LABELS,
        weights=_effective_weights(bh, effective_defender),
    )[0]

    # 3. Resolve action
    if action in ("3PT", "MID", "LAYUP"):
        result = resolve_shot(bh, effective_defender, action, zone)
        state.action_log.append({"text": result.description, "details": result.breakdown})
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
        # ── Pick the best zone to drive to ──────────────────────────────────
        target_x, target_y, target_zone, shot_type_at_target, target_label, _ = (
            _best_drive_target(bh, all_defenders)
        )

        result = resolve_drive(bh, effective_defender, target_x, target_y, target_label)
        state.action_log.append({"text": result.description, "details": result.breakdown})

        # Build defender chase animation data.
        # Defender positions captured here are *after* update_defense (step 1).
        def_from_x: Optional[float] = None
        def_from_y: Optional[float] = None
        def_to_x:   Optional[float] = None
        def_to_y:   Optional[float] = None
        def_name:   Optional[str]   = None
        if defender and defender.is_on_court():
            def_name   = defender.name
            def_from_x = defender.x
            def_from_y = defender.y
            if result.success:
                # Chase target: CONTEST_RADIUS ft behind the ball handler's
                # new position, approaching from the defender's current angle.
                ddx = defender.x - result.new_x
                ddy = defender.y - result.new_y
                ddist = math.sqrt(ddx * ddx + ddy * ddy)
                if ddist > 0.01:
                    scale    = CONTEST_RADIUS / ddist
                    def_to_x = min(50.0, max(0.0, result.new_x + ddx * scale))
                    def_to_y = min(47.0, max(0.0, result.new_y + ddy * scale))
                else:
                    def_to_x, def_to_y = def_from_x, def_from_y
                # Commit the chase position so it persists into the next state.
                defender.place(def_to_x, def_to_y)
            else:
                # Failed drive — defender stays put (slight reactive shuffle).
                def_to_x, def_to_y = def_from_x, def_from_y

        state.last_annotation = {
            "type":           "DRIVE",
            "from_x":         bh.x,
            "from_y":         bh.y,
            "to_x":           result.new_x,
            "to_y":           result.new_y,
            "success":        result.success,
            "driver_name":    bh.name,
            "defender_name":  def_name,
            "defender_from_x": def_from_x,
            "defender_from_y": def_from_y,
            "defender_to_x":  def_to_x,
            "defender_to_y":  def_to_y,
        }
        if result.success:
            bh.place(result.new_x, result.new_y)

            if target_zone == CourtZone.RESTRICTED_AREA:
                # ── Auto-chain: layup from the rim ───────────────────────────────
                on_court_defs = [d for d in all_defenders if d.is_on_court()]
                rim_defender = (
                    min(
                        on_court_defs,
                        key=lambda d: math.sqrt(
                            (d.x - _BASKET_X) ** 2 + (d.y - _BASKET_Y) ** 2
                        ),
                    )
                    if on_court_defs else None
                )
                layup = resolve_shot(bh, rim_defender, "LAYUP", CourtZone.RESTRICTED_AREA)
                state.action_log.append({"text": layup.description, "details": layup.breakdown})
                state.score   = 2 if layup.made else 0
                state.outcome = "MADE_2" if layup.made else "MISSED"
                state.is_over = True
                state.last_annotation["layup"] = {
                    "made":   layup.made,
                    "from_x": bh.x,
                    "from_y": bh.y,
                }
            # For PAINT / MID_RANGE targets: player is now in a better zone;
            # possession continues and they'll pick their next action next step.
        # On failure: no turnover — ball handler keeps the ball and picks again next step.

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


def effective_weights(ball_handler: Player, defender: Optional[Player]) -> list[float]:
    """Public wrapper around _effective_weights for use in the UI."""
    return _effective_weights(ball_handler, defender)
