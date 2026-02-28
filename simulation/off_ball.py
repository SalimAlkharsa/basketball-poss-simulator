"""
simulation/off_ball.py
----------------------
Off-ball action resolvers: cuts, off-ball screens, and on-ball screens
(pick-and-roll / pick-and-pop).

Design principles
-----------------
* All resolvers are **pure functions** — they return result dataclasses and
  never mutate any player or state.  The engine owns all mutations.
* Probability of *attempting* an off-ball action is derived entirely from
  existing player attributes + a position-based factor, so no new CSV
  columns are needed.
* Cut destinations are chosen to maximise separation from the nearest
  defender, producing emergent "lifting" behaviour when players are deep in
  the paint and must move outward to open the floor.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from models.court import CourtZone, get_zone
from simulation.utils import player_dist, contest_factor, CONTEST_RADIUS, DEFENDER_SNAP_OFFSET

if TYPE_CHECKING:
    from models.player import Player

# ── Constants ──────────────────────────────────────────────────────────────────

_BASKET_X = 25.0
_BASKET_Y = 5.25

# How far around a defender a screener needs to be to potentially set the screen
SCREEN_RANGE = 6.0      # ft — screener must be within this of the screen position
# Radius within which a cut is checked for "getting open"
CUT_CONTEST_RADIUS = 5.0  # ft — wider than shot contest; defender needs a head start

# ── Off-ball tendency model ────────────────────────────────────────────────────

@dataclass
class OffBallTendencies:
    """All position-based probability weights that govern off-ball behaviour.

    Stored as a mutable dataclass so an LLM integration (or any external
    agent) can read and adjust individual values at runtime without touching
    source code.

    Fields
    ------
    cut_factors : dict[pos, float]
        Multiplied by a player's ``drive_effectiveness`` to get the raw
        *attempt* weight for cutting on each step.  Higher → position cuts
        more frequently.
    screen_factors : dict[pos, float]
        Multiplied by a player's ``strength`` to get the raw *attempt*
        weight for setting a screen.  Higher → position screens more.
    pop_probabilities : dict[pos, float]
        After a successful on-ball screen: probability the screener *pops*
        to a 3-pt spot instead of rolling to the basket.
    base_stay : float
        Fixed unnormalised weight added to every player's "stay / space"
        option.  Raise it to make off-ball movement less frequent overall;
        lower it for a more active off-ball game.
    default_cut_factor : float
        Fallback used when a position string is not found in ``cut_factors``.
    default_screen_factor : float
        Fallback used when a position string is not found in ``screen_factors``.
    default_pop_prob : float
        Fallback used when a position string is not found in
        ``pop_probabilities``.
    """
    cut_factors:       dict = field(default_factory=lambda: {
        "PG": 0.32, "SG": 0.28, "SF": 0.24, "PF": 0.16, "C": 0.08,
    })
    screen_factors:    dict = field(default_factory=lambda: {
        "PG": 0.06, "SG": 0.10, "SF": 0.14, "PF": 0.24, "C": 0.30,
    })
    pop_probabilities: dict = field(default_factory=lambda: {
        "PG": 0.55, "SG": 0.50, "SF": 0.45, "PF": 0.35, "C": 0.20,
    })
    base_stay:             float = 0.50
    default_cut_factor:    float = 0.20
    default_screen_factor: float = 0.12
    default_pop_prob:      float = 0.35


# Module-level default instance.  Mutate this directly to apply global changes;
# pass a custom instance to individual resolvers for per-possession overrides.
TENDENCIES = OffBallTendencies()

# ── Cut destination candidates ─────────────────────────────────────────────────
# These represent target (x, y) spots the cutter can move to, keyed by their
# source zone.  The resolver picks the candidate with maximum separation from
# the nearest on-court defender (the "open pocket").

_BASKET_CUT_SPOTS = [
    (25.0, 5.0),   # straight-line backdoor / basket seal
    (21.0, 6.0),   # left of the rim
    (29.0, 6.0),   # right of the rim
    (22.0, 9.0),   # mid-paint left
    (28.0, 9.0),   # mid-paint right
]

_PAINT_SPOTS = [
    (18.0, 11.0),  # mid-post left
    (32.0, 11.0),  # mid-post right
    (25.0, 13.0),  # high post / elbow
    (18.0, 15.0),  # elbow left
    (32.0, 15.0),  # elbow right
]

# "Lift" spots: exiting the paint/RA to open the floor
_LIFT_SPOTS = [
    (10.0, 22.0),  # left wing above-break (gives emergent "lift")
    (40.0, 22.0),  # right wing above-break
    (25.0, 28.0),  # top of the key
    (15.0, 20.0),  # left short-corner high
    (35.0, 20.0),  # right short-corner high
    ( 3.0, 10.0),  # left corner 3
    (47.0, 10.0),  # right corner 3
]

# 3-pt spots used for screener pop
_POP_SPOTS = [
    ( 3.0, 10.0),  # left corner 3
    (47.0, 10.0),  # right corner 3
    (10.0, 22.0),  # left wing
    (40.0, 22.0),  # right wing
    (25.0, 28.0),  # top of key
]

_THREE_PT_ZONES = frozenset({
    CourtZone.CORNER_3_LEFT,
    CourtZone.CORNER_3_RIGHT,
    CourtZone.WING_3_LEFT,
    CourtZone.WING_3_RIGHT,
    CourtZone.TOP_OF_KEY_3,
})

# Maps a player's current zone to the list of cut targets they should consider.
_CUT_CANDIDATES_BY_ZONE: dict = {
    CourtZone.TOP_OF_KEY_3:    _BASKET_CUT_SPOTS + _PAINT_SPOTS,
    CourtZone.WING_3_LEFT:     _BASKET_CUT_SPOTS + _PAINT_SPOTS,
    CourtZone.WING_3_RIGHT:    _BASKET_CUT_SPOTS + _PAINT_SPOTS,
    CourtZone.CORNER_3_LEFT:   _BASKET_CUT_SPOTS + _PAINT_SPOTS,
    CourtZone.CORNER_3_RIGHT:  _BASKET_CUT_SPOTS + _PAINT_SPOTS,
    CourtZone.MID_RANGE:       _BASKET_CUT_SPOTS,
    # Players in the paint/RA "lift" to create spacing — this drives emergent behaviour
    CourtZone.PAINT:           _LIFT_SPOTS,
    CourtZone.RESTRICTED_AREA: _LIFT_SPOTS,
}

# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class CutResult:
    """Result of an off-ball cut."""
    player_name: str
    success: bool               # True → got open (beat defender to the spot)
    from_x: float
    from_y: float
    to_x: float
    to_y: float
    destination_zone: Optional[CourtZone]
    description: str


@dataclass
class ScreenResult:
    """Result of an **off-ball** screen (setting for a teammate)."""
    screener_name: str
    screener_from_x: float
    screener_from_y: float
    target_name: str            # teammate being freed
    success: bool               # screen set cleanly
    screen_x: float             # screener's new position
    screen_y: float
    description: str


@dataclass
class OnBallScreenResult:
    """Result of an **on-ball** screen (pick-and-roll / pick-and-pop)."""
    screener_name: str
    screener_from_x: float
    screener_from_y: float
    ball_handler_name: str
    success: bool
    screen_x: float
    screen_y: float
    roll_or_pop: str            # "ROLL" or "POP"
    final_x: float              # screener's final position after roll/pop
    final_y: float
    final_zone: Optional[CourtZone]
    description: str


# ── Off-ball decision weights ──────────────────────────────────────────────────

def off_ball_decision_weights(
    player,
    tendencies: Optional[OffBallTendencies] = None,
) -> tuple[float, float, float]:
    """Return (w_cut, w_screen, w_stay) for an off-ball offensive player.

    Weights are unnormalised; the caller feeds them into random.choices.

    Args:
        player:     The off-ball offensive player.
        tendencies: ``OffBallTendencies`` instance to read from.  Defaults
                    to the module-level ``TENDENCIES`` singleton.
    """
    t        = tendencies if tendencies is not None else TENDENCIES
    pos_str  = player.position.value
    w_cut    = player.offense.drive_effectiveness * t.cut_factors.get(pos_str, t.default_cut_factor)
    w_screen = player.offense.strength            * t.screen_factors.get(pos_str, t.default_screen_factor)
    w_stay   = t.base_stay
    return w_cut, w_screen, w_stay


# ── Internal helpers ───────────────────────────────────────────────────────────

def _best_cut_destination(cutter, all_defenders: list) -> tuple[float, float]:
    """Return the (x, y) cut target with the most separation from any defender.

    Only considers spots that are meaningfully different from the cutter's
    current position (> 1 ft away).  Falls back to the rim if nothing fits.
    """
    candidates = _CUT_CANDIDATES_BY_ZONE.get(cutter.zone, _BASKET_CUT_SPOTS)
    on_court_defs = [d for d in all_defenders if d.is_on_court()]

    best_spot = None
    best_separation = -1.0

    for tx, ty in candidates:
        # Skip spots too close to where the player already is
        if math.sqrt((tx - cutter.x) ** 2 + (ty - cutter.y) ** 2) < 1.5:
            continue
        separation = (
            min(math.sqrt((d.x - tx) ** 2 + (d.y - ty) ** 2) for d in on_court_defs)
            if on_court_defs else 99.0
        )
        if separation > best_separation:
            best_separation = separation
            best_spot = (tx, ty)

    return best_spot if best_spot is not None else (_BASKET_X, _BASKET_Y + 0.5)


def _screen_position(
    screener,
    target,
    target_defender,
) -> tuple[float, float]:
    """Compute the (x, y) spot where the screener stands to pin the defender.

    Places the screener 1.5 ft in front of the defender (on the side facing
    the offensive target), blocking the defender's direct path to the target.
    """
    dx = target.x - target_defender.x
    dy = target.y - target_defender.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        return (
            min(50.0, max(0.0, target_defender.x + 1.0)),
            target_defender.y,
        )
    sx = min(50.0, max(0.0, target_defender.x + (dx / dist) * 1.5))
    sy = min(47.0, max(0.0, target_defender.y + (dy / dist) * 1.5))
    return sx, sy


def _best_pop_spot(all_defenders: list) -> tuple[float, float]:
    """Return the 3-pt pop spot with most separation from any defender."""
    on_court_defs = [d for d in all_defenders if d.is_on_court()]
    best = _POP_SPOTS[0]
    best_sep = -1.0
    for px, py in _POP_SPOTS:
        sep = (
            min(math.sqrt((d.x - px) ** 2 + (d.y - py) ** 2) for d in on_court_defs)
            if on_court_defs else 99.0
        )
        if sep > best_sep:
            best_sep = sep
            best = (px, py)
    return best


# ── Resolvers ──────────────────────────────────────────────────────────────────

def resolve_cut(
    cutter,
    cutter_defender,
    all_defenders: list,
) -> CutResult:
    """Resolve an off-ball cut.

    The cutter moves to the most open spot reachable from their current zone.
    P(getting open) is gated by the cutter's drive_effectiveness vs the
    defender's speed over a wider contest radius (CUT_CONTEST_RADIUS).

    The engine moves the cutter regardless of success.  On success the cutter
    has separation; on failure the defender shadows them to the new spot.
    """
    from_x, from_y = cutter.x, cutter.y
    to_x, to_y = _best_cut_destination(cutter, all_defenders)

    base_p = cutter.offense.drive_effectiveness
    if cutter_defender is not None and cutter_defender.is_on_court():
        def_dist  = player_dist(cutter, cutter_defender)
        contest   = contest_factor(def_dist, cutter_defender.defense.speed, CUT_CONTEST_RADIUS)
        prob_open = base_p * contest
        def_desc  = f" past {cutter_defender.name}"
    else:
        prob_open = base_p
        def_desc  = ""

    success = random.random() < prob_open
    dest_zone = get_zone(to_x, to_y)
    zone_label = dest_zone.value if dest_zone else "unknown"

    if success:
        description = (
            f"[cut] {cutter.name} cuts to {zone_label}{def_desc} — open! "
            f"({prob_open:.0%})"
        )
    else:
        description = (
            f"[cut] {cutter.name} cuts to {zone_label}{def_desc} — covered. "
            f"({prob_open:.0%})"
        )
    return CutResult(
        player_name=cutter.name,
        success=success,
        from_x=from_x,
        from_y=from_y,
        to_x=to_x,
        to_y=to_y,
        destination_zone=dest_zone,
        description=description,
    )


def resolve_off_ball_screen(
    screener,
    target,
    target_defender,
) -> ScreenResult:
    """Resolve an off-ball screen for a teammate.

    The screener moves to stand in front of the target's defender.
    P(screen set) = screener's strength × reach_factor (penalised by travel
    distance so a C sprinting from half-court can't magically be in position).

    On success the engine should add the target's defender to
    `state.screened_defenders` so it benefits the target on their next action.
    """
    from_x, from_y = screener.x, screener.y

    if target_defender is None or not target_defender.is_on_court():
        return ScreenResult(
            screener_name=screener.name,
            screener_from_x=from_x,
            screener_from_y=from_y,
            target_name=target.name,
            success=False,
            screen_x=from_x,
            screen_y=from_y,
            description=(
                f"[screen] {screener.name} tries to screen for {target.name} "
                f"— no defender to pin."
            ),
        )

    sx, sy = _screen_position(screener, target, target_defender)
    dist_to_screen = math.sqrt((sx - from_x) ** 2 + (sy - from_y) ** 2)
    reach_factor   = max(0.25, 1.0 - dist_to_screen / 22.0)
    prob           = screener.offense.strength * reach_factor
    success        = random.random() < prob

    if success:
        description = (
            f"[screen] {screener.name} screens for {target.name} "
            f"— {target_defender.name} is pinned! ({prob:.0%})"
        )
    else:
        description = (
            f"[screen] {screener.name} screens for {target.name} "
            f"— {target_defender.name} fights through. ({prob:.0%})"
        )
    return ScreenResult(
        screener_name=screener.name,
        screener_from_x=from_x,
        screener_from_y=from_y,
        target_name=target.name,
        success=success,
        screen_x=sx,
        screen_y=sy,
        description=description,
    )


def resolve_on_ball_screen(
    screener,
    ball_handler,
    bh_defender,
    all_defenders: list,
    tendencies: Optional[OffBallTendencies] = None,
) -> OnBallScreenResult:
    """Resolve a pick-and-roll or pick-and-pop.

    The screener places themselves between the ball handler and their
    defender.  On success the ball handler's defender is screened (engine adds
    them to `state.screened_defenders`).

    The screener then **rolls** toward the basket or **pops** to the most
    open 3-pt spot.  Roll/pop choice is position-gated: bigs tend to roll,
    wings/guards tend to pop.
    """
    from_x, from_y = screener.x, screener.y

    if bh_defender is None or not bh_defender.is_on_court():
        return OnBallScreenResult(
            screener_name=screener.name,
            screener_from_x=from_x,
            screener_from_y=from_y,
            ball_handler_name=ball_handler.name,
            success=False,
            screen_x=from_x,
            screen_y=from_y,
            roll_or_pop="ROLL",
            final_x=from_x,
            final_y=from_y,
            final_zone=screener.zone,
            description=(
                f"[on-ball screen] {screener.name} sets on-ball screen "
                f"for {ball_handler.name} — no defender to free up."
            ),
        )

    sx, sy = _screen_position(screener, ball_handler, bh_defender)
    dist_to_screen = math.sqrt((sx - from_x) ** 2 + (sy - from_y) ** 2)
    reach_factor   = max(0.25, 1.0 - dist_to_screen / 22.0)
    prob           = screener.offense.strength * reach_factor
    success        = random.random() < prob

    # Roll vs pop decision
    t           = tendencies if tendencies is not None else TENDENCIES
    pos_str     = screener.position.value
    p_pop       = t.pop_probabilities.get(pos_str, t.default_pop_prob)
    roll_or_pop = "POP" if random.random() < p_pop else "ROLL"

    if roll_or_pop == "ROLL":
        # Drive toward the basket from the screen position
        angle   = math.atan2(_BASKET_Y - sy, _BASKET_X - sx)
        final_x = min(50.0, max(0.0, _BASKET_X + math.cos(angle + math.pi) * 2.8))
        final_y = min(47.0, max(0.0, _BASKET_Y + math.sin(angle + math.pi) * 2.8))
    else:
        final_x, final_y = _best_pop_spot(all_defenders)

    final_zone = get_zone(final_x, final_y)
    zone_label = final_zone.value if final_zone else "unknown"
    action_desc = (
        f"rolls to the {zone_label}"
        if roll_or_pop == "ROLL"
        else f"pops to the {zone_label}"
    )

    if success:
        description = (
            f"[on-ball screen] {screener.name} screens for {ball_handler.name} "
            f"— {bh_defender.name} pinned! "
            f"{screener.name} {action_desc}. ({prob:.0%})"
        )
    else:
        description = (
            f"[on-ball screen] {screener.name} screens for {ball_handler.name} "
            f"— {bh_defender.name} fights through. "
            f"{screener.name} {action_desc}. ({prob:.0%})"
        )

    return OnBallScreenResult(
        screener_name=screener.name,
        screener_from_x=from_x,
        screener_from_y=from_y,
        ball_handler_name=ball_handler.name,
        success=success,
        screen_x=sx,
        screen_y=sy,
        roll_or_pop=roll_or_pop,
        final_x=final_x,
        final_y=final_y,
        final_zone=final_zone,
        description=description,
    )
