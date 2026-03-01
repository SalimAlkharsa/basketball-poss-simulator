"""
simulation/actions.py
---------------------
Probabilistic action resolvers: shot, drive, and pass.

Each resolver is a pure function returning a result dataclass.
No state is mutated here — the engine layer owns mutations.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from models.court import CourtZone
from simulation.utils import (
    player_dist,
    dist_to_segment,
    contest_factor,
    distance_decay,
    CONTEST_RADIUS,
    INTERCEPT_RADIUS,
    DRIVE_CLOSE_THRESHOLD,
)

# ── Constants ──────────────────────────────────────────────────────────────────

_BASKET_X      = 25.0
_BASKET_Y      = 5.25
_RA_LAND_RADIUS = 2.4   # ft from basket — solidly inside the 4-ft RA arc

_THREE_PT_ZONES = frozenset({
    CourtZone.CORNER_3_LEFT,
    CourtZone.CORNER_3_RIGHT,
    CourtZone.WING_3_LEFT,
    CourtZone.WING_3_RIGHT,
    CourtZone.TOP_OF_KEY_3,
})

# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ShotResult:
    made: bool
    zone: Optional[CourtZone]
    contested: bool
    prob: float
    description: str
    breakdown: list = field(default_factory=list)


@dataclass
class DriveResult:
    success: bool
    new_x: float
    new_y: float
    prob: float
    description: str
    breakdown: list = field(default_factory=list)


@dataclass
class PassResult:
    success: bool
    recipient: object       # Player | None
    interceptor: object     # Player | None
    prob: float
    description: str


# ── Internal helpers ───────────────────────────────────────────────────────────

def _capped_movement(from_x: float, from_y: float, to_x: float, to_y: float, max_step: float = 40.0) -> tuple[float, float]:
    """Move from one point toward another, capped at max_step feet.

    Args:
        from_x, from_y: Starting position
        to_x, to_y: Target position
        max_step: Maximum distance to move (default 40.0 ft)

    Returns:
        New (x, y) position, clamped to court bounds [0,50] × [0,47]
    """
    dx = to_x - from_x
    dy = to_y - from_y
    dist = math.sqrt(dx * dx + dy * dy)

    if dist <= max_step:
        # Already within max_step, move all the way
        new_x = min(50.0, max(0.0, to_x))
        new_y = min(47.0, max(0.0, to_y))
    else:
        # Move max_step toward target
        ratio = max_step / dist
        new_x = min(50.0, max(0.0, from_x + dx * ratio))
        new_y = min(47.0, max(0.0, from_y + dy * ratio))

    return new_x, new_y


def _base_shot_prob(attacker, shot_type: str) -> float:
    """Return the attacker's base make probability for the given shot type."""
    if shot_type == "3PT":
        return attacker.offense.three_pt_shooting
    if shot_type == "LAYUP":
        return attacker.offense.layup
    return attacker.offense.mid_range_shooting   # "MID"


def _defense_attr_for_zone(defender, zone: Optional[CourtZone]) -> float:
    """Choose rim_protection for close-range zones, outside_defense otherwise."""
    if zone in (CourtZone.RESTRICTED_AREA, CourtZone.PAINT):
        return defender.defense.rim_protection
    return defender.defense.outside_defense


def _expected_shot_value(player, all_defenders: list) -> float:
    """Estimated expected value for a player attempting a shot at their location.

    Used by resolve_pass to rank candidate pass recipients.
    """
    if not player.is_on_court():
        return 0.0
    zone = player.zone
    # Pick shot type by what zone they're in (best single shot they'd take)
    if zone in _THREE_PT_ZONES:
        shot_type = "3PT"
    elif zone == CourtZone.RESTRICTED_AREA:
        shot_type = "LAYUP"
    else:
        shot_type = "MID"

    base = _base_shot_prob(player, shot_type)
    point_value = 3.0 if shot_type == "3PT" else 2.0
    active_defs = [d for d in all_defenders if d.is_on_court()]
    if not active_defs:
        return base * point_value

    nearest = min(active_defs, key=lambda d: player_dist(player, d))
    d_dist = player_dist(player, nearest)
    defense_attr = _defense_attr_for_zone(nearest, zone)
    contest = contest_factor(d_dist, defense_attr, CONTEST_RADIUS)
    return base * contest * point_value


# ── Resolvers ──────────────────────────────────────────────────────────────────

def resolve_shot(attacker, defender, shot_type: str, zone: Optional[CourtZone]) -> ShotResult:
    """Resolve a shot attempt.

    Args:
        attacker:   Offensive player taking the shot.
        defender:   Their matched defender (may be None → uncontested).
        shot_type:  "3PT", "MID", or "LAYUP".
        zone:       Court zone the attacker occupies (for contest type selection).
    """
    base = _base_shot_prob(attacker, shot_type)
    decay = distance_decay(attacker.x, attacker.y)

    if defender is not None and defender.is_on_court():
        d_dist = player_dist(attacker, defender)
        contested = d_dist < CONTEST_RADIUS
        defense_attr = _defense_attr_for_zone(defender, zone)
        contest = contest_factor(d_dist, defense_attr, CONTEST_RADIUS)
        contest_desc = f" (contested by {defender.name})" if contested else ""
    else:
        d_dist = float("inf")
        contested = False
        contest = 1.0
        contest_desc = ""

    prob = base * contest * decay
    made = random.random() < prob

    # ── Build breakdown ───────────────────────────────────────────────────────
    if defender is not None and defender.is_on_court():
        def_type = "rim prot" if zone in (CourtZone.RESTRICTED_AREA, CourtZone.PAINT) else "out def"
        breakdown = [
            f"base ({shot_type}): {base:.0%}",
            f"{defender.name} {d_dist:.1f} ft away → contest: 1 − {defense_attr:.0%} × (1 − {d_dist:.1f}/{CONTEST_RADIUS:.1f}) = {contest:.0%}  [{def_type}]",
        ]
        if decay < 1.0:
            breakdown.append(f"distance decay: {decay:.0%}")
        formula = f"{base:.0%} × {contest:.0%}" + (f" × {decay:.0%}" if decay < 1.0 else "")
        breakdown.append(f"prob = {formula} = {prob:.0%}")
    else:
        breakdown = [f"base ({shot_type}): {base:.0%} (uncontested)"]
        if decay < 1.0:
            breakdown.append(f"distance decay: {decay:.0%}")
        breakdown.append(f"prob = {prob:.0%}")

    verb = (
        "3-pointer" if shot_type == "3PT"
        else "layup" if shot_type == "LAYUP"
        else "mid-range jumper"
    )
    result_desc = "Good!" if made else "No good."
    description = (
        f"{attacker.name} attempts a {verb}{contest_desc} — {result_desc} ({prob:.0%})"
    )
    return ShotResult(made=made, zone=zone, contested=contested, prob=prob, description=description, breakdown=breakdown)


def resolve_drive(
    attacker,
    defender,
    target_x: float,
    target_y: float,
    target_label: str = "basket",
) -> DriveResult:
    """Resolve a drive toward a chosen target position.

    The attacker's matched defender contests the initial penetration.
    On success the attacker is placed at (target_x, target_y).
    On failure the attacker keeps their current position (no turnover — resets).
    """
    base = attacker.offense.drive_effectiveness

    if defender is not None and defender.is_on_court():
        d_dist = player_dist(attacker, defender)
        contest = contest_factor(d_dist, defender.defense.speed, DRIVE_CLOSE_THRESHOLD)
        contest_desc = f" past {defender.name}"
        breakdown = [
            f"drive eff: {base:.0%}",
            f"{defender.name} {d_dist:.1f} ft away → contest: 1 − {defender.defense.speed:.0%} × (1 − {d_dist:.1f}/{DRIVE_CLOSE_THRESHOLD:.1f}) = {contest:.0%}  [speed]",
            f"prob = {base:.0%} × {contest:.0%} = {base * contest:.0%}",
        ]
    else:
        d_dist = float("inf")
        contest = 1.0
        contest_desc = ""
        breakdown = [
            f"drive eff: {base:.0%} (no defender)",
            f"prob = {base:.0%}",
        ]

    prob = base * contest
    success = random.random() < prob

    if success:
        # Cap movement to 40 ft per step toward the target
        new_x, new_y = _capped_movement(attacker.x, attacker.y, target_x, target_y, max_step=40.0)
        description = (
            f"{attacker.name} drives to the {target_label}{contest_desc} "
            f"— gets through! ({prob:.0%})"
        )
    else:
        new_x, new_y = attacker.x, attacker.y
        description = (
            f"{attacker.name} drives toward the {target_label}{contest_desc} "
            f"— stopped, resets. ({prob:.0%})"
        )

    return DriveResult(success=success, new_x=new_x, new_y=new_y, prob=prob, description=description, breakdown=breakdown)


def resolve_pass(ball_handler, teammates: list, all_defenders: list) -> PassResult:
    """Resolve a pass to the teammate with the highest expected shot value.

    An interception occurs if a defender lies within INTERCEPT_RADIUS of the
    pass lane. The first defender that rolls a successful deflection intercepts.
    """
    eligible = [t for t in teammates if t.is_on_court() and t is not ball_handler]
    if not eligible:
        return PassResult(
            success=False,
            recipient=None,
            interceptor=None,
            prob=0.0,
            description=f"{ball_handler.name} has no eligible targets — turnover.",
        )

    target = max(eligible, key=lambda t: _expected_shot_value(t, all_defenders))

    # Interception check: scan every active defender against the pass lane
    active_defs = [d for d in all_defenders if d.is_on_court()]
    interceptor = None
    for defender in active_defs:
        lane_dist = dist_to_segment(
            defender.x, defender.y,
            ball_handler.x, ball_handler.y,
            target.x, target.y,
        )
        if lane_dist < INTERCEPT_RADIUS:
            threat = defender.defense.deflections * (1.0 - lane_dist / INTERCEPT_RADIUS)
            if random.random() < threat:
                interceptor = defender
                description = (
                    f"{ball_handler.name} → {target.name} — "
                    f"intercepted by {interceptor.name}! ({threat:.0%})"
                )
                return PassResult(
                    success=False,
                    recipient=target,
                    interceptor=interceptor,
                    prob=1.0 - threat,
                    description=description,
                )

    description = f"{ball_handler.name} passes to {target.name}."
    return PassResult(success=True, recipient=target, interceptor=None, prob=1.0, description=description)
