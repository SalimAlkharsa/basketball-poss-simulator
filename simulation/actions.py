"""
simulation/actions.py
---------------------
Probabilistic action resolvers: shot, drive, and pass.

Each resolver is a pure function returning a result dataclass.
No state is mutated here — the engine layer owns mutations.
"""

import random
from dataclasses import dataclass
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

_BASKET_X = 25.0
_BASKET_Y = 5.25

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


@dataclass
class DriveResult:
    success: bool
    new_x: float
    new_y: float
    prob: float
    description: str


@dataclass
class PassResult:
    success: bool
    recipient: object       # Player | None
    interceptor: object     # Player | None
    prob: float
    description: str


# ── Internal helpers ───────────────────────────────────────────────────────────

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
    active_defs = [d for d in all_defenders if d.is_on_court()]
    if not active_defs:
        return base

    nearest = min(active_defs, key=lambda d: player_dist(player, d))
    d_dist = player_dist(player, nearest)
    defense_attr = _defense_attr_for_zone(nearest, zone)
    contest = contest_factor(d_dist, defense_attr, CONTEST_RADIUS)
    return base * contest


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

    verb = (
        "3-pointer" if shot_type == "3PT"
        else "layup" if shot_type == "LAYUP"
        else "mid-range jumper"
    )
    result_desc = "Good!" if made else "No good."
    description = (
        f"{attacker.name} attempts a {verb}{contest_desc} — {result_desc} ({prob:.0%})"
    )
    return ShotResult(made=made, zone=zone, contested=contested, prob=prob, description=description)


def resolve_drive(attacker, defender) -> DriveResult:
    """Resolve a drive to the basket.

    Success: attacker moves halfway toward the basket.
    Failure: turnover, attacker position unchanged.
    """
    base = attacker.offense.drive_effectiveness

    if defender is not None and defender.is_on_court():
        d_dist = player_dist(attacker, defender)
        contest = contest_factor(d_dist, defender.defense.speed, DRIVE_CLOSE_THRESHOLD)
        contest_desc = f" — challenged by {defender.name}"
    else:
        contest = 1.0
        contest_desc = ""

    prob = base * contest
    success = random.random() < prob

    if success:
        new_x = min(50.0, max(0.0, (attacker.x + _BASKET_X) / 2))
        new_y = min(47.0, max(0.0, (attacker.y + _BASKET_Y) / 2))
        description = (
            f"{attacker.name} drives{contest_desc} — gets through! ({prob:.0%})"
        )
    else:
        new_x, new_y = attacker.x, attacker.y
        description = (
            f"{attacker.name} drives{contest_desc} — stopped, resets. ({prob:.0%})"
        )

    return DriveResult(success=success, new_x=new_x, new_y=new_y, prob=prob, description=description)


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
