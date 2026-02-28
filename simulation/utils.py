"""
simulation/utils.py
-------------------
Pure geometry helpers and proximity constants for the action engine.
No model imports — safe to import from anywhere.
"""

import math

# ── Proximity constants (feet) ─────────────────────────────────────────────────

CONTEST_RADIUS        = 2.0   # Defender within this distance contests a shot or layup
INTERCEPT_RADIUS      = 1.2   # Defender within this distance of a pass lane threatens interception
DRIVE_CLOSE_THRESHOLD = 2.0   # Defender within this distance contests a drive
DEFENDER_SNAP_OFFSET  = 1.0   # Gap left between defender and offensive player after snapping


# ── Geometry ───────────────────────────────────────────────────────────────────

def player_dist(p1, p2) -> float:
    """Euclidean distance between two on-court players."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def dist_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Minimum distance from point (px, py) to line segment (ax, ay) → (bx, by)."""
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def basket_dist(x: float, y: float) -> float:
    """Distance from (x, y) to the basket at (25, 5.25)."""
    return math.sqrt((x - 25.0) ** 2 + (y - 5.25) ** 2)


# ── Probability modifiers ──────────────────────────────────────────────────────

def contest_factor(
    defender_dist: float,
    defense_attr: float,
    radius: float,
) -> float:
    """Return a [0, 1] success multiplier based on defender proximity.

    At dist >= radius  →  no contest, factor = 1.0
    At dist = 0        →  max contest, factor = 1.0 − defense_attr
    Linear interpolation between.
    """
    if defender_dist >= radius:
        return 1.0
    t = defender_dist / radius          # 0 = fully contested, 1 = uncontested
    return 1.0 - defense_attr * (1.0 - t)


def distance_decay(x: float, y: float) -> float:
    """Return a [0, 1] multiplier that penalises shots taken beyond the 3-point arc.

    Shots on or inside the arc return 1.0.
    Decay is linear past the arc, reaching 0 at 10 ft beyond the arc radius.
    """
    d = basket_dist(x, y)
    overshoot = d - 23.75           # 23.75 ft = THREE_PT_RADIUS from models/court.py
    if overshoot <= 0:
        return 1.0
    return max(0.0, 1.0 - overshoot / 10.0)
