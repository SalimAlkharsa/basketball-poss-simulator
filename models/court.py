"""
models/court.py
---------------
Court zone definitions and coordinate-to-zone mapping.
"""

import math
from enum import Enum


class CourtZone(Enum):
    RESTRICTED_AREA = "RESTRICTED_AREA"
    PAINT = "PAINT"
    MID_RANGE = "MID_RANGE"
    CORNER_3_LEFT = "CORNER_3_LEFT"
    CORNER_3_RIGHT = "CORNER_3_RIGHT"
    WING_3_LEFT = "WING_3_LEFT"
    WING_3_RIGHT = "WING_3_RIGHT"
    TOP_OF_KEY_3 = "TOP_OF_KEY_3"
    BACKCOURT = "BACKCOURT"


# Court geometry constants (must match drawing/court.py)
BASKET_X = 25.0
BASKET_Y = 5.25
THREE_PT_RADIUS = 23.75
RESTRICTED_RADIUS = 4.0
CORNER_3_X_LEFT = 3.0
CORNER_3_X_RIGHT = 47.0
# y where the corner straight ends and the arc begins
CORNER_3_Y_MAX = math.sqrt(THREE_PT_RADIUS**2 - (BASKET_X - CORNER_3_X_RIGHT)**2) + BASKET_Y
PAINT_X_LEFT = 17.0
PAINT_X_RIGHT = 33.0
PAINT_Y_MAX = 19.0
# x boundaries that separate wing from top-of-key above the break
TOP_KEY_X_LEFT = 17.0
TOP_KEY_X_RIGHT = 33.0
COURT_W = 50.0
COURT_D = 47.0
HALF_COURT_Y = 47.0


def _dist_from_basket(x: float, y: float) -> float:
    return math.sqrt((x - BASKET_X) ** 2 + (y - BASKET_Y) ** 2)


def _outside_three_arc(x: float, y: float) -> bool:
    return _dist_from_basket(x, y) >= THREE_PT_RADIUS


def get_zone(x: float, y: float) -> CourtZone:
    """Return the CourtZone for a given (x, y) court coordinate.

    Coordinate system: origin at bottom-left baseline corner.
    Width 0–50 ft, depth 0–47 ft. Basket at (25, 5.25).
    """
    # Backcourt: beyond half-court line or out of bounds
    if y > HALF_COURT_Y or x < 0 or x > COURT_W or y < 0:
        return CourtZone.BACKCOURT

    dist = _dist_from_basket(x, y)

    # Restricted area: ≤4 ft arc from basket
    if dist <= RESTRICTED_RADIUS:
        return CourtZone.RESTRICTED_AREA

    # Paint: free-throw lane, outside restricted area
    if PAINT_X_LEFT <= x <= PAINT_X_RIGHT and 0 <= y <= PAINT_Y_MAX:
        return CourtZone.PAINT

    # Corner 3s: within 3 ft of sideline, below corner cutoff y
    if y <= CORNER_3_Y_MAX:
        if x <= CORNER_3_X_LEFT:
            return CourtZone.CORNER_3_LEFT
        if x >= CORNER_3_X_RIGHT:
            return CourtZone.CORNER_3_RIGHT

    # Outside the 3-point arc
    if _outside_three_arc(x, y):
        if TOP_KEY_X_LEFT <= x <= TOP_KEY_X_RIGHT:
            return CourtZone.TOP_OF_KEY_3
        if x < TOP_KEY_X_LEFT:
            return CourtZone.WING_3_LEFT
        return CourtZone.WING_3_RIGHT

    # Inside arc, not paint, not restricted — mid-range
    return CourtZone.MID_RANGE
