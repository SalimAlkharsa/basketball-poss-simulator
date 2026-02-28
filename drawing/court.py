"""
drawing/court.py
----------------
Utilities for rendering a basketball court with matplotlib.

NBA half-court dimensions (all values in feet):
  Court    : 50 ft wide × 47 ft deep
  Origin   : bottom-left corner of the baseline
  Basket   : (25, 5.25)
  Paint    : 16 ft wide × 19 ft tall, centered on basket x
  FT circle: radius 6, centered at (25, 19)
  3-pt arc : radius 23.75, corner straights 3 ft from sideline
  RA arc   : radius 4, curves toward baseline
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle

# ── Court constants ────────────────────────────────────────────────────────────

COURT_W      = 50.0    # width (ft)
COURT_D      = 47.0    # depth / half-court length (ft)

BASKET_X     = 25.0    # basket centre x
BASKET_Y     = 5.25    # basket centre y (from baseline)
BASKET_R     = 0.75    # rim radius

BACKBOARD_Y  = 4.0     # backboard y
BACKBOARD_LX = 22.0    # backboard left x
BACKBOARD_RX = 28.0    # backboard right x

PAINT_X      = 17.0    # left edge of paint
PAINT_W      = 16.0    # paint width
PAINT_H      = 19.0    # paint height (= FT line y)

FT_Y         = 19.0    # free-throw line y
FT_RADIUS    = 6.0     # free-throw circle radius

THREE_RADIUS = 23.75   # 3-point arc radius
CORNER_X     = 3.0     # corner-3 distance from sideline

RA_RADIUS    = 4.0     # restricted-area arc radius

# ── Colour palette ─────────────────────────────────────────────────────────────

COURT_COLOR  = "#C68642"   # hardwood tan
PAINT_COLOR  = "#A0522D"   # slightly darker lane
LINE_COLOR   = "white"
RIM_COLOR    = "orange"
BG_COLOR     = "#1E1E1E"   # figure background

LINE_WIDTH   = 1.5


# ── Drawing helpers ─────────────────────────────────────────────────────────────

def _draw_boundary(ax: plt.Axes) -> None:
    ax.add_patch(mpatches.Rectangle(
        (0, 0), COURT_W, COURT_D,
        linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor=COURT_COLOR,
    ))


def _draw_paint(ax: plt.Axes) -> None:
    ax.add_patch(mpatches.Rectangle(
        (PAINT_X, 0), PAINT_W, PAINT_H,
        linewidth=LINE_WIDTH, edgecolor=LINE_COLOR, facecolor=PAINT_COLOR,
    ))


def _draw_ft_line_and_circle(ax: plt.Axes) -> None:
    # Free-throw line
    ax.plot([PAINT_X, PAINT_X + PAINT_W], [FT_Y, FT_Y],
            color=LINE_COLOR, linewidth=LINE_WIDTH)

    # Upper half solid, lower half dashed (per NBA convention)
    diam = FT_RADIUS * 2
    ax.add_patch(Arc((BASKET_X, FT_Y), diam, diam,
                     angle=0, theta1=0, theta2=180,
                     color=LINE_COLOR, linewidth=LINE_WIDTH))
    ax.add_patch(Arc((BASKET_X, FT_Y), diam, diam,
                     angle=0, theta1=180, theta2=360,
                     color=LINE_COLOR, linewidth=LINE_WIDTH, linestyle="dashed"))


def _draw_three_point_line(ax: plt.Axes) -> None:
    # Corner y where the arc meets the straight
    corner_y = BASKET_Y + np.sqrt(THREE_RADIUS**2 - (BASKET_X - CORNER_X)**2)

    # Corner straights
    ax.plot([CORNER_X, CORNER_X], [0, corner_y],
            color=LINE_COLOR, linewidth=LINE_WIDTH)
    ax.plot([COURT_W - CORNER_X, COURT_W - CORNER_X], [0, corner_y],
            color=LINE_COLOR, linewidth=LINE_WIDTH)

    # Arc sweep angles (CCW from +x axis)
    theta_left  = np.degrees(np.arctan2(corner_y - BASKET_Y, CORNER_X - BASKET_X))
    theta_right = np.degrees(np.arctan2(corner_y - BASKET_Y,
                                        (COURT_W - CORNER_X) - BASKET_X))
    ax.add_patch(Arc((BASKET_X, BASKET_Y),
                     THREE_RADIUS * 2, THREE_RADIUS * 2,
                     angle=0, theta1=theta_right, theta2=theta_left,
                     color=LINE_COLOR, linewidth=LINE_WIDTH))


def _draw_restricted_area(ax: plt.Axes) -> None:
    ax.add_patch(Arc((BASKET_X, BASKET_Y), RA_RADIUS * 2, RA_RADIUS * 2,
                     angle=0, theta1=180, theta2=360,
                     color=LINE_COLOR, linewidth=LINE_WIDTH))


def _draw_basket(ax: plt.Axes) -> None:
    ax.add_patch(Circle((BASKET_X, BASKET_Y), BASKET_R,
                        linewidth=LINE_WIDTH, edgecolor=RIM_COLOR, facecolor="none"))
    ax.plot([BACKBOARD_LX, BACKBOARD_RX], [BACKBOARD_Y, BACKBOARD_Y],
            color=LINE_COLOR, linewidth=LINE_WIDTH + 0.5)


def _draw_halfcourt_line(ax: plt.Axes) -> None:
    ax.plot([0, COURT_W], [COURT_D, COURT_D],
            color=LINE_COLOR, linewidth=LINE_WIDTH)


# ── Public API ─────────────────────────────────────────────────────────────────

def draw_half_court() -> plt.Figure:
    """Return a matplotlib Figure of an NBA half-court."""
    fig, ax = plt.subplots(figsize=(5, 4.7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(COURT_COLOR)

    _draw_boundary(ax)
    _draw_paint(ax)
    _draw_ft_line_and_circle(ax)
    _draw_three_point_line(ax)
    _draw_restricted_area(ax)
    _draw_basket(ax)
    _draw_halfcourt_line(ax)

    ax.set_xlim(-1, COURT_W + 1)
    ax.set_ylim(-1, COURT_D + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig
