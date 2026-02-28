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
from matplotlib.patches import Arc, Circle, Wedge

from simulation.utils import CONTEST_RADIUS

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


# ── Debug zone overlay ─────────────────────────────────────────────────────────

def _draw_zone_overlays(ax: plt.Axes) -> None:
    """Draw semi-transparent colored patches for each court zone."""
    # Precompute geometry shared with models/court.py
    corner_y = BASKET_Y + np.sqrt(THREE_RADIUS**2 - (BASKET_X - CORNER_X)**2)
    top_key_x_left = 17.0
    top_key_x_right = 33.0

    # Sweep angles for the 3-point arc
    theta_left  = np.degrees(np.arctan2(corner_y - BASKET_Y, CORNER_X - BASKET_X))
    theta_right = np.degrees(np.arctan2(corner_y - BASKET_Y,
                                        (COURT_W - CORNER_X) - BASKET_X))

    def _filled_arc_patch(cx, cy, r, t1, t2, color, alpha):
        """Return a filled Wedge (pie slice) patch."""
        return Wedge((cx, cy), r, t1, t2, facecolor=color, alpha=alpha, edgecolor="none")

    # Restricted area (filled circle up to RA_RADIUS, lower half only shown)
    ax.add_patch(Wedge((BASKET_X, BASKET_Y), RA_RADIUS, 0, 360,
                       facecolor="#FF4444", alpha=0.4, edgecolor="none"))

    # Paint (rectangle minus restricted area — approximate with rectangle, RA shown on top)
    ax.add_patch(mpatches.Rectangle(
        (PAINT_X, 0), PAINT_W, PAINT_H,
        facecolor="#FF8C00", alpha=0.3, edgecolor="none", zorder=1,
    ))

    # For above-break zones, use a rasterized pixel approach via imshow
    # Build a grid and classify each pixel
    xs = np.linspace(0, COURT_W, 200)
    ys = np.linspace(0, COURT_D, 188)
    XX, YY = np.meshgrid(xs, ys)
    dist = np.sqrt((XX - BASKET_X)**2 + (YY - BASKET_Y)**2)
    outside_arc = dist > THREE_RADIUS
    in_corner_left  = (XX <= CORNER_X) & (YY <= corner_y)
    in_corner_right = (XX >= COURT_W - CORNER_X) & (YY <= corner_y)
    above_break = outside_arc & ~in_corner_left & ~in_corner_right & (YY <= COURT_D)

    # Wing left (above-break, x < top_key_x_left)
    wing_left  = above_break & (XX < top_key_x_left)
    # Wing right (above-break, x > top_key_x_right)
    wing_right = above_break & (XX > top_key_x_right)
    # Top of key (above-break, top_key_x_left ≤ x ≤ top_key_x_right)
    top_key    = above_break & (XX >= top_key_x_left) & (XX <= top_key_x_right)

    # Mid-range: inside arc, not paint, not restricted, not corner, y ≤ COURT_D
    in_ra    = dist <= RA_RADIUS
    in_paint = (XX >= PAINT_X) & (XX <= PAINT_X + PAINT_W) & (YY <= PAINT_H)
    mid_range = (~outside_arc) & (~in_ra) & (~in_paint) & (~in_corner_left) & (~in_corner_right) & (YY <= COURT_D)

    # Build RGBA image
    rgba = np.zeros((*XX.shape, 4))
    def _apply(mask, color_hex, alpha):
        r = int(color_hex[1:3], 16) / 255
        g = int(color_hex[3:5], 16) / 255
        b = int(color_hex[5:7], 16) / 255
        rgba[mask] = [r, g, b, alpha]

    _apply(mid_range,  "#FFD700", 0.3)
    _apply(wing_left,  "#9B59B6", 0.3)
    _apply(wing_right, "#1ABC9C", 0.3)
    _apply(top_key,    "#32CD32", 0.3)

    ax.imshow(rgba, extent=[0, COURT_W, 0, COURT_D], origin="lower",
              aspect="auto", zorder=2, interpolation="nearest")

    # Corner 3 patches drawn after imshow so they render on top
    ax.add_patch(mpatches.Rectangle(
        (0, 0), CORNER_X, corner_y,
        facecolor="#4169E1", alpha=0.3, edgecolor="none", zorder=3,
    ))
    ax.add_patch(mpatches.Rectangle(
        (COURT_W - CORNER_X, 0), CORNER_X, corner_y,
        facecolor="#4169E1", alpha=0.3, edgecolor="none", zorder=3,
    ))


# ── Player rendering ───────────────────────────────────────────────────────────

def _draw_defender_radii(ax: plt.Axes, players: list) -> None:
    """Draw a dashed contest-radius circle around each on-court defender."""
    for player in players:
        if not player.is_on_court():
            continue
        from models.player import Role
        if player.role != Role.DEFENSE:
            continue
        circle = Circle(
            (player.x, player.y),
            CONTEST_RADIUS,
            fill=False,
            linestyle="--",
            edgecolor="#FF4444",
            linewidth=0.8,
            alpha=0.6,
            zorder=6,
        )
        ax.add_patch(circle)


def _draw_players(ax: plt.Axes, players: list, ball_handler_name: str = None) -> None:
    """Draw colored dots + name labels for each player with a court location.

    The ball handler is rendered as a larger star marker in gold to make
    possession obvious at a glance.
    """
    for player in players:
        if not player.is_on_court():
            continue
        color = "#3399FF" if player.team == "Blue Team" else "#FF4444"
        is_ball_handler = (ball_handler_name is not None and player.name == ball_handler_name)
        if is_ball_handler:
            ax.plot(player.x, player.y, "*", color="#FFD700", markersize=14,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=12)
        else:
            ax.plot(player.x, player.y, "o", color=color, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=10)
        ax.text(player.x, player.y + 1.2, player.name.split()[-1],
                color="white", fontsize=5, ha="center", va="bottom",
                fontweight="bold", zorder=13)


# ── Public API ─────────────────────────────────────────────────────────────────

def draw_half_court(
    debug: bool = False,
    players: list = None,
    ball_handler_name: str = None,
) -> plt.Figure:
    """Return a matplotlib Figure of an NBA half-court.

    Args:
        debug: Draw semi-transparent zone overlays and defender contest radii.
        players: Optional list of Player objects to render as dots.
        ball_handler_name: Name of the current ball handler; rendered as a star.
    """
    fig, ax = plt.subplots(figsize=(5, 4.7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(COURT_COLOR)

    _draw_boundary(ax)
    _draw_paint(ax)

    if debug:
        _draw_zone_overlays(ax)

    _draw_ft_line_and_circle(ax)
    _draw_three_point_line(ax)
    _draw_restricted_area(ax)
    _draw_basket(ax)
    _draw_halfcourt_line(ax)

    if players:
        if debug:
            _draw_defender_radii(ax, players)
        _draw_players(ax, players, ball_handler_name=ball_handler_name)

    ax.set_xlim(-1, COURT_W + 1)
    ax.set_ylim(-1, COURT_D + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig
