"""
coaching/controller.py
----------------------
NormalizationController: applies a CoachingDecision to live Player objects
and the OffBallTendencies singleton.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.player import Tendencies
from coaching.schemas import CoachingDecision, SafeTendencies, ALLOWED_ZONES
from simulation.off_ball import TENDENCIES as OFF_BALL_TENDENCIES

if TYPE_CHECKING:
    from models.player import Player

# ---------------------------------------------------------------------------
# Zone → canonical (x, y) lookup table
# RESTRICTED_AREA and BACKCOURT are intentionally excluded.
# ---------------------------------------------------------------------------

ZONE_POSITIONS: dict[str, tuple[float, float]] = {
    "PAINT":           (25.0, 13.0),
    "MID_RANGE":       (25.0, 18.0),
    "CORNER_3_LEFT":   (3.0,  10.0),
    "CORNER_3_RIGHT":  (47.0, 10.0),
    "WING_3_LEFT":     (10.0, 22.0),
    "WING_3_RIGHT":    (40.0, 22.0),
    "TOP_OF_KEY_3":    (25.0, 26.0),
}

_VALID_POSITIONS = {"PG", "SG", "SF", "PF", "C"}


class NormalizationController:
    def __init__(self, players: list["Player"]):
        self.player_map = {p.name: p for p in players}

    def apply(self, decision: CoachingDecision) -> tuple[list[str], dict[str, tuple[float, float]]]:
        """
        Apply a CoachingDecision to live player objects and the off-ball singleton.

        Returns:
            logs: list of log strings describing what was changed
            coached_positions: dict[player_name -> (x, y)] for re-apply after new_possession
        """
        logs: list[str] = [f'Coach: "{decision.timeout_message}"']
        coached_positions: dict[str, tuple[float, float]] = {}

        # --- On-ball tendency updates (L1 normalized per player) ---
        for raw in decision.adjustments:
            player = self.player_map.get(raw.player_name)
            if player is None:
                logs.append(f"[WARN] Unknown player: {raw.player_name} — skipped.")
                continue
            cur = player.tendencies
            safe = SafeTendencies(
                player_name=raw.player_name,
                tendency_three=raw.tendency_three  if raw.tendency_three  is not None else cur.tendency_three,
                tendency_mid=raw.tendency_mid      if raw.tendency_mid    is not None else cur.tendency_mid,
                tendency_drive=raw.tendency_drive  if raw.tendency_drive  is not None else cur.tendency_drive,
                tendency_pass=raw.tendency_pass    if raw.tendency_pass   is not None else cur.tendency_pass,
                tendency_layup=raw.tendency_layup  if raw.tendency_layup  is not None else cur.tendency_layup,
            )
            player.tendencies = Tendencies(
                tendency_three=safe.tendency_three,
                tendency_mid=safe.tendency_mid,
                tendency_drive=safe.tendency_drive,
                tendency_pass=safe.tendency_pass,
                tendency_layup=safe.tendency_layup,
            )
            logs.append(
                f"Updated {raw.player_name}: "
                f"3PT={safe.tendency_three:.2f} MID={safe.tendency_mid:.2f} "
                f"DRV={safe.tendency_drive:.2f} PAS={safe.tendency_pass:.2f} "
                f"LAY={safe.tendency_layup:.2f}"
            )

        # --- Off-ball tendency updates (direct mutation of TENDENCIES singleton) ---
        ob = decision.off_ball

        for pos, val in ob.cut_factors.items():
            if pos in _VALID_POSITIONS:
                OFF_BALL_TENDENCIES.cut_factors[pos] = max(0.0, float(val))
                logs.append(f"Off-ball: cut_factor[{pos}] → {val:.2f}")

        for pos, val in ob.screen_factors.items():
            if pos in _VALID_POSITIONS:
                OFF_BALL_TENDENCIES.screen_factors[pos] = max(0.0, float(val))
                logs.append(f"Off-ball: screen_factor[{pos}] → {val:.2f}")

        for pos, val in ob.pop_probabilities.items():
            if pos in _VALID_POSITIONS:
                OFF_BALL_TENDENCIES.pop_probabilities[pos] = max(0.0, min(1.0, float(val)))
                logs.append(f"Off-ball: pop_prob[{pos}] → {val:.2f}")

        if ob.base_stay is not None:
            OFF_BALL_TENDENCIES.base_stay = max(0.0, float(ob.base_stay))
            logs.append(f"Off-ball: base_stay → {ob.base_stay:.2f}")

        # --- Positioning updates ---
        used_zones: set[str] = set()
        for assignment in decision.positioning:
            player = self.player_map.get(assignment.player_name)
            if player is None:
                logs.append(f"[WARN] Unknown player: {assignment.player_name} — skipped.")
                continue
            zone_name = assignment.zone.upper()
            if zone_name not in ZONE_POSITIONS:
                logs.append(
                    f"[WARN] Invalid/blocked zone '{zone_name}' for {assignment.player_name} — skipped."
                )
                continue
            if zone_name in used_zones:
                logs.append(
                    f"[WARN] Zone {zone_name} already assigned — skipping {assignment.player_name}."
                )
                continue
            x, y = ZONE_POSITIONS[zone_name]
            try:
                player.place(x, y)
                used_zones.add(zone_name)
                coached_positions[assignment.player_name] = (x, y)
                logs.append(
                    f"Repositioned {assignment.player_name} → {zone_name} ({x}, {y})"
                )
            except ValueError as e:
                logs.append(f"[WARN] Could not place {assignment.player_name}: {e}")

        return logs, coached_positions
