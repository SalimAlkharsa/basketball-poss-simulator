"""
models/player.py
----------------
Player data model with offensive/defensive attributes, tendencies, and position/role.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Position(str, Enum):
    """Basketball positions."""
    PG = "PG"  # Point Guard
    SG = "SG"  # Shooting Guard
    SF = "SF"  # Small Forward
    PF = "PF"  # Power Forward
    C = "C"    # Center


class Role(str, Enum):
    """On-court role during a possession."""
    OFFENSE = "OFFENSE"
    DEFENSE = "DEFENSE"


@dataclass
class OffensiveAttributes:
    """Offensive player capabilities (all 0.0–1.0)."""
    three_pt_shooting: float
    mid_range_shooting: float
    drive_effectiveness: float
    passing: float
    layup: float
    strength: float

    def __post_init__(self) -> None:
        """Validate that all attributes are in [0.0, 1.0]."""
        attrs = {
            "three_pt_shooting": self.three_pt_shooting,
            "mid_range_shooting": self.mid_range_shooting,
            "drive_effectiveness": self.drive_effectiveness,
            "passing": self.passing,
            "layup": self.layup,
            "strength": self.strength,
        }
        for name, value in attrs.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"OffensiveAttributes.{name} must be in [0.0, 1.0], got {value}"
                )


@dataclass
class DefensiveAttributes:
    """Defensive player capabilities (all 0.0–1.0)."""
    outside_defense: float
    speed: float
    deflections: float
    rim_protection: float

    def __post_init__(self) -> None:
        """Validate that all attributes are in [0.0, 1.0]."""
        attrs = {
            "outside_defense": self.outside_defense,
            "speed": self.speed,
            "deflections": self.deflections,
            "rim_protection": self.rim_protection,
        }
        for name, value in attrs.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"DefensiveAttributes.{name} must be in [0.0, 1.0], got {value}"
                )


@dataclass
class Tendencies:
    """Offensive action tendencies (must sum to 1.0 ± 1e-6)."""
    tendency_three: float
    tendency_mid: float
    tendency_drive: float
    tendency_pass: float
    tendency_layup: float

    ACTION_LABELS = ["3PT", "MID", "DRIVE", "PASS", "LAYUP"]

    def __post_init__(self) -> None:
        """Validate that all tendencies are non-negative and sum to 1.0."""
        attrs = {
            "tendency_three": self.tendency_three,
            "tendency_mid": self.tendency_mid,
            "tendency_drive": self.tendency_drive,
            "tendency_pass": self.tendency_pass,
            "tendency_layup": self.tendency_layup,
        }

        # Check all are non-negative
        for name, value in attrs.items():
            if value < 0.0:
                raise ValueError(
                    f"Tendencies.{name} must be non-negative, got {value}"
                )

        # Check sum is 1.0 ± 1e-6
        total = sum(attrs.values())
        if not (1.0 - 1e-6 <= total <= 1.0 + 1e-6):
            raise ValueError(
                f"Tendencies must sum to 1.0 ± 1e-6, got {total}"
            )

    def as_weights(self) -> list[float]:
        """Return tendencies as a list for use with random.choices()."""
        return [
            self.tendency_three,
            self.tendency_mid,
            self.tendency_drive,
            self.tendency_pass,
            self.tendency_layup,
        ]


@dataclass
class Player:
    """A basketball player with position, role, and attributes."""
    name: str
    team: str
    position: Position
    role: Role
    offense: OffensiveAttributes
    defense: DefensiveAttributes
    tendencies: Tendencies
    x: Optional[float] = None
    y: Optional[float] = None

    def __post_init__(self) -> None:
        """Coerce position and role from strings if needed."""
        if isinstance(self.position, str):
            self.position = Position(self.position)
        if isinstance(self.role, str):
            self.role = Role(self.role)

    def is_on_court(self) -> bool:
        """Check if player has a valid court location."""
        return self.x is not None and self.y is not None

    def place(self, x: float, y: float) -> None:
        """Place player at (x, y) on court. Raises ValueError if out of bounds."""
        if not (0.0 <= x <= 50.0 and 0.0 <= y <= 47.0):
            raise ValueError(
                f"Court coordinates must be within [0,50]×[0,47], got ({x}, {y})"
            )
        self.x = x
        self.y = y

    @property
    def zone(self):
        """Return the CourtZone for the player's current position, or None if off-court."""
        if not self.is_on_court():
            return None
        from models.court import get_zone
        return get_zone(self.x, self.y)

    def clear_location(self) -> None:
        """Remove player's court location."""
        self.x = None
        self.y = None

    def __repr__(self) -> str:
        """String representation."""
        return f"Player({self.name}, {self.position.value}, {self.team})"
