"""
models/team.py
--------------
Team data model with player roster management.
"""

from dataclasses import dataclass, field
from typing import Iterator

from models.player import Player, Position, Role


@dataclass
class Team:
    """A basketball team with player roster."""
    name: str
    players: list[Player] = field(default_factory=list)

    MAX_PLAYERS = 5

    def add_player(self, player: Player) -> None:
        """Add a player to the team roster.

        Raises ValueError if team is already at MAX_PLAYERS.
        """
        if len(self.players) >= self.MAX_PLAYERS:
            raise ValueError(
                f"Team {self.name} is full (max {self.MAX_PLAYERS} players)"
            )
        self.players.append(player)

    def set_role(self, role: Role) -> None:
        """Assign a role to all players on the team."""
        for player in self.players:
            player.role = role

    @property
    def offense_players(self) -> list[Player]:
        """Return all players on offense."""
        return [p for p in self.players if p.role == Role.OFFENSE]

    @property
    def defense_players(self) -> list[Player]:
        """Return all players on defense."""
        return [p for p in self.players if p.role == Role.DEFENSE]

    def is_full(self) -> bool:
        """Check if team has MAX_PLAYERS players."""
        return len(self.players) == self.MAX_PLAYERS

    def validate(self) -> None:
        """Assert that team has exactly MAX_PLAYERS players.

        Raises AssertionError if not valid.
        """
        assert len(self.players) == self.MAX_PLAYERS, (
            f"Team {self.name} has {len(self.players)} players, "
            f"expected {self.MAX_PLAYERS}"
        )

    def player_by_name(self, name: str) -> Player:
        """Find a player by name.

        Raises ValueError if player not found.
        """
        for player in self.players:
            if player.name == name:
                return player
        raise ValueError(f"Player {name} not found on team {self.name}")

    def place_at_defaults(self) -> None:
        """Place all 5 players at default starting positions based on role + position.

        OFFENSE uses 5-out wing spacing; DEFENSE mirrors man-to-man.
        """
        offense_defaults = {
            Position.PG: (25.0, 23.0),
            Position.SG: (38.0, 23.0),
            Position.SF: (44.0, 10.0),
            Position.PF: (12.0, 23.0),
            Position.C:  (6.0,  10.0),
        }
        defense_defaults = {
            Position.PG: (25.0, 22.0),
            Position.SG: (37.0, 22.0),
            Position.SF: (43.0, 11.0),
            Position.PF: (13.0, 22.0),
            Position.C:  (7.0,  11.0),
        }
        for player in self.players:
            if player.role == Role.OFFENSE:
                coords = offense_defaults.get(player.position)
            else:
                coords = defense_defaults.get(player.position)
            if coords:
                player.place(*coords)

    def __iter__(self) -> Iterator[Player]:
        """Iterate over players."""
        return iter(self.players)

    def __len__(self) -> int:
        """Return number of players."""
        return len(self.players)

    def __repr__(self) -> str:
        """String representation."""
        return f"Team({self.name}, {len(self.players)} players)"
