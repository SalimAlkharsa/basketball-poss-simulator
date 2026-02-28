"""
data/loader.py
--------------
CSV loading logic for teams and players.
"""

import csv
from pathlib import Path
from typing import Any

from models.player import (
    Player,
    Position,
    Role,
    OffensiveAttributes,
    DefensiveAttributes,
    Tendencies,
)
from models.team import Team


DEFAULT_CSV = Path(__file__).parent.parent / "config" / "players.csv"


def _parse_float(value: str, column: str) -> float:
    """Parse a string to float with column-aware error reporting."""
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Column '{column}': cannot parse '{value}' as float")


def _player_from_row(row: dict[str, str]) -> Player:
    """Construct a Player from a flat CSV row."""
    # Parse offensive attributes
    offense = OffensiveAttributes(
        three_pt_shooting=_parse_float(row["off__three_pt_shooting"], "off__three_pt_shooting"),
        mid_range_shooting=_parse_float(row["off__mid_range_shooting"], "off__mid_range_shooting"),
        drive_effectiveness=_parse_float(row["off__drive_effectiveness"], "off__drive_effectiveness"),
        passing=_parse_float(row["off__passing"], "off__passing"),
        layup=_parse_float(row["off__layup"], "off__layup"),
        strength=_parse_float(row["off__strength"], "off__strength"),
    )

    # Parse defensive attributes
    defense = DefensiveAttributes(
        outside_defense=_parse_float(row["def__outside_defense"], "def__outside_defense"),
        speed=_parse_float(row["def__speed"], "def__speed"),
        deflections=_parse_float(row["def__deflections"], "def__deflections"),
        rim_protection=_parse_float(row["def__rim_protection"], "def__rim_protection"),
    )

    # Parse tendencies
    tendencies = Tendencies(
        tendency_three=_parse_float(row["tend__three"], "tend__three"),
        tendency_mid=_parse_float(row["tend__mid"], "tend__mid"),
        tendency_drive=_parse_float(row["tend__drive"], "tend__drive"),
        tendency_pass=_parse_float(row["tend__pass"], "tend__pass"),
        tendency_layup=_parse_float(row["tend__layup"], "tend__layup"),
    )

    # Create and return player
    player = Player(
        name=row["name"],
        team=row["team"],
        position=Position(row["position"]),
        role=Role(row["role"]),
        offense=offense,
        defense=defense,
        tendencies=tendencies,
    )

    return player


def load_players_flat(csv_path: Path = DEFAULT_CSV) -> list[Player]:
    """Load all players from CSV as a flat list."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    players = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            players.append(_player_from_row(row))

    return players


def load_teams(csv_path: Path = DEFAULT_CSV) -> tuple[Team, Team]:
    """Load teams from CSV.

    Returns tuple of (team_a, team_b) with exactly 5 players each.
    Raises ValueError if CSV doesn't contain exactly 2 teams × 5 players.
    """
    players = load_players_flat(csv_path)

    # Group players by team name
    teams_dict: dict[str, list[Player]] = {}
    for player in players:
        if player.team not in teams_dict:
            teams_dict[player.team] = []
        teams_dict[player.team].append(player)

    # Validate exactly 2 teams
    if len(teams_dict) != 2:
        raise ValueError(
            f"Expected exactly 2 teams, got {len(teams_dict)}: {list(teams_dict.keys())}"
        )

    # Create teams and add players
    team_names = sorted(teams_dict.keys())
    teams = []
    for team_name in team_names:
        team = Team(name=team_name)
        for player in teams_dict[team_name]:
            team.add_player(player)
        team.validate()
        teams.append(team)

    return tuple(teams)  # type: ignore
