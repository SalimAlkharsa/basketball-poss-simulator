# Basketball Possession Simulator

## Project Overview

A Streamlit-based basketball possession simulator that renders an NBA half-court and simulates possession logic between two teams (Blue Team on offense, Red Team on defense).

## Current Status

- ✅ Court rendering complete (`drawing/court.py`)
- ✅ Player and team data models complete (`models/player.py`, `models/team.py`)
- ✅ CSV data loading complete (`data/loader.py`, `config/players.csv`)
- ✅ Court zone system complete (`models/court.py`)
- ✅ Player zone mapping complete (`Player.zone` property, `place()` bounds validation)
- ✅ Default player placement complete (`Team.place_at_defaults()`)
- ✅ Debug mode: zone overlays + defender contest-radius circles (`draw_half_court(debug=True)`)
- ✅ Action resolution engine complete (`simulation/actions.py`)
- ✅ Possession state machine complete (`simulation/engine.py`)
- ✅ Man-to-man defensive repositioning (speed-gated, per-step)
- ✅ Basketball graphic rendered on court, animated on every step
- ✅ Streamlit UI: Step / New possession controls, action log, player attributes
- 🔄 Next: Advanced offensive actions (cuts, off-ball movement, screens)

## Architecture

### Directory Structure

```
├── app.py                          # Main Streamlit app (session state, animation loop)
├── drawing/
│   └── court.py                    # Court rendering, basketball graphic, debug overlays
├── models/
│   ├── __init__.py
│   ├── court.py                    # CourtZone enum + get_zone(x, y)
│   ├── player.py                   # Player, Position, Role, Attributes, Tendencies
│   └── team.py                     # Team roster management
├── simulation/
│   ├── __init__.py
│   ├── utils.py                    # Geometry helpers + proximity constants
│   ├── actions.py                  # Probabilistic resolvers: shot, drive, pass
│   └── engine.py                   # PossessionState, step_possession, man-to-man defense
├── data/
│   ├── __init__.py
│   └── loader.py                   # CSV loading → load_teams()
└── config/
    └── players.csv                 # 10 players (5 Blue, 5 Red)
```

### Core Classes

**Court Model** (`models/court.py`):

- `CourtZone` enum: RESTRICTED_AREA, PAINT, MID_RANGE, CORNER_3_LEFT, CORNER_3_RIGHT, WING_3_LEFT, WING_3_RIGHT, TOP_OF_KEY_3, BACKCOURT
- `get_zone(x, y) → CourtZone`: maps any court coordinate to its zone
- Constants: BASKET_X/Y, THREE_PT_RADIUS, RESTRICTED_RADIUS, CORNER_3_Y_MAX (~14.13 ft), PAINT bounds, TOP_KEY bounds

**Player Model** (`models/player.py`):

- `Player`: name, team, position, role, offense, defense, tendencies, x, y
  - Methods: `is_on_court()`, `place(x, y)` (validates bounds [0,50]×[0,47]), `clear_location()`
  - Property: `zone` → `CourtZone | None` (None if off-court)
  - Auto-coerces `position` and `role` from strings (CSV compatibility)
- `OffensiveAttributes`: three_pt_shooting, mid_range_shooting, drive_effectiveness, passing, layup, strength
  - All values in [0.0, 1.0], validated in `__post_init__`
- `DefensiveAttributes`: outside_defense, speed, deflections, rim_protection
  - All values in [0.0, 1.0], validated in `__post_init__`
- `Tendencies`: tendency_three, tendency_mid, tendency_drive, tendency_pass, tendency_layup
  - Must sum to 1.0 ± 1e-6, validated in `__post_init__`
  - Method: `as_weights()` → list for `random.choices()`
  - Constant: `ACTION_LABELS = ["3PT", "MID", "DRIVE", "PASS", "LAYUP"]`
- `Position` enum: PG, SG, SF, PF, C
- `Role` enum: OFFENSE, DEFENSE

**Team Model** (`models/team.py`):

- `Team`: name, players (list), MAX_PLAYERS=5
  - Methods: `add_player()`, `set_role()`, `validate()`, `is_full()`, `player_by_name()`, `place_at_defaults()`
  - `place_at_defaults()`: positions all 5 players by role+position (OFFENSE=5-out, DEFENSE=man-to-man overlay)
  - Properties: `offense_players`, `defense_players`
  - Supports iteration and len()

### Simulation Module (`simulation/`)

**`simulation/utils.py`** — Pure geometry, no model imports:

- `player_dist(p1, p2) → float` — Euclidean between two on-court players
- `dist_to_segment(px, py, ax, ay, bx, by) → float` — point-to-segment for pass interception
- `basket_dist(x, y) → float` — distance from coordinate to basket
- `contest_factor(defender_dist, defense_attr, radius) → float` — `[0,1]` success multiplier; linear from `1-defense_attr` at 0 ft to `1.0` at radius
- `distance_decay(x, y) → float` — penalises shots taken beyond the arc; linear to 0 at 10 ft overshoot
- Constants: `CONTEST_RADIUS=2.0`, `INTERCEPT_RADIUS=1.2`, `DRIVE_CLOSE_THRESHOLD=2.0`, `DEFENDER_SNAP_OFFSET=1.0`

**`simulation/actions.py`** — Stateless resolvers returning result dataclasses:

| Resolver                                               | Offensive stat                                       | Defensive counter                                             | Result type   |
| ------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------- | ------------- |
| `resolve_shot(attacker, defender, shot_type, zone)`    | `three_pt_shooting` / `mid_range_shooting` / `layup` | `outside_defense` (perimeter) or `rim_protection` (RA/PAINT)  | `ShotResult`  |
| `resolve_drive(attacker, defender)`                    | `drive_effectiveness`                                | `speed`                                                       | `DriveResult` |
| `resolve_pass(ball_handler, teammates, all_defenders)` | —                                                    | `deflections` (interception roll per defender near pass lane) | `PassResult`  |

Pass target selection: `max(teammates, key=expected_shot_value)` where `expected_shot_value = shooting_attr × contest_factor(nearest_defender)`.

**`simulation/engine.py`** — Possession orchestration:

- `PossessionState`: `ball_handler`, `matchups`, `action_log`, `score`, `is_over`, `outcome`, `last_annotation`
  - `outcome` values: `"MADE_2"` | `"MADE_3"` | `"MISSED"` | `"TURNOVER"` | `"INTERCEPTED"`
  - `last_annotation` keys: `type SHOT → {from_x, from_y, shot_type, made}`, `type PASS/DRIVE → {from_x, from_y, to_x, to_y}`
- `build_matchups(offense_team, defense_team) → Matchups` — pairs by matching `Position`
- `_effective_weights(ball_handler, defender) → list[float]` — raw tendencies for shoot/drive multiplied by expected success probability; PASS weight unmodified; renormalised
- `update_defense(matchups)` — each defender closes toward their man with probability = `defender.defense.speed`; snaps to `CONTEST_RADIUS - DEFENDER_SNAP_OFFSET` ft behind man
- `step_possession(state, blue_team, red_team) → PossessionState` — one full action cycle; hard cap at 12 steps
- `new_possession(blue_team, red_team) → PossessionState` — resets positions, ball to PG

**`drawing/court.py`** additions:

- `_draw_basketball(ax, bx, by)` — orange circle with seam lines, `zorder=15/16`
- `_draw_defender_radii(ax, players)` — dashed red circles at `CONTEST_RADIUS` around each defender (debug only)
- `draw_half_court(debug, players, ball_pos)` — `ball_pos: tuple` replaces old `ball_handler_name`

**Animation in `app.py`:**

- `_animation_frames(annotation) → list[tuple]` — 24 interpolated `(bx, by)` positions
  - PASS / DRIVE: linear interpolation
  - SHOT: linear + sine arc toward basket (arc height 8 ft)
- Frame delay `1/24 s` → 24 FPS, ~1 s total animation
- Court rendered into `st.empty()` placeholder; frames overwrite in-place

### Tendency × Probability Model

For each shoot/drive action the effective weight fed to `random.choices()` is:

$$w_{\text{eff}} = w_{\text{raw}} \times P(\text{success} \mid \text{contest, distance})$$

PASS weight is left unmodified. All weights are renormalised to sum to 1 after multiplication.

### Teams & Players

**Blue Team** (OFFENSE role):

- Player 1 (PG): High shooting, high passing
- Player 2 (SG): High shooting, medium drive
- Player 3 (SF): Balanced offense, high drive
- Player 4 (PF): Facilitator, high passing
- Player 5 (C): Inside player, high layup/strength

**Red Team** (DEFENSE role):

- Player 6 (PG): Balanced, defensive focus
- Player 7 (SG): Perimeter defender
- Player 8 (SF): Versatile, high rim protection
- Player 9 (PF): Paint defender
- Player 10 (C): Rim protector

## Court Coordinate System

From `drawing/court.py`:

- Width: 50 ft (x: 0–50)
- Depth: 47 ft (y: 0–47)
- Origin: bottom-left corner of baseline
- Basket: (25, 5.25)
- Use these coordinates when placing players or rendering possessions

## CSV Format

Column naming convention in `config/players.csv`:

- `off__*`: Offensive attributes (six columns)
- `def__*`: Defensive attributes (four columns)
- `tend__*`: Tendencies (five columns)

Example row:

```
name,team,position,role,off__three_pt_shooting,off__mid_range_shooting,...,tend__three,tend__mid,...
Player 1,Blue Team,PG,OFFENSE,0.92,0.78,...,0.45,0.15,...
```

## Validation Chain

| Layer                               | What                            | When             |
| ----------------------------------- | ------------------------------- | ---------------- |
| `OffensiveAttributes.__post_init__` | All 6 values in [0.0, 1.0]      | Construction     |
| `DefensiveAttributes.__post_init__` | All 4 values in [0.0, 1.0]      | Construction     |
| `Tendencies.__post_init__`          | Non-negative, sum == 1.0 ± 1e-6 | Construction     |
| `Player.__post_init__`              | Coerce/validate enums           | Construction     |
| `Player.place`                      | x∈[0,50], y∈[0,47]              | Placement        |
| `Team.add_player`                   | Max 5 players                   | Add time         |
| `Team.validate`                     | Exactly 5 players               | Called by loader |
| `load_teams`                        | Exactly 2 teams                 | After full parse |

## Usage

### Loading Teams

```python
from data.loader import load_teams

blue_team, red_team = load_teams()
print(blue_team)  # Team(Blue Team, 5 players)
print(red_team)   # Team(Red Team, 5 players)
```

### Accessing Players

```python
for player in blue_team:
    print(player.name, player.position, player.tendencies.as_weights())

# Place player on court
blue_team.players[0].place(x=25.0, y=10.0)
```

### Checking Player Attributes

```python
player = blue_team.players[0]
print(player.offense.three_pt_shooting)        # 0.92
print(player.defense.outside_defense)          # 0.72
print(player.tendencies.ACTION_LABELS)         # ["3PT", "MID", "DRIVE", "PASS", "LAYUP"]
```

## Development Notes

- No external ML/AI dependencies, only stdlib (dataclasses, enum, pathlib, csv)
- Streamlit app loads without errors (once streamlit is installed)
- All validations run at construction time, preventing invalid game state
- Enum strings auto-coerce in Player `__post_init__` for seamless CSV loading
- Tendency sums validated with floating-point tolerance (1e-6)

## Court Zone System

### Zones (`models/court.py`)

| Zone              | Description                     | Boundary                |
| ----------------- | ------------------------------- | ----------------------- |
| `RESTRICTED_AREA` | ≤4 ft from basket               | dist(x,y, 25, 5.25) ≤ 4 |
| `PAINT`           | Free-throw lane, outside RA     | x∈[17,33], y∈[0,19]     |
| `MID_RANGE`       | Inside arc, not paint/RA/corner | catch-all inside arc    |
| `CORNER_3_LEFT`   | Left corner 3                   | x≤3, y≤14.13            |
| `CORNER_3_RIGHT`  | Right corner 3                  | x≥47, y≤14.13           |
| `WING_3_LEFT`     | Left wing above-the-break       | outside arc, x<17       |
| `WING_3_RIGHT`    | Right wing above-the-break      | outside arc, x>33       |
| `TOP_OF_KEY_3`    | Top of key                      | outside arc, x∈[17,33]  |
| `BACKCOURT`       | Beyond half-court / OOB         | y>47 or x<0/x>50        |

### Default Player Positions

**OFFENSE** (5-out): PG(25,23), SG(38,23), SF(44,10), PF(12,23), C(6,10)

**DEFENSE** (man-to-man): PG(25,22), SG(37,22), SF(43,11), PF(13,22), C(7,11)

### Debug Mode

`draw_half_court(debug=True, players=[...])` renders colored zone overlays and player dots.
Toggled via sidebar checkbox in `app.py`. Zone per player shown in sidebar.

## Next Steps

1. Off-ball player movement (cuts, relocations after pass)
2. Screen and roll action type
3. Auto-run mode (simulate full possession without manual stepping)
4. Score tracking across multiple possessions
5. Defensive scheme alternatives (zone, help defence)
