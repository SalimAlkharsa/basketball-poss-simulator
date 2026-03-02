# Basketball Possession Simulator

A rule-based basketball possession simulator where a language model acts as the head coach — analyzing game logs, reasoning about strategy, and issuing tactical adjustments in real time.

Built for the Mistral Hackathon as a test bench for evaluating LLM strategic reasoning in a structured, stochastic environment.

**[Watch the demo →](https://youtu.be/fDhFEfUN5Vc)**

---

## What It Is

This is intended to act as an open-ended simulation environment with:

- **Deterministic resolution mechanics** grounded in player attributes and geometry
- **Structured randomness** via probabilistic outcomes on every action
- **An LLM coaching layer** that observes new information, reasons, and adapts to conditional results

In an example run, the offense starts scoring around **0.60 points per possession**. After several LLM-driven timeout adjustments, it climbs to **~1.18–1.20 PPP** — a realistic offensive efficiency ceiling — by shifting tendencies, repositioning players, and tuning off-ball activity.

---

## The Environment

### Teams

| Team       | Role    | Players  |
|------------|---------|----------|
| Blue Team  | Offense | PG, SG, SF, PF, C |
| Red Team   | Defense | PG, SG, SF, PF, C (naive man-to-man) |

### Player Attributes

**Offense:** `3PT`, `MID`, `DRV` (drive effectiveness), `PAS`, `LAY`

**Defense:** `PERIM` (perimeter defense), `RIM` (rim protection), `SPD` (on-ball speed), `DEF` (deflections)

### Action Resolution

Every action outcome is computed from first principles:

| Action | Formula |
|--------|---------|
| **Shot** | `shooter_attr × distance_decay × contest_factor` |
| **Drive** | `DRV × (1 - defender_SPD × proximity_factor)` |
| **Pass** | Target = max(teammates, key=expected_shot_value); interception = defender DEF × lane proximity |
| **Cut** | `DRV × (1 - defender_SPD × contest)` |
| **Screen** | `screener_STR × reach_factor`; screened defender is pinned on success |

Contest factors are linear over a 6 ft radius. Defenders use `RIM` near the basket and `PERIM` on the perimeter. Distance decay penalizes shots beyond the arc.

---

## The Intelligence Layer

Every 30 possessions, the simulation calls a **timeout**.

The LLM (`ministral-8b-latest` by default) receives:

- Full possession action logs (narrative format)
- A scouting report of all player attributes (offense and defense)
- Current on-ball tendencies and off-ball multipliers
- Current player zone positions
- The exact resolution mechanics described in natural language

It must respond with:

1. **`## TACTICAL ANALYSIS`** — step-by-step chain-of-thought reasoning
2. **`## DATA_START ... ## DATA_END`** — structured JSON with:
   - Updated per-player on-ball tendencies
   - Off-ball multipliers (`cut_factors`, `screen_factors`, `pop_probabilities`, `base_stay`)
   - Optional player repositioning (zone assignments)

The system normalizes tendency outputs — the model emits raw weights, not distributions.

---

## Architecture

```
├── app.py                          # Streamlit UI + animation loop + coaching controller
├── drawing/
│   └── court.py                    # Court renderer (matplotlib), debug zone overlays
├── models/
│   ├── court.py                    # CourtZone enum + get_zone(x, y)
│   ├── player.py                   # Player, Attributes, Tendencies dataclasses
│   └── team.py                     # Team roster + placement
├── simulation/
│   ├── utils.py                    # Geometry helpers (contest_factor, distance_decay, etc.)
│   ├── actions.py                  # Stateless resolvers: shot, drive, pass
│   ├── off_ball.py                 # Off-ball resolvers: cut, screen (P&R / P&P)
│   └── engine.py                   # PossessionState, step_possession, man-to-man defense
├── coaching/
│   ├── agent.py                    # CoachingAgent: Mistral API call + CoT extraction
│   ├── analytics.py                # Possession recording + narrative/delta builders
│   ├── controller.py               # NormalizationController: applies decisions to model state
│   └── schemas.py                  # CoachingDecision Pydantic schema
├── data/
│   └── loader.py                   # CSV → load_teams()
└── config/
    └── players.csv                 # 10 players with calibrated attributes
```

### Key Design Decisions

**Separation of concerns:** The simulation layer has zero knowledge of the coaching layer. `simulation/` is purely mechanical; `coaching/` purely strategic.

**Chain-of-thought before JSON:** The prompt enforces `## TACTICAL ANALYSIS` before `## DATA_START`, ensuring the model reasons before committing to numbers. The JSON is extracted via regex regardless of whether the model wraps it in code fences.

**Soft normalization:** Tendency values from the LLM are treated as raw weights and normalized to sum to 1.0. This means the model can express relative preferences without the burden of exact arithmetic.

**Animation system:** 24-frame interpolated animations for passes, drives, shots (with a sine arc toward the basket), and off-ball cuts/screens — rendered into a `st.empty()` placeholder at 24 FPS.

---

## Court Coordinate System

- Width: **50 ft** (x: 0–50)
- Depth: **47 ft** (y: 0–47)
- Origin: bottom-left baseline corner
- Basket: **(25, 5.25)**

### Zones

| Zone | Location |
|------|----------|
| `RESTRICTED_AREA` | ≤4 ft from basket |
| `PAINT` | Free-throw lane (outside RA) |
| `MID_RANGE` | Inside arc, not paint |
| `CORNER_3_LEFT/RIGHT` | Corner 3-point area (y ≤ 14.13 ft) |
| `WING_3_LEFT/RIGHT` | Above-the-break wing |
| `TOP_OF_KEY_3` | Top of key arc |
| `BACKCOURT` | Beyond half-court / OOB |

---

## Running It

```bash
pip install -r requirements.txt
# Add MISTRAL_API_KEY to .env
streamlit run app.py
```

Controls:
- **Step** — advance one action in the current possession
- **New Possession** — reset player positions and start fresh
- **Auto-run** — simulate possessions continuously; timeout fires every 30
- **Debug mode** — toggle zone overlays and defender contest-radius circles

---

## Why This Matters

This is effectively **RL-style optimization via strategic language reasoning** rather than gradient updates.

It tests whether a language model can:
- Interpret structured probabilistic logs
- Model action-outcome relationships from first principles
- Adapt strategy to opponent strengths without being told what to change
- Generalize adjustments across stochastic outcomes

The simulation provides ground truth. The model provides judgment. The loop is the experiment.

Potential extensions: LLM vs LLM coaching, multi-possession strategy memory, defensive scheme adaptation, leaderboards across models.

---

## Tech Stack

- **Streamlit** — UI and state management
- **Matplotlib** — court rendering and animation
- **Mistral AI** (`ministral-8b-latest`) — coaching agent
- **Pydantic** — structured output validation
- Pure Python stdlib for simulation logic (no ML dependencies in the sim layer)