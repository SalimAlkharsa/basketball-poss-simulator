"""
coaching/agent.py
-----------------
CoachingAgent: calls the Mistral API with a CoT prompt and extracts a
structured CoachingDecision from the response using regex + JSON parsing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mistralai import Mistral

from coaching.schemas import CoachingDecision
from simulation.off_ball import TENDENCIES

if TYPE_CHECKING:
    from models.player import Player

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Head Coach of the Blue Team basketball team.
Analyze the game state below and provide tactical adjustments.

You MUST respond in two clearly separated sections:

## TACTICAL ANALYSIS
[Your verbal reasoning: which players are struggling, why, and what to change.
Reference the strategy document and specific game trends. Think step-by-step.]

## DATA_START
(no code fences — raw JSON only)
{
  "timeout_message": "...",
  "adjustments": [
    {
      "player_name": "Player 1",
      "tendency_three": 0.35,
      "tendency_mid": 0.10,
      "tendency_drive": 0.25,
      "tendency_pass": 0.20,
      "tendency_layup": 0.10
    }
  ],
  "off_ball": {
    "cut_factors": {},
    "screen_factors": {},
    "pop_probabilities": {},
    "base_stay": null
  },
  "positioning": []
}
## DATA_END

IMPORTANT:
- Values in "adjustments" are ABSOLUTE new on-ball tendencies (not deltas), each >= 0.
  You do NOT need to ensure they sum to 1.0 — the system will normalize.
- Values in "off_ball.cut_factors" and "off_ball.screen_factors" are multipliers (any float >= 0).
  Higher = that position cuts/screens more often.
- Values in "off_ball.pop_probabilities" are probabilities in [0.0, 1.0].
  Higher = screener more likely to pop to 3PT after setting an on-ball screen.
- "off_ball.base_stay" is a float >= 0. Lower = more frequent off-ball actions overall.
- Omit keys in off_ball dicts to leave them unchanged. Set base_stay to null if no change needed.
- Only include players in "adjustments" where a meaningful change is warranted.
- "positioning" (optional): Reassign players to different starting zones for the next possession.
  Valid zones: PAINT, MID_RANGE, CORNER_3_LEFT, CORNER_3_RIGHT, WING_3_LEFT, WING_3_RIGHT, TOP_OF_KEY_3.
  Example: [{{"player_name": "Player 3", "zone": "CORNER_3_RIGHT"}}]
  Only move players when it makes clear tactical sense. Do not place two players in the same zone.
- Write ## TACTICAL ANALYSIS first. Never output JSON before your analysis.
- Keep the TACTICAL ANALYSIS concise (under 400 words). You MUST always reach the ## DATA_START block.
"""


def _build_tendencies_table(players: list["Player"]) -> str:
    """Build a plain-text grid of current on-ball tendencies."""
    header = f"{'Player':<20} | {'3PT':>5} | {'MID':>5} | {'DRV':>5} | {'PAS':>5} | {'LAY':>5}"
    sep = "-" * len(header)
    rows = [header, sep]
    for p in players:
        t = p.tendencies
        rows.append(
            f"{p.name + ' (' + p.position.value + ')':<20} "
            f"| {t.tendency_three:>5.2f} "
            f"| {t.tendency_mid:>5.2f} "
            f"| {t.tendency_drive:>5.2f} "
            f"| {t.tendency_pass:>5.2f} "
            f"| {t.tendency_layup:>5.2f}"
        )
    return "\n".join(rows)


def _build_off_ball_section() -> str:
    """Serialize the live TENDENCIES singleton for the prompt."""
    t = TENDENCIES
    positions = ["PG", "SG", "SF", "PF", "C"]

    def _fmt(d: dict) -> str:
        return "  ".join(f"{p}={d.get(p, '?'):.2f}" for p in positions)

    return (
        f"cut_factors:       {_fmt(t.cut_factors)}\n"
        f"screen_factors:    {_fmt(t.screen_factors)}\n"
        f"pop_probabilities: {_fmt(t.pop_probabilities)}\n"
        f"base_stay: {t.base_stay:.2f}  (lower = more off-ball movement)"
    )


def _build_positioning_section(players: list["Player"]) -> str:
    """Show current player zones derived from player.zone property."""
    lines = []
    for p in players:
        zone = p.zone
        zone_label = zone.value if zone else "Unknown"
        lines.append(f"{p.name} ({p.position.value}) → {zone_label}")
    return "\n".join(lines)


class CoachingAgent:
    def __init__(self, api_key: str, model: str = "ministral-8b-latest"):
        self.client = Mistral(api_key=api_key)
        self.model = model
        strategy_path = Path("strategies/strategy.md")
        self.strategy = strategy_path.read_text() if strategy_path.exists() else ""

    def call(
        self,
        narrative: str,
        players: list["Player"],
    ) -> tuple[str, CoachingDecision]:
        """
        Returns (full_cot_text, parsed_decision).
        Raises ValueError if JSON block is missing or malformed.
        """
        messages = self._build_messages(narrative, players)
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=3000,
        )
        full_text = response.choices[0].message.content
        decision = self._extract_decision(full_text)
        return full_text, decision

    def _build_messages(
        self,
        narrative: str,
        players: list["Player"],
    ) -> list[dict]:
        tendencies_table = _build_tendencies_table(players)
        off_ball_section = _build_off_ball_section()
        positioning_section = _build_positioning_section(players)

        user_content = (
            f"=== GAME NARRATIVE ===\n{narrative}\n\n"
            f"=== STRATEGY ===\n{self.strategy}\n\n"
            f"=== CURRENT ON-BALL TENDENCIES ===\n{tendencies_table}\n\n"
            f"=== CURRENT OFF-BALL TENDENCIES ===\n{off_ball_section}\n\n"
            f"=== CURRENT POSITIONING ===\n{positioning_section}"
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _extract_decision(self, text: str) -> CoachingDecision:
        """Regex-extract JSON block between ## DATA_START and ## DATA_END.

        Tolerates an optional ```json ... ``` code fence inside the block.
        Also handles responses that never emit ## DATA_END (truncated output)
        by falling back to the first complete top-level JSON object after
        ## DATA_START.
        """
        # Grab everything after ## DATA_START (up to ## DATA_END if present)
        start_match = re.search(r"## DATA_START\s*", text)
        if not start_match:
            raise ValueError(
                "LLM response missing DATA_START/DATA_END block.\n"
                f"Full response:\n{text}"
            )
        after_start = text[start_match.end():]
        end_idx = after_start.find("## DATA_END")
        raw_block = after_start[:end_idx].strip() if end_idx != -1 else after_start.strip()

        # Strip optional markdown code fence
        raw_block = re.sub(r"^```(?:json)?\s*", "", raw_block)
        raw_block = re.sub(r"\s*```\s*$", "", raw_block)
        raw_block = raw_block.strip()

        # Extract first complete top-level JSON object
        match = re.search(r"\{.*\}", raw_block, re.DOTALL)
        if not match:
            raise ValueError(
                "LLM response missing DATA_START/DATA_END block.\n"
                f"Full response:\n{text}"
            )
        raw_json = match.group(0)
        data = json.loads(raw_json)
        return CoachingDecision.model_validate(data)
