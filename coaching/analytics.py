"""
coaching/analytics.py
---------------------
PPP (points per possession) tracker and narrative delta builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.engine import PossessionState


@dataclass
class PossessionRecord:
    outcome: str        # "MADE_2" | "MADE_3" | "MISSED" | "TURNOVER" | "INTERCEPTED"
    score: int          # 0, 2, or 3
    steps: int          # len(action_log) at end of possession
    action_log: list    # raw action_log from PossessionState


@dataclass
class PPPWindow:
    total_possessions: int
    total_points: int
    ppp: float
    prev_ppp: float
    ppp_delta: float
    outcome_counts: dict
    recent_outcomes: list


def record_possession(state: "PossessionState") -> PossessionRecord:
    """Create a PossessionRecord from a completed PossessionState."""
    return PossessionRecord(
        outcome=state.outcome or "MISSED",
        score=state.score,
        steps=len(state.action_log),
        action_log=list(state.action_log),
    )


def _compute_ppp(records: list[PossessionRecord]) -> float:
    if not records:
        return 0.0
    total = sum(r.score for r in records)
    return total / len(records)


def _count_turnovers_in_log(records: list[PossessionRecord]) -> int:
    """Count turnover-causing actions (DRIVE, PASS turnovers) in the last 3 possessions."""
    count = 0
    for rec in records[-3:]:
        for entry in rec.action_log:
            text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
            if "turnover" in text.lower() or "intercepted" in text.lower():
                count += 1
    return count


def _count_shot_types(records: list[PossessionRecord]) -> dict:
    """Count shot attempt types across action logs."""
    counts = {"3PT": 0, "MID": 0, "LAYUP": 0, "total": 0}
    for rec in records:
        for entry in rec.action_log:
            text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
            text_lower = text.lower()
            if "3pt" in text_lower or "three" in text_lower:
                counts["3PT"] += 1
                counts["total"] += 1
            elif "mid" in text_lower or "mid-range" in text_lower:
                counts["MID"] += 1
                counts["total"] += 1
            elif "layup" in text_lower or "lay-up" in text_lower:
                counts["LAYUP"] += 1
                counts["total"] += 1
    return counts


def build_narrative_delta(records: list[PossessionRecord]) -> str:
    """
    Compute PPP metrics and build a human-readable narrative paragraph
    summarizing recent game trends.
    """
    if not records:
        return "No possession history yet."

    current_ppp = _compute_ppp(records)
    prev_records = records[-5:-1] if len(records) > 1 else []
    prev_ppp = _compute_ppp(prev_records) if prev_records else current_ppp
    ppp_delta = current_ppp - prev_ppp

    outcome_counts: dict[str, int] = {
        "MADE_2": 0,
        "MADE_3": 0,
        "MISSED": 0,
        "TURNOVER": 0,
        "INTERCEPTED": 0,
    }
    for rec in records:
        if rec.outcome in outcome_counts:
            outcome_counts[rec.outcome] += 1

    recent_outcomes = [rec.outcome for rec in records[-5:]]

    turnovers_last3 = _count_turnovers_in_log(records)
    shot_counts = _count_shot_types(records)

    # PPP direction
    if abs(ppp_delta) < 0.05:
        ppp_dir = "≈ stable"
    elif ppp_delta > 0:
        ppp_dir = f"▲ up {ppp_delta:+.2f} from prior window ({prev_ppp:.2f})"
    else:
        ppp_dir = f"▼ down {ppp_delta:+.2f} from prior window ({prev_ppp:.2f})"

    parts = [
        f"PPP: {current_ppp:.2f} ({ppp_dir}).",
        f"Last 5 outcomes: {', '.join(recent_outcomes)}.",
    ]

    if turnovers_last3 > 0:
        parts.append(
            f"Last 3 possessions had {turnovers_last3} turnover-causing action(s) (drives/passes)."
        )

    total_shots = shot_counts["total"]
    if total_shots > 0:
        pct_3pt  = shot_counts["3PT"]  / total_shots * 100
        pct_mid  = shot_counts["MID"]  / total_shots * 100
        pct_lay  = shot_counts["LAYUP"] / total_shots * 100
        parts.append(
            f"Shot distribution: 3PT {pct_3pt:.0f}%, mid-range {pct_mid:.0f}%, layup {pct_lay:.0f}%."
        )

    made_3s = outcome_counts["MADE_3"]
    if made_3s == 0 and len(records) >= 3:
        parts.append(f"No made 3-pointers in the last {len(records)} possession(s).")

    total_to = outcome_counts["TURNOVER"] + outcome_counts["INTERCEPTED"]
    if total_to > 0:
        parts.append(f"Total turnovers across session: {total_to}.")

    return " ".join(parts)


def build_action_logs_text(records: list[PossessionRecord]) -> str:
    """
    Format the raw action logs from recent possessions to feed into the LLM.
    """
    if not records:
        return "No possession history yet."

    lines = []
    for i, rec in enumerate(records, start=1):
        lines.append(f"Possession {i} (Outcome: {rec.outcome}, Score: {rec.score}):")
        for log in rec.action_log:
            text = log.get("text", "") if isinstance(log, dict) else str(log)
            lines.append(f" - {text}")
        lines.append("")
    
    return "\n".join(lines).strip()
