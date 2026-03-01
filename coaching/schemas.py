"""
coaching/schemas.py
-------------------
Pydantic models for raw LLM output and validated/normalized tendencies.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic.functional_validators import AfterValidator


# ---------------------------------------------------------------------------
# Raw LLM suggestion (pre-normalization)
# ---------------------------------------------------------------------------

class RawTendencySuggestion(BaseModel):
    player_name: str
    tendency_three: float = Field(..., ge=0.0)
    tendency_mid: float = Field(..., ge=0.0)
    tendency_drive: float = Field(..., ge=0.0)
    tendency_pass: float = Field(..., ge=0.0)
    tendency_layup: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Safe tendency after L1 normalization
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


ClampedFloat = Annotated[float, AfterValidator(_clamp)]


class SafeTendencies(BaseModel):
    player_name: str
    tendency_three: ClampedFloat
    tendency_mid: ClampedFloat
    tendency_drive: ClampedFloat
    tendency_pass: ClampedFloat
    tendency_layup: ClampedFloat

    @model_validator(mode="after")
    def l1_normalize(self) -> "SafeTendencies":
        keys = [
            "tendency_three",
            "tendency_mid",
            "tendency_drive",
            "tendency_pass",
            "tendency_layup",
        ]
        total = sum(getattr(self, k) for k in keys)
        if total <= 0:
            # Uniform fallback
            for k in keys:
                setattr(self, k, 0.2)
        else:
            for k in keys:
                setattr(self, k, getattr(self, k) / total)
        return self


# ---------------------------------------------------------------------------
# Off-ball tendency suggestion
# ---------------------------------------------------------------------------

class RawOffBallSuggestion(BaseModel):
    """
    Suggested updates to the OffBallTendencies singleton.
    All values are optional — omit a position to leave it unchanged.
    cut_factors and screen_factors are per-position multipliers.
    pop_probabilities are [0,1] per position.
    base_stay is a global float; lower = more active off-ball movement.
    """
    cut_factors: dict = Field(default_factory=dict)
    screen_factors: dict = Field(default_factory=dict)
    pop_probabilities: dict = Field(default_factory=dict)
    base_stay: Optional[float] = None


# ---------------------------------------------------------------------------
# Player zone assignment
# ---------------------------------------------------------------------------

ALLOWED_ZONES = {
    "PAINT",
    "MID_RANGE",
    "CORNER_3_LEFT",
    "CORNER_3_RIGHT",
    "WING_3_LEFT",
    "WING_3_RIGHT",
    "TOP_OF_KEY_3",
}


class PlayerZoneAssignment(BaseModel):
    player_name: str
    zone: str  # Must be one of ALLOWED_ZONES; validated in controller for graceful skip


# ---------------------------------------------------------------------------
# Full coaching output
# ---------------------------------------------------------------------------

class CoachingDecision(BaseModel):
    adjustments: list[RawTendencySuggestion]
    off_ball: RawOffBallSuggestion = Field(default_factory=RawOffBallSuggestion)
    positioning: list[PlayerZoneAssignment] = Field(default_factory=list)
    timeout_message: str
