"""
schemas.py
==========
Pydantic validation schemas for all API request bodies.
Ensures data integrity before reaching the ML pipeline.
"""

from pydantic import BaseModel, field_validator, model_validator
from typing import Literal


# Import team IDs at module level for validation
try:
    from ml_pipeline import TEAM_NAMES
    _VALID_TEAM_IDS = set(TEAM_NAMES.keys())
except ImportError:
    _VALID_TEAM_IDS = set()


class PredictionRequest(BaseModel):
    """Validates POST /api/predict request body."""

    team1: int
    team2: int
    home_team: Literal["team1", "team2", "neutral"] = "team1"

    @field_validator("team1", "team2")
    @classmethod
    def validate_team_id(cls, v):
        if _VALID_TEAM_IDS and v not in _VALID_TEAM_IDS:
            raise ValueError(
                f"Team ID {v} inválido. "
                f"IDs válidos: {sorted(_VALID_TEAM_IDS)[:5]}... (30 total)"
            )
        return v

    @model_validator(mode="after")
    def teams_must_be_different(self):
        if self.team1 == self.team2:
            raise ValueError("team1 y team2 deben ser equipos diferentes")
        return self


class HeadToHeadRequest(BaseModel):
    """Validates POST /api/head-to-head request body."""

    team1: int
    team2: int

    @field_validator("team1", "team2")
    @classmethod
    def validate_team_id(cls, v):
        if _VALID_TEAM_IDS and v not in _VALID_TEAM_IDS:
            raise ValueError(f"Team ID {v} inválido")
        return v

    @model_validator(mode="after")
    def teams_must_be_different(self):
        if self.team1 == self.team2:
            raise ValueError("Equipos deben ser diferentes")
        return self


class AdminUpdateRequest(BaseModel):
    """Validates admin update request."""

    secret: str
