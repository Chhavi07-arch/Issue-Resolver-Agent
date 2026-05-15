"""Pydantic schema for Debug agent output."""

from typing import List

from pydantic import BaseModel, Field


class DebugResult(BaseModel):
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_fix_approach: str
    escalate: bool
    reasoning: str
    relevant_files: List[str] = Field(default_factory=list)
