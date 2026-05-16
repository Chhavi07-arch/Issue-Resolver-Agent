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

    # Structured handoff for Fix PR agent — populated when the LLM path runs
    suspected_symbols: List[str] = Field(
        default_factory=list,
        description=(
            "Function, class, method, or field names from the root cause analysis, "
            "copied verbatim from file contents. Used by Fix PR to locate the exact "
            "code section to patch. Empty if no specific symbols were identifiable."
        ),
    )
    repair_strategy: str = Field(
        default="",
        description=(
            "Concrete, symbol-specific repair instruction: what to change, where, and why. "
            "More precise than suggested_fix_approach — must reference actual identifiers "
            "from the evidence. Empty when root cause is ambiguous."
        ),
    )
    diagnosis_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the DIAGNOSIS (root cause identification), independent of "
            "whether an automated patch can be applied. High diagnosis_confidence with "
            "a failed patch triggers a structured escalation comment rather than silence."
        ),
    )
