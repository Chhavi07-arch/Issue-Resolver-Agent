"""Pydantic schemas for the Fix PR agent output."""

from typing import List

from pydantic import BaseModel, Field


class FileEdit(BaseModel):
    """A single surgical find-and-replace edit targeting one file."""

    path: str = Field(description="Relative file path to modify (must exist in repo context)")
    change_summary: str = Field(description="One sentence describing what this edit does")
    find_snippet: str = Field(
        description="Exact verbatim text to find in the file — must be a substring of the actual file content"
    )
    replace_with: str = Field(
        description="Replacement text — the minimal change that fixes the issue"
    )


class FixResult(BaseModel):
    """Structured output from the Fix PR agent."""

    patch_plan: str = Field(
        description="2-3 sentence description of the overall fix strategy"
    )
    files_to_modify: List[str] = Field(
        description="List of file paths that need changes (max 2)",
        max_length=2,
        default_factory=list,
    )
    proposed_edits: List[FileEdit] = Field(
        description="The actual code edits to apply (one per file, max 2)",
        default_factory=list,
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this fix correctly addresses the root cause",
    )
    validation_notes: str = Field(
        description="Notes explaining why the fix is or is not ready for a PR",
        default="",
    )
    ready_for_pr: bool = Field(
        description="True only when the fix is specific, grounded, and safe to apply",
        default=False,
    )
