"""Pydantic schemas for the Fix PR agent output."""

from typing import List, Optional

from pydantic import BaseModel, Field


class FileEdit(BaseModel):
    """A single surgical find-and-replace edit targeting one file."""

    path: str = Field(description="Relative file path to modify (must exist in repo context)")
    change_summary: str = Field(description="One sentence describing what this edit does")
    source_lines: str = Field(
        description=(
            "Line range in the numbered file listing where find_snippet was located, "
            "e.g. '23-27'. Used to confirm the snippet was copied from a specific "
            "location, not reconstructed from memory. Format: 'START-END'."
        ),
        default="unknown",
    )
    find_snippet: str = Field(
        description=(
            "Exact verbatim text copied character-for-character from the numbered file "
            "listing above (WITHOUT the 'NNN | ' line-number prefix). "
            "Must be a contiguous block from a single location — never merge lines "
            "from different parts of the file."
        )
    )
    replace_with: str = Field(
        description=(
            "Minimal replacement for find_snippet. Only the lines needed to fix the bug "
            "should differ from find_snippet. Preserve indentation and style exactly."
        )
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


class LineAnchoredEdit(BaseModel):
    """Edit specified by line numbers. The system extracts find_snippet from the file."""

    path: str = Field(description="Relative file path")
    start_line: int = Field(ge=1, description="First line to replace (1-based, from NNN | prefix)")
    end_line: int = Field(ge=1, description="Last line to replace (1-based, inclusive)")
    replace_with: str = Field(description="Corrected code (no NNN | prefixes)")
    change_summary: str = Field(default="", description="One sentence: what this edit does")


class LinePatchResult(BaseModel):
    """Mode B patch output: line-anchored edits extracted from numbered file listing."""

    patch_plan: str
    edits: List[LineAnchoredEdit] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    validation_notes: str = Field(default="")
    ready_for_pr: bool = Field(default=False)
