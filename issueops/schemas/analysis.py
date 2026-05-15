"""Pydantic schema for Issue Analyzer agent output."""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    DOCS = "docs"
    QUESTION = "question"
    OTHER = "other"


class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueAnalysis(BaseModel):
    issue_type: IssueType
    severity: IssueSeverity
    keywords: List[str] = Field(default_factory=list)
    suspected_files: List[str] = Field(default_factory=list)
    stack_traces: List[str] = Field(default_factory=list)
    reproduction_hints: List[str] = Field(default_factory=list)
    duplicate_likelihood: float = Field(ge=0.0, le=1.0, default=0.0)
    summary: str
