"""Pydantic schemas for GitHub API response models."""

from typing import Any, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Issue / PR / Comment  (used for write responses)
# ---------------------------------------------------------------------------

class GitHubIssue(BaseModel):
    id: int
    number: int
    title: str
    body: Optional[str] = None
    state: str
    html_url: str
    user_login: Optional[str] = None


class GitHubPR(BaseModel):
    id: int
    number: int
    title: str
    html_url: str
    state: str
    draft: bool


class GitHubComment(BaseModel):
    id: int
    html_url: str
    body: str


class GitHubBranch(BaseModel):
    ref: str
    sha: str


class GitHubCommit(BaseModel):
    sha: str
    html_url: str


# ---------------------------------------------------------------------------
# Read / search result shapes  (used by repo_context agent)
# ---------------------------------------------------------------------------

class CommitSummary(BaseModel):
    sha: str
    message: str
    author: str
    date: str
    html_url: str


class CodeSearchItem(BaseModel):
    path: str
    html_url: str
    snippet: Optional[str] = None   # text fragment from text-match header


class IssueSearchItem(BaseModel):
    number: int
    title: str
    state: str
    html_url: str
    body_preview: Optional[str] = None
