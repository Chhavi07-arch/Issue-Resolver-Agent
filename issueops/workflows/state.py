"""Shared LangGraph workflow state model."""

from typing import Any, Optional
from typing_extensions import NotRequired, TypedDict


class WorkflowState(TypedDict):
    # Issue identity
    issue_id: int
    repo_owner: str
    repo_name: str
    issue_title: str
    issue_body: str

    # Agent outputs (populated as workflow progresses)
    analysis: Optional[dict[str, Any]]
    repo_context: Optional[dict[str, Any]]
    debug_result: Optional[dict[str, Any]]
    fix_result: Optional[dict[str, Any]]

    # Final outputs
    pr_url: Optional[str]
    issue_comment_url: Optional[str]

    # Execution metadata
    errors: list[str]
    current_step: str

    # Control flags
    disable_llm: NotRequired[bool]       # set True to force heuristic fallback (testing)
    dry_run_writes: NotRequired[bool]    # set True to skip GitHub write operations (local testing)
