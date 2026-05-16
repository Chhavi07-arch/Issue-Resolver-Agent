"""Planner Agent — validates state and logs workflow start."""

import logging
from typing import Any

from issueops.tools.omium_tracing import trace
from issueops.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

_STR_FIELDS = ("repo_owner", "repo_name", "issue_title")


@trace("plan")
async def plan(state: WorkflowState) -> dict[str, Any]:
    """Validate required fields are present and log workflow start.

    Does NOT perform issue analysis itself — only coordinates entry.
    """
    # issue_id=0 is valid for synthetic tests; check strings for empty explicitly
    missing = [f for f in _STR_FIELDS if not state.get(f)]
    if state.get("issue_id") is None:
        missing.append("issue_id")
    if missing:
        logger.error("Planner: missing required fields: %s", missing)
        return {
            "current_step": "aborted",
            "errors": state.get("errors", []) + [f"Missing fields: {missing}"],
        }

    logger.info(
        "Planner: workflow started — issue #%s '%s' in %s/%s",
        state["issue_id"],
        state["issue_title"],
        state["repo_owner"],
        state["repo_name"],
    )
    return {"current_step": "planned"}
