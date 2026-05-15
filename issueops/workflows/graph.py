"""LangGraph workflow — fully wired with agent nodes."""

import logging

from langgraph.graph import END, START, StateGraph

from issueops.agents.analyzer import analyze_issue
from issueops.agents.debug import debug_root_cause
from issueops.agents.fix_pr import escalate_to_comment, generate_fix_and_pr
from issueops.agents.planner import plan
from issueops.agents.repo_context import gather_repo_context
from issueops.workflows.state import WorkflowState

logger = logging.getLogger(__name__)


def confidence_router(state: WorkflowState) -> str:
    """Route to fix or escalate based on debug agent confidence."""
    from issueops.config.settings import settings

    debug_result = state.get("debug_result") or {}
    confidence = debug_result.get("confidence", 0.0)
    escalate = debug_result.get("escalate", True)

    decision = "escalate" if (escalate or confidence < settings.confidence_threshold) else "fix"
    logger.info(
        "Router: confidence=%.2f threshold=%.2f decision=%s",
        confidence, settings.confidence_threshold, decision,
    )
    return decision


def build_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("plan", plan)
    graph.add_node("analyze_issue", analyze_issue)
    graph.add_node("gather_repo_context", gather_repo_context)
    graph.add_node("debug_root_cause", debug_root_cause)
    graph.add_node("generate_fix_and_pr", generate_fix_and_pr)
    graph.add_node("escalate_to_comment", escalate_to_comment)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "analyze_issue")
    graph.add_edge("analyze_issue", "gather_repo_context")
    graph.add_edge("gather_repo_context", "debug_root_cause")
    graph.add_conditional_edges(
        "debug_root_cause",
        confidence_router,
        {"fix": "generate_fix_and_pr", "escalate": "escalate_to_comment"},
    )
    graph.add_edge("generate_fix_and_pr", END)
    graph.add_edge("escalate_to_comment", END)

    return graph


workflow = build_graph().compile()
