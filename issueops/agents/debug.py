"""Debug Agent — LLM-primary root cause analysis with heuristic fallback."""

import json
import logging
from pathlib import Path
from typing import Any

from issueops.config.settings import settings
from issueops.schemas.debug import DebugResult
from issueops.tools.omium_tracing import checkpoint, trace
from issueops.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "debug.txt"

# Heuristic signal sets (fallback only)
_HIGH_CONFIDENCE_SIGNALS = {
    "crash", "error", "exception", "traceback", "attributeerror",
    "typeerror", "valueerror", "keyerror", "none", "null",
}
_LOW_CONFIDENCE_SIGNALS = {"slow", "sometimes", "intermittent", "maybe", "unclear", "investigate"}

_FILE_SNIPPET_LIMIT = 1500   # chars per file in the debug prompt
_MAX_SNIPPET_FILES = 2


# ---------------------------------------------------------------------------
# Evidence formatter
# ---------------------------------------------------------------------------

def _format_evidence(repo_context: dict[str, Any]) -> tuple[str, str, str]:
    """Return (file_snippets_text, recent_commits_text, related_issues_text)."""

    # File snippets — trimmed to keep prompt size manageable
    snippets = repo_context.get("file_snippets") or {}
    snippet_parts: list[str] = []
    for path, content in list(snippets.items())[:_MAX_SNIPPET_FILES]:
        trimmed = content[:_FILE_SNIPPET_LIMIT]
        ellipsis = "...[truncated]" if len(content) > _FILE_SNIPPET_LIMIT else ""
        snippet_parts.append(f"### {path}\n```\n{trimmed}{ellipsis}\n```")
    file_snippets_text = "\n\n".join(snippet_parts) if snippet_parts else "(no file contents available)"

    # Commits
    commits = repo_context.get("recent_commits") or []
    if commits:
        commit_lines = [
            f"- {c.get('sha', '')[:8]} by {c.get('author', '?')} on {c.get('date', '?')[:10]}: {c.get('message', '')}"
            for c in commits[:5]
        ]
        recent_commits_text = "\n".join(commit_lines)
    else:
        recent_commits_text = "(no recent commits available)"

    # Related issues
    issues = repo_context.get("related_issues") or []
    if issues:
        issue_lines = [
            f"- #{i.get('number')} [{i.get('state')}] {i.get('title', '')}"
            for i in issues[:5]
        ]
        related_issues_text = "\n".join(issue_lines)
    else:
        related_issues_text = "(no related issues found)"

    return file_snippets_text, recent_commits_text, related_issues_text


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

async def _debug_with_llm(state: WorkflowState) -> DebugResult:
    from issueops.tools.llm import generate_structured

    analysis = state.get("analysis") or {}
    repo_context = state.get("repo_context") or {}

    file_snippets_text, recent_commits_text, related_issues_text = _format_evidence(repo_context)

    template = _PROMPT_PATH.read_text()
    prompt = (
        template
        .replace("{issue_title}", state["issue_title"])
        .replace("{issue_body}", state["issue_body"] or "(no body)")
        .replace("{analysis}", json.dumps(analysis, indent=2))
        .replace("{file_snippets}", file_snippets_text)
        .replace("{recent_commits}", recent_commits_text)
        .replace("{related_issues}", related_issues_text)
    )

    return await generate_structured(prompt, DebugResult)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _debug_heuristic(state: WorkflowState) -> DebugResult:
    title = state["issue_title"].lower()
    body = (state["issue_body"] or "").lower()
    text = title + " " + body

    analysis = state.get("analysis") or {}
    repo_context = state.get("repo_context") or {}
    relevant_files = (
        analysis.get("suspected_files")
        or repo_context.get("relevant_files")
        or ["src/main.py"]
    )
    stack_traces = analysis.get("stack_traces", [])

    high_hit = any(sig in text for sig in _HIGH_CONFIDENCE_SIGNALS)
    low_hit = any(sig in text for sig in _LOW_CONFIDENCE_SIGNALS)

    if high_hit and not low_hit:
        confidence = 0.85
        root_cause = (
            f"Unhandled exception or crash in the code path described by issue #{state['issue_id']}."
        )
        if stack_traces:
            root_cause += f" Stack evidence: {stack_traces[0][:200]}"
        fix_approach = (
            "Add null/None guard and input validation before the failing operation. "
            "Wrap with appropriate error handling."
        )
    else:
        confidence = 0.45
        root_cause = (
            f"Ambiguous issue: '{state['issue_title']}'. "
            "No concrete error signal — requires further investigation."
        )
        fix_approach = (
            "Reproduce the issue locally, add debug logging around the suspected area, "
            "and gather more context before proposing a fix."
        )

    recent_messages = [
        c["message"] for c in (repo_context.get("recent_commits") or [])[:2]
    ]
    reasoning = (
        f"Heuristic: high_signal={high_hit}, low_signal={low_hit}. "
        f"Files: {relevant_files[:3]}. Recent commits: {recent_messages}."
    )

    return DebugResult(
        root_cause=root_cause,
        confidence=confidence,
        suggested_fix_approach=fix_approach,
        escalate=confidence < settings.confidence_threshold,
        reasoning=reasoning,
        relevant_files=relevant_files[:5],
    )


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

@trace("debug_root_cause")
async def debug_root_cause(state: WorkflowState) -> dict[str, Any]:
    """Root cause analysis.

    Primary: Gemini 2.5 Flash reasons over issue + repo evidence.
    Fallback: keyword heuristics (used when LLM unavailable or disabled).
    """
    logger.info("Debug: investigating issue #%s", state["issue_id"])

    use_llm = settings.llm_available and not state.get("disable_llm", False)
    source = "heuristic"

    if use_llm:
        try:
            result = await _debug_with_llm(state)
            source = "llm"
            logger.info(
                "Debug: LLM success — confidence=%.2f escalate=%s",
                result.confidence, result.escalate,
            )
        except Exception as exc:
            logger.warning("Debug: LLM failed (%s: %s), using heuristic fallback", type(exc).__name__, exc)
            result = _debug_heuristic(state)
    else:
        reason = "no API key" if not settings.llm_available else "disabled via flag"
        logger.info("Debug: skipping LLM (%s), using heuristics", reason)
        result = _debug_heuristic(state)

    # Enforce threshold: force escalate=True when confidence is too low.
    if result.confidence < settings.confidence_threshold:
        result = result.model_copy(update={"escalate": True})
    # Clear over-conservative escalation: if confidence is sufficient AND specific files
    # are identified, trust the reasoning and allow the fix path to proceed.
    elif result.escalate and result.confidence >= settings.confidence_threshold and result.relevant_files:
        logger.info(
            "Debug: clearing over-conservative escalate — confidence=%.2f files=%s",
            result.confidence, result.relevant_files,
        )
        result = result.model_copy(update={"escalate": False})

    logger.info(
        "Debug: done — source=%s confidence=%.2f escalate=%s files=%d",
        source, result.confidence, result.escalate, len(result.relevant_files),
    )

    outcome = {"debug_result": result.model_dump(), "current_step": "debugged"}
    await checkpoint("after_debugging")
    return outcome
