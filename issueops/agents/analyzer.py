"""Issue Analyzer Agent — LLM-primary with deterministic heuristic fallback."""

import logging
import re
from pathlib import Path
from typing import Any

from issueops.config.settings import settings
from issueops.schemas.analysis import IssueAnalysis, IssueSeverity, IssueType
from issueops.tools.omium_tracing import checkpoint, trace
from issueops.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyzer.txt"

# Heuristic signal sets (fallback only)
_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "when", "from",
    "have", "into", "been", "will", "also", "just", "more", "not",
}
_BUG_SIGNALS = {"bug", "error", "crash", "fail", "broken", "exception", "traceback", "wrong", "unexpected"}
_FEATURE_SIGNALS = {"feature", "add", "support", "request", "enhancement", "improve", "allow", "enable"}
_DOCS_SIGNALS = {"doc", "documentation", "readme", "example", "guide", "typo", "comment"}
_CRITICAL_SIGNALS = {"crash", "critical", "production", "down", "outage", "data loss"}


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

async def _analyze_with_llm(state: WorkflowState) -> IssueAnalysis:
    from issueops.tools.llm import LLMError, generate_structured

    template = _PROMPT_PATH.read_text()
    prompt = template.replace("{issue_title}", state["issue_title"]).replace(
        "{issue_body}", state["issue_body"] or "(no body)"
    )

    return await generate_structured(prompt, IssueAnalysis)


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _analyze_heuristic(state: WorkflowState) -> IssueAnalysis:
    title = state["issue_title"].lower()
    body = (state["issue_body"] or "").lower()
    text = title + " " + body

    if any(w in text for w in _BUG_SIGNALS):
        issue_type = IssueType.BUG
        severity = IssueSeverity.CRITICAL if any(w in text for w in _CRITICAL_SIGNALS) else IssueSeverity.HIGH
    elif any(w in text for w in _FEATURE_SIGNALS):
        issue_type = IssueType.FEATURE
        severity = IssueSeverity.LOW
    elif any(w in text for w in _DOCS_SIGNALS):
        issue_type = IssueType.DOCS
        severity = IssueSeverity.LOW
    else:
        issue_type = IssueType.OTHER
        severity = IssueSeverity.MEDIUM

    raw_words = re.findall(r'\b[a-z][a-z0-9]{3,}\b', text)
    seen: set[str] = set()
    keywords: list[str] = []
    for w in raw_words:
        if w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            keywords.append(w)
        if len(keywords) == 10:
            break

    file_matches = re.findall(r'\b[\w./\-]+\.(?:py|js|ts|go|java|rs|rb|cpp|c|h)\b', body)
    suspected_files = list(dict.fromkeys(file_matches))[:5]

    stack_traces: list[str] = []
    if "traceback" in body or 'file "' in body or "at line" in body:
        lines = (state["issue_body"] or "").splitlines()
        trace_lines = [ln.strip() for ln in lines if re.search(r'File "|^\s+at |error:|exception', ln, re.I)]
        if trace_lines:
            stack_traces = ["\n".join(trace_lines[:8])]

    repro_hints: list[str] = []
    for ln in (state["issue_body"] or "").splitlines():
        stripped = ln.strip()
        if stripped and (stripped.startswith(("1.", "2.", "3.", "-", "*", "Step")) or "steps to reproduce" in stripped.lower()):
            repro_hints.append(stripped)

    return IssueAnalysis(
        issue_type=issue_type,
        severity=severity,
        keywords=keywords,
        suspected_files=suspected_files,
        stack_traces=stack_traces,
        reproduction_hints=repro_hints[:5],
        duplicate_likelihood=0.1,
        summary=f"[{issue_type.value.upper()}] {state['issue_title'][:80]}",
    )


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

@trace("analyze_issue")
async def analyze_issue(state: WorkflowState) -> dict[str, Any]:
    """Classify issue type, severity, and extract structured metadata.

    Primary: Gemini 2.5 Flash via generate_structured.
    Fallback: deterministic regex heuristics (used when LLM is unavailable or disabled).
    """
    logger.info("Analyzer: starting for issue #%s", state["issue_id"])

    use_llm = settings.llm_available and not state.get("disable_llm", False)
    source = "heuristic"

    if use_llm:
        try:
            analysis = await _analyze_with_llm(state)
            source = "llm"
            logger.info(
                "Analyzer: LLM success — type=%s severity=%s keywords=%d files=%d",
                analysis.issue_type.value, analysis.severity.value,
                len(analysis.keywords), len(analysis.suspected_files),
            )
        except Exception as exc:
            logger.warning("Analyzer: LLM failed (%s: %s), using heuristic fallback", type(exc).__name__, exc)
            analysis = _analyze_heuristic(state)
    else:
        reason = "no API key" if not settings.llm_available else "disabled via flag"
        logger.info("Analyzer: skipping LLM (%s), using heuristics", reason)
        analysis = _analyze_heuristic(state)

    logger.info(
        "Analyzer: done — source=%s type=%s severity=%s keywords=%d files=%d traces=%d",
        source, analysis.issue_type.value, analysis.severity.value,
        len(analysis.keywords), len(analysis.suspected_files), len(analysis.stack_traces),
    )

    result = {"analysis": analysis.model_dump(), "current_step": "analyzed"}
    await checkpoint("after_issue_parsing")
    return result
