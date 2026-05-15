"""Fix PR Agent — real LLM code generation, patch validation, and GitHub write operations.

Write path (Phase 6):
  ready_for_pr=True  + dry_run_writes=False → branch + commit + draft PR + issue comment
  ready_for_pr=False + dry_run_writes=False → escalation comment only

Dry-run path (default in local runner):
  ready_for_pr=True  + dry_run_writes=True  → mock URLs, no GitHub writes
  ready_for_pr=False + dry_run_writes=True  → mock comment URL, no GitHub writes
"""

import logging
from pathlib import Path
from typing import Any

from issueops.config.settings import settings
from issueops.schemas.fix import FileEdit, FixResult
from issueops.tools import github as gh
from issueops.tools.patch_builder import apply_edit_to_content, apply_and_diff
from issueops.tools.patch_validator import diff_size_ok, validate_fix_result
from issueops.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "fix_pr.txt"
_FILE_CONTENT_LIMIT = 3000   # chars per file in the fix prompt


# ---------------------------------------------------------------------------
# Evidence formatter
# ---------------------------------------------------------------------------

def _format_file_contents(repo_context: dict[str, Any]) -> str:
    snippets = repo_context.get("file_snippets") or {}
    if not snippets:
        return "(no file contents available — cannot generate grounded fix)"

    parts: list[str] = []
    for path, content in list(snippets.items())[:2]:
        truncated = content[:_FILE_CONTENT_LIMIT]
        note = "\n...[truncated — fix must target lines shown above]" if len(content) > _FILE_CONTENT_LIMIT else ""
        parts.append(f"### {path}\n```\n{truncated}{note}\n```")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

async def _generate_fix_with_llm(state: WorkflowState) -> FixResult:
    from issueops.tools.llm import generate_structured

    debug_result = state.get("debug_result") or {}
    repo_context = state.get("repo_context") or {}
    file_contents_text = _format_file_contents(repo_context)

    template = _PROMPT_PATH.read_text()
    prompt = (
        template
        .replace("{issue_title}", state["issue_title"])
        .replace("{issue_body}", state["issue_body"] or "(no body)")
        .replace("{root_cause}", debug_result.get("root_cause", "unknown"))
        .replace("{suggested_fix_approach}", debug_result.get("suggested_fix_approach", "unknown"))
        .replace("{file_contents}", file_contents_text)
    )

    return await generate_structured(prompt, FixResult)


# ---------------------------------------------------------------------------
# Validation + diff pipeline
# ---------------------------------------------------------------------------

def _run_validation(
    fix_result: FixResult,
    repo_context: dict[str, Any],
) -> tuple[FixResult, dict[str, str]]:
    """Validate all edits, build diffs, return (updated_fix_result, diffs)."""
    file_snippets = repo_context.get("file_snippets") or {}
    diffs: dict[str, str] = {}
    validation_issues: list[str] = []

    all_valid, notes = validate_fix_result(fix_result, file_snippets)

    if all_valid:
        for edit in fix_result.proposed_edits:
            content = file_snippets.get(edit.path)
            if content is None:
                continue
            diff, applied = apply_and_diff(content, edit)
            if not applied or diff is None:
                all_valid = False
                validation_issues.append(f"{edit.path}: patch application failed unexpectedly")
                continue

            size_ok, size_note = diff_size_ok(diff)
            if not size_ok:
                all_valid = False
                validation_issues.append(f"{edit.path}: {size_note}")
            else:
                diffs[edit.path] = diff
                logger.info("patch_builder: diff built for %s (%s)", edit.path, size_note)

    final_notes = notes
    if validation_issues:
        final_notes = "; ".join([notes] + validation_issues) if notes else "; ".join(validation_issues)

    updated = fix_result.model_copy(update={
        "ready_for_pr": all_valid and bool(diffs),
        "validation_notes": final_notes,
    })

    return updated, diffs


# ---------------------------------------------------------------------------
# Debug-guided file enrichment
# ---------------------------------------------------------------------------

async def _enrich_snippets_with_debug_files(
    state: WorkflowState,
    repo_context: dict[str, Any],
) -> dict[str, Any]:
    """Fetch files named in debug_result.relevant_files that are absent from repo_context.

    Recovers from retrieval misses: the debug agent may correctly identify the root-cause
    file even when repo_context retrieved the wrong set (e.g., auth files instead of domain
    service files). Without this, the fix agent would receive no relevant code to edit.
    """
    debug_result = state.get("debug_result") or {}
    debug_files: list[str] = debug_result.get("relevant_files") or []
    if not debug_files:
        return repo_context

    owner = state["repo_owner"]
    repo = state["repo_name"]
    snippets: dict[str, str] = dict(repo_context.get("file_snippets") or {})
    enriched = False

    for path in debug_files[:3]:
        if path in snippets:
            continue
        try:
            content = await gh.get_file_contents(owner, repo, path)
            if content:
                snippets[path] = content[:_FILE_CONTENT_LIMIT]
                enriched = True
                logger.info("FixPR: enriched context with debug-identified file %s", path)
        except Exception as exc:
            logger.warning("FixPR: could not fetch debug file %s: %s", path, exc)

    if not enriched:
        return repo_context

    return {**repo_context, "file_snippets": snippets}


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fix_fallback(state: WorkflowState, reason: str) -> tuple[FixResult, dict[str, str]]:
    """Return a safe non-ready FixResult when LLM is unavailable."""
    debug_result = state.get("debug_result") or {}
    files = debug_result.get("relevant_files", [])

    result = FixResult(
        patch_plan=f"Automated fix unavailable ({reason}). Manual review required.",
        files_to_modify=files[:2],
        proposed_edits=[],
        confidence=0.0,
        validation_notes=reason,
        ready_for_pr=False,
    )
    return result, {}


# ---------------------------------------------------------------------------
# PR metadata builder
# ---------------------------------------------------------------------------

def _build_pr_metadata(
    state: WorkflowState,
    fix_result: FixResult,
    diffs: dict[str, str],
) -> dict[str, str]:
    issue_id = state["issue_id"]
    debug_result = state.get("debug_result") or {}
    changed = fix_result.files_to_modify or list(diffs.keys())

    diff_section = ""
    if diffs:
        diff_section = "\n\n### Generated Diff\n```diff\n"
        for path, diff_text in diffs.items():
            diff_section += f"# {path}\n{diff_text[:800]}\n"
        diff_section += "```"

    pr_body = (
        f"## IssueOps Autonomous Fix\n\n"
        f"Resolves #{issue_id}\n\n"
        f"### Root Cause\n{debug_result.get('root_cause', 'See analysis')}\n\n"
        f"### Fix Strategy\n{fix_result.patch_plan}\n\n"
        f"### Files Changed\n"
        + "\n".join(f"- `{f}`" for f in changed)
        + diff_section
        + "\n\n"
        "> **Draft PR** — generated autonomously by IssueOps. "
        "Human review required before merge."
    )

    return {
        "branch": f"issueops/fix-issue-{issue_id}",
        "pr_title": f"fix: resolve issue #{issue_id} — {state['issue_title'][:55]}",
        "pr_body": pr_body,
        "commit_message": f"fix: address root cause from issue #{issue_id}\n\n{fix_result.patch_plan[:200]}",
    }


# ---------------------------------------------------------------------------
# Comment body builders
# ---------------------------------------------------------------------------

def _build_success_comment(fix_result: FixResult, pr_url: str) -> str:
    return (
        f"## IssueOps — Draft PR Created\n\n"
        f"A draft PR has been automatically generated for review: {pr_url}\n\n"
        f"**Confidence:** {fix_result.confidence:.0%}\n\n"
        f"**Fix Summary:** {fix_result.patch_plan[:300]}\n\n"
        f"> Human review required before merge. "
        f"This fix was generated autonomously — do not merge without verifying the diff."
    )


def _build_failure_comment(issue_id: int, error: str) -> str:
    return (
        f"## IssueOps — Write Error\n\n"
        f"The fix for issue #{issue_id} was generated and validated, "
        f"but an error occurred while creating the PR.\n\n"
        f"**Error:** {error}\n\n"
        f"Manual intervention required."
    )


def _build_low_confidence_comment(
    issue_id: int,
    confidence: float,
    root_cause: str,
) -> str:
    return (
        f"## IssueOps — Investigation Complete\n\n"
        f"IssueOps investigated issue #{issue_id} but confidence was too low "
        f"({confidence:.0%}) to generate an automated fix.\n\n"
        f"**Partial analysis:**\n{root_cause}\n\n"
        f"A human engineer should review and address this issue."
    )


# ---------------------------------------------------------------------------
# Live write pipeline
# ---------------------------------------------------------------------------

async def _execute_writes(
    state: WorkflowState,
    fix_result: FixResult,
    diffs: dict[str, str],
    pr_meta: dict[str, str],
) -> dict[str, Any]:
    """Execute real GitHub write operations.

    Flow: detect base branch → create branch → apply edits + commit files
          → create draft PR → comment issue.

    Returns a state-update dict. On GitHubWriteError, posts a failure comment
    and returns a write_failed state.
    """
    owner = state["repo_owner"]
    repo = state["repo_name"]
    issue_id = state["issue_id"]
    branch_name = pr_meta["branch"]
    existing_errors = list(state.get("errors") or [])

    try:
        # 1. Detect base branch
        base_branch = await gh.get_default_branch(owner, repo)
        logger.info("FixPR writes: base_branch=%s", base_branch)

        # 2. Create feature branch
        await gh.create_branch(owner, repo, base_branch, branch_name)

        # 3. Apply each validated edit to the full file and commit
        for edit in fix_result.proposed_edits:
            # Re-fetch full content — repo_context copy may be truncated
            full_content = await gh.get_file_contents(owner, repo, edit.path)
            if full_content is None:
                raise gh.GitHubWriteError(
                    f"Could not re-fetch '{edit.path}' from {owner}/{repo} for writing"
                )

            new_content, applied = apply_edit_to_content(full_content, edit)
            if not applied:
                raise gh.GitHubWriteError(
                    f"Edit did not apply to full content of '{edit.path}' — "
                    "snippet may differ from current file"
                )

            await gh.create_or_update_file(
                owner, repo, edit.path, new_content,
                branch_name, pr_meta["commit_message"],
            )

        # 4. Open draft PR
        pr_data = await gh.create_draft_pr(
            owner, repo,
            pr_meta["pr_title"], pr_meta["pr_body"],
            branch_name, base_branch,
        )
        pr_url: str = pr_data.get("html_url", "")
        pr_number: int = pr_data.get("number", 0)

        # 5. Comment on original issue
        comment_data = await gh.comment_on_issue(
            owner, repo, issue_id, _build_success_comment(fix_result, pr_url)
        )
        comment_url: str = comment_data.get("html_url", "")

        logger.info("FixPR writes: complete — PR #%d %s", pr_number, pr_url)

        return {
            "fix_result": {
                **fix_result.model_dump(),
                "diffs": diffs,
                **pr_meta,
                "mock_writes": False,
            },
            "pr_url": pr_url,
            "issue_comment_url": comment_url,
            "current_step": "completed",
        }

    except gh.GitHubWriteError as exc:
        logger.error("FixPR writes: failed — %s", exc)

        # Best-effort failure comment
        failure_comment_url: str | None = None
        try:
            comment_data = await gh.comment_on_issue(
                owner, repo, issue_id, _build_failure_comment(issue_id, str(exc))
            )
            failure_comment_url = comment_data.get("html_url", "")
        except Exception as inner:
            logger.warning("FixPR writes: could not post failure comment: %s", inner)

        return {
            "fix_result": {
                **fix_result.model_dump(),
                "diffs": diffs,
                **pr_meta,
                "write_error": str(exc),
                "mock_writes": False,
            },
            "pr_url": None,
            "issue_comment_url": failure_comment_url,
            "current_step": "write_failed",
            "errors": existing_errors + [str(exc)],
        }


# ---------------------------------------------------------------------------
# Agent entry points
# ---------------------------------------------------------------------------

async def generate_fix_and_pr(state: WorkflowState) -> dict[str, Any]:
    """Generate real code fix, validate, and (optionally) execute GitHub writes.

    dry_run_writes=True  (default in local runner): validate + diff, no GitHub writes.
    dry_run_writes=False (webhook / --live-writes):  full write pipeline.

    Safety gate: GitHub writes only proceed when fix_result.ready_for_pr=True.
    """
    issue_id = state["issue_id"]
    owner = state["repo_owner"]
    repo = state["repo_name"]
    dry_run = state.get("dry_run_writes", True)
    logger.info(
        "FixPR: starting for issue #%s in %s/%s  dry_run=%s",
        issue_id, owner, repo, dry_run,
    )

    use_llm = settings.llm_available and not state.get("disable_llm", False)
    repo_context = state.get("repo_context") or {}
    llm_succeeded = False
    raw_fix: FixResult
    diffs_final: dict[str, str] = {}

    # --- 0. Enrich repo_context with any files the debug agent identified but
    #        repo_context didn't retrieve (e.g. auth files fetched instead of domain files) ---
    if use_llm:
        repo_context = await _enrich_snippets_with_debug_files(state, repo_context)

    # --- 1. Generate fix ---
    if use_llm:
        try:
            raw_fix = await _generate_fix_with_llm(state)
            llm_succeeded = True
            logger.info(
                "FixPR: LLM generated fix — files=%s confidence=%.2f",
                raw_fix.files_to_modify, raw_fix.confidence,
            )
        except Exception as exc:
            logger.warning(
                "FixPR: LLM failed (%s: %s), using fallback",
                type(exc).__name__, exc,
            )
            raw_fix, diffs_final = _fix_fallback(state, f"LLM error: {type(exc).__name__}")
    else:
        reason = "no API key" if not settings.llm_available else "disabled via flag"
        logger.info("FixPR: skipping LLM (%s)", reason)
        raw_fix, diffs_final = _fix_fallback(state, f"LLM unavailable: {reason}")

    # --- 2. Validate + build diffs (only when LLM produced a result) ---
    if llm_succeeded:
        fix_result, diffs_final = _run_validation(raw_fix, repo_context)
    else:
        fix_result = raw_fix

    logger.info(
        "FixPR: validation done — ready_for_pr=%s diffs=%d notes='%s'",
        fix_result.ready_for_pr,
        len(diffs_final),
        fix_result.validation_notes[:80],
    )

    # --- 3. Build PR metadata ---
    pr_meta = _build_pr_metadata(state, fix_result, diffs_final)

    # --- 4. Safety gate: escalate if not ready ---
    if not fix_result.ready_for_pr:
        debug_result = state.get("debug_result") or {}
        low_conf_comment = _build_low_confidence_comment(
            issue_id,
            fix_result.confidence,
            debug_result.get("root_cause", "Analysis unavailable"),
        )

        if not dry_run:
            try:
                comment_data = await gh.comment_on_issue(owner, repo, issue_id, low_conf_comment)
                comment_url: str | None = comment_data.get("html_url", "")
            except gh.GitHubWriteError as exc:
                logger.warning("FixPR: could not post low-confidence comment: %s", exc)
                comment_url = None
        else:
            comment_url = f"https://github.com/{owner}/{repo}/issues/{issue_id}#issuecomment-dry-run"
            logger.info("FixPR [dry run]: would post low-confidence comment to %s", comment_url)

        return {
            "fix_result": {
                **fix_result.model_dump(),
                "diffs": diffs_final,
                **pr_meta,
                "mock_writes": dry_run,
            },
            "issue_comment_url": comment_url,
            "current_step": "fix_proposed_needs_review",
        }

    # --- 5. Execute writes or return dry-run result ---
    if not dry_run:
        return await _execute_writes(state, fix_result, diffs_final, pr_meta)

    # Dry-run: return mock URLs for testing
    pr_number = 1000 + (issue_id if isinstance(issue_id, int) else 0)
    mock_pr_url = f"https://github.com/{owner}/{repo}/pull/{pr_number}"
    mock_comment_url = f"https://github.com/{owner}/{repo}/issues/{issue_id}#issuecomment-dry-run"
    logger.info(
        "FixPR [dry run]: would create branch=%s and PR at %s",
        pr_meta["branch"], mock_pr_url,
    )

    return {
        "fix_result": {
            **fix_result.model_dump(),
            "diffs": diffs_final,
            **pr_meta,
            "mock_writes": True,
        },
        "pr_url": mock_pr_url,
        "issue_comment_url": mock_comment_url,
        "current_step": "completed",
    }


async def escalate_to_comment(state: WorkflowState) -> dict[str, Any]:
    """Flag issue for manual review when debug confidence is too low."""
    issue_id = state["issue_id"]
    owner = state["repo_owner"]
    repo = state["repo_name"]
    debug_result = state.get("debug_result") or {}
    confidence = debug_result.get("confidence", 0.0)
    dry_run = state.get("dry_run_writes", True)

    logger.info(
        "Escalation: low confidence (%.2f) for issue #%s — manual review required",
        confidence, issue_id,
    )

    comment_body = _build_low_confidence_comment(
        issue_id, confidence, debug_result.get("root_cause", "unknown")
    )

    if not dry_run:
        try:
            comment_data = await gh.comment_on_issue(owner, repo, issue_id, comment_body)
            comment_url: str = comment_data.get("html_url", "")
        except gh.GitHubWriteError as exc:
            logger.warning("Escalation: could not post comment: %s", exc)
            comment_url = f"https://github.com/{owner}/{repo}/issues/{issue_id}#issuecomment-failed"
    else:
        comment_url = f"https://github.com/{owner}/{repo}/issues/{issue_id}#issuecomment-dry-run"
        logger.info("Escalation [dry run]: would post escalation comment to %s", comment_url)

    escalation_result = {
        "escalated": True,
        "reason": "confidence below threshold — manual review required",
        "confidence": confidence,
        "root_cause": debug_result.get("root_cause", "unknown"),
        "comment_body": comment_body,
        "mock_writes": dry_run,
    }

    return {
        "fix_result": escalation_result,
        "issue_comment_url": comment_url,
        "current_step": "escalated",
    }
