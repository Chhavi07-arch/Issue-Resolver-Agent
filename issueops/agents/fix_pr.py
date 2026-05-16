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
from issueops.schemas.fix import FileEdit, FixResult, LinePatchResult, LineAnchoredEdit
from issueops.tools import github as gh
from issueops.tools.bug_detectors import detect_in_file
from issueops.tools.patch_builder import apply_edit_to_content, apply_and_diff, line_anchored_to_file_edit
from issueops.tools.patch_validator import diff_size_ok, validate_fix_result
from issueops.tools.patch_syntax import check_brace_balance, recover_from_prepend, validate_replacement_syntax
from issueops.tools.patch_verifier import verify_patch
from issueops.tools.omium_tracing import checkpoint, trace
from issueops.tools.symbol_search import extract_context_around_line, find_definition_line
from issueops.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "fix_pr.txt"
_FILE_CONTENT_LIMIT = 3000   # chars per file in the fix prompt


# ---------------------------------------------------------------------------
# Evidence formatter
# ---------------------------------------------------------------------------

def _format_file_contents(
    repo_context: dict[str, Any],
    suspected_symbols: list[str] | None = None,
) -> str:
    """Format file contents for the fix prompt with line-number anchors.

    When suspected_symbols are provided, attempts to locate the most relevant
    function/method and shows that section (with accurate file-relative line
    numbers) rather than always truncating from line 1.  This gives the LLM
    the exact lines to edit even when the relevant code is deep in the file.
    """
    snippets = repo_context.get("file_snippets") or {}
    if not snippets:
        return "(no file contents available — cannot generate grounded fix)"

    parts: list[str] = []
    for path, content in list(snippets.items())[:2]:
        total_lines = len(content.splitlines())
        shown_section: str | None = None
        sec_start = 1

        # Symbol-targeted extraction: find the relevant function, not just lines 1-N
        if suspected_symbols:
            for sym in suspected_symbols[:6]:
                def_line = find_definition_line(sym, content)
                if def_line is None:
                    continue
                section_text, sec_start, sec_end = extract_context_around_line(
                    content, def_line, max_lines=40
                )
                if section_text.strip():
                    header = (
                        f"[lines {sec_start}–{sec_end} of {total_lines} "
                        f"— context for symbol '{sym}']"
                    )
                    numbered_lines = [
                        f"{sec_start + i:4d} | {line}"
                        for i, line in enumerate(section_text.splitlines())
                    ]
                    numbered = "\n".join(numbered_lines)
                    parts.append(f"### {path}\n{header}\n```\n{numbered}\n```")
                    shown_section = sym
                    break

        if shown_section is not None:
            # Telemetry: symbol resolved to a definition line
            for sym in suspected_symbols[:6]:
                def_line = find_definition_line(sym, content)
                if def_line is not None:
                    _, s_start, s_end = extract_context_around_line(content, def_line, max_lines=40)
                    logger.info(
                        "[SYMBOL_RESOLVE] symbol=%s line=%d file=%s context_lines=%d-%d",
                        sym, def_line, path, s_start, s_end,
                    )
                    break
            continue

        # Default: first _FILE_CONTENT_LIMIT chars, with accurate line numbers
        truncated = content[:_FILE_CONTENT_LIMIT]
        is_truncated = len(content) > _FILE_CONTENT_LIMIT
        numbered_lines = [
            f"{i:4d} | {line}"
            for i, line in enumerate(truncated.splitlines(), start=1)
        ]
        numbered = "\n".join(numbered_lines)
        footer = (
            f"\n... [file truncated — shown lines 1-{len(numbered_lines)} of {total_lines}; "
            "fix must target lines shown above]"
        ) if is_truncated else ""
        parts.append(f"### {path}\n```\n{numbered}{footer}\n```")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

async def _generate_fix_with_llm(
    state: WorkflowState,
    repo_context: dict[str, Any] | None = None,
    prior_error: str | None = None,
) -> FixResult:
    from issueops.tools.llm import generate_structured

    debug_result = state.get("debug_result") or {}
    if repo_context is None:
        repo_context = state.get("repo_context") or {}

    suspected_symbols = debug_result.get("suspected_symbols") or []
    file_contents_text = _format_file_contents(repo_context, suspected_symbols)

    # Prefer the more specific repair_strategy over suggested_fix_approach
    fix_approach = (
        debug_result.get("repair_strategy")
        or debug_result.get("suggested_fix_approach")
        or "unknown"
    )

    template = _PROMPT_PATH.read_text()
    prompt = (
        template
        .replace("{issue_title}", state["issue_title"])
        .replace("{issue_body}", state["issue_body"] or "(no body)")
        .replace("{root_cause}", debug_result.get("root_cause", "unknown"))
        .replace("{suggested_fix_approach}", fix_approach)
        .replace("{file_contents}", file_contents_text)
    )

    # Retry feedback: when Mode A is being re-run after a validation failure,
    # surface the specific rejection reason and reiterate the format rule the
    # model is most likely to have violated. This is the Aider-style feedback
    # loop — one targeted retry recovers a high fraction of format failures.
    if prior_error:
        prompt += (
            "\n\n---\n\n"
            "## RETRY — your previous response was rejected\n\n"
            f"**Rejection reason:** {prior_error}\n\n"
            "Re-read the **CRITICAL — common mistake to avoid** section above. "
            "The single most common cause of rejection is this pattern:\n\n"
            "- `find_snippet` contains the buggy lines.\n"
            "- `replace_with` contains the SAME buggy lines AND the fix appended after.\n\n"
            "That doubles the buggy code instead of replacing it. `replace_with` must be a "
            "**complete drop-in replacement** for `find_snippet` — the lines as they should "
            "look AFTER the fix is applied, with the bug actually removed or corrected in place.\n\n"
            "If the fix is to insert a guard *before* the buggy line, the guard must appear "
            "*before* the buggy line in `replace_with`, with the buggy line still present "
            "(unchanged) after it. Do NOT include the original buggy line plus a separate "
            "fixed version.\n\n"
            "Produce a corrected edit now. If you cannot, return an empty `proposed_edits` list."
        )

    return await generate_structured(prompt, FixResult)


# ---------------------------------------------------------------------------
# Mode B: line-anchored patching
# ---------------------------------------------------------------------------

_ANCHORED_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "fix_pr_anchored.txt"
_SURGICAL_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "fix_pr_surgical.txt"


async def _generate_fix_mode_b(
    state: WorkflowState,
    repo_context: dict[str, Any],
) -> tuple[FixResult, dict[str, str]]:
    """Mode B: line-anchored patching. LLM cites line numbers; system extracts find_snippet."""
    from issueops.tools.llm import generate_structured

    debug_result = state.get("debug_result") or {}
    suspected_symbols = debug_result.get("suspected_symbols") or []
    file_contents_text = _format_file_contents(repo_context, suspected_symbols)
    fix_approach = (
        debug_result.get("repair_strategy")
        or debug_result.get("suggested_fix_approach")
        or "unknown"
    )

    template = _ANCHORED_PROMPT_PATH.read_text()
    prompt = (
        template
        .replace("{issue_title}", state["issue_title"])
        .replace("{issue_body}", state["issue_body"] or "(no body)")
        .replace("{root_cause}", debug_result.get("root_cause", "unknown"))
        .replace("{suggested_fix_approach}", fix_approach)
        .replace("{file_contents}", file_contents_text)
    )

    line_result: LinePatchResult = await generate_structured(prompt, LinePatchResult)
    logger.info(
        "[PATCH_MODE] mode=B edits=%d confidence=%.2f",
        len(line_result.edits), line_result.confidence,
    )

    # Convert line-anchored edits to FileEdits using actual file content
    file_snippets = repo_context.get("file_snippets") or {}
    file_edits = []
    for anchored in line_result.edits:
        content = file_snippets.get(anchored.path)
        if content is None:
            logger.warning(
                "[PATCH_MODE] mode=B file=%s: content not in snippets", anchored.path
            )
            continue
        fe = line_anchored_to_file_edit(anchored, content)
        if fe is None:
            logger.warning(
                "[PATCH_MODE] mode=B file=%s: line range %d-%d out of bounds",
                anchored.path, anchored.start_line, anchored.end_line,
            )
            continue
        file_edits.append(fe)

    if not file_edits:
        logger.warning("[PATCH_EMPTY_RESPONSE] mode=B: no valid line-anchored edits after conversion")
        return _fix_fallback(state, "Mode B: no valid line ranges")

    # Build a FixResult from the line-anchored edits
    fix = FixResult(
        patch_plan=line_result.patch_plan,
        files_to_modify=[e.path for e in file_edits],
        proposed_edits=file_edits,
        confidence=line_result.confidence,
        validation_notes=line_result.validation_notes,
        ready_for_pr=False,
    )
    return _run_validation(fix, repo_context)


# ---------------------------------------------------------------------------
# Mode C: surgical patch — single target file, forced edit output
# ---------------------------------------------------------------------------

async def _generate_fix_mode_c(
    state: WorkflowState,
    repo_context: dict[str, Any],
) -> tuple["FixResult", dict[str, str]]:
    """Mode C: surgical patch targeting one file. LLM is required to produce at least one edit."""
    from issueops.tools.llm import generate_structured

    debug_result = state.get("debug_result") or {}
    file_snippets = repo_context.get("file_snippets") or {}

    # Primary target: prefer debug-identified files that are present in snippets
    relevant_files: list[str] = debug_result.get("relevant_files") or []
    target_file = next((f for f in relevant_files if f in file_snippets), None)
    if target_file is None:
        target_file = next(iter(file_snippets), None)

    if not target_file:
        logger.warning("[PATCH_MODE_C] no target file available in snippets")
        return _fix_fallback(state, "Mode C: no target file available")

    content = file_snippets[target_file]
    total_lines = len(content.splitlines())
    numbered_lines = [
        f"{i:4d} | {line}"
        for i, line in enumerate(content.splitlines(), start=1)
    ]
    file_contents_text = f"### {target_file}\n```\n" + "\n".join(numbered_lines) + "\n```"

    diagnosis_confidence = debug_result.get("diagnosis_confidence", 0.0)
    fix_approach = (
        debug_result.get("repair_strategy")
        or debug_result.get("suggested_fix_approach")
        or "unknown"
    )

    template = _SURGICAL_PROMPT_PATH.read_text()
    prompt = (
        template
        .replace("{issue_title}", state["issue_title"])
        .replace("{issue_body}", state["issue_body"] or "(no body)")
        .replace("{root_cause}", debug_result.get("root_cause", "unknown"))
        .replace("{suggested_fix_approach}", fix_approach)
        .replace("{target_file}", target_file)
        .replace("{file_contents}", file_contents_text)
        .replace("{diagnosis_confidence}", f"{diagnosis_confidence:.0%}")
    )

    line_result: LinePatchResult = await generate_structured(prompt, LinePatchResult)
    logger.info(
        "[PATCH_MODE_C] target=%s edits=%d confidence=%.2f",
        target_file, len(line_result.edits), line_result.confidence,
    )

    if not line_result.edits:
        logger.warning(
            "[PATCH_EMPTY_RESPONSE] mode=C target=%s LLM returned empty edits despite surgical constraint",
            target_file,
        )
        return _fix_fallback(state, "Mode C: LLM returned no edits despite surgical constraint")

    # Convert line-anchored edits to FileEdits — normalise path to target_file
    file_edits = []
    for anchored in line_result.edits:
        if anchored.path != target_file:
            logger.warning(
                "[PATCH_MODE_C] LLM cited path=%s but target is %s — normalizing",
                anchored.path, target_file,
            )
            anchored = anchored.model_copy(update={"path": target_file})
        fe = line_anchored_to_file_edit(anchored, content)
        if fe is None:
            logger.warning(
                "[PATCH_MODE_C] file=%s: line range %d-%d out of bounds (file has %d lines)",
                target_file, anchored.start_line, anchored.end_line, total_lines,
            )
            continue
        file_edits.append(fe)

    if not file_edits:
        logger.warning("[PATCH_MODE_C] no valid edits after line range conversion for %s", target_file)
        return _fix_fallback(state, "Mode C: line ranges out of bounds")

    fix = FixResult(
        patch_plan=line_result.patch_plan,
        files_to_modify=[e.path for e in file_edits],
        proposed_edits=file_edits,
        confidence=line_result.confidence,
        validation_notes=line_result.validation_notes,
        ready_for_pr=False,
    )
    return _run_validation(fix, repo_context)


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
    logger.info(
        "[PATCH_VALIDATION] result=%s files=%d notes='%s'",
        "pass" if all_valid else "fail",
        len(fix_result.proposed_edits),
        notes[:120],
    )

    if all_valid:
        for edit in fix_result.proposed_edits:
            content = file_snippets.get(edit.path)
            if content is None:
                continue

            # Syntax / structural gate — runs before any file I/O
            syntax_ok, syntax_note = validate_replacement_syntax(edit)

            # Prepend false-positive recovery: append-after-match is a legitimate
            # edit shape that the static detector cannot distinguish from a
            # duplicate-then-correct bug. Apply the patch and verify the result.
            if not syntax_ok and syntax_note.startswith("[prepend]"):
                recovered_ok, recover_note = recover_from_prepend(edit, content)
                if recovered_ok:
                    logger.info(
                        "[PATCH_RECOVERY] [prepend] recovered as insertion for %s — %s",
                        edit.path, recover_note,
                    )
                    syntax_ok = True
                    syntax_note = f"[recovered_prepend] {recover_note}"
                else:
                    logger.warning(
                        "[PATCH_RECOVERY] [prepend] recovery failed for %s — %s",
                        edit.path, recover_note,
                    )

            if not syntax_ok:
                all_valid = False
                validation_issues.append(f"{edit.path}: {syntax_note}")
                logger.warning(
                    "patch_syntax rejected edit for %s: %s", edit.path, syntax_note
                )
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
                continue

            # Post-apply brace balance — verify the patched file is still coherent
            patched, _ = apply_edit_to_content(content, edit)
            balance_ok, balance_note = check_brace_balance(content, patched, edit.path)
            if not balance_ok:
                all_valid = False
                validation_issues.append(f"{edit.path}: {balance_note}")
                logger.warning(
                    "patch_syntax post-apply balance check failed for %s: %s",
                    edit.path, balance_note,
                )
                continue

            diffs[edit.path] = diff
            logger.info("patch_builder: diff built for %s (%s)", edit.path, size_note)
            logger.info(
                "[PATCH_VALIDATION] result=pass method=diff_build file=%s size=%s",
                edit.path, size_note,
            )

    final_notes = notes
    if validation_issues:
        final_notes = "; ".join([notes] + validation_issues) if notes else "; ".join(validation_issues)

    updated = fix_result.model_copy(update={
        "ready_for_pr": all_valid and bool(diffs),
        "validation_notes": final_notes,
    })

    return updated, diffs


# ---------------------------------------------------------------------------
# Stage A: Deterministic detector-based fix
# ---------------------------------------------------------------------------

def _try_detector_fix(
    state: WorkflowState,
    repo_context: dict[str, Any],
) -> tuple["FixResult | None", dict[str, str]]:
    """Try pattern-based bug detectors before invoking the LLM.

    Returns (FixResult, diffs) if a detector fires with sufficient confidence,
    (None, {}) otherwise.  Detectors are synchronous and require no LLM call.
    """
    debug_result = state.get("debug_result") or {}
    root_cause: str = debug_result.get("root_cause", "")
    suspected_symbols: list[str] = debug_result.get("suspected_symbols") or []
    repair_strategy: str = debug_result.get("repair_strategy") or ""
    file_snippets: dict[str, str] = repo_context.get("file_snippets") or {}

    for path, content in file_snippets.items():
        det = detect_in_file(path, content, root_cause, suspected_symbols, repair_strategy)
        if det is None:
            continue

        logger.info(
            "[DETECTOR_HIT] pattern=%s file=%s confidence=%.2f",
            det.pattern_name, path, det.confidence,
        )

        edit = FileEdit(
            path=path,
            change_summary=det.description,
            source_lines="detector",
            find_snippet=det.find_snippet,
            replace_with=det.replace_with,
        )

        diff, applied = apply_and_diff(content, edit)
        if not applied or diff is None:
            logger.warning(
                "FixPR detector: match for %s but apply_and_diff failed — skipping",
                path,
            )
            continue

        fix_result = FixResult(
            patch_plan=f"Deterministic fix ({det.pattern_name}): {det.description}",
            files_to_modify=[path],
            proposed_edits=[edit],
            confidence=det.confidence,
            validation_notes=f"Applied via deterministic detector '{det.pattern_name}'",
            ready_for_pr=True,
        )
        logger.info(
            "FixPR detector: '%s' generated a validated fix for %s (confidence=%.2f)",
            det.pattern_name, path, det.confidence,
        )
        return fix_result, {path: diff}

    return None, {}


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
# Diff-substance accounting (no-op guard)
# ---------------------------------------------------------------------------

# Bracket/punctuation-only fragments that don't represent a real code change
_TRIVIAL_DIFF_BODIES: frozenset[str] = frozenset({
    "{", "}", "(", ")", "[", "]",
    "});", "},", "};", "),", ");",
    "})", ")", "]", "[",
    "pass", "...", ":",
})


def _count_substantive_diff_lines(diff_text: str) -> int:
    """Count +/- lines in a unified diff that represent a real code change.

    Excludes:
      - the file header lines (``+++ a/x``, ``--- b/x``)
      - whitespace-only changes
      - comment-only lines (Python ``#``, JS/Java ``//``)
      - bracket/punctuation-only lines (``}``, ``});``, etc.)

    Used by the no-op guard: when the debug agent reported high diagnosis
    confidence in a real bug (e.g. add a None-guard, fix a None equality),
    the resulting diff must change real code lines, not just merge whitespace.
    A trivial diff under a confident diagnosis is almost always a fallback
    mode shipping cosmetic noise instead of the actual fix.
    """
    count = 0
    for line in diff_text.splitlines():
        if not line or line[0] not in ("+", "-"):
            continue
        if line.startswith(("+++", "---")):
            continue
        body = line[1:].strip()
        if not body:
            continue
        if body in _TRIVIAL_DIFF_BODIES:
            continue
        if body.startswith("#") or body.startswith("//"):
            continue
        count += 1
    return count


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
# Diagnosis-only PR helpers
# ---------------------------------------------------------------------------

def _build_diagnosis_only_content(
    issue_id: int | str,
    debug_result: dict[str, Any],
    state: WorkflowState,
) -> str:
    """Generate the markdown diagnosis file committed to the diagnosis-only PR branch."""
    root_cause = debug_result.get("root_cause", "See issue")
    repair_strategy = debug_result.get("repair_strategy", "")
    suspected_symbols = debug_result.get("suspected_symbols") or []
    relevant_files = debug_result.get("relevant_files") or []
    diagnosis_confidence = debug_result.get("diagnosis_confidence", 0.0)

    lines = [
        f"# IssueOps Diagnosis — Issue #{issue_id}",
        "",
        f"**Issue:** {state['issue_title']}",
        "",
        f"**Root Cause** (confidence: {diagnosis_confidence:.0%}):",
        root_cause,
        "",
    ]
    if repair_strategy:
        lines += [f"**Suggested Fix:**", repair_strategy, ""]
    if suspected_symbols:
        lines.append(f"**Relevant Symbols:** `{'`, `'.join(suspected_symbols[:6])}`")
        lines.append("")
    if relevant_files:
        lines.append("**Files to Investigate:**")
        lines += [f"- `{f}`" for f in relevant_files[:4]]
        lines.append("")
    lines += [
        "---",
        "*This PR was created automatically by IssueOps.*",
        "*No production code was modified — this is a diagnosis-only draft for human review.*",
    ]
    return "\n".join(lines)


def _build_diagnosis_pr_metadata(
    state: WorkflowState,
    debug_result: dict[str, Any],
) -> dict[str, str]:
    issue_id = state["issue_id"]
    root_cause = debug_result.get("root_cause", "See analysis")
    repair_strategy = debug_result.get("repair_strategy", "")
    diagnosis_confidence = debug_result.get("diagnosis_confidence", 0.0)

    pr_body = (
        f"## IssueOps Diagnosis PR — Issue #{issue_id}\n\n"
        f"Automated patch generation failed, but root cause was identified "
        f"with **{diagnosis_confidence:.0%} confidence**.\n\n"
        f"### Root Cause\n{root_cause}\n\n"
    )
    if repair_strategy:
        pr_body += f"### Suggested Human Fix\n{repair_strategy}\n\n"
    pr_body += (
        "> **Diagnosis-Only Draft PR** — no production code was modified automatically.\n"
        "> A human engineer should implement the suggested fix and push to this branch."
    )

    return {
        "branch": f"issueops/fix-issue-{issue_id}",
        "pr_title": f"diagnosis: issue #{issue_id} — {state['issue_title'][:50]}",
        "pr_body": pr_body,
        "commit_message": f"docs: IssueOps diagnosis notes for issue #{issue_id}",
        "diagnosis_file": f".issueops/diagnosis-issue-{issue_id}.md",
    }


async def _execute_diagnosis_only_pr(
    state: WorkflowState,
    debug_result: dict[str, Any],
    pr_meta: dict[str, str],
) -> dict[str, Any]:
    """Create branch + diagnosis notes file + draft PR + issue comment."""
    owner = state["repo_owner"]
    repo = state["repo_name"]
    issue_id = state["issue_id"]
    branch_name = pr_meta["branch"]
    existing_errors = list(state.get("errors") or [])

    diagnosis_content = _build_diagnosis_only_content(issue_id, debug_result, state)

    try:
        base_branch = await gh.get_default_branch(owner, repo)
        await gh.create_branch(owner, repo, base_branch, branch_name)

        await gh.create_or_update_file(
            owner, repo, pr_meta["diagnosis_file"], diagnosis_content,
            branch_name, pr_meta["commit_message"],
        )

        pr_data = await gh.create_draft_pr(
            owner, repo,
            pr_meta["pr_title"], pr_meta["pr_body"],
            branch_name, base_branch,
        )
        pr_url: str = pr_data.get("html_url", "")
        pr_number: int = pr_data.get("number", 0)

        diagnosis_confidence = debug_result.get("diagnosis_confidence", 0.0)
        comment_body = (
            f"## IssueOps — Diagnosis PR Opened\n\n"
            f"Automated patch generation failed, but root cause was identified "
            f"({diagnosis_confidence:.0%} confidence).\n\n"
            f"A diagnosis-only draft PR has been opened for human review: {pr_url}\n\n"
            f"**Root Cause:** {debug_result.get('root_cause', 'See PR')[:300]}\n\n"
            f"> No code was modified automatically. Human implementation required."
        )
        comment_data = await gh.comment_on_issue(owner, repo, issue_id, comment_body)
        comment_url: str = comment_data.get("html_url", "")

        logger.info("[DIAGNOSIS_ONLY_PR] created PR #%d %s", pr_number, pr_url)
        return {
            "fix_result": {
                "patch_plan": debug_result.get("repair_strategy", "Diagnosis only — no automated fix"),
                "files_to_modify": [],
                "proposed_edits": [],
                "confidence": 0.0,
                "validation_notes": "Diagnosis-only PR — no code edits applied",
                "ready_for_pr": False,
                "diagnosis_only": True,
                **pr_meta,
                "mock_writes": False,
            },
            "pr_url": pr_url,
            "issue_comment_url": comment_url,
            "current_step": "completed",
        }

    except gh.GitHubWriteError as exc:
        logger.error("[DIAGNOSIS_ONLY_PR] write failed: %s", exc)
        return {
            "fix_result": {"diagnosis_only": True, "write_error": str(exc), "mock_writes": False},
            "pr_url": None,
            "issue_comment_url": None,
            "current_step": "write_failed",
            "errors": existing_errors + [str(exc)],
        }


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
    suspected_symbols: list[str] | None = None,
    repair_strategy: str = "",
    diagnosis_confidence: float = 0.0,
    relevant_files: list[str] | None = None,
) -> str:
    body = (
        f"## IssueOps — Investigation Complete\n\n"
        f"IssueOps investigated issue #{issue_id} but patch confidence was too low "
        f"({confidence:.0%}) to generate an automated fix.\n\n"
        f"**Root Cause Analysis** (diagnosis confidence: {diagnosis_confidence:.0%}):\n"
        f"{root_cause}\n\n"
    )
    if repair_strategy:
        body += f"**Suggested Fix:**\n{repair_strategy}\n\n"
    if suspected_symbols:
        body += f"**Relevant symbols:** `{'`, `'.join(suspected_symbols[:6])}`\n\n"
    if relevant_files:
        body += "**Files to investigate:**\n" + "\n".join(f"- `{f}`" for f in relevant_files[:4]) + "\n\n"
    body += "> Automated patch was not applied — human review required."
    return body


# ---------------------------------------------------------------------------
# Live write pipeline
# ---------------------------------------------------------------------------

@trace("execute_writes")
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

        # 2. Resolve a non-colliding branch name (handles reopen-to-retest workflows
        # where a previous run already created issueops/fix-issue-N). The resolver
        # auto-appends -v2/-v3/... and updates the PR title to reflect the iteration.
        resolved_branch = await gh.find_available_branch_name(owner, repo, branch_name)
        if resolved_branch != branch_name:
            logger.info(
                "FixPR writes: branch '%s' already exists — using '%s' instead",
                branch_name, resolved_branch,
            )
            branch_name = resolved_branch
            pr_meta = {**pr_meta, "branch": resolved_branch}
            # Reflect iteration in the PR title so the new draft is distinguishable
            # from prior attempts in the GitHub UI.
            suffix = resolved_branch.rsplit("-", 1)[-1]
            if suffix.startswith("v") and suffix[1:].isdigit():
                base_title = pr_meta.get("pr_title", "")
                pr_meta = {**pr_meta, "pr_title": f"{base_title} ({suffix})"}

        # 3. Create feature branch
        await gh.create_branch(owner, repo, base_branch, branch_name)

        # 4. Apply each validated edit to the full file and commit
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

            # Post-apply verification
            verification = await verify_patch(edit.path, new_content)
            logger.info(
                "[PATCH_VERIFICATION] result=%s method=%s file=%s detail=%s",
                "pass" if verification.passed else "fail",
                verification.method, edit.path, verification.detail[:100],
            )
            if not verification.passed:
                logger.warning(
                    "[PATCH_VERIFICATION] verification failed but proceeding — human review required"
                )
                # Apply confidence_multiplier to fix_result.confidence
                fix_result = fix_result.model_copy(update={
                    "confidence": fix_result.confidence * verification.confidence_multiplier,
                })

            await gh.create_or_update_file(
                owner, repo, edit.path, new_content,
                branch_name, pr_meta["commit_message"],
            )

        # 5. Open draft PR
        pr_data = await gh.create_draft_pr(
            owner, repo,
            pr_meta["pr_title"], pr_meta["pr_body"],
            branch_name, base_branch,
        )
        pr_url: str = pr_data.get("html_url", "")
        pr_number: int = pr_data.get("number", 0)

        # 6. Comment on original issue
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

@trace("generate_fix_and_pr")
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

    await checkpoint("before_fix_generation")
    use_llm = settings.llm_available and not state.get("disable_llm", False)
    repo_context = state.get("repo_context") or {}
    diffs_final: dict[str, str] = {}

    # --- 0. Enrich repo_context with debug-identified files missing from retrieval ---
    repo_context = await _enrich_snippets_with_debug_files(state, repo_context)

    # --- 1a. Stage A: Deterministic detectors (no LLM, zero hallucination risk) ---
    fix_result, diffs_final = _try_detector_fix(state, repo_context)

    if fix_result is not None:
        logger.info(
            "FixPR: detector stage succeeded — skipping LLM (confidence=%.2f)",
            fix_result.confidence,
        )
    else:
        # --- 1b. Stage B: LLM-generated fix with symbol-targeted context ---
        llm_succeeded = False
        raw_fix: FixResult

        if use_llm:
            try:
                raw_fix = await _generate_fix_with_llm(state, repo_context)
                llm_succeeded = True
                logger.info(
                    "FixPR: LLM generated fix — files=%s confidence=%.2f",
                    raw_fix.files_to_modify, raw_fix.confidence,
                )
            except Exception as exc:
                exc_name = type(exc).__name__
                if "ValidationError" in exc_name or "validation_error" in exc_name.lower():
                    logger.warning("[PATCH_SCHEMA_FAIL] mode=A schema validation failed: %s", exc)
                else:
                    logger.warning("FixPR: LLM failed (%s: %s), using fallback", exc_name, exc)
                raw_fix, diffs_final = _fix_fallback(state, f"LLM error: {exc_name}")
        else:
            reason = "no API key" if not settings.llm_available else "disabled via flag"
            logger.info("FixPR: skipping LLM (%s)", reason)
            raw_fix, diffs_final = _fix_fallback(state, f"LLM unavailable: {reason}")

        # --- 2. Validate + build diffs (only when LLM produced a result) ---
        if llm_succeeded:
            if not raw_fix.proposed_edits:
                logger.warning(
                    "[PATCH_EMPTY_RESPONSE] mode=A LLM returned empty edits — confidence=%.2f",
                    raw_fix.confidence,
                )
            fix_result, diffs_final = _run_validation(raw_fix, repo_context)
            logger.info(
                "[PATCH_MODE] mode=A result=%s confidence=%.2f",
                "ready" if fix_result.ready_for_pr else "not_ready",
                fix_result.confidence,
            )
        else:
            fix_result = raw_fix

        # --- 2b. Mode A retry: feed the validation error back and try once more ---
        # Aider-style feedback loop. Only worth attempting when Mode A produced
        # edits that failed validation (vs returning no edits at all) — otherwise
        # the retry has no anchor and Mode B is a better next step.
        if (
            not fix_result.ready_for_pr
            and use_llm
            and llm_succeeded
            and raw_fix.proposed_edits
        ):
            retry_error = fix_result.validation_notes or "(no specific reason)"
            logger.info(
                "[PATCH_RETRY] mode=A failed — retrying once with feedback: '%s'",
                retry_error[:120],
            )
            try:
                retry_raw = await _generate_fix_with_llm(
                    state, repo_context, prior_error=retry_error
                )
                if retry_raw.proposed_edits:
                    retry_fix_result, retry_diffs = _run_validation(retry_raw, repo_context)
                    if retry_fix_result.ready_for_pr:
                        logger.info(
                            "[PATCH_RETRY] mode=A retry succeeded — confidence=%.2f",
                            retry_fix_result.confidence,
                        )
                        fix_result, diffs_final = retry_fix_result, retry_diffs
                    else:
                        logger.info(
                            "[PATCH_RETRY] mode=A retry also failed: %s",
                            retry_fix_result.validation_notes[:120],
                        )
                else:
                    logger.info("[PATCH_RETRY] mode=A retry produced no edits")
            except Exception as exc:
                exc_name = type(exc).__name__
                if "ValidationError" in exc_name or "validation_error" in exc_name.lower():
                    logger.warning("[PATCH_SCHEMA_FAIL] mode=A retry schema validation failed: %s", exc)
                else:
                    logger.warning(
                        "[PATCH_RETRY] mode=A retry errored (%s: %s) — falling through to mode=B",
                        exc_name, exc,
                    )

        # --- 1c. Stage B fallback: line-anchored patching ---
        if not fix_result.ready_for_pr and use_llm:
            logger.info("[PATCH_MODE] mode=A failed, trying mode=B (line-anchored)")
            try:
                fix_result, diffs_final = await _generate_fix_mode_b(state, repo_context)
                logger.info(
                    "[PATCH_MODE] mode=B result=%s confidence=%.2f",
                    "ready" if fix_result.ready_for_pr else "not_ready",
                    fix_result.confidence,
                )
            except Exception as exc:
                exc_name = type(exc).__name__
                if "ValidationError" in exc_name or "validation_error" in exc_name.lower():
                    logger.warning("[PATCH_SCHEMA_FAIL] mode=B schema validation failed: %s", exc)
                else:
                    logger.warning(
                        "[PATCH_MODE] mode=B failed (%s: %s), keeping mode=A result",
                        exc_name, exc,
                    )

        # --- 1d. Stage C: surgical patch — forced edit, single target file ---
        if not fix_result.ready_for_pr and use_llm:
            _dbg = state.get("debug_result") or {}
            _diag_conf = _dbg.get("diagnosis_confidence", 0.0)
            if _diag_conf >= settings.confidence_threshold:
                logger.info(
                    "[PATCH_RETRY_REASON] modes A+B produced no edits but "
                    "diagnosis_confidence=%.2f >= threshold=%.2f — attempting Mode C (surgical)",
                    _diag_conf, settings.confidence_threshold,
                )
                try:
                    fix_result, diffs_final = await _generate_fix_mode_c(state, repo_context)
                    logger.info(
                        "[PATCH_MODE_C] result=%s confidence=%.2f",
                        "ready" if fix_result.ready_for_pr else "not_ready",
                        fix_result.confidence,
                    )
                except Exception as exc:
                    exc_name = type(exc).__name__
                    if "ValidationError" in exc_name or "validation_error" in exc_name.lower():
                        logger.warning("[PATCH_SCHEMA_FAIL] mode=C schema validation failed: %s", exc)
                    else:
                        logger.warning(
                            "[PATCH_MODE_C] failed (%s: %s), keeping previous result",
                            exc_name, exc,
                        )
            else:
                logger.info(
                    "[PATCH_RETRY_REASON] diagnosis_confidence=%.2f < threshold=%.2f — Mode C not attempted",
                    _diag_conf, settings.confidence_threshold,
                )

    # --- 2c. No-op guard: reject trivially small patches under a confident diagnosis ---
    # When the debug agent identified a specific real bug with high confidence,
    # shipping a 1-2 line cosmetic change is worse than escalating. This blocks
    # the failure mode where Mode B / Mode C ship a near-no-op just to satisfy
    # the validator after the actually-correct Mode A edit was rejected.
    if fix_result.ready_for_pr and diffs_final:
        debug_state = state.get("debug_result") or {}
        diagnosis_confidence = debug_state.get("diagnosis_confidence", 0.0)
        if diagnosis_confidence >= 0.75:
            sub_lines = sum(
                _count_substantive_diff_lines(d) for d in diffs_final.values()
            )
            if sub_lines < 3:
                logger.warning(
                    "[PATCH_NOOP_GUARD] diagnosis_confidence=%.2f but diff has only %d "
                    "substantive line(s) — refusing to ship a trivial patch; escalating instead",
                    diagnosis_confidence, sub_lines,
                )
                fix_result = fix_result.model_copy(update={
                    "ready_for_pr": False,
                    "validation_notes": (
                        (fix_result.validation_notes + " | " if fix_result.validation_notes else "")
                        + f"NOOP_GUARD: diff has {sub_lines} substantive line(s); "
                        f"diagnosis_confidence={diagnosis_confidence:.2f} requires >=3"
                    ),
                })
                diffs_final = {}

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
        diagnosis_confidence = debug_result.get("diagnosis_confidence", 0.0)
        logger.info(
            "[PR_REASON] decision=escalate confidence=%.2f diagnosis_confidence=%.2f",
            fix_result.confidence,
            diagnosis_confidence,
        )

        # Diagnosis-only PR: open a PR with root cause notes when all patches failed
        # but diagnosis confidence is high enough and feature flag is enabled.
        if (
            settings.allow_diagnosis_only_pr
            and diagnosis_confidence >= 0.75
            and fix_result.confidence == 0.0
        ):
            logger.info(
                "[PR_REASON] diagnosis_only_pr=eligible diagnosis_confidence=%.2f — opening diagnosis PR",
                diagnosis_confidence,
            )
            diag_meta = _build_diagnosis_pr_metadata(state, debug_result)
            if not dry_run:
                return await _execute_diagnosis_only_pr(state, debug_result, diag_meta)
            pr_number = 1000 + (issue_id if isinstance(issue_id, int) else 0)
            mock_pr_url = f"https://github.com/{owner}/{repo}/pull/{pr_number}"
            mock_comment_url = f"https://github.com/{owner}/{repo}/issues/{issue_id}#issuecomment-dry-run"
            logger.info("[DIAGNOSIS_ONLY_PR] dry run: would create diagnosis PR at %s", mock_pr_url)
            return {
                "fix_result": {"diagnosis_only": True, "mock_writes": True, **diag_meta},
                "pr_url": mock_pr_url,
                "issue_comment_url": mock_comment_url,
                "current_step": "completed",
            }

        low_conf_comment = _build_low_confidence_comment(
            issue_id,
            fix_result.confidence,
            debug_result.get("root_cause", "Analysis unavailable"),
            suspected_symbols=debug_result.get("suspected_symbols"),
            repair_strategy=debug_result.get("repair_strategy", ""),
            diagnosis_confidence=debug_result.get("diagnosis_confidence", 0.0),
            relevant_files=debug_result.get("relevant_files"),
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
    logger.info(
        "[PR_REASON] decision=create confidence=%.2f branch=%s",
        fix_result.confidence, pr_meta["branch"],
    )
    await checkpoint("before_pr_creation")
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
        issue_id,
        confidence,
        debug_result.get("root_cause", "unknown"),
        suspected_symbols=debug_result.get("suspected_symbols"),
        repair_strategy=debug_result.get("repair_strategy", ""),
        diagnosis_confidence=debug_result.get("diagnosis_confidence", 0.0),
        relevant_files=debug_result.get("relevant_files"),
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
