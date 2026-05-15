"""Patch validator — checks FileEdit objects before applying them."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from issueops.schemas.fix import FileEdit, FixResult
from issueops.tools.patch_match import find_snippet

logger = logging.getLogger(__name__)

_MAX_FILES = 2
_MAX_SNIPPET_CHARS = 2000
_MAX_DIFF_LINES = 150


@dataclass
class EditValidation:
    valid: bool
    reasons: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        self.reasons.append(reason)


def validate_edit(edit: FileEdit, file_content: Optional[str]) -> EditValidation:
    """Validate a single FileEdit against the available file content."""
    result = EditValidation(valid=True)

    if not edit.path or not edit.path.strip():
        result.valid = False
        result.add("path is empty")
        return result

    if not edit.find_snippet or not edit.find_snippet.strip():
        result.valid = False
        result.add(f"{edit.path}: find_snippet is empty")
        return result

    if len(edit.find_snippet) > _MAX_SNIPPET_CHARS:
        result.valid = False
        result.add(
            f"{edit.path}: find_snippet is {len(edit.find_snippet)} chars "
            f"(max {_MAX_SNIPPET_CHARS}) — not surgical"
        )
        return result

    if edit.find_snippet == edit.replace_with:
        result.valid = False
        result.add(f"{edit.path}: find_snippet and replace_with are identical — no-op edit")
        return result

    if file_content is None:
        result.valid = False
        result.add(
            f"{edit.path}: file content not in repo context — "
            "cannot verify snippet; file may not exist or was not fetched"
        )
        return result

    match = find_snippet(edit.find_snippet, file_content)
    if match is None:
        result.valid = False
        result.add(
            f"{edit.path}: find_snippet not found in file content "
            "(tried exact match, whitespace-normalized match, and fuzzy match) — "
            "snippet may reference the wrong code block, or the file was truncated "
            "before the relevant section"
        )
        return result

    if match.method != "exact":
        logger.info(
            "patch_validator: %s — accepted via %s match (whitespace tolerance)",
            edit.path, match.method,
        )

    return result


def validate_fix_result(
    fix_result: FixResult,
    file_snippets: dict[str, str],
) -> tuple[bool, str]:
    """Validate all edits in a FixResult.

    Returns (all_valid, human_readable_notes).
    """
    if not fix_result.proposed_edits:
        return False, "No proposed edits generated"

    if len(fix_result.files_to_modify) > _MAX_FILES:
        return False, (
            f"Too many files targeted ({len(fix_result.files_to_modify)} > {_MAX_FILES})"
        )

    notes: list[str] = []
    all_valid = True

    for edit in fix_result.proposed_edits:
        content = file_snippets.get(edit.path)
        ev = validate_edit(edit, content)
        if ev.valid:
            notes.append(f"{edit.path}: OK")
            logger.info("patch_validator: %s — VALID", edit.path)
        else:
            all_valid = False
            for r in ev.reasons:
                notes.append(f"INVALID: {r}")
                logger.warning("patch_validator: %s — %s", edit.path, r)

    return all_valid, "; ".join(notes)


def diff_size_ok(diff: str) -> tuple[bool, str]:
    """Check that a diff is not unreasonably large."""
    lines = diff.splitlines()
    if len(lines) > _MAX_DIFF_LINES:
        return False, f"diff is {len(lines)} lines (max {_MAX_DIFF_LINES}) — edit is not surgical"
    return True, f"{len(lines)} diff lines"
