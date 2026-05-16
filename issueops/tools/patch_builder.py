"""Patch builder — applies FileEdit objects and generates unified diffs."""

import difflib
import logging
from typing import Optional

from issueops.schemas.fix import FileEdit, LineAnchoredEdit
from issueops.tools.patch_match import find_snippet

logger = logging.getLogger(__name__)


def apply_edit(original_content: str, edit: FileEdit) -> tuple[str, bool]:
    """Apply a single find-and-replace edit.

    Returns (patched_content, was_applied).
    Replaces only the first occurrence to keep edits surgical.

    Matching is whitespace-tolerant: tries exact → normalized-whitespace →
    fuzzy (difflib) in order.  The replacement always targets the verbatim
    original text so no accidental normalization is introduced into the file.
    """
    match = find_snippet(edit.find_snippet, original_content)
    if match is None:
        logger.debug(
            "patch_builder: snippet not found in %s (tried exact/normalized/fuzzy)",
            edit.path,
        )
        return original_content, False

    if match.method != "exact":
        logger.info(
            "patch_builder: applied via %s match for %s", match.method, edit.path
        )

    patched = original_content.replace(match.matched_text, edit.replace_with, 1)
    return patched, True


def build_diff(path: str, original: str, patched: str) -> str:
    """Generate a unified diff string between original and patched content."""
    orig_lines = original.splitlines()
    patch_lines = patched.splitlines()

    diff_iter = difflib.unified_diff(
        orig_lines,
        patch_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff_iter)


def apply_and_diff(original_content: str, edit: FileEdit) -> tuple[Optional[str], bool]:
    """Apply edit and return (unified_diff_string, success).

    Returns (None, False) if the snippet is not found.
    """
    patched, applied = apply_edit(original_content, edit)
    if not applied:
        return None, False
    diff = build_diff(edit.path, original_content, patched)
    return diff, True


def apply_edit_to_content(original_content: str, edit: FileEdit) -> tuple[str, bool]:
    """Apply edit and return (modified_file_content, success).

    Returns (original_content, False) if the snippet is not found.
    Intended for callers that need the full patched content to write back to disk or GitHub.
    """
    return apply_edit(original_content, edit)


def line_anchored_to_file_edit(edit: LineAnchoredEdit, content: str) -> Optional[FileEdit]:
    """Extract find_snippet from file content at specified line range, return FileEdit.

    The start_line and end_line are 1-based (matching the NNN | prefix from the prompt).
    Returns None if line numbers are out of range.
    """
    lines = content.splitlines(keepends=True)
    total = len(lines)

    # Convert 1-based inclusive range to 0-based slice indices
    start_idx = edit.start_line - 1
    end_idx = edit.end_line  # end_line is inclusive, so slice up to end_line (exclusive)

    if start_idx < 0 or end_idx > total or start_idx >= end_idx:
        logger.warning(
            "line_anchored_to_file_edit: line range %d-%d out of bounds (file has %d lines) for %s",
            edit.start_line, edit.end_line, total, edit.path,
        )
        return None

    find_snippet_text = "".join(lines[start_idx:end_idx])

    return FileEdit(
        path=edit.path,
        change_summary=edit.change_summary,
        source_lines=f"{edit.start_line}-{edit.end_line}",
        find_snippet=find_snippet_text,
        replace_with=edit.replace_with,
    )


def apply_line_anchored_edit(content: str, edit: LineAnchoredEdit) -> tuple[str, bool]:
    """Apply a line-anchored edit by replacing the specified line range.

    Returns (patched_content, was_applied).
    Uses 1-based inclusive line numbers matching the NNN | prompt prefix.
    Returns (content, False) if line numbers are out of range.
    """
    lines = content.splitlines(keepends=True)
    total = len(lines)

    start_idx = edit.start_line - 1
    end_idx = edit.end_line  # inclusive → exclusive slice

    if start_idx < 0 or end_idx > total or start_idx >= end_idx:
        logger.warning(
            "apply_line_anchored_edit: line range %d-%d out of bounds (file has %d lines) for %s",
            edit.start_line, edit.end_line, total, edit.path,
        )
        return content, False

    # Ensure replace_with ends with newline to keep subsequent lines intact
    replacement = edit.replace_with
    if replacement and not replacement.endswith("\n"):
        replacement += "\n"

    patched_lines = lines[:start_idx] + [replacement] + lines[end_idx:]
    patched = "".join(patched_lines)

    logger.info(
        "apply_line_anchored_edit: replaced lines %d-%d in %s",
        edit.start_line, edit.end_line, edit.path,
    )
    return patched, True
