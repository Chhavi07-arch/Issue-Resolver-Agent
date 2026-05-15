"""Patch builder — applies FileEdit objects and generates unified diffs."""

import difflib
import logging
from typing import Optional

from issueops.schemas.fix import FileEdit

logger = logging.getLogger(__name__)


def apply_edit(original_content: str, edit: FileEdit) -> tuple[str, bool]:
    """Apply a single find-and-replace edit.

    Returns (patched_content, was_applied).
    Replaces only the first occurrence to keep edits surgical.
    """
    if edit.find_snippet not in original_content:
        logger.debug("patch_builder: snippet not found in %s", edit.path)
        return original_content, False

    patched = original_content.replace(edit.find_snippet, edit.replace_with, 1)
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
