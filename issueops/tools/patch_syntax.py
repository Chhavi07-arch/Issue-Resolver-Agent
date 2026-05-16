"""Lightweight syntax and structural validation for generated code replacements.

Two public entry points:

  validate_replacement_syntax(edit)
      Checks replace_with for obvious defects before patch application.
      Language-aware where practical; uses stdlib only (no external parsers).

  check_brace_balance(original, patched, path)
      After applying an edit, verifies the full patched file still has
      balanced braces (for brace-scoped languages).

Design principles:
  - False negatives (missed defects) are acceptable — the fuzzy patch
    matcher and GitHub API provide downstream safety nets.
  - False positives (rejecting valid edits) are costly — they escalate
    a correctly-diagnosed bug to human review. Conservative thresholds
    throughout.
  - No external dependencies. Python AST is stdlib. All other checks
    are regex / character counting.
  - Language detection from file extension only — no repo configuration.
"""

import ast
import logging
import re
import textwrap
from typing import Optional

from issueops.schemas.fix import FileEdit

logger = logging.getLogger(__name__)

# ─── Language / extension mapping ────────────────────────────────────────────

_LANGUAGE_MAP: dict[str, str] = {
    ".java":  "java",
    ".kt":    "java",   # Kotlin — same brace semantics
    ".scala": "java",
    ".cs":    "java",   # C# — same brace semantics
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "javascript",
    ".jsx":   "javascript",
    ".tsx":   "javascript",
    ".go":    "go",     # Go uses braces; separate from Java to allow future specialisation
    ".rs":    "java",   # Rust — same brace semantics
}

# Languages where a complete file must have balanced {} — used by check_brace_balance
_BRACE_SCOPED_EXTS = frozenset({
    ".java", ".kt", ".scala", ".cs", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
})


def _ext(path: str) -> str:
    dot = path.rfind(".")
    return path[dot:].lower() if dot != -1 else ""


def _language(path: str) -> str:
    return _LANGUAGE_MAP.get(_ext(path), "unknown")


# ─── String-literal-aware bracket counter ────────────────────────────────────

def _net_brackets(text: str, open_c: str, close_c: str) -> int:
    """Return net-open count for a bracket pair, skipping characters inside strings.

    Handles single-quoted and double-quoted string literals with backslash
    escapes.  Known limitation: does not handle triple-quoted Python strings
    or JS/TS backtick template literals — accepted trade-off for simplicity.
    In practice, format strings like ``"msg {}"`` and SLF4J ``"log {}"`` use
    symmetric ``{}`` so their raw delta is 0 and they don't distort the count.
    """
    net = 0
    in_str = False
    str_char = ""
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < len(text):
                i += 2          # skip escaped character
                continue
            if c == str_char:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str = True
                str_char = c
            elif c == open_c:
                net += 1
            elif c == close_c:
                net -= 1
        i += 1
    return net


# ─── Individual standalone checks ────────────────────────────────────────────

def _check_brace_delta(find_snippet: str, replace_with: str) -> tuple[bool, str]:
    """replace_with must preserve the same net-open ``{}`` count as find_snippet.

    A mismatch means the replacement opens a scope it doesn't close (or vice
    versa), which corrupts the surrounding code structure.

    Only ``{}`` is checked here (not ``()`` or ``[]``) because asymmetric
    paren/bracket counts are common and intentional in partial expressions,
    method-call continuations, and array literals.
    """
    d_find = _net_brackets(find_snippet, "{", "}")
    d_rep  = _net_brackets(replace_with,  "{", "}")
    if d_find != d_rep:
        direction = "adds" if d_rep > d_find else "removes"
        count = abs(d_rep - d_find)
        return False, (
            f"brace imbalance: find_snippet net-open={d_find:+d}, "
            f"replace_with net-open={d_rep:+d} — the edit {direction} "
            f"{count} unmatched brace(s). Verify every '{{' is closed and "
            f"every '}}' has a matching '{{' within the replacement."
        )
    return True, "ok"


def _check_no_snippet_prepended(find_snippet: str, replace_with: str) -> tuple[bool, str]:
    """Detect the LLM artifact of appending the fix *after* the original code.

    When a model fails to understand find/replace semantics, it sometimes
    emits: ``replace_with = find_snippet + corrected_version``.  The
    resulting patch would double the original code.

    Uses normalized comparison (indentation-insensitive) so different
    indentation styles don't mask this pattern.
    """
    # Skip if snippet is trivially short (risk of false positive too high)
    if len(find_snippet.strip()) < 20 or len(replace_with) <= len(find_snippet):
        return True, "ok"

    def _norm(t: str) -> str:
        return "\n".join(" ".join(l.split()) for l in t.splitlines() if l.strip())

    snip_norm = _norm(find_snippet)
    rep_norm  = _norm(replace_with)

    # A significant length gap ensures we're not just seeing a prefix coincidence
    if rep_norm.startswith(snip_norm) and len(rep_norm) > len(snip_norm) + 10:
        return False, (
            "replace_with begins with find_snippet — the original code was not "
            "removed; new code was appended after it instead. "
            "replace_with must contain only the corrected code, "
            "not the original followed by the correction."
        )
    return True, "ok"


def _check_duplicate_fragments(replace_with: str) -> tuple[bool, str]:
    """Detect repeated code blocks inside replace_with.

    Slides a window of ``_BLOCK`` significant lines looking for the same
    sequence appearing twice.  Catches the artifact where the LLM includes
    both the original version and the fixed version of a block.

    Lines shorter than ``_MIN_LEN`` characters are excluded from the window
    (braces-only lines, blank lines, trivial one-liners — these can repeat
    legitimately in boilerplate).
    """
    _BLOCK   = 3
    _MIN_LEN = 12

    sig = [
        l.strip()
        for l in replace_with.splitlines()
        if l.strip() and len(l.strip()) >= _MIN_LEN
    ]

    if len(sig) < _BLOCK * 2:
        return True, "ok"

    seen: set[tuple[str, ...]] = set()
    for i in range(len(sig) - _BLOCK + 1):
        block = tuple(sig[i : i + _BLOCK])
        if block in seen:
            preview = sig[i][:60]
            return False, (
                f"duplicate code block in replace_with — the same {_BLOCK}-line "
                f"sequence appears more than once (near '{preview}'). "
                "This is typically a concatenation artifact where both the "
                "original and the corrected code were included."
            )
        seen.add(block)
    return True, "ok"


# Per-language impossible token patterns: (regex, human description)
_IMPOSSIBLE_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "all": [
        # Adjacent close/open block boundary with only whitespace between —
        # a strong signal that two separate code fragments were concatenated.
        (
            r"\}\s*\n\s*\{",
            "adjacent '}{' block boundaries (a '}' line immediately followed by a "
            "'{' line with nothing between) — likely two separate fragments joined",
        ),
    ],
    "java": [
        (r";;",                    "double semicolon ';;'"),
        (r"\bpublic\s+public\b",   "duplicate 'public' modifier"),
        (r"\bprivate\s+private\b", "duplicate 'private' modifier"),
        (r"\bstatic\s+static\b",   "duplicate 'static' modifier"),
        (r"\bvoid\s+void\b",       "duplicate 'void' type"),
        (r"\breturn\s+return\b",   "duplicate 'return' keyword"),
    ],
    "python": [
        (r"\bdef\s+def\b",       "duplicate 'def' keyword"),
        (r"\bclass\s+class\b",   "duplicate 'class' keyword"),
        (r"\breturn\s+return\b", "duplicate 'return' keyword"),
        (r"\bimport\s+import\b", "duplicate 'import' keyword"),
    ],
    "javascript": [
        (r";;",                      "double semicolon ';;'"),
        (r"\bfunction\s+function\b", "duplicate 'function' keyword"),
        (r"\bconst\s+const\b",       "duplicate 'const' keyword"),
        (r"\bvar\s+var\b",           "duplicate 'var' keyword"),
        (r"\breturn\s+return\b",     "duplicate 'return' keyword"),
    ],
}


def _check_impossible_tokens(replace_with: str, language: str) -> tuple[bool, str]:
    """Regex scan for token sequences that cannot occur in valid source code."""
    patterns = (
        _IMPOSSIBLE_PATTERNS.get("all", [])
        + _IMPOSSIBLE_PATTERNS.get(language, [])
    )
    for pattern, description in patterns:
        if re.search(pattern, replace_with):
            return False, f"impossible token pattern detected: {description}"
    return True, "ok"


def _check_python_ast(replace_with: str) -> tuple[bool, str]:
    """Parse replace_with as Python using the stdlib ``ast`` module.

    ``textwrap.dedent`` removes common leading indentation so a snippet
    extracted from inside a method body can be parsed as standalone code.

    This is the only check that uses a real parser — all others are
    heuristic.  For Python files it provides exact SyntaxError / IndentationError
    detection with zero false positives.
    """
    try:
        ast.parse(textwrap.dedent(replace_with))
        return True, "ok"
    except IndentationError as exc:
        return False, f"Python indentation error on line {exc.lineno}: {exc.msg}"
    except SyntaxError as exc:
        return False, f"Python syntax error on line {exc.lineno}: {exc.msg}"


# ─── Public API — standalone validation ──────────────────────────────────────

def validate_replacement_syntax(edit: FileEdit) -> tuple[bool, str]:
    """Run all applicable checks on edit.replace_with before patch application.

    Checks are run in order; the first failure is returned immediately
    (subsequent checks are not run, preserving a single clear error message).

    Returns (is_valid, human-readable note).
    """
    lang = _language(edit.path)

    # Brace delta — skip for Python (ast handles structure there)
    if lang != "python":
        ok, note = _check_brace_delta(edit.find_snippet, edit.replace_with)
        if not ok:
            logger.warning("patch_syntax [brace_delta] %s: %s", edit.path, note)
            return False, f"[brace_delta] {note}"

    ok, note = _check_no_snippet_prepended(edit.find_snippet, edit.replace_with)
    if not ok:
        logger.warning("patch_syntax [prepend] %s: %s", edit.path, note)
        return False, f"[prepend] {note}"

    ok, note = _check_duplicate_fragments(edit.replace_with)
    if not ok:
        logger.warning("patch_syntax [duplicate_fragment] %s: %s", edit.path, note)
        return False, f"[duplicate_fragment] {note}"

    ok, note = _check_impossible_tokens(edit.replace_with, lang)
    if not ok:
        logger.warning("patch_syntax [impossible_token] %s: %s", edit.path, note)
        return False, f"[impossible_token] {note}"

    if lang == "python":
        ok, note = _check_python_ast(edit.replace_with)
        if not ok:
            logger.warning("patch_syntax [python_ast] %s: %s", edit.path, note)
            return False, f"[python_ast] {note}"

    logger.debug("patch_syntax: %s passed all checks (lang=%s)", edit.path, lang)
    return True, "ok"


# ─── Recovery — [prepend] false-positive ─────────────────────────────────────

def recover_from_prepend(edit: FileEdit, content: str) -> tuple[bool, str]:
    """Attempt to rehabilitate an edit that failed the [prepend] check.

    The prepend detector fires whenever ``replace_with`` begins with
    ``find_snippet`` and is at least 10 chars longer. Two distinct LLM behaviours
    produce this pattern:

      (a) Append-after-match (intentional): the model wants to insert new code
          immediately after the matched block. Applying the patch produces a
          structurally valid file — the matched lines stay, new code is added
          after them. This is a legitimate edit shape.

      (b) Duplicate-then-correct (genuinely broken): the model emitted the
          original lines AND a corrected copy. The resulting file is corrupt.

    Distinguishing (a) from (b) statically is unreliable. Instead, we apply the
    patch and verify the *output* with the same per-language structural checks
    used elsewhere: Python AST parse for ``.py``, brace balance for braced
    languages. Output that parses cleanly is accepted as recovery.

    Returns (recovered_ok, human_readable_note). Caller should only invoke this
    after ``validate_replacement_syntax`` returns a ``[prepend]`` failure.
    """
    from issueops.tools.patch_match import find_snippet as _find_snippet

    # Anti-pattern: replace_with literally contains find_snippet more than
    # once. This means the model duplicated the original block (with or
    # without modifications inside the second copy). Even if the resulting
    # file parses, it doubles the buggy logic. Reject before applying.
    if edit.replace_with.count(edit.find_snippet) > 1:
        return False, (
            "recovery rejected: replace_with contains find_snippet more than "
            "once — the model duplicated the original block instead of inserting "
            "new code after it"
        )

    match = _find_snippet(edit.find_snippet, content)
    if match is None:
        return False, "recovery aborted: could not locate find_snippet in content"

    patched = content.replace(match.matched_text, edit.replace_with, 1)

    lang = _language(edit.path)

    if lang == "python":
        try:
            ast.parse(patched)
            return True, "post-apply Python AST parse ok"
        except IndentationError as exc:
            return False, f"post-apply Python indentation error on line {exc.lineno}: {exc.msg}"
        except SyntaxError as exc:
            return False, f"post-apply Python syntax error on line {exc.lineno}: {exc.msg}"

    if _ext(edit.path) in _BRACE_SCOPED_EXTS:
        balanced_ok, balance_note = check_brace_balance(content, patched, edit.path)
        if not balanced_ok:
            return False, f"post-apply check failed: {balance_note}"
        return True, "post-apply brace balance ok"

    # No structural post-apply check available for this language — accept
    # conservatively. The downstream diff-size check still bounds blast radius.
    return True, "no per-language post-apply check available; accepted on diff-size guard only"


# ─── Public API — post-apply balance check ───────────────────────────────────

def check_brace_balance(
    original: str,
    patched: str,
    path: str,
) -> tuple[bool, str]:
    """Verify the full patched file still has balanced braces.

    Only runs for brace-scoped languages (.java, .js, .ts, etc.).
    Skips if the original file was already unbalanced — we don't flag
    pre-existing structural issues.

    Uses raw character counts (no string-literal parsing) on the full file.
    This is safe in practice because format strings like ``"msg {}"`` and
    ``SLF4J "value {}"`` use symmetric ``{}`` and contribute 0 net delta.
    """
    if _ext(path) not in _BRACE_SCOPED_EXTS:
        return True, "ok"

    orig_delta    = original.count("{") - original.count("}")
    patched_delta = patched.count("{")  - patched.count("}")

    # If the original was already imbalanced, we can't attribute the imbalance
    # to this edit — skip the check to avoid a false rejection.
    if orig_delta != 0:
        logger.debug(
            "patch_syntax [brace_balance] %s: original already imbalanced (%+d) — skipped",
            path, orig_delta,
        )
        return True, "ok"

    if patched_delta != 0:
        direction = "extra '{'" if patched_delta > 0 else "extra '}'"
        logger.warning(
            "patch_syntax [brace_balance] %s: edit introduced imbalance (%+d)",
            path, patched_delta,
        )
        return False, (
            f"[brace_balance] edit introduced {abs(patched_delta)} unmatched brace(s) "
            f"({direction}) into the file — the patched file has {abs(patched_delta)} "
            f"more '{{' than '}}'. Verify the replacement closes all scopes it opens."
        )

    return True, "ok"
