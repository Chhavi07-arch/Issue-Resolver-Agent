"""Whitespace-tolerant snippet matching for patch validation and application.

Four progressive levels, each attempted before falling to the next:

  Level 1 — exact, line-boundary:
      verbatim substring match, but only where the snippet starts at position 0
      or immediately after a newline.  Prevents a 2-space snippet from matching
      inside an 8-space indented line (which would corrupt indentation on apply).

  Level 2 — normalized:
      strip leading/trailing whitespace per line, collapse internal whitespace
      runs to a single space, then require exact line-sequence equality.
      Handles indentation changes, trailing spaces, and CRLF vs LF.

  Level 3 — fuzzy:
      difflib SequenceMatcher on the CHARACTER-level joined normalized text of
      sliding content windows.  Character-level (not line-level) comparison
      correctly scores small snippets where one line has minor token differences
      (e.g. "a==b" vs "a == b") without inflating scores on structurally
      different code.

      Only attempted when the snippet has >= _FUZZY_MIN_LINES non-blank lines.
      Only accepted when ratio >= _FUZZY_MIN_RATIO.

  Level 4 — short-snippet fuzzy:
      For 1-2 line snippets that cannot use Level 3 (too short for multi-line
      windowing), finds the single line or consecutive pair with the best
      difflib ratio against the snippet.  Threshold _SHORT_FUZZY_MIN_RATIO.

Safety invariants:
  - Levels 1 and 2 require structural equality — no approximation.
  - Level 3 uses a conservative ratio threshold plus a minimum-lines guard.
  - Level 4 uses a separate (also conservative) threshold for short snippets.
  - All levels return the ORIGINAL (unnormalized) matched text so callers
    can do a literal str.replace() that never introduces normalization
    artifacts into the file.
  - Only the first match is ever returned (surgical edits stay surgical).
"""

import difflib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Tuning constants ────────────────────────────────────────────────────────

# Fuzzy match: reject unless normalized character similarity meets this threshold.
# 0.85 is conservative — two structurally different code blocks rarely reach it.
_FUZZY_MIN_RATIO: float = 0.85

# Fuzzy match: only attempted when snippet has at least this many non-blank lines.
# Lowered from 3 to 2 so 2-line snippets can also use Level 3.
_FUZZY_MIN_LINES: int = 2

# Fuzzy match: try window sizes snippet_lines ± this slack to absorb blank-line variance.
_FUZZY_SLACK: int = 2

# Short-snippet fuzzy (Level 4): ratio threshold for 1-2 line snippets.
_SHORT_FUZZY_MIN_RATIO: float = 0.82


# ─── Result type ─────────────────────────────────────────────────────────────

class SnippetMatch:
    __slots__ = ("matched_text", "start_line", "end_line", "method")

    def __init__(self, matched_text: str, start_line: int, end_line: int, method: str):
        self.matched_text = matched_text
        self.start_line = start_line
        self.end_line = end_line
        self.method = method

    def __repr__(self) -> str:
        return (
            f"SnippetMatch(method={self.method!r}, lines={self.start_line}-{self.end_line},"
            f" len={len(self.matched_text)})"
        )


# ─── Normalization helpers ────────────────────────────────────────────────────

def _norm(line: str) -> str:
    """Strip leading/trailing whitespace and collapse internal runs to one space."""
    return " ".join(line.split())


# ─── Public API ──────────────────────────────────────────────────────────────

def find_snippet(
    snippet: str,
    content: str,
    anchor_symbols: list[str] | None = None,
) -> Optional[SnippetMatch]:
    """Find snippet in content with progressive whitespace tolerance.

    Returns a SnippetMatch whose .matched_text is the verbatim substring of
    content corresponding to the snippet, or None if no match is found at any
    level.

    Use matched_text as the first argument to str.replace() so edits always
    target original text, never normalized text.

    Args:
        snippet: The code block to locate.
        content: Full file content to search in.
        anchor_symbols: Optional list of symbol names.  When provided, the
            search first focuses on lines near those symbols (nearest-symbol
            fallback) before falling through to full-file scanning.
    """
    if not snippet or not content:
        return None

    # ── Level 1: exact at line boundary ──────────────────────────────────────
    match = _exact_match(snippet, content)
    if match is not None:
        return match

    # ── Level 2: whitespace-normalized line match ─────────────────────────────
    match = _normalized_match(snippet, content)
    if match is not None:
        logger.info(
            "[PATCH_MATCH] normalized match at lines %d-%d "
            "(exact match failed — whitespace difference)",
            match.start_line, match.end_line,
        )
        return match

    # ── Nearest-symbol pre-scan (anchor_symbols) ─────────────────────────────
    # When caller provides anchor symbols, try to find the snippet near those
    # symbol definitions first.  This narrows the search window and increases
    # confidence for short snippets that appear in multiple locations.
    if anchor_symbols:
        match = _anchor_symbol_match(snippet, content, anchor_symbols)
        if match is not None:
            return match

    # ── Level 3: fuzzy (character-level difflib) ──────────────────────────────
    match = _fuzzy_match(snippet, content)
    if match is not None:
        return match

    # ── Level 4: short-snippet fuzzy (1-2 line snippets) ─────────────────────
    match = _short_snippet_fuzzy_match(snippet, content)
    return match


# ─── Level 1: exact, line-boundary ───────────────────────────────────────────

def _exact_match(snippet: str, content: str) -> Optional[SnippetMatch]:
    """Return first occurrence of snippet that starts at position 0 or after \\n.

    The line-boundary requirement prevents a less-indented snippet from matching
    mid-way through a more-indented line (e.g. '  if ...' matching inside
    '        if ...' at offset 6), which would corrupt indentation on apply.
    """
    search_from = 0
    while True:
        idx = content.find(snippet, search_from)
        if idx == -1:
            return None
        if idx == 0 or content[idx - 1] == "\n":
            pre = content[:idx]
            start = pre.count("\n")
            end = start + snippet.count("\n") + (0 if snippet.endswith("\n") else 1)
            return SnippetMatch(snippet, start, end, "exact")
        # Occurrence is mid-line — try next
        search_from = idx + 1


# ─── Level 2: normalized ─────────────────────────────────────────────────────

def _normalized_match(snippet: str, content: str) -> Optional[SnippetMatch]:
    """Exact match on per-line normalized text.

    Leading/trailing blank lines of the snippet are trimmed before comparison —
    they carry no structural meaning and LLMs frequently omit/add them.
    """
    content_lines_raw = content.splitlines(keepends=True)
    norm_content = [_norm(l) for l in content.splitlines()]

    snip_norm = [_norm(l) for l in snippet.splitlines()]

    # Trim surrounding blank lines from snippet
    lo = 0
    while lo < len(snip_norm) and not snip_norm[lo]:
        lo += 1
    hi = len(snip_norm)
    while hi > lo and not snip_norm[hi - 1]:
        hi -= 1
    trimmed = snip_norm[lo:hi]

    if not trimmed:
        return None

    n = len(trimmed)
    for i in range(len(norm_content) - n + 1):
        if norm_content[i : i + n] == trimmed:
            matched_text = "".join(content_lines_raw[i : i + n])
            return SnippetMatch(matched_text, i, i + n, "normalized")

    return None


# ─── Level 3: fuzzy (character-level) ────────────────────────────────────────

def _fuzzy_match(snippet: str, content: str) -> Optional[SnippetMatch]:
    """Character-level difflib fuzzy match.

    Joins normalized non-blank lines into a single string for comparison.
    Character-level scoring (vs line-level) correctly handles small snippets
    where one line has minor token differences (e.g. spacing around operators)
    — a 1-line change in a 4-line snippet registers ~0.95 at char level but
    only ~0.75 at list-element level.

    A sliding window of size snippet_lines ± _FUZZY_SLACK is used so that
    minor blank-line count differences between LLM output and actual file
    do not block a valid match.
    """
    snip_nb = [_norm(l) for l in snippet.splitlines() if l.strip()]

    if len(snip_nb) < _FUZZY_MIN_LINES:
        logger.debug(
            "patch_match: fuzzy skipped — only %d non-blank lines in snippet (min %d)",
            len(snip_nb), _FUZZY_MIN_LINES,
        )
        return None

    snip_joined = "\n".join(snip_nb)

    content_lines_raw = content.splitlines(keepends=True)
    content_norm = [_norm(l) for l in content.splitlines()]
    total_lines = len(content_norm)
    base_n = len(snippet.splitlines())  # expected window height

    best_ratio = 0.0
    best_start = -1
    best_end = -1

    seen_deltas: set[int] = set()
    for slack in range(_FUZZY_SLACK + 1):
        for delta in ([0] if slack == 0 else [slack, -slack]):
            if delta in seen_deltas:
                continue
            seen_deltas.add(delta)
            win_size = base_n + delta
            if win_size <= 0 or win_size > total_lines:
                continue
            for i in range(total_lines - win_size + 1):
                window_nb = [l for l in content_norm[i : i + win_size] if l]
                window_joined = "\n".join(window_nb)
                ratio = difflib.SequenceMatcher(None, snip_joined, window_joined).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                    best_end = i + win_size

    if best_ratio >= _FUZZY_MIN_RATIO and best_start >= 0:
        matched_text = "".join(content_lines_raw[best_start:best_end])
        logger.info(
            "[PATCH_MATCH] fuzzy match at lines %d-%d, ratio=%.3f (threshold=%.2f)",
            best_start, best_end, best_ratio, _FUZZY_MIN_RATIO,
        )
        return SnippetMatch(matched_text, best_start, best_end, "fuzzy")

    logger.debug(
        "[PATCH_MATCH] no Level-3 match — best fuzzy ratio=%.3f (threshold=%.2f)",
        best_ratio, _FUZZY_MIN_RATIO,
    )
    return None


# ─── Nearest-symbol anchor pre-scan ──────────────────────────────────────────

def _anchor_symbol_match(
    snippet: str, content: str, anchor_symbols: list[str]
) -> Optional[SnippetMatch]:
    """Search near lines containing anchor symbols before full-file scan.

    Tries exact and normalized matching restricted to a ±50-line window around
    each anchor symbol occurrence.  Returns None if no match is found in any
    window (caller falls through to full-file fuzzy).
    """
    content_lines_raw = content.splitlines(keepends=True)
    content_lines = content.splitlines()

    for sym in anchor_symbols:
        if not sym:
            continue
        for i, line in enumerate(content_lines):
            if sym not in line:
                continue
            # Narrow window around this symbol occurrence
            win_start = max(0, i - 50)
            win_end = min(len(content_lines), i + 50)
            window_content = "".join(content_lines_raw[win_start:win_end])

            # Try exact then normalized in the window
            m = _exact_match(snippet, window_content)
            if m is not None:
                # Adjust line numbers to be file-relative
                adj = SnippetMatch(
                    m.matched_text,
                    m.start_line + win_start,
                    m.end_line + win_start,
                    "anchor_exact",
                )
                logger.info(
                    "[PATCH_MATCH] anchor_exact match near symbol '%s' at lines %d-%d",
                    sym, adj.start_line, adj.end_line,
                )
                return adj

            m = _normalized_match(snippet, window_content)
            if m is not None:
                adj = SnippetMatch(
                    m.matched_text,
                    m.start_line + win_start,
                    m.end_line + win_start,
                    "anchor_normalized",
                )
                logger.info(
                    "[PATCH_MATCH] anchor_normalized match near symbol '%s' at lines %d-%d",
                    sym, adj.start_line, adj.end_line,
                )
                return adj

    return None


# ─── Level 4: short-snippet fuzzy (1-2 non-blank lines) ──────────────────────

def _short_snippet_fuzzy_match(snippet: str, content: str) -> Optional[SnippetMatch]:
    """Level 4 fuzzy match for 1-2 line snippets.

    Level 3 requires _FUZZY_MIN_LINES non-blank lines and uses multi-line
    windows.  For 1-2 line snippets that still have no match after Levels 1-3,
    this level:
      - 1-line snippets: scans every line in content, picks best difflib ratio.
      - 2-line snippets: scans every consecutive pair, picks best difflib ratio.

    Threshold: _SHORT_FUZZY_MIN_RATIO (0.82 by default).
    """
    snip_nb = [_norm(l) for l in snippet.splitlines() if l.strip()]
    n_lines = len(snip_nb)

    if n_lines == 0 or n_lines > 2:
        # Level 4 only handles 1-2 line snippets; longer handled by Level 3
        return None

    snip_joined = "\n".join(snip_nb)
    content_lines_raw = content.splitlines(keepends=True)
    content_norm = [_norm(l) for l in content.splitlines()]
    total = len(content_norm)

    best_ratio = 0.0
    best_start = -1

    if n_lines == 1:
        for i, line in enumerate(content_norm):
            if not line:
                continue
            ratio = difflib.SequenceMatcher(None, snip_joined, line).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
        best_end = best_start + 1
    else:  # n_lines == 2
        for i in range(total - 1):
            pair_nb = [l for l in content_norm[i : i + 2] if l]
            if not pair_nb:
                continue
            pair_joined = "\n".join(pair_nb)
            ratio = difflib.SequenceMatcher(None, snip_joined, pair_joined).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
        best_end = best_start + 2

    if best_ratio >= _SHORT_FUZZY_MIN_RATIO and best_start >= 0:
        matched_text = "".join(content_lines_raw[best_start:best_end])
        logger.info(
            "[PATCH_MATCH] short_fuzzy match at lines %d-%d, ratio=%.3f "
            "(threshold=%.2f, snippet_lines=%d)",
            best_start, best_end, best_ratio, _SHORT_FUZZY_MIN_RATIO, n_lines,
        )
        return SnippetMatch(matched_text, best_start, best_end, "short_fuzzy")

    logger.debug(
        "[PATCH_MATCH] Level-4 short_fuzzy: no match — best ratio=%.3f "
        "(threshold=%.2f, snippet_lines=%d)",
        best_ratio, _SHORT_FUZZY_MIN_RATIO, n_lines,
    )
    return None
