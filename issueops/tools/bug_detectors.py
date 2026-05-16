"""Deterministic code bug detectors.

Given a file's content and the debug agent's output (root_cause, suspected_symbols,
repair_strategy), each detector locates a specific anti-pattern and produces a
FileEdit without LLM involvement.

Design goals:
  - Zero false positives preferred over catching every case.
  - find_snippet is ALWAYS copied verbatim from file content — no reconstruction.
  - Language-aware via file extension, framework-agnostic otherwise.
  - Detectors use root_cause / suspected_symbols to focus the search, not to infer
    things the code doesn't show. If the debug result didn't signal a pattern,
    the detector does not run.
  - Each detector is independent and stateless.

Public API:
  detect_in_file(path, content, root_cause, suspected_symbols, repair_strategy)
      → Optional[DetectorMatch]
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectorMatch:
    find_snippet: str    # verbatim text lifted from file content
    replace_with: str    # corrected replacement
    confidence: float
    description: str
    pattern_name: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ext(path: str) -> str:
    dot = path.rfind(".")
    return path[dot:].lower() if dot != -1 else ""


def _is_java_like(path: str) -> bool:
    return _ext(path) in {".java", ".kt", ".scala", ".cs"}


def _is_python(path: str) -> bool:
    return _ext(path) == ".py"


def _get_line(content: str, pos: int) -> str:
    """Return the full source line containing byte position pos."""
    start = content.rfind('\n', 0, pos) + 1
    end = content.find('\n', pos)
    return content[start:(end if end != -1 else len(content))]


def _is_comment_line(line: str) -> bool:
    s = line.lstrip()
    return s.startswith('//') or s.startswith('*') or s.startswith('#')


# ─── Detector 1: Java boxed-type == comparison ────────────────────────────────

def _detect_boxed_equality(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect == used to compare Java boxed / object types.

    Only fires when:
      - root_cause mentions == / equal / boxed (prevents noise on unrelated issues)
      - a suspected_symbol is found in a == expression in the file
    """
    if not _is_java_like(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("==", "equal", "boxed", "long", "integer", "object")):
        return None

    # Build search set: suspected symbols + boxed-typed fields discovered in the file
    search_syms = set(suspected_symbols)
    boxed_decl_re = re.compile(
        r'\b(?:Long|Integer|Double|Float|Short|Byte|Boolean|String)\s+(\w+)\b'
    )
    for m in boxed_decl_re.finditer(content):
        search_syms.add(m.group(1))

    if not search_syms:
        return None

    for sym in sorted(search_syms):
        if len(sym) < 2:
            continue

        # Capture full dotted-name on each side: (qualifier.)*sym == right
        # and left == (qualifier.)*sym.
        # Using named groups so we can reconstruct the exact expression.
        p1 = re.compile(
            r'(?<![=!<>])\b((?:\w+\.)*' + re.escape(sym) + r')\s*==\s*([\w.]+)(?!=)'
        )
        p2 = re.compile(
            r'(?<![=!<>])\b([\w.]+)\s*==\s*((?:\w+\.)*' + re.escape(sym) + r')(?!=)(?!\w)'
        )

        for pattern in (p1, p2):
            for m in pattern.finditer(content):
                line = _get_line(content, m.start())
                if _is_comment_line(line):
                    continue

                left, right = m.group(1).strip(), m.group(2).strip()
                old_expr = f'{left} == {right}'
                new_expr = f'Objects.equals({left}, {right})'

                if old_expr not in line:
                    continue

                find_snippet = line
                replace_with = line.replace(old_expr, new_expr, 1)
                if find_snippet == replace_with:
                    continue

                logger.debug("boxed_equality: %s → Objects.equals() in %s", old_expr, path)
                return DetectorMatch(
                    find_snippet=find_snippet,
                    replace_with=replace_with,
                    confidence=0.85,
                    description=f"'{sym}' compared with == — use Objects.equals() for boxed/object type",
                    pattern_name="boxed_equality",
                )

    return None


# ─── Detector 2: @Cacheable missing user identity in key ─────────────────────

def _detect_cacheable_missing_user_key(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect @Cacheable without userId/principal in the key — cross-user data bleed."""
    if not _is_java_like(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("cache", "user", "tenant", "session", "another user", "wrong user", "shared")):
        return None

    user_terms = {"userid", "principal", "username", "user_id", "tenantid", "tenant_id"}

    cacheable_re = re.compile(r'@Cacheable\s*\([^)]*\)', re.DOTALL)
    for m in cacheable_re.finditer(content):
        annotation = m.group(0)
        ann_lower = annotation.lower()

        # Skip if already scoped to a user
        if any(t in ann_lower for t in user_terms):
            continue

        # Must have an explicit key expression (value-only caches are skipped — riskier)
        # Capture the exact key=... fragment as it appears (with or without spaces)
        key_match = re.search(r'(key\s*=\s*"([^"]*)")', annotation)
        if not key_match:
            continue

        key_fragment = key_match.group(1)   # exact text, e.g. 'key="#status"'
        current_key = key_match.group(2)    # just the value, e.g. '#status'

        # Grab the full source line containing this annotation
        line_start = content.rfind('\n', 0, m.start()) + 1
        ann_end_line = content.find('\n', m.end())
        if ann_end_line == -1:
            ann_end_line = len(content)
        find_snippet = content[line_start:ann_end_line]

        if '#' in current_key:
            # SpEL expression: use single-quoted separator to avoid breaking outer double quotes
            new_key = current_key + " + '_' + #userId"
        else:
            # Literal key: wrap it in SpEL concat with userId
            new_key = "'" + current_key + "' + '_' + #userId"

        # Build replacement preserving the exact key=... format from the source
        new_key_fragment = key_fragment.replace(current_key, new_key, 1)
        replace_with = find_snippet.replace(key_fragment, new_key_fragment, 1)
        if find_snippet == replace_with:
            continue

        return DetectorMatch(
            find_snippet=find_snippet,
            replace_with=replace_with,
            confidence=0.80,
            description=(
                f"@Cacheable key '{current_key}' lacks user identity — "
                "cached result is shared across different users"
            ),
            pattern_name="cacheable_missing_user_key",
        )

    return None


# ─── Detector 3: Python mutable default argument ─────────────────────────────

def _detect_mutable_default_arg(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect Python def with mutable default argument (list/dict/set literal)."""
    if not _is_python(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("mutable", "default", "shared", "list", "dict", "[]", "{}", "argument")):
        return None

    mutable_re = re.compile(
        r'(def\s+\w+\s*\([^)]*?\b(\w+)\s*=\s*(?:\[\s*\]|\{\s*\}|list\(\)|dict\(\)|set\(\)))'
    )
    for m in mutable_re.finditer(content):
        param_name = m.group(2)
        line_start = content.rfind('\n', 0, m.start()) + 1
        line_end = content.find('\n', m.end())
        if line_end == -1:
            line_end = len(content)

        find_snippet = content[line_start:line_end]
        replace_with = re.sub(
            r'\b' + re.escape(param_name) + r'\s*=\s*(?:\[\s*\]|\{\s*\}|list\(\)|dict\(\)|set\(\))',
            f'{param_name}=None',
            find_snippet,
            count=1,
        )
        if find_snippet == replace_with:
            continue

        return DetectorMatch(
            find_snippet=find_snippet,
            replace_with=replace_with,
            confidence=0.88,
            description=f"Mutable default argument '{param_name}=[]' — shared state across calls",
            pattern_name="mutable_default_arg",
        )

    return None


# ─── Detector 4: Inverted ownership / guard condition ─────────────────────────

def _detect_inverted_condition(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect an inverted condition on a suspected symbol.

    Matches: if (!sym) or if (sym != other) or if (other != sym)
    when root_cause explicitly mentions inversion / negation / wrong direction.
    Only fires when a suspected symbol is involved to avoid false positives.
    """
    if not _is_java_like(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("inverted", "negated", "wrong condition", "!equals", "!= ", "backwards", "reversed")):
        return None

    for sym in suspected_symbols:
        if len(sym) < 3:
            continue

        # Pattern A: if (!sym)
        neg_re = re.compile(r'if\s*\(!\s*' + re.escape(sym) + r'\b')
        for m in neg_re.finditer(content):
            line = _get_line(content, m.start())
            if _is_comment_line(line):
                continue
            find_snippet = line
            replace_with = line.replace('!' + sym, sym, 1)
            if find_snippet == replace_with:
                old = '! ' + sym
                replace_with = line.replace(old, sym, 1)
            if find_snippet != replace_with:
                return DetectorMatch(
                    find_snippet=find_snippet,
                    replace_with=replace_with,
                    confidence=0.75,
                    description=f"Inverted condition '!{sym}' — guard is checking the wrong polarity",
                    pattern_name="inverted_condition",
                )

        # Pattern B: (qualifier.)?sym != other  or  other != (qualifier.)?sym
        p_neq1 = re.compile(r'(?<![=!<>])\b((?:\w+\.)*' + re.escape(sym) + r')\s*!=\s*([\w.]+)(?!=)')
        p_neq2 = re.compile(r'(?<![=!<>])\b([\w.]+)\s*!=\s*((?:\w+\.)*' + re.escape(sym) + r')(?!=)(?!\w)')

        for neq_re in (p_neq1, p_neq2):
            for m in neq_re.finditer(content):
                line = _get_line(content, m.start())
                if _is_comment_line(line):
                    continue
                left, right = m.group(1).strip(), m.group(2).strip()
                old_expr = f'{left} != {right}'
                new_expr = f'{left} == {right}'
                if old_expr not in line:
                    continue
                find_snippet = line
                replace_with = line.replace(old_expr, new_expr, 1)
                if find_snippet != replace_with:
                    return DetectorMatch(
                        find_snippet=find_snippet,
                        replace_with=replace_with,
                        confidence=0.72,
                        description=f"Condition '{old_expr}' may be inverted — should be ==",
                        pattern_name="inverted_condition",
                    )

    return None


# ─── Detector 5: JPA sort field name mismatch ────────────────────────────────

def _detect_jpa_sort_mismatch(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect wrong field name string in JPA sort/query expressions.

    Only fires when:
      - .java or .kt file
      - root_cause mentions property/sort/field/JPA related keywords
      - a suspected_symbol (camelCase, lowercase start) appears in a sort expression
    """
    if not _is_java_like(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("property", "found for type", "sort", "no property", "jpa", "field")):
        return None

    # Pattern for sort expressions containing string literals
    sort_patterns = [
        re.compile(r'Sort\.by\s*\(\s*"([^"]+)"\s*\)'),
        re.compile(r'Sort\.by\s*\(\s*Sort\.Order\.\w+\s*\(\s*"([^"]+)"\s*\)\s*\)'),
        re.compile(r'PageRequest\.of\s*\([^)]*Sort\.by\s*\(\s*"([^"]+)"\s*\)'),
        re.compile(r'@SortDefault\s*\([^)]*sort\s*=\s*"([^"]+)"'),
        re.compile(r'\.findBy\w*OrderBy(\w+)\s*\('),
    ]

    for sym in suspected_symbols:
        if not sym or not sym[0].islower() or len(sym) < 2:
            continue

        for pattern in sort_patterns:
            for m in pattern.finditer(content):
                field_in_sort = m.group(1) if m.lastindex and m.lastindex >= 1 else None
                if field_in_sort is None:
                    continue

                line = _get_line(content, m.start())
                if _is_comment_line(line):
                    continue

                # The wrong field must be in the expression and the correct sym must differ
                if field_in_sort == sym:
                    continue  # already correct

                # Only fire when the wrong field appears in sort and sym is a plausible fix
                old_str = f'"{field_in_sort}"'
                new_str = f'"{sym}"'
                if old_str not in line:
                    continue

                find_snippet = line
                replace_with = line.replace(old_str, new_str, 1)
                if find_snippet == replace_with:
                    continue

                logger.debug(
                    "jpa_sort_mismatch: '%s' → '%s' in sort expression in %s",
                    field_in_sort, sym, path,
                )
                return DetectorMatch(
                    find_snippet=find_snippet,
                    replace_with=replace_with,
                    confidence=0.82,
                    description=(
                        f"JPA sort expression uses field name '{field_in_sort}' — "
                        f"should be '{sym}' based on the entity class"
                    ),
                    pattern_name="jpa_sort_mismatch",
                )

    return None


# ─── Detector 6: @Cacheable without key expression (shared for all callers) ──

def _detect_hardcoded_cache_value(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect @Cacheable with literal name but no key= expression.

    Without key=, all callers share the same cached result regardless of
    method arguments — classic cross-user data bleed.
    """
    if not _is_java_like(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("cache", "shared", "user", "tenant")):
        return None

    # Match @Cacheable("name") or @Cacheable(value="name") without key=
    # Pattern A: @Cacheable("literal")
    pattern_a = re.compile(r'@Cacheable\s*\(\s*"([^"]+)"\s*\)')
    # Pattern B: @Cacheable(value="literal") — no key= present
    pattern_b = re.compile(r'@Cacheable\s*\(\s*value\s*=\s*"([^"]+)"\s*\)')

    for pattern in (pattern_a, pattern_b):
        for m in pattern.finditer(content):
            annotation = m.group(0)
            # Already has a key= — skip
            if 'key=' in annotation or 'key =' in annotation:
                continue

            cache_name = m.group(1)
            line = _get_line(content, m.start())
            if _is_comment_line(line):
                continue

            # Build replacement: add key="#userId"
            if pattern is pattern_a:
                # @Cacheable("name") → @Cacheable(value="name", key="#userId")
                old_fragment = f'@Cacheable("{cache_name}")'
                new_fragment = f'@Cacheable(value="{cache_name}", key="#userId")'
            else:
                # @Cacheable(value="name") → @Cacheable(value="name", key="#userId")
                old_fragment = f'value="{cache_name}"'
                new_fragment = f'value="{cache_name}", key="#userId"'

            find_snippet = line
            replace_with = line.replace(old_fragment, new_fragment, 1)
            if find_snippet == replace_with:
                continue

            logger.debug(
                "hardcoded_cache_value: @Cacheable('%s') missing key= in %s",
                cache_name, path,
            )
            return DetectorMatch(
                find_snippet=find_snippet,
                replace_with=replace_with,
                confidence=0.78,
                description=(
                    f"@Cacheable(\"{cache_name}\") has no key= expression — "
                    "all callers share the same cached result (cross-user data bleed)"
                ),
                pattern_name="hardcoded_cache_value",
            )

    return None


# ─── Detector 7: Python mutation during iteration ─────────────────────────────

def _detect_mutation_during_iteration(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect list mutation inside a for loop over that same list.

    Catches: for x in items: items.remove(x)  or  items.append(...)
    Fix: copy the list — for x in list(items): items.remove(x)
    """
    if not _is_python(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("mutation", "modify", "remove", "iteration",
                                    "concurrent modification", "list changed")):
        return None

    lines = content.splitlines()
    for i, line in enumerate(lines):
        # Detect: for VAR in LIST:
        for_m = re.match(r'^(\s*)for\s+(\w+)\s+in\s+(\w+)\s*:', line)
        if not for_m:
            continue

        indent = for_m.group(1)
        loop_list = for_m.group(3)
        body_indent = indent + "    "

        # Look at next 10 lines for mutation of the same list
        for j in range(i + 1, min(i + 11, len(lines))):
            body_line = lines[j]
            if not body_line.startswith(body_indent):
                break  # left the loop body

            # Check for list mutation: LIST.remove, LIST.append, del LIST[
            if (
                re.search(r'\b' + re.escape(loop_list) + r'\.(remove|append|pop|insert|extend)\s*\(', body_line)
                or re.search(r'del\s+' + re.escape(loop_list) + r'\s*\[', body_line)
            ):
                # Fix the for line: wrap list name in list()
                old_for = line
                new_for = re.sub(
                    r'(for\s+\w+\s+in\s+)' + re.escape(loop_list) + r'\s*:',
                    lambda mo: mo.group(1) + f'list({loop_list}):',
                    line,
                    count=1,
                )
                if old_for == new_for:
                    continue

                logger.debug(
                    "mutation_during_iteration: '%s' mutated inside 'for … in %s' in %s",
                    loop_list, loop_list, path,
                )
                return DetectorMatch(
                    find_snippet=old_for,
                    replace_with=new_for,
                    confidence=0.85,
                    description=(
                        f"List '{loop_list}' is mutated inside a 'for … in {loop_list}' loop — "
                        "iterate over a copy with list() to avoid RuntimeError"
                    ),
                    pattern_name="mutation_during_iteration",
                )

    return None


# ─── Detector 8: Python division by zero risk ─────────────────────────────────

def _detect_division_by_zero_risk(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect suspected symbol used as denominator without a zero-guard."""
    if not _is_python(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("division", "zero", "denominator", "normalize", "zerodivisionerror")):
        return None

    for sym in suspected_symbols:
        if not sym or len(sym) < 2:
            continue

        # Pattern: result = expr / sym  (sym in denominator position)
        pattern = re.compile(
            r'^(\s*\w[\w.]*\s*=\s*[\w\s.()+\-*]*)\s*/\s*(' + re.escape(sym) + r')\s*$',
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            line = _get_line(content, m.start())
            if _is_comment_line(line):
                continue

            # Skip if there's already a zero-guard on the same or previous line
            start_pos = content.rfind('\n', 0, m.start()) + 1
            prev_line_start = content.rfind('\n', 0, start_pos - 1) + 1
            prev_line = content[prev_line_start:start_pos].strip()
            if sym in prev_line and ('!= 0' in prev_line or '> 0' in prev_line or 'if' in prev_line):
                continue

            # Build the safe replacement: ternary guard
            stripped = line.rstrip()
            # Match the assignment target and expression
            assign_m = re.match(
                r'^(\s*)(\w[\w.]*)\s*=\s*(.+)\s*/\s*' + re.escape(sym) + r'\s*$',
                stripped,
            )
            if assign_m is None:
                continue

            leading = assign_m.group(1)
            target = assign_m.group(2)
            numerator = assign_m.group(3).strip()
            old_line = line
            new_line = (
                f"{leading}{target} = {numerator} / {sym} if {sym} != 0 else 0\n"
            )

            logger.debug(
                "division_by_zero_risk: '%s' used as denominator without guard in %s",
                sym, path,
            )
            return DetectorMatch(
                find_snippet=old_line,
                replace_with=new_line,
                confidence=0.78,
                description=(
                    f"'{sym}' used as denominator without a zero-guard — "
                    f"use ternary: '{target} = {numerator} / {sym} if {sym} != 0 else 0'"
                ),
                pattern_name="division_by_zero_risk",
            )

    return None


# ─── Detector 9: JavaScript/TypeScript loose equality (== / !=) ───────────────

def _detect_js_loose_equality(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect loose == / != in JS/TS files where a suspected symbol is involved."""
    ext = _ext(path)
    if ext not in {".js", ".ts", ".jsx", ".tsx"}:
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("==", "equality", "type coercion", "strict", "===", "loose")):
        return None

    for sym in suspected_symbols:
        if not sym or len(sym) < 2:
            continue

        # Look for sym == expr or expr == sym (but not === or !==)
        # Negative lookbehind/lookahead to avoid matching === or !==
        patterns = [
            re.compile(r'(?<![=!<>])\b' + re.escape(sym) + r'\s*(?<!=)==(?!=)\s*[\w"\'`.]'),
            re.compile(r'[\w"\'`.]\s*(?<!=)==(?!=)\s*\b' + re.escape(sym) + r'\b'),
            re.compile(r'(?<![=!<>])\b' + re.escape(sym) + r'\s*!=(?!=)\s*[\w"\'`.]'),
            re.compile(r'[\w"\'`.]\s*!=(?!=)\s*\b' + re.escape(sym) + r'\b'),
        ]

        for pat in patterns:
            for m in pat.finditer(content):
                line = _get_line(content, m.start())
                if _is_comment_line(line):
                    continue

                # Replace == with === and != with !==
                # Only replace the non-strict operators, not existing === / !==
                old_line = line
                new_line = re.sub(r'(?<![=!<>])==(?!=)', '===', line)
                new_line = re.sub(r'!=(?!=)', '!==', new_line)

                if old_line == new_line:
                    continue

                logger.debug(
                    "js_loose_equality: replaced == with === in %s for symbol '%s'",
                    path, sym,
                )
                return DetectorMatch(
                    find_snippet=old_line,
                    replace_with=new_line,
                    confidence=0.82,
                    description=(
                        f"Loose equality operator == used with '{sym}' — "
                        "use === for strict comparison to avoid type coercion bugs"
                    ),
                    pattern_name="js_loose_equality",
                )

    return None


# ─── Detector 10: Python None equality anti-pattern ──────────────────────────

def _detect_none_equality_python(
    content: str,
    path: str,
    root_cause: str,
    suspected_symbols: list[str],
) -> Optional[DetectorMatch]:
    """Detect x == None or x != None — should use 'is None' / 'is not None'."""
    if not _is_python(path):
        return None

    rc = root_cause.lower()
    if not any(kw in rc for kw in ("is none", "== none", "identity", "singleton", "none")):
        return None

    for sym in suspected_symbols:
        if not sym or len(sym) < 2:
            continue

        patterns = [
            (
                re.compile(r'\b' + re.escape(sym) + r'\s*==\s*None\b'),
                lambda line, s=sym: re.sub(
                    r'\b' + re.escape(s) + r'\s*==\s*None\b',
                    f'{s} is None',
                    line,
                    count=1,
                ),
            ),
            (
                re.compile(r'\b' + re.escape(sym) + r'\s*!=\s*None\b'),
                lambda line, s=sym: re.sub(
                    r'\b' + re.escape(s) + r'\s*!=\s*None\b',
                    f'{s} is not None',
                    line,
                    count=1,
                ),
            ),
            (
                re.compile(r'\bNone\s*==\s*' + re.escape(sym) + r'\b'),
                lambda line, s=sym: re.sub(
                    r'\bNone\s*==\s*' + re.escape(s) + r'\b',
                    f'{s} is None',
                    line,
                    count=1,
                ),
            ),
            (
                re.compile(r'\bNone\s*!=\s*' + re.escape(sym) + r'\b'),
                lambda line, s=sym: re.sub(
                    r'\bNone\s*!=\s*' + re.escape(s) + r'\b',
                    f'{s} is not None',
                    line,
                    count=1,
                ),
            ),
        ]

        for pat, replacer in patterns:
            for m in pat.finditer(content):
                line = _get_line(content, m.start())
                if _is_comment_line(line):
                    continue

                old_line = line
                new_line = replacer(line)
                if old_line == new_line:
                    continue

                logger.debug(
                    "none_equality_python: '%s' compared with == None in %s",
                    sym, path,
                )
                return DetectorMatch(
                    find_snippet=old_line,
                    replace_with=new_line,
                    confidence=0.85,
                    description=(
                        f"'{sym}' compared with == None / != None — "
                        "use 'is None' / 'is not None' for identity check"
                    ),
                    pattern_name="none_equality_python",
                )

    return None


# ─── Public API ───────────────────────────────────────────────────────────────

_DETECTORS = [
    _detect_boxed_equality,
    _detect_cacheable_missing_user_key,
    _detect_mutable_default_arg,
    _detect_inverted_condition,
    _detect_jpa_sort_mismatch,
    _detect_hardcoded_cache_value,
    _detect_mutation_during_iteration,
    _detect_division_by_zero_risk,
    _detect_js_loose_equality,
    _detect_none_equality_python,
]

_MIN_CONFIDENCE = 0.75


def detect_in_file(
    path: str,
    content: str,
    root_cause: str,
    suspected_symbols: list[str],
    repair_strategy: str = "",
) -> Optional[DetectorMatch]:
    """Run all deterministic detectors against a single file.

    Returns the first DetectorMatch with confidence >= 0.75, or None.

    Detectors use root_cause and suspected_symbols to decide whether to scan —
    they do not blindly search for every anti-pattern in every file.
    """
    combined_context = f"{root_cause} {repair_strategy}"

    for detector in _DETECTORS:
        try:
            match = detector(content, path, combined_context, suspected_symbols)
            if match is not None and match.confidence >= _MIN_CONFIDENCE:
                logger.info(
                    "bug_detector [%s]: matched in %s (confidence=%.2f)",
                    match.pattern_name, path, match.confidence,
                )
                return match
        except Exception as exc:
            logger.warning(
                "bug_detector: %s raised %s: %s",
                detector.__name__, type(exc).__name__, exc,
            )

    return None
