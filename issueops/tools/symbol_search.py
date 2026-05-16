"""Symbol extraction and function-context retrieval.

Used by the Fix PR agent to locate specific code sections based on identifiers
produced by the Debug agent (function names, class names, field names).

No external dependencies — regex only.
"""

import re
from typing import Optional

_IDENTIFIER_RE = re.compile(
    r'\b([A-Z][a-zA-Z0-9]{2,}|[a-z][a-zA-Z0-9_]{2,}|[A-Z][A-Z0-9_]{3,})\b'
)

_STOP_WORDS = frozenset({
    "the", "and", "for", "this", "that", "with", "from", "when", "where",
    "which", "have", "been", "will", "not", "can", "but", "are", "was",
    "has", "its", "any", "all", "one", "two", "etc", "also", "null", "none",
    "true", "false", "new", "return", "class", "void", "int", "long", "string",
    "list", "map", "set", "get", "put", "add", "remove", "size", "type",
    "public", "private", "static", "final", "import", "package", "extends",
    "super", "throw", "throws", "try", "catch", "switch", "case", "break",
    "while", "def", "async", "await", "function", "const", "let", "var",
    "export", "interface", "enum", "object", "value", "result", "error",
    "data", "item", "user", "name", "time", "date", "code", "file", "path",
    "text", "note", "info", "issue", "fix", "bug", "test", "true", "false",
})


def extract_symbols_from_text(text: str) -> list[str]:
    """Extract likely code identifiers from prose text (issue body, root cause).

    Returns identifiers in order of first appearance, deduplicated.
    Filters stop words and tokens shorter than 4 chars.
    """
    symbols: list[str] = []
    seen: set[str] = set()
    for m in _IDENTIFIER_RE.finditer(text):
        sym = m.group(1)
        if sym.lower() in _STOP_WORDS:
            continue
        if len(sym) < 4:
            continue
        if sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    return symbols


def find_definition_line(symbol: str, content: str) -> Optional[int]:
    """Return 1-based line number of the most likely definition site of symbol.

    Searches for function/method/class/field definitions before plain usages.
    Returns None if not found anywhere in the file.
    """
    lines = content.splitlines()

    # Priority 1: keyword-introduced definition (def, class, function, method signature)
    def_re = re.compile(
        r'\b(?:def|class|function|interface|enum|struct|record)\s+' + re.escape(symbol) + r'\b'
        r'|'
        # JS/TS: const name = (async )? (...) =>
        r'\b(?:const|let|var)\s+' + re.escape(symbol) + r'\s*='
        r'|'
        # Java-style: ReturnType name( or modifiers ReturnType name(
        r'\b' + re.escape(symbol) + r'\s*\([^;{]*(?:\{|$)'
    )

    # Priority 2: any occurrence
    any_re = re.compile(r'\b' + re.escape(symbol) + r'\b')

    first_occurrence: Optional[int] = None
    for i, line in enumerate(lines):
        if def_re.search(line):
            return i + 1
        if first_occurrence is None and any_re.search(line):
            first_occurrence = i + 1

    return first_occurrence


def extract_context_around_line(
    content: str,
    target_line: int,
    max_lines: int = 35,
) -> tuple[str, int, int]:
    """Extract code context around target_line (1-based).

    For brace-scoped files: walks back to the nearest function/class start
    and forward to the matching close brace.  Capped at max_lines.

    Returns (text, start_line_1based, end_line_1based).
    """
    lines = content.splitlines()
    total = len(lines)
    idx = max(0, min(target_line - 1, total - 1))  # 0-based

    # Walk backwards to the function/class/method start (up to 25 lines)
    start_idx = max(0, idx - 3)
    for i in range(idx, max(-1, idx - 25), -1):
        line = lines[i]
        # Python def/class
        if re.match(r'^\s*(?:async\s+)?def\s+\w+', line):
            start_idx = i
            break
        if re.match(r'^\s*class\s+\w+', line):
            start_idx = i
            break
        # Java/C# annotations, access modifiers
        if re.match(r'^\s*@\w+', line):
            start_idx = i
            break
        if re.match(r'^\s*(?:public|private|protected|static|final|synchronized|override|virtual)\b', line):
            start_idx = i
            break
        # Generic: line with ( that ends with { — method signature
        stripped = line.strip()
        if '(' in stripped and stripped.endswith('{'):
            start_idx = i
            break

    # Determine end: track brace depth from start_idx, cap at max_lines
    end_idx = min(total - 1, start_idx + max_lines - 1)

    if '{' in content:
        depth = 0
        in_block = False
        for i in range(start_idx, min(total, start_idx + max_lines + 15)):
            for ch in lines[i]:
                if ch == '{':
                    depth += 1
                    in_block = True
                elif ch == '}':
                    depth -= 1
            if in_block and depth <= 0:
                end_idx = min(i, start_idx + max_lines - 1)
                break

    start_line = start_idx + 1
    end_line = end_idx + 1
    return '\n'.join(lines[start_idx:end_idx + 1]), start_line, end_line
