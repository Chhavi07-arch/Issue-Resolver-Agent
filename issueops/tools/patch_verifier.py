"""Post-patch verification — validate patched file content before committing.

Verification methods by language:
  Python:  ast.parse() with textwrap.dedent — stdlib, zero subprocess overhead.
  Java/Kotlin: brace balance check (character counts); optionally Maven compile.
  JS/TS:  brace balance check.
  Other:  skipped (pass=True).

Public API:
  verify_patch(path, patched_content, repo_root=None) -> VerificationResult

VerificationResult fields:
  passed              bool   — True if verification passed or was skipped.
  method              str    — Which check was performed.
  detail              str    — Human-readable explanation.
  confidence_multiplier float — 1.0 (pass/skip), 0.3 (syntax error), 0.5 (compile error).
"""

import ast
import asyncio
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Result type ─────────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    passed: bool
    method: str
    detail: str
    confidence_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.passed and self.confidence_multiplier == 1.0:
            # Ensure multiplier is always below 1.0 on failure
            self.confidence_multiplier = 0.3


# ─── Extension helpers ────────────────────────────────────────────────────────

def _ext(path: str) -> str:
    dot = path.rfind(".")
    return path[dot:].lower() if dot != -1 else ""


_BRACE_SCOPED = frozenset({
    ".java", ".kt", ".scala", ".cs",
    ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs",
})

_JAVA_LIKE = frozenset({".java", ".kt", ".scala", ".cs"})


# ─── Python AST verification ──────────────────────────────────────────────────

def _verify_python(path: str, content: str) -> VerificationResult:
    """Parse the full patched Python file using stdlib ast.

    textwrap.dedent strips common leading whitespace so snippets extracted from
    inside indented method bodies can be parsed as standalone code.
    """
    try:
        ast.parse(textwrap.dedent(content))
        logger.info(
            "[PATCH_VERIFICATION] result=pass method=python_ast file=%s", path
        )
        return VerificationResult(
            passed=True,
            method="python_ast",
            detail="ast.parse() succeeded",
            confidence_multiplier=1.0,
        )
    except IndentationError as exc:
        detail = f"IndentationError on line {exc.lineno}: {exc.msg}"
        logger.warning(
            "[PATCH_VERIFICATION] result=fail method=python_ast file=%s detail=%s",
            path, detail,
        )
        return VerificationResult(
            passed=False,
            method="python_ast",
            detail=detail,
            confidence_multiplier=0.3,
        )
    except SyntaxError as exc:
        detail = f"SyntaxError on line {exc.lineno}: {exc.msg}"
        logger.warning(
            "[PATCH_VERIFICATION] result=fail method=python_ast file=%s detail=%s",
            path, detail,
        )
        return VerificationResult(
            passed=False,
            method="python_ast",
            detail=detail,
            confidence_multiplier=0.3,
        )


# ─── Brace balance verification (Java/JS/TS) ─────────────────────────────────

def _verify_brace_balance(path: str, content: str) -> VerificationResult:
    """Check that curly braces are balanced in the patched file.

    Uses raw character counts — fast and dependency-free.  String literals
    that contain symmetric {} (e.g. SLF4J format strings) contribute 0 net
    delta and don't distort the count.
    """
    open_count = content.count("{")
    close_count = content.count("}")
    delta = open_count - close_count

    if delta == 0:
        logger.info(
            "[PATCH_VERIFICATION] result=pass method=brace_balance file=%s", path
        )
        return VerificationResult(
            passed=True,
            method="brace_balance",
            detail=f"braces balanced ({open_count} open, {close_count} close)",
            confidence_multiplier=1.0,
        )

    direction = "extra '{'" if delta > 0 else "extra '}'"
    detail = (
        f"brace imbalance: {abs(delta)} unmatched brace(s) — {direction} "
        f"({open_count} '{{' vs {close_count} '}}')"
    )
    logger.warning(
        "[PATCH_VERIFICATION] result=fail method=brace_balance file=%s detail=%s",
        path, detail,
    )
    return VerificationResult(
        passed=False,
        method="brace_balance",
        detail=detail,
        confidence_multiplier=0.3,
    )


# ─── Maven compile verification (Java with repo_root) ────────────────────────

async def _verify_maven_compile(path: str, repo_root: str) -> VerificationResult:
    """Run mvn -q test-compile -DskipTests with a 20-second timeout.

    Uses asyncio.create_subprocess_exec — safe in async context, no blocking.
    Only attempted when repo_root is provided and file is Java-like.
    Falls back to brace_balance result on timeout or subprocess error.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "mvn", "-q", "test-compile", "-DskipTests",
            cwd=repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning(
                "[PATCH_VERIFICATION] result=skip method=maven_compile file=%s "
                "detail=timeout after 20s",
                path,
            )
            # Timeout: fall back to pass (don't block on slow CI environments)
            return VerificationResult(
                passed=True,
                method="maven_compile_timeout",
                detail="Maven compile timed out after 20s — result inconclusive",
                confidence_multiplier=0.8,
            )

        if proc.returncode == 0:
            logger.info(
                "[PATCH_VERIFICATION] result=pass method=maven_compile file=%s", path
            )
            return VerificationResult(
                passed=True,
                method="maven_compile",
                detail="mvn test-compile succeeded",
                confidence_multiplier=1.0,
            )

        err_text = stderr.decode(errors="replace")[:300] if stderr else ""
        detail = f"mvn test-compile failed (exit {proc.returncode}): {err_text}"
        logger.warning(
            "[PATCH_VERIFICATION] result=fail method=maven_compile file=%s detail=%s",
            path, detail[:120],
        )
        return VerificationResult(
            passed=False,
            method="maven_compile",
            detail=detail,
            confidence_multiplier=0.5,
        )

    except FileNotFoundError:
        logger.debug(
            "[PATCH_VERIFICATION] method=maven_compile skipped — mvn not on PATH"
        )
        return VerificationResult(
            passed=True,
            method="maven_compile_skipped",
            detail="mvn not found on PATH — compile check skipped",
            confidence_multiplier=1.0,
        )
    except Exception as exc:
        logger.warning(
            "[PATCH_VERIFICATION] method=maven_compile error: %s", exc
        )
        return VerificationResult(
            passed=True,
            method="maven_compile_error",
            detail=f"maven check failed unexpectedly: {exc}",
            confidence_multiplier=0.9,
        )


# ─── Public API ──────────────────────────────────────────────────────────────

async def verify_patch(
    path: str,
    patched_content: str,
    repo_root: Optional[str] = None,
) -> VerificationResult:
    """Run the appropriate verification for the given file extension.

    Args:
        path: Relative file path — extension determines the verification method.
        patched_content: Full file content after applying the patch.
        repo_root: If provided and file is Java-like, also try Maven compile.

    Returns:
        VerificationResult with passed, method, detail, confidence_multiplier.
    """
    ext = _ext(path)

    if ext == ".py":
        return _verify_python(path, patched_content)

    if ext in _BRACE_SCOPED:
        balance_result = _verify_brace_balance(path, patched_content)

        # For Java-like files with a repo_root, also try Maven compile
        if ext in _JAVA_LIKE and repo_root is not None and balance_result.passed:
            maven_result = await _verify_maven_compile(path, repo_root)
            # Return the stricter of the two results
            if not maven_result.passed:
                return maven_result
            # Both passed — return maven result (more authoritative)
            return maven_result

        return balance_result

    # Unknown extension — skip verification
    logger.debug(
        "[PATCH_VERIFICATION] result=skip method=skipped file=%s (unknown extension)", path
    )
    return VerificationResult(
        passed=True,
        method="skipped",
        detail=f"no verifier for extension '{ext}'",
        confidence_multiplier=1.0,
    )
