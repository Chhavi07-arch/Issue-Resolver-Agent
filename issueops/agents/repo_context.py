"""Repo Context Agent — structure-aware repository grounding.

Discovery priority:
  1. Explicit files from stack traces (analyzer output)
  2. Tree-based scoring: fetch full file tree, score paths against detected
     subsystems and keywords — no code search needed when this succeeds
  3. Code search fallback: semantic framework patterns first, then lexical
     keyword queries (the original strategy, preserved for robustness)

Any individual API failure is caught; the workflow continues with partial data.
"""

import logging
import re
from typing import Any

from issueops.tools import github as gh
from issueops.workflows.state import WorkflowState

logger = logging.getLogger(__name__)

_MAX_FILES = 3        # max files to fetch content for (debug/fix agents use top 2)
_MAX_COMMITS = 3
_MAX_SEARCH_HITS = 5
_MAX_RELATED_ISSUES = 3
_FILE_CONTENT_CHARS = 3000
_MAX_TREE_SCORE_PATHS = 1500  # cap tree paths scored per run

# ---------------------------------------------------------------------------
# Subsystem signal tables
# ---------------------------------------------------------------------------

# Maps subsystem name → issue text fragments that suggest it is involved.
# All signals are matched case-insensitively against lowercased text.
_SUBSYSTEM_SIGNALS: dict[str, list[str]] = {
    "auth":        ["auth", "login", "logout", "token", "jwt", "session",
                    "401", "unauthorized", "credential", "oauth", "password", "bearer", "sign in"],
    "security":    ["security", "403", "forbidden", "permission", "role", "privilege",
                    "csrf", "cors", "acl", "access control", "access denied"],
    "cache":       ["cache", "stale", "invalidat", "redis", "memcach",
                    "ttl", "expir", "cached", "hot reload", "outdated",
                    # Symptom-based signals for cache isolation bugs (no mention of "cache" in issue)
                    "intermittent",    # non-deterministic result → shared/cached state
                    "concurrent",      # concurrent requests differ → race or cache hit
                    "another user",    # cross-user data → cache key missing user identity
                    "wrong user",      # same pattern
                    "other user",      # same pattern
                    "inconsistent",    # inconsistent results → shared state
                    "bleed",           # "bleeding" across requests → isolation failure
                    "privacy",         # privacy violation → cache isolation bug
                    "isolation",       # data isolation failure
                    "tenant",          # multi-tenant leakage
                    ],
    "rate_limit":  ["rate limit", "429", "throttl", "quota", "too many request", "backoff", "rate-limit"],
    "middleware":  ["middleware", "filter", "interceptor", "pipeline", "hook", "chain"],
    "validation":  ["validat", "400", "bad request", "required field", "invalid",
                    "schema", "dto", "payload", "request body", "constraint"],
    "controller":  ["controller", "router", "route", "endpoint", "handler", "api", "rest"],
    "service":     ["service", "business logic", "use case", "usecase"],
    "repository":  ["repositor", "database", "query", "sql", "orm",
                    "persist", "storage", "dao", "findby", "select"],
    "model":       ["model", "entity", "domain", "field", "column", "schema"],
    "config":      ["config", "setting", "environment", "env var", "propert", "yaml", "application.yml"],
    "worker":      ["worker", "job", "queue", "background",
                    "cron", "scheduler", "celery", "async task"],
                    # "task" intentionally omitted — fires on domain entities (TaskService, /api/tasks)
    "websocket":   ["websocket", "ws ", "socket.io", "realtime", "live update"],
}

# Maps subsystem name → path fragments to match against file tree entries.
# Checked against lowercased filename stem and full path separately.
_SUBSYSTEM_PATH_PATTERNS: dict[str, list[str]] = {
    "auth":        ["auth", "jwt", "token", "session", "login", "oauth", "credential", "identity"],
    "security":    ["securit", "auth", "cors", "csrf", "filter", "access", "acl"],
    "cache":       ["cache", "redis", "memcach"],
    "rate_limit":  ["rate", "throttl", "limiter", "ratelimit"],
    "middleware":  ["middleware", "filter", "interceptor"],
    "validation":  ["valid", "dto", "schema", "request", "validator", "constraint"],
    "controller":  ["controller", "router", "route", "handler", "view", "resource", "api", "rest"],
    "service":     ["service", "usecase", "use_case", "application", "facade"],
    "repository":  ["repositor", "repo", "dao", "store", "storage", "mapper", "persistence"],
    "model":       ["model", "entity", "domain", "bean", "dto"],
    "config":      ["config", "setting", "properties", "configuration", "application"],
    "worker":      ["worker", "job", "task", "queue", "scheduler", "consumer", "processor"],
    "websocket":   ["websocket", "socket", "ws"],
}

# Maps subsystem name → identifiable code-level symbols for search queries.
# Used only when tree-based discovery yields nothing.
_SUBSYSTEM_CODE_PATTERNS: dict[str, list[str]] = {
    "auth":        ["@PreAuthorize", "JwtFilter", "AuthenticationService",
                    "authenticate(", "auth_required", "@login_required"],
    "security":    ["SecurityConfig", "@EnableWebSecurity", "hasRole(",
                    "filterChain", "corsConfiguration", "ROLE_"],
    "cache":       ["@Cacheable", "CacheManager", "redisTemplate",
                    "cache.get(", ".cache(", "cache_key"],
    "rate_limit":  ["RateLimiter", "@RateLimit", "throttle(", "rateLimiter", "rate_limit("],
    "middleware":  ["addInterceptor", "HandlerInterceptor", "app.use(",
                    "@app.middleware", "WebMvcConfigurer"],
    "validation":  ["@Valid", "@NotNull", "@NotBlank", "bindingResult", "validate("],
    "controller":  ["@RestController", "@GetMapping", "@PostMapping",
                    "router.get(", "@app.route(", "@app.get(", "def get(self"],
    "service":     ["@Service", "@Injectable", "@Component", "class.*Service"],
    "repository":  ["@Repository", "JpaRepository", "CrudRepository",
                    ".findBy", "session.query("],
    "config":      ["@Configuration", "@Bean", "@Value(", "os.environ", "config.get("],
    "worker":      ["@Scheduled", "@Async", "@celery.task", "BackgroundTasks(", "Cron"],
}

_SOURCE_EXTENSIONS = frozenset({
    ".py", ".java", ".js", ".ts", ".go", ".rb", ".rs",
    ".kt", ".scala", ".php", ".cs", ".cpp", ".c",
})
_CONFIG_EXTENSIONS = frozenset({
    ".yml", ".yaml", ".toml", ".properties", ".xml",
    # .json and .env intentionally omitted — too common in docs/fixtures
})

# Directory segments that indicate non-production-code paths.
# Checked at path-segment boundaries, not as substrings, to prevent
# "latest" triggering on "test" or "normal" triggering on "orm".
_SKIP_DIR_SEGMENTS = frozenset({
    # Test code
    "test", "tests", "spec", "specs", "mock", "mocks",
    "fixture", "fixtures",
    # Build and vendored output
    "target", "build", "dist", "node_modules", "vendor",
    "generated", "__pycache__", ".git",
    # Schema/RPC definitions
    "proto", "thrift",
    # Documentation (not application code)
    "docs", "doc", "docs_src",
    # Database migrations (schema changes, not application logic)
    "migration", "migrations",
    # NOTE: "samples", "demo", "examples" intentionally omitted —
    # they appear in legitimate package names (e.g. org.springframework.samples)
})

# Filename suffixes that should always be skipped regardless of directory
_SKIP_FILENAME_SUFFIXES = (".min.js", ".min.css", ".lock", ".sum", ".pb.go")


def _should_skip(path: str) -> bool:
    """True for paths unlikely to contain bug-relevant production code.

    Uses segment matching (not substring) to avoid false positives like
    'latest' matching 'test' or 'normal' matching 'orm'.
    """
    parts = path.lower().replace("\\", "/").split("/")
    filename = parts[-1]

    # Hidden files
    if filename.startswith("."):
        return True

    # Known non-source filename patterns
    if any(filename.endswith(s) for s in _SKIP_FILENAME_SUFFIXES):
        return True

    # Any directory component is a known non-source segment
    for segment in parts[:-1]:  # directories only, not the filename
        if segment in _SKIP_DIR_SEGMENTS:
            return True

    # Test files by naming convention (filename only)
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    if stem.endswith("test") or stem.endswith("tests") or stem.endswith("spec") or \
       stem.startswith("test_") or stem.endswith("_test"):
        return True

    return False


# ---------------------------------------------------------------------------
# Subsystem inference
# ---------------------------------------------------------------------------

def _expand_camel(token: str) -> list[str]:
    """Split a CamelCase/PascalCase identifier into component words.

    "OrderService" → ["orderservice", "order", "service"]
    "createOrder"  → ["createorder", "create", "order"]
    "jwt"          → ["jwt"]  (no-op for lowercase)
    """
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", token)
    result = [token.lower()]
    if len(parts) > 1:
        result.extend(p.lower() for p in parts)
    return result


def _infer_subsystems(analysis: dict[str, Any]) -> list[str]:
    """Rank subsystems likely involved given issue analysis.

    Scores each subsystem by how many of its signal tokens appear in the
    combined issue text (keywords + stack traces + summary). Returns up to 4
    subsystems in descending score order.

    CamelCase keywords are expanded so "OrderService" contributes both
    "order" and "service" as matchable tokens.
    """
    raw_keywords = analysis.get("keywords") or []
    # Expand camelCase so "AuthFilter" contributes "auth" and "filter" individually
    expanded_tokens: list[str] = []
    for kw in raw_keywords:
        expanded_tokens.extend(_expand_camel(kw))

    traces_text = " ".join(analysis.get("stack_traces") or []).lower()
    summary_text = analysis.get("summary", "").lower()
    combined = " ".join(expanded_tokens) + " " + traces_text + " " + summary_text

    scores: dict[str, int] = {}
    for subsystem, signals in _SUBSYSTEM_SIGNALS.items():
        score = 0
        for sig in signals:
            if " " in sig or "-" in sig:
                # Multi-word phrase — substring match is fine (specific enough)
                if sig in combined:
                    score += 2
            else:
                # Single token — require word-boundary start to prevent false positives
                # e.g. "ttl" must not fire inside "throttle", "orm" not inside "normal"
                if re.search(r"\b" + re.escape(sig), combined):
                    score += 2
        if score:
            scores[subsystem] = score

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)

    # When cache is the top subsystem, the bug lives in the service layer
    # (cache annotations like @Cacheable sit on service methods, not controllers)
    if ranked and ranked[0] == "cache" and "service" not in ranked:
        ranked.insert(1, "service")

    # Bugs almost always manifest in a controller/handler — ensure it's in scope
    issue_type = analysis.get("issue_type", "")
    if issue_type == "bug" and "controller" not in ranked:
        ranked.append("controller")

    return ranked[:4]


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------

def _detect_framework(tree_paths: list[str]) -> str | None:
    """Infer project framework from the repository file tree."""
    joined = " ".join(tree_paths[:300]).lower()

    has_java = any(p.endswith(".java") for p in tree_paths[:200])
    has_py = any(p.endswith(".py") for p in tree_paths[:200])
    has_ts = any(p.endswith(".ts") for p in tree_paths[:200])
    has_js = any(p.endswith(".js") for p in tree_paths[:200])
    has_go = any(p.endswith(".go") for p in tree_paths[:200])

    if has_java:
        if "pom.xml" in joined or "build.gradle" in joined:
            return "java_spring"
        return "java"
    if has_py:
        if "django" in joined or "urls.py" in joined or "views.py" in joined:
            return "python_django"
        if "fastapi" in joined or "uvicorn" in joined:
            return "python_fastapi"
        return "python"
    if has_ts:
        if "nestjs" in joined or "nest" in joined:
            return "node_nestjs"
        return "node_typescript"
    if has_js:
        if "express" in joined:
            return "node_express"
        return "node"
    if has_go:
        return "go"
    return None


# ---------------------------------------------------------------------------
# Tree-based file scoring
# ---------------------------------------------------------------------------

def _score_file(
    path: str,
    subsystems: list[str],
    keywords: list[str],
    endpoint_entities: list[str] = (),
) -> float:
    """Compute a relevance score for a repository file path.

    Returns -1.0 for paths that should be skipped (tests, build artifacts).
    Returns 0 or higher for candidates — higher means more likely relevant.
    """
    if _should_skip(path):
        return -1.0

    # Only score source and config files
    dot_idx = path.rfind(".")
    ext = path[dot_idx:].lower() if dot_idx != -1 else ""
    if ext not in _SOURCE_EXTENSIONS and ext not in _CONFIG_EXTENSIONS:
        return -1.0

    lower_path = path.lower()
    # Filename stem (no directory prefix, no extension)
    slash_idx = lower_path.rfind("/")
    filename = lower_path[slash_idx + 1:] if slash_idx != -1 else lower_path
    stem = filename[:filename.rfind(".")] if "." in filename else filename

    score = 0.0

    # Subsystem path pattern matching — ranked by detection confidence
    for rank, subsystem in enumerate(subsystems):
        weight = 1.0 / (rank + 1)  # 1.0, 0.5, 0.33, 0.25 for ranks 0–3
        patterns = _SUBSYSTEM_PATH_PATTERNS.get(subsystem, [])
        stem_matched = False
        for pat in patterns:
            if pat in stem:
                score += 3.0 * weight
                stem_matched = True
                break
        if not stem_matched:
            for pat in patterns:
                if pat in lower_path:
                    score += 1.5 * weight
                    break

    # Endpoint entity match — URL path directly names the controller domain.
    # Scored higher than subsystem pattern (3.0) because the endpoint is unambiguous:
    # /api/tasks → TaskController, TaskService win over AuthController regardless of
    # which subsystem keywords happened to fire from the issue text.
    for entity in endpoint_entities:
        if len(entity) < 3:
            continue
        if entity in stem:
            score += 4.0
        elif entity in lower_path:
            score += 1.5

    # Keyword match against filename stem (strong signal — identifier naming)
    for kw in keywords:
        kw_lower = kw.lower()
        if len(kw_lower) < 3:
            continue
        if kw_lower in stem:
            score += 2.0
        elif kw_lower in lower_path:
            score += 0.5

    # Slight preference for source files over config
    if ext in _SOURCE_EXTENSIONS:
        score += 0.3

    return score


def _select_files_from_tree(
    tree_paths: list[str],
    subsystems: list[str],
    keywords: list[str],
    max_files: int = _MAX_FILES,
    endpoint_entities: list[str] = (),
) -> list[str]:
    """Score all tree paths and return the top candidates."""
    if not tree_paths:
        return []

    scored: list[tuple[float, str]] = []
    for path in tree_paths[:_MAX_TREE_SCORE_PATHS]:
        s = _score_file(path, subsystems, keywords, endpoint_entities)
        if s > 0:
            scored.append((s, path))

    scored.sort(reverse=True)

    top = [p for _, p in scored[:max_files]]
    if top:
        logger.info(
            "RepoContext: tree scoring → top candidates (scores): %s",
            [(p, round(s, 1)) for s, p in scored[:max_files]],
        )
    return top


# ---------------------------------------------------------------------------
# Code search query builders
# ---------------------------------------------------------------------------

def _build_fallback_queries(keywords: list[str]) -> list[str]:
    """Return an ordered list of code-aware search queries.

    Preference: precise code identifiers first, English prose last.
    GitHub code search works on token matching, not semantic similarity.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = q.strip()
        if q and q not in seen and len(q) >= 2:
            seen.add(q)
            queries.append(q)

    # Exception/Error class names — appear verbatim in tracebacks and handlers
    for k in keywords:
        if k[0].isupper() and len(k) > 3 and k not in ("None", "True", "False"):
            add(k)

    # snake_case tokens — likely function or variable names
    for k in keywords:
        if "_" in k and k == k.lower():
            add(k)

    # Short lowercase identifiers (3–10 chars)
    for k in keywords:
        if k == k.lower() and 3 <= len(k) <= 10:
            add(k)

    # Any remaining keyword individually
    for k in keywords[:6]:
        add(k)

    # Two-word fallback combining best code-like tokens
    code_like = [q for q in queries if len(q) <= 15]
    if len(code_like) >= 2:
        add(f"{code_like[0]} {code_like[1]}")

    # Code-pattern probes from error signals
    kw_text = " ".join(keywords).lower()
    if any(x in kw_text for x in ("indexerror", "index", "bound", "range")):
        add("len(")
    if any(x in kw_text for x in ("nonetype", "attributeerror", "strip", "lower", "none")):
        add("None")
        add(".strip(")

    return queries


def _build_semantic_queries(subsystems: list[str], keywords: list[str]) -> list[str]:
    """Generate framework-aware code search queries from detected subsystems.

    These find files by the identifiers they contain rather than filename, useful
    when tree-based scoring finds nothing or the repo tree is unavailable.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)

    for subsystem in subsystems[:2]:  # top 2 subsystems
        patterns = _SUBSYSTEM_CODE_PATTERNS.get(subsystem, [])
        for pat in patterns[:3]:  # top 3 per subsystem
            add(pat)

    return queries


# ---------------------------------------------------------------------------
# Endpoint / error-message signal extraction
# ---------------------------------------------------------------------------

# Path segments that are API infrastructure, not domain entities
_URL_SKIP_SEGMENTS = frozenset({
    "api", "v1", "v2", "v3", "v4", "rest", "graphql",
    "http", "https", "com", "org", "github", "io", "www",
})

_URL_SEG_RE = re.compile(r'/([a-z][a-z0-9_-]{1,30})(?=[/{?\s]|$)', re.IGNORECASE)

_QUOTED_IDENT_RE = re.compile(r"""['"`]([a-zA-Z][a-zA-Z0-9_.]{1,40})['"`]""")

# Spring Data JPA: "No property 'createdDate' found for type 'Task'"
_JPA_FIELD_ERROR_RE = re.compile(
    r"no property\s+'?([a-zA-Z][a-zA-Z0-9]+)'?\s+found for type\s+'?([a-zA-Z][a-zA-Z0-9]+)'?",
    re.IGNORECASE,
)
_NULL_POINTER_RE = re.compile(r"NullPointerException|NPE(?:\b|$)", re.IGNORECASE)
_CLASS_CAST_RE = re.compile(r"ClassCastException|cannot cast\b", re.IGNORECASE)

_COMMON_IDENTS = frozenset({
    "null", "true", "false", "undefined", "string", "int",
    "boolean", "object", "list", "map", "void", "class",
})


def _extract_endpoint_entities(title: str, body: str) -> list[str]:
    """Extract domain entity names from URL paths mentioned in issue text.

    GET /api/tasks/{id}          → ["tasks", "task"]
    /api/users/{id}/orders/{oid} → ["users", "user", "orders", "order"]

    These are used to boost file scoring for the matching domain controller/service,
    overriding auth-keyword noise when the endpoint clearly names the domain.
    """
    combined = (title + " " + body).lower()
    seen: set[str] = set()
    entities: list[str] = []

    for m in _URL_SEG_RE.finditer(combined):
        seg = m.group(1).lower()
        if seg in _URL_SKIP_SEGMENTS or len(seg) < 2:
            continue
        if seg not in seen:
            seen.add(seg)
            entities.append(seg)
        # Naive singularization: "tasks" → "task"
        if seg.endswith("s") and len(seg) > 3:
            singular = seg[:-1]
            if singular not in seen:
                seen.add(singular)
                entities.append(singular)

    return entities[:6]


def _extract_error_identifiers(title: str, body: str) -> list[str]:
    """Extract quoted code identifiers from error messages in issue text.

    "No property 'createdDate' found for type 'Task'" → ["createdDate", "Task"]
    Injects domain-specific names the LLM keyword extractor may have missed.
    """
    combined = title + " " + body
    seen: set[str] = set()
    identifiers: list[str] = []

    def add(ident: str) -> None:
        if ident and ident.lower() not in _COMMON_IDENTS and ident not in seen:
            seen.add(ident)
            identifiers.append(ident)

    # High-priority: JPA field-not-found pattern (captures field + type names)
    for m in _JPA_FIELD_ERROR_RE.finditer(combined):
        add(m.group(1))
        add(m.group(2))

    # General: quoted camelCase/PascalCase/snake_case identifiers
    for m in _QUOTED_IDENT_RE.finditer(combined):
        ident = m.group(1)
        if "." not in ident or ident.endswith((".java", ".py", ".js", ".ts")):
            if re.search(r"[A-Z_]", ident) or "_" in ident:
                add(ident)

    return identifiers[:8]


def _detect_framework_errors(title: str, body: str) -> list[str]:
    """Return subsystems implied by recognizable framework error patterns.

    Deterministic — higher confidence than keyword heuristics.
    e.g. "No property X found for type Y" → model + repository (not auth).
    """
    combined = title + " " + body
    implied: list[str] = []

    if _JPA_FIELD_ERROR_RE.search(combined):
        # JPA derived-query or sort field does not match entity field → model/repo layer
        implied.extend(["model", "repository", "controller"])
    if _NULL_POINTER_RE.search(combined) and "service" not in implied:
        implied.append("service")
    if _CLASS_CAST_RE.search(combined) and "service" not in implied:
        implied.append("service")

    return implied


# ---------------------------------------------------------------------------
# Call-chain traversal
# ---------------------------------------------------------------------------

# Subsystems where annotation-based code search outperforms filename scoring
# (i.e. the relevant symbol is a decorator/annotation, not in the filename)
_ANNOTATION_HEAVY_SUBS = frozenset({"cache", "service", "repository"})

_LAYERED_SUFFIXES = (
    "controller", "service", "repository", "handler",
    "resource", "router", "manager",
)


def _extract_entity_prefix(filename: str) -> str | None:
    """Extract the domain entity prefix from a layered class filename.

    "TaskController.java" → "task"
    "OrderService.py"     → "order"
    "UserRepository.kt"   → "user"
    Returns None if the filename doesn't follow the pattern or prefix is < 3 chars.
    """
    stem = filename.rsplit(".", 1)[0].lower()
    for suffix in _LAYERED_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix) + 2:
            return stem[: -len(suffix)]
    return None


def _find_chain_files(
    tree_paths: list[str],
    target_files: list[str],
    subsystems: list[str],
) -> list[str]:
    """Find service/repository files that share a domain entity with already-found files.

    E.g. if target_files contains "TaskController.java", extract "task" and
    find "TaskService.java", "TaskRepository.java" in the tree.
    """
    entities: set[str] = set()
    for path in target_files:
        filename = path.rsplit("/", 1)[-1]
        entity = _extract_entity_prefix(filename)
        if entity and len(entity) >= 3:
            entities.add(entity)

    if not entities:
        return []

    # Include model/entity so call-chain reaches domain objects (e.g. Task.java when
    # field-name mismatches or entity-level bugs are suspected)
    chain_subs = {"service", "repository", "cache", "model"}
    chain_patterns = [
        pat
        for sub in subsystems
        if sub in chain_subs
        for pat in (_SUBSYSTEM_PATH_PATTERNS.get(sub, [])[:3])
    ]

    seen = set(target_files)
    prioritised: list[str] = []
    secondary: list[str] = []

    for path in tree_paths:
        if _should_skip(path) or path in seen:
            continue
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        for entity in entities:
            if entity in stem:
                seen.add(path)
                if any(pat in stem for pat in chain_patterns):
                    prioritised.append(path)
                else:
                    secondary.append(path)
                break

    return (prioritised + secondary)[:3]


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

async def gather_repo_context(state: WorkflowState) -> dict[str, Any]:
    """Gather repository evidence using structure-aware retrieval.

    Discovery chain:
      1. Explicit files from stack traces (always tried first)
      2. Repo tree fetch → subsystem-aware file scoring
      3. Code search fallback (semantic patterns, then lexical keywords)

    Any individual API failure is caught; partial=True is set if any step
    fails but the workflow always continues.
    """
    owner = state["repo_owner"]
    repo = state["repo_name"]
    analysis = state.get("analysis") or {}
    suspected_files: list[str] = analysis.get("suspected_files") or []
    keywords: list[str] = analysis.get("keywords") or []

    logger.info("RepoContext: starting for %s/%s", owner, repo)

    file_snippets: dict[str, str] = {}
    code_search_results: list[dict[str, Any]] = []
    recent_commits: list[dict[str, Any]] = []
    related_issues: list[dict[str, Any]] = []
    agent_errors: list[str] = []

    # Raw issue text (available directly — more reliable than LLM-extracted keywords
    # for structured patterns like URL paths and error messages)
    title_text: str = state.get("issue_title", "")
    body_text: str = state.get("issue_body", "") or ""

    # ------------------------------------------------------------------
    # 1. Infer subsystems from issue analysis + raw text signals
    # ------------------------------------------------------------------

    # Extract endpoint entities (/api/tasks/{id} → ["tasks", "task"])
    endpoint_entities = _extract_endpoint_entities(title_text, body_text)

    # Extract error identifiers ("No property 'createdDate'" → ["createdDate", "Task"])
    error_identifiers = _extract_error_identifiers(title_text, body_text)

    # Augment LLM-extracted keywords with error identifiers for subsystem + file scoring
    augmented_keywords = list(dict.fromkeys(keywords + error_identifiers))

    # Run subsystem inference on augmented keyword set
    augmented_analysis = {**analysis, "keywords": augmented_keywords} if error_identifiers else analysis
    subsystems = _infer_subsystems(augmented_analysis)

    # Prepend subsystems implied by deterministic framework error patterns.
    # These are higher-confidence than keyword heuristics — e.g. the JPA "No property X
    # found for type Y" error can only come from the model/repository layer.
    framework_error_subs = _detect_framework_errors(title_text, body_text)
    if framework_error_subs:
        for sub in reversed(framework_error_subs):
            if sub not in subsystems:
                subsystems.insert(0, sub)
        subsystems = subsystems[:4]

    # Use augmented keywords everywhere downstream
    keywords = augmented_keywords

    if subsystems:
        logger.info("RepoContext: subsystems inferred — %s", subsystems)
    if endpoint_entities:
        logger.info("RepoContext: endpoint entities — %s", endpoint_entities)
    if error_identifiers:
        logger.info("RepoContext: error identifiers — %s", error_identifiers)

    # ------------------------------------------------------------------
    # 2. Fetch repo file tree (single API call)
    # ------------------------------------------------------------------
    tree_paths: list[str] = []
    framework: str | None = None

    try:
        tree_paths = await gh.list_repo_tree(owner, repo)
        framework = _detect_framework(tree_paths)
        logger.info(
            "RepoContext: tree fetched — %d paths, framework=%s",
            len(tree_paths), framework,
        )
    except Exception as exc:
        logger.warning("RepoContext: tree fetch failed (%s) — continuing without", exc)
        agent_errors.append(f"tree fetch failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Build target file list
    # ------------------------------------------------------------------
    # Priority 1: explicit files from stack traces
    target_files: list[str] = list(dict.fromkeys(suspected_files[:_MAX_FILES]))

    # Priority 2: tree-based scoring to fill remaining slots
    if len(target_files) < _MAX_FILES and tree_paths:
        remaining = _MAX_FILES - len(target_files) + 1  # +1 for 404 redundancy
        candidates = _select_files_from_tree(
            tree_paths, subsystems, keywords,
            max_files=remaining, endpoint_entities=endpoint_entities,
        )
        for c in candidates:
            if c not in target_files:
                target_files.append(c)
                if len(target_files) >= _MAX_FILES:
                    break

    # Priority 2b: supplementary annotation-based search when top subsystem is cache/service/repo
    # and the tree didn't surface a file matching that subsystem's path patterns.
    # This handles cases where the bug is in a @Cacheable/@Service method whose filename
    # doesn't obviously signal the subsystem (e.g. TaskService.java for a cache bug).
    if subsystems and subsystems[0] in _ANNOTATION_HEAVY_SUBS and keywords:
        top_sub = subsystems[0]
        top_patterns = _SUBSYSTEM_PATH_PATTERNS.get(top_sub, [])
        already_covered = any(
            any(pat in f.lower() for pat in top_patterns)
            for f in target_files
        )
        if not already_covered:
            for query in _build_semantic_queries([top_sub], keywords)[:2]:
                logger.info(
                    "RepoContext: supplementary search for %s subsystem: %r", top_sub, query
                )
                try:
                    hits = await gh.search_code(owner, repo, query, max_results=_MAX_SEARCH_HITS)
                except Exception as exc:
                    logger.warning("RepoContext: supplementary search %r failed: %s", query, exc)
                    agent_errors.append(f"supplementary search failed ({query!r}): {exc}")
                    continue
                if hits:
                    new_paths = [
                        h["path"]
                        for h in hits
                        if h.get("path") and h["path"] not in set(target_files)
                    ]
                    if new_paths:
                        target_files = new_paths[:2] + target_files  # prepend: annotation match is specific
                        logger.info(
                            "RepoContext: supplementary search prepended %s", new_paths[:2]
                        )
                        break

    # Priority 2c: call-chain expansion — if we found a controller, also find its service/repo
    if target_files and tree_paths:
        chain_files = _find_chain_files(tree_paths, target_files, subsystems)
        for cf in chain_files:
            if cf not in target_files:
                target_files.append(cf)
                logger.info("RepoContext: call-chain expanded → %s", cf)
            if len(target_files) >= _MAX_FILES + 1:  # +1 tolerance since debug agent caps at 2
                break

    # Priority 3: code search (semantic then lexical) when tree gave nothing
    if not target_files and keywords:
        all_queries = _build_semantic_queries(subsystems, keywords) + _build_fallback_queries(keywords)

        for query in all_queries:
            logger.info("RepoContext: no files yet — trying code search %r", query)
            try:
                hits = await gh.search_code(owner, repo, query, max_results=_MAX_SEARCH_HITS)
            except Exception as exc:
                logger.warning("RepoContext: search %r failed: %s", query, exc)
                agent_errors.append(f"code search failed ({query!r}): {exc}")
                continue

            if not hits:
                logger.info("RepoContext: search %r → 0 hits", query)
                continue

            code_search_results = hits
            logger.info("RepoContext: search %r → %d hits", query, len(hits))
            seen_paths: set[str] = set(target_files)
            for h in hits:
                p = h.get("path", "")
                if p and p not in seen_paths:
                    seen_paths.add(p)
                    target_files.append(p)
                if len(target_files) >= _MAX_FILES:
                    break
            break

    # ------------------------------------------------------------------
    # 4. Fetch file contents
    # ------------------------------------------------------------------
    for file_path in target_files[:_MAX_FILES]:
        try:
            content = await gh.get_file_contents(owner, repo, file_path)
            if content is not None:
                file_snippets[file_path] = content[:_FILE_CONTENT_CHARS]
                logger.info(
                    "RepoContext: fetched %s (%d chars)", file_path, len(file_snippets[file_path])
                )
            else:
                logger.info("RepoContext: no content for %s (404 or binary)", file_path)
        except Exception as exc:
            msg = f"file read failed for {file_path}: {exc}"
            logger.warning("RepoContext: %s", msg)
            agent_errors.append(msg)

    # ------------------------------------------------------------------
    # 5. Fetch commits for primary file
    # ------------------------------------------------------------------
    primary_file = target_files[0] if target_files else None
    try:
        recent_commits = await gh.get_recent_commits(
            owner, repo, path=primary_file, limit=_MAX_COMMITS
        )
        logger.info("RepoContext: got %d commits (path=%s)", len(recent_commits), primary_file)
    except Exception as exc:
        msg = f"commit fetch failed: {exc}"
        logger.warning("RepoContext: %s", msg)
        agent_errors.append(msg)

    # ------------------------------------------------------------------
    # 6. Search related issues
    # ------------------------------------------------------------------
    if keywords:
        issue_query = " ".join(keywords[:2])
        try:
            related_issues = await gh.search_issues(
                owner, repo, issue_query, max_results=_MAX_RELATED_ISSUES
            )
            logger.info("RepoContext: got %d related issues", len(related_issues))
        except Exception as exc:
            msg = f"issue search failed: {exc}"
            logger.warning("RepoContext: %s", msg)
            agent_errors.append(msg)

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    relevant_files = list(file_snippets.keys()) or target_files

    repo_context: dict[str, Any] = {
        "relevant_files": relevant_files,
        "file_snippets": file_snippets,
        "recent_commits": recent_commits,
        "related_issues": related_issues,
        "code_search_results": code_search_results,
        "partial": bool(agent_errors),
        "errors": agent_errors,
        "repo_structure": {
            "framework": framework,
            "subsystems_detected": subsystems,
            "tree_size": len(tree_paths),
        },
    }

    logger.info(
        "RepoContext: done — files=%d snippets=%d commits=%d issues=%d "
        "subsystems=%s framework=%s partial=%s",
        len(relevant_files), len(file_snippets), len(recent_commits),
        len(related_issues), subsystems, framework, repo_context["partial"],
    )

    return {"repo_context": repo_context, "current_step": "repo_context_gathered"}
