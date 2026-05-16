"""GitHub REST API wrappers using httpx.

READ functions  — fully implemented.
WRITE functions — real implementation (Phase 6).
"""

import base64
import logging
from typing import Any, Optional

import httpx

from issueops.config.settings import settings

logger = logging.getLogger(__name__)

_STANDARD_ACCEPT = "application/vnd.github+json"
_TEXT_MATCH_ACCEPT = "application/vnd.github.text-match+json"


class GitHubWriteError(Exception):
    """Raised when a GitHub write operation fails unrecoverably."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_client(accept: str = _STANDARD_ACCEPT) -> httpx.AsyncClient:
    headers: dict[str, str] = {
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return httpx.AsyncClient(
        base_url=settings.github_api_base,
        headers=headers,
        timeout=httpx.Timeout(settings.http_timeout),
        follow_redirects=True,
    )


async def _get(
    path: str,
    params: Optional[dict[str, Any]] = None,
    accept: str = _STANDARD_ACCEPT,
) -> tuple[int, Any]:
    """GET request → (status_code, parsed_json | None).

    Never raises — all errors are caught and logged.
    Returns (0, None) on network/timeout failure.
    """
    async with _make_client(accept) as client:
        try:
            resp = await client.get(path, params=params)
        except httpx.TimeoutException:
            logger.warning("GitHub API timeout: GET %s", path)
            return 0, None
        except httpx.RequestError as exc:
            logger.warning("GitHub API request error: GET %s — %s", path, exc)
            return 0, None

    status = resp.status_code

    if status == 401:
        logger.error("GitHub API 401 Unauthorized — check GITHUB_TOKEN")
        return status, None

    if status == 403:
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        if remaining == "0" or "rate limit" in resp.text.lower():
            logger.warning("GitHub rate limit exceeded for %s", path)
        else:
            logger.warning("GitHub API 403 Forbidden for %s", path)
        return status, None

    if status == 404:
        logger.debug("GitHub API 404 Not Found: %s", path)
        return status, None

    if status == 422:
        logger.warning("GitHub API 422 Unprocessable for %s: %s", path, resp.text[:200])
        return status, None

    if status == 429:
        logger.warning("GitHub API 429 Too Many Requests for %s", path)
        return status, None

    if not resp.is_success:
        logger.warning("GitHub API %d for %s: %s", status, path, resp.text[:200])
        return status, None

    try:
        return status, resp.json()
    except Exception:
        return status, None


async def _write(
    method: str,
    path: str,
    json_body: dict[str, Any],
) -> tuple[int, Any]:
    """POST/PUT → (status_code, parsed_json | None).

    Never raises — all errors are caught and logged.
    Returns (0, None) on network/timeout failure.
    """
    async with _make_client() as client:
        try:
            if method == "POST":
                resp = await client.post(path, json=json_body)
            else:  # PUT
                resp = await client.put(path, json=json_body)
        except httpx.TimeoutException:
            logger.warning("GitHub API timeout: %s %s", method, path)
            return 0, None
        except httpx.RequestError as exc:
            logger.warning("GitHub API request error: %s %s — %s", method, path, exc)
            return 0, None

    status = resp.status_code
    try:
        data = resp.json()
    except Exception:
        data = None
    return status, data


# ---------------------------------------------------------------------------
# READ — fully implemented
# ---------------------------------------------------------------------------

async def get_issue(owner: str, repo: str, issue_number: int) -> Optional[dict[str, Any]]:
    """Fetch a single issue. Returns None on any failure."""
    logger.debug("get_issue: %s/%s#%d", owner, repo, issue_number)
    status, data = await _get(f"/repos/{owner}/{repo}/issues/{issue_number}")
    if data and isinstance(data, dict) and "number" in data:
        return {
            "id": data["id"],
            "number": data["number"],
            "title": data.get("title", ""),
            "body": data.get("body") or "",
            "state": data.get("state", ""),
            "html_url": data.get("html_url", ""),
            "user_login": (data.get("user") or {}).get("login"),
        }
    return None


async def search_code(
    owner: str,
    repo: str,
    query: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search code in a repo. Returns list of {path, html_url, snippet}."""
    if not query.strip():
        return []

    q = f"{query} repo:{owner}/{repo}"
    logger.debug("search_code: q=%r", q)

    status, data = await _get(
        "/search/code",
        params={"q": q, "per_page": max_results},
        accept=_TEXT_MATCH_ACCEPT,
    )
    if not data or "items" not in data:
        return []

    results = []
    for item in data["items"][:max_results]:
        snippet: Optional[str] = None
        matches = item.get("text_matches") or []
        if matches:
            snippet = matches[0].get("fragment", "")

        results.append({
            "path": item.get("path", ""),
            "html_url": item.get("html_url", ""),
            "snippet": snippet,
        })

    logger.debug("search_code: %d hits for %r", len(results), query)
    return results


async def get_file_contents(
    owner: str,
    repo: str,
    path: str,
    ref: Optional[str] = None,
) -> Optional[str]:
    """Fetch decoded text content of a file. Returns None on failure or binary."""
    params: dict[str, Any] = {}
    if ref:
        params["ref"] = ref

    logger.debug("get_file_contents: %s/%s %s ref=%s", owner, repo, path, ref)
    status, data = await _get(f"/repos/{owner}/{repo}/contents/{path}", params=params or None)

    if not data or not isinstance(data, dict):
        return None

    if data.get("type") != "file":
        logger.debug("get_file_contents: %s is not a file (type=%s)", path, data.get("type"))
        return None

    encoding = data.get("encoding", "")
    raw = data.get("content", "")

    if encoding == "base64":
        try:
            decoded = base64.b64decode(raw.replace("\n", ""))
            return decoded.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("get_file_contents: decode error for %s — %s", path, exc)
            return None

    return raw or None


async def get_recent_commits(
    owner: str,
    repo: str,
    path: Optional[str] = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Fetch recent commits, optionally filtered to a file path."""
    params: dict[str, Any] = {"per_page": min(limit, 10)}
    if path:
        params["path"] = path

    logger.debug("get_recent_commits: %s/%s path=%s limit=%d", owner, repo, path, limit)
    status, data = await _get(f"/repos/{owner}/{repo}/commits", params=params)

    if not data or not isinstance(data, list):
        return []

    commits = []
    for item in data[:limit]:
        commit = item.get("commit", {})
        author_meta = commit.get("author") or {}
        gh_author = item.get("author") or {}
        commits.append({
            "sha": item.get("sha", "")[:8],
            "message": commit.get("message", "").splitlines()[0],
            "author": gh_author.get("login") or author_meta.get("name", "unknown"),
            "date": author_meta.get("date", ""),
            "html_url": item.get("html_url", ""),
        })

    logger.debug("get_recent_commits: got %d commits", len(commits))
    return commits


async def search_issues(
    owner: str,
    repo: str,
    query: str,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search issues in a repo. Returns list of {number, title, state, html_url, body_preview}."""
    if not query.strip():
        return []

    q = f"{query} repo:{owner}/{repo} is:issue"
    logger.debug("search_issues: q=%r", q)

    status, data = await _get(
        "/search/issues",
        params={"q": q, "per_page": max_results, "sort": "relevance"},
    )
    if not data or "items" not in data:
        return []

    results = []
    for item in data["items"][:max_results]:
        body = item.get("body") or ""
        results.append({
            "number": item.get("number"),
            "title": item.get("title", ""),
            "state": item.get("state", ""),
            "html_url": item.get("html_url", ""),
            "body_preview": body[:200] if body else None,
        })

    logger.debug("search_issues: %d results for %r", len(results), query)
    return results


async def list_repo_tree(owner: str, repo: str, max_paths: int = 2000) -> list[str]:
    """Return a flat list of file paths from the repository using the Git Trees API.

    Fetches the default branch tree recursively in a single API call.
    Returns [] on any failure (rate limit, empty repo, network error).
    Large repos may return a truncated tree — partial results are still useful.
    """
    default_branch = await get_default_branch(owner, repo)

    logger.debug("list_repo_tree: %s/%s branch=%s", owner, repo, default_branch)
    status, tree_data = await _get(
        f"/repos/{owner}/{repo}/git/trees/{default_branch}",
        params={"recursive": "1"},
    )

    if not tree_data or not isinstance(tree_data, dict) or "tree" not in tree_data:
        logger.debug("list_repo_tree: no tree data for %s/%s (status=%d)", owner, repo, status)
        return []

    if tree_data.get("truncated"):
        logger.info(
            "list_repo_tree: tree truncated for %s/%s — using first %d paths",
            owner, repo, max_paths,
        )

    paths = [
        item["path"]
        for item in tree_data["tree"][:max_paths]
        if item.get("type") == "blob" and item.get("path")
    ]
    logger.debug("list_repo_tree: %d file paths for %s/%s", len(paths), owner, repo)
    return paths


# ---------------------------------------------------------------------------
# READ helpers for writes
# ---------------------------------------------------------------------------

async def get_default_branch(owner: str, repo: str) -> str:
    """Return the repo's default branch. Falls back to settings.default_base_branch."""
    status, data = await _get(f"/repos/{owner}/{repo}")
    if data and isinstance(data, dict):
        branch = data.get("default_branch")
        if branch:
            logger.debug("get_default_branch: %s/%s → %s", owner, repo, branch)
            return branch
    logger.warning(
        "get_default_branch: could not determine default branch for %s/%s; using '%s'",
        owner, repo, settings.default_base_branch,
    )
    return settings.default_base_branch


async def _get_branch_sha(owner: str, repo: str, branch: str) -> Optional[str]:
    """Return the HEAD commit SHA of a branch. Returns None if not found."""
    status, data = await _get(f"/repos/{owner}/{repo}/git/refs/heads/{branch}")
    if data and isinstance(data, dict):
        return (data.get("object") or {}).get("sha")
    return None


async def _get_file_sha(owner: str, repo: str, path: str, branch: str) -> Optional[str]:
    """Return the blob SHA of a file on a branch (required for updates). Returns None if not found."""
    status, data = await _get(
        f"/repos/{owner}/{repo}/contents/{path}",
        params={"ref": branch},
    )
    if data and isinstance(data, dict) and data.get("type") == "file":
        return data.get("sha")
    return None


# ---------------------------------------------------------------------------
# WRITE — real implementations (Phase 6)
# ---------------------------------------------------------------------------

async def create_branch(
    owner: str,
    repo: str,
    base_branch: str,
    new_branch: str,
) -> dict[str, Any]:
    """Create new_branch from the HEAD of base_branch.

    Raises GitHubWriteError on failure.
    """
    sha = await _get_branch_sha(owner, repo, base_branch)
    if not sha:
        raise GitHubWriteError(
            f"Cannot create branch: could not get HEAD SHA for '{base_branch}' in {owner}/{repo}"
        )

    logger.info(
        "create_branch: %s/%s  %s → %s  (base %s)",
        owner, repo, base_branch, new_branch, sha[:8],
    )
    status, data = await _write(
        "POST",
        f"/repos/{owner}/{repo}/git/refs",
        {"ref": f"refs/heads/{new_branch}", "sha": sha},
    )

    if status not in (200, 201):
        msg = (data or {}).get("message", "") if isinstance(data, dict) else str(data)
        raise GitHubWriteError(f"create_branch failed ({status}): {msg}")

    logger.info("create_branch: created branch '%s'", new_branch)
    return data  # type: ignore[return-value]


async def branch_exists(owner: str, repo: str, branch: str) -> bool:
    """Return True if the branch ref exists in the repo."""
    sha = await _get_branch_sha(owner, repo, branch)
    return bool(sha)


async def find_available_branch_name(
    owner: str,
    repo: str,
    base_name: str,
    max_attempts: int = 20,
) -> str:
    """Return a branch name that does not yet exist on the remote.

    Tries ``base_name`` first; if it already exists, tries ``base_name-v2``,
    ``base_name-v3``, ... up to ``max_attempts``. Falls back to a millisecond
    timestamp suffix if all numbered slots are taken.

    Used by the fix-PR pipeline so re-runs against the same issue produce
    fresh branches instead of failing with 422 "Reference already exists".
    A network failure during probing is treated as "branch does not exist"
    to avoid blocking writes; the caller's create_branch will surface any
    real conflict.
    """
    try:
        if not await branch_exists(owner, repo, base_name):
            return base_name
    except Exception as exc:
        logger.warning(
            "find_available_branch_name: probe failed for '%s' (%s) — assuming available",
            base_name, exc,
        )
        return base_name

    for v in range(2, max_attempts + 1):
        candidate = f"{base_name}-v{v}"
        try:
            if not await branch_exists(owner, repo, candidate):
                logger.info(
                    "find_available_branch_name: '%s' taken — using '%s'",
                    base_name, candidate,
                )
                return candidate
        except Exception as exc:
            logger.warning(
                "find_available_branch_name: probe failed for '%s' (%s) — using it anyway",
                candidate, exc,
            )
            return candidate

    # All numbered slots taken — fall back to timestamp
    import time
    fallback = f"{base_name}-t{int(time.time())}"
    logger.warning(
        "find_available_branch_name: all -v2..-v%d taken for '%s' — using timestamp fallback '%s'",
        max_attempts, base_name, fallback,
    )
    return fallback


async def create_or_update_file(
    owner: str,
    repo: str,
    path: str,
    new_content: str,
    branch: str,
    commit_message: str,
) -> dict[str, Any]:
    """Create or update a single file on branch via a new commit.

    Fetches the existing blob SHA automatically (required by GitHub for updates).
    Raises GitHubWriteError on failure.
    """
    content_b64 = base64.b64encode(new_content.encode("utf-8")).decode("ascii")

    body: dict[str, Any] = {
        "message": commit_message,
        "content": content_b64,
        "branch": branch,
    }

    existing_sha = await _get_file_sha(owner, repo, path, branch)
    if existing_sha:
        body["sha"] = existing_sha
        logger.info(
            "create_or_update_file: updating %s on %s (blob sha %s)",
            path, branch, existing_sha[:8],
        )
    else:
        logger.info("create_or_update_file: creating new file %s on %s", path, branch)

    status, data = await _write("PUT", f"/repos/{owner}/{repo}/contents/{path}", body)

    if status not in (200, 201):
        msg = (data or {}).get("message", "") if isinstance(data, dict) else str(data)
        raise GitHubWriteError(f"create_or_update_file failed for '{path}' ({status}): {msg}")

    commit_sha = ""
    if isinstance(data, dict):
        commit_sha = ((data.get("commit") or {}).get("sha") or "")[:8]
    logger.info("create_or_update_file: committed %s (sha %s)", path, commit_sha)
    return data  # type: ignore[return-value]


async def create_draft_pr(
    owner: str,
    repo: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str,
) -> dict[str, Any]:
    """Open a draft pull request.

    Raises GitHubWriteError on failure.
    """
    logger.info(
        "create_draft_pr: %s/%s  head=%s  base=%s",
        owner, repo, head_branch, base_branch,
    )
    status, data = await _write(
        "POST",
        f"/repos/{owner}/{repo}/pulls",
        {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": True,
        },
    )

    if status not in (200, 201):
        msg = (data or {}).get("message", "") if isinstance(data, dict) else str(data)
        raise GitHubWriteError(f"create_draft_pr failed ({status}): {msg}")

    pr_number = data.get("number", 0) if isinstance(data, dict) else 0
    pr_url = data.get("html_url", "") if isinstance(data, dict) else ""
    logger.info("create_draft_pr: opened draft PR #%d — %s", pr_number, pr_url)
    return data  # type: ignore[return-value]


async def comment_on_issue(
    owner: str,
    repo: str,
    issue_number: int,
    body: str,
) -> dict[str, Any]:
    """Post a comment on an issue.

    Raises GitHubWriteError on failure.
    """
    logger.info("comment_on_issue: %s/%s#%d", owner, repo, issue_number)
    status, data = await _write(
        "POST",
        f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
        {"body": body},
    )

    if status not in (200, 201):
        msg = (data or {}).get("message", "") if isinstance(data, dict) else str(data)
        raise GitHubWriteError(f"comment_on_issue failed ({status}): {msg}")

    comment_url = data.get("html_url", "") if isinstance(data, dict) else ""
    logger.info("comment_on_issue: posted — %s", comment_url)
    return data  # type: ignore[return-value]
