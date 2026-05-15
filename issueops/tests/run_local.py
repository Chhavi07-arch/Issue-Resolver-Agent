"""Local test runner — invoke the LangGraph workflow directly without a server.

Usage (mock payload, LLM enabled if GEMINI_API_KEY set, dry-run writes):
    python issueops/tests/run_local.py

Usage (force heuristic fallback, no LLM):
    python issueops/tests/run_local.py --no-llm

Usage (real repo, synthetic issue):
    python issueops/tests/run_local.py --owner octocat --repo Hello-World

Usage (real repo, real issue):
    python issueops/tests/run_local.py --owner psf --repo requests --issue 6725

Usage (live GitHub writes — requires GITHUB_TOKEN and a sacrificial repo):
    python issueops/tests/run_local.py --owner my-org --repo my-repo --issue 1 --live-writes
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure the project root (parent of issueops/) is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from issueops.tools import github as gh
from issueops.workflows.graph import workflow
from issueops.workflows.state import WorkflowState

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("run_local")


# ---------------------------------------------------------------------------
# State builders
# ---------------------------------------------------------------------------

def _state_from_payload_file(dry_run_writes: bool = True) -> WorkflowState:
    payload_path = Path(__file__).parent / "test_payload.json"
    with open(payload_path) as f:
        payload = json.load(f)
    issue = payload["issue"]
    repo = payload["repository"]
    return _build_state(
        issue_id=issue["number"],
        owner=repo["owner"]["login"],
        repo=repo["name"],
        title=issue["title"],
        body=issue.get("body") or "",
        dry_run_writes=dry_run_writes,
    )


async def _state_from_real_issue(
    owner: str, repo: str, issue_number: int, dry_run_writes: bool = True
) -> WorkflowState:
    logger.info("Fetching real issue %s/%s#%d from GitHub...", owner, repo, issue_number)
    data = await gh.get_issue(owner, repo, issue_number)
    if not data:
        logger.error("Could not fetch issue — check GITHUB_TOKEN and issue number")
        sys.exit(1)
    return _build_state(
        issue_id=data["number"],
        owner=owner,
        repo=repo,
        title=data["title"],
        body=data.get("body") or "",
        dry_run_writes=dry_run_writes,
    )


def _state_from_synthetic(
    owner: str, repo: str, disable_llm: bool = False, dry_run_writes: bool = True
) -> WorkflowState:
    """Synthetic issue that exercises the FIX path (high-confidence signals)."""
    return _build_state(
        issue_id=99,
        owner=owner,
        repo=repo,
        disable_llm=disable_llm,
        dry_run_writes=dry_run_writes,
        title="App crashes with AttributeError when processing empty input",
        body=(
            "The application crashes with an AttributeError: "
            "'NoneType' object has no attribute 'strip' when a user submits an empty form.\n\n"
            "Traceback (most recent call last):\n"
            "  File \"src/main.py\", line 42, in handle_request\n"
            "    result = user_input.strip().lower()\n"
            "AttributeError: 'NoneType' object has no attribute 'strip'\n\n"
            "Steps to reproduce:\n"
            "1. Send a request with an empty body\n"
            "2. Observe 500 error in logs"
        ),
    )


def _build_state(
    issue_id: int, owner: str, repo: str, title: str, body: str,
    disable_llm: bool = False,
    dry_run_writes: bool = True,
) -> WorkflowState:
    return {
        "issue_id": issue_id,
        "repo_owner": owner,
        "repo_name": repo,
        "issue_title": title,
        "issue_body": body,
        "analysis": None,
        "repo_context": None,
        "debug_result": None,
        "fix_result": None,
        "pr_url": None,
        "issue_comment_url": None,
        "errors": [],
        "current_step": "received",
        "disable_llm": disable_llm,
        "dry_run_writes": dry_run_writes,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(result: dict, elapsed: float) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("ISSUEOPS WORKFLOW RESULT")
    print(sep)
    print(f"  Repo        : {result.get('repo_owner')}/{result.get('repo_name')}")
    print(f"  Issue       : #{result.get('issue_id')} — {result.get('issue_title')}")
    print(f"  Final step  : {result.get('current_step')}")
    print(f"  Write mode  : {'DRY RUN (no GitHub writes)' if result.get('dry_run_writes', True) else 'LIVE'}")
    print(f"  Elapsed     : {elapsed:.2f}s")
    print()

    rc = result.get("repo_context") or {}
    if rc:
        print(f"  Repo context:")
        print(f"    files     : {rc.get('relevant_files', [])}")
        print(f"    commits   : {len(rc.get('recent_commits', []))}")
        print(f"    issues    : {len(rc.get('related_issues', []))}")
        print(f"    search    : {len(rc.get('code_search_results', []))}")
        if rc.get("partial"):
            print(f"    partial   : {rc.get('errors', [])}")
        print()

    dr = result.get("debug_result") or {}
    if dr:
        print(f"  Debug:")
        print(f"    confidence: {dr.get('confidence', 0):.0%}")
        print(f"    escalated : {dr.get('escalate', True)}")
        print(f"    root cause: {dr.get('root_cause', 'N/A')[:100]}")
        print()

    fr = result.get("fix_result") or {}

    # ---- Fix / escalation section ----
    if fr.get("escalated"):
        print(f"  Escalated   : {fr.get('reason')}")
    elif result.get("current_step") == "write_failed":
        print(f"  Write error : {fr.get('write_error')}")
    else:
        ready = fr.get("ready_for_pr", False)
        mock = fr.get("mock_writes", True)
        mode_tag = "[DRY RUN]" if mock else "[LIVE]"
        status = "READY FOR PR" if ready else "needs review"
        print(f"  Fix {mode_tag}  [{status}]")
        print(f"    confidence   : {fr.get('confidence', 0):.0%}")
        print(f"    ready_for_pr : {ready}")
        if fr.get("patch_plan"):
            print(f"    plan         : {fr['patch_plan'][:120]}")
        if fr.get("files_to_modify"):
            print(f"    files        : {fr['files_to_modify']}")
        if fr.get("validation_notes"):
            print(f"    validation   : {fr['validation_notes'][:120]}")

        # Diff preview
        diffs = fr.get("diffs") or {}
        if diffs:
            print()
            for path, diff_text in diffs.items():
                preview_lines = diff_text.splitlines()[:25]
                print(f"    --- Diff: {path} ({len(diff_text.splitlines())} lines) ---")
                for line in preview_lines:
                    print(f"    {line}")
                if len(diff_text.splitlines()) > 25:
                    print(f"    ... ({len(diff_text.splitlines()) - 25} more lines)")
        else:
            print("    (no diffs — fix not validated against file content)")

        if fr.get("branch"):
            print()
            print(f"    branch       : {fr['branch']}")
        if fr.get("pr_title"):
            print(f"    pr_title     : {fr['pr_title']}")

    if result.get("pr_url"):
        mock_tag = " [dry run]" if (fr.get("mock_writes", True)) else ""
        print(f"\n  PR URL      : {result['pr_url']}{mock_tag}")
    if result.get("issue_comment_url"):
        mock_tag = " [dry run]" if (fr.get("mock_writes", True)) else ""
        print(f"  Comment URL : {result['issue_comment_url']}{mock_tag}")

    errors = result.get("errors") or []
    if errors:
        print(f"\n  ERRORS: {errors}")

    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="IssueOps local workflow runner")
    parser.add_argument("--owner", help="GitHub repo owner (enables real API mode)")
    parser.add_argument("--repo", help="GitHub repo name")
    parser.add_argument("--issue", type=int, help="Issue number to fetch from GitHub")
    parser.add_argument("--title", help="Custom issue title (overrides synthetic issue)")
    parser.add_argument("--body", help="Custom issue body (overrides synthetic issue)")
    parser.add_argument(
        "--no-full-state",
        action="store_true",
        help="Skip printing full state JSON (cleaner output)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM and force heuristic fallback (useful for offline testing)",
    )
    parser.add_argument(
        "--live-writes",
        action="store_true",
        help=(
            "Enable real GitHub write operations (branch, commit, PR, comment). "
            "Requires GITHUB_TOKEN. Use against a sacrificial/test repo only. "
            "Default is dry-run (no writes)."
        ),
    )
    args = parser.parse_args()

    dry_run_writes = not args.live_writes
    llm_label = " [LLM DISABLED]" if args.no_llm else ""
    write_label = " [DRY RUN]" if dry_run_writes else " [LIVE WRITES]"

    if not dry_run_writes:
        logger.warning(
            "Live writes enabled — will create real branch, commit, PR, and comment"
        )

    # Build initial state
    if args.owner and args.repo and args.issue:
        initial_state = await _state_from_real_issue(
            args.owner, args.repo, args.issue, dry_run_writes=dry_run_writes
        )
        initial_state["disable_llm"] = args.no_llm
        mode = f"real issue {args.owner}/{args.repo}#{args.issue}{llm_label}{write_label}"
    elif args.owner and args.repo and (args.title or args.body):
        initial_state = _build_state(
            issue_id=99,
            owner=args.owner,
            repo=args.repo,
            title=args.title or "(no title)",
            body=args.body or "",
            disable_llm=args.no_llm,
            dry_run_writes=dry_run_writes,
        )
        mode = f"custom issue against {args.owner}/{args.repo}{llm_label}{write_label}"
    elif args.owner and args.repo:
        initial_state = _state_from_synthetic(
            args.owner, args.repo, disable_llm=args.no_llm, dry_run_writes=dry_run_writes
        )
        mode = f"synthetic issue against {args.owner}/{args.repo}{llm_label}{write_label}"
    else:
        initial_state = _state_from_payload_file(dry_run_writes=dry_run_writes)
        initial_state["disable_llm"] = args.no_llm
        mode = f"mock payload (tests/test_payload.json){llm_label}{write_label}"

    logger.info("Mode: %s", mode)
    logger.info(
        "Starting workflow — issue #%s: %s",
        initial_state["issue_id"],
        initial_state["issue_title"],
    )

    start = time.monotonic()
    result = await workflow.ainvoke(initial_state)
    elapsed = time.monotonic() - start

    if not args.no_full_state:
        print("\n--- FULL FINAL STATE ---")
        print(json.dumps(result, indent=2, default=str))

    _print_summary(result, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
