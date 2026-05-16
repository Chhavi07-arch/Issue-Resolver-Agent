"""FastAPI application — webhook entry point."""

import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, status

from issueops.config.settings import settings
from issueops.tools.omium_tracing import create_run, init_omium
from issueops.workflows.state import WorkflowState

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("IssueOps starting — confidence_threshold=%.2f", settings.confidence_threshold)
    init_omium(
        api_key=settings.omium_api_key,
        api_base_url=settings.omium_api_url or None,
    )
    yield
    logger.info("IssueOps shutting down")


app = FastAPI(title="IssueOps", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Webhook signature validation
# ---------------------------------------------------------------------------

def _verify_signature(payload: bytes, signature_header: str | None) -> None:
    """Raise 401 if the GitHub HMAC-SHA256 signature is invalid."""
    if not settings.github_webhook_secret:
        logger.warning("Webhook secret not set — skipping signature validation (dev mode)")
        return

    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing signature")

    expected = hmac.new(
        settings.github_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(f"sha256={expected}", signature_header):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")


# ---------------------------------------------------------------------------
# Background workflow runner
# ---------------------------------------------------------------------------

async def _run_workflow(initial_state: WorkflowState) -> None:
    """Invoke the LangGraph workflow asynchronously."""
    from issueops.workflows.graph import workflow

    issue_id = initial_state["issue_id"]
    start = time.monotonic()

    try:
        logger.info(
            "Workflow starting — issue=#%s repo=%s/%s title='%s'",
            issue_id,
            initial_state["repo_owner"],
            initial_state["repo_name"],
            initial_state["issue_title"],
        )

        await create_run(issue_id, initial_state["issue_title"])
        result = await workflow.ainvoke(initial_state)

        elapsed = time.monotonic() - start
        logger.info(
            "Workflow complete — issue=#%s step=%s pr_url=%s comment_url=%s elapsed=%.1fs",
            issue_id,
            result.get("current_step"),
            result.get("pr_url") or "N/A",
            result.get("issue_comment_url") or "N/A",
            elapsed,
        )

    except Exception:
        elapsed = time.monotonic() - start
        logger.exception(
            "Workflow failed — issue=#%s elapsed=%.1fs", issue_id, elapsed
        )


# ---------------------------------------------------------------------------
# GitHub webhook endpoint
# ---------------------------------------------------------------------------

@app.post("/webhook/github", status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
):
    """Receive GitHub webhook, validate signature, dispatch async workflow."""
    payload = await request.body()
    _verify_signature(payload, x_hub_signature_256)

    event = x_github_event or "unknown"
    logger.info("Webhook received — event=%s", event)

    if event != "issues":
        logger.debug("Ignoring non-issues event: %s", event)
        return {"ignored": True, "event": event}

    data = json.loads(payload)
    action = data.get("action")

    if action not in ("opened", "reopened"):
        logger.debug("Ignoring issues event with action=%s", action)
        return {"ignored": True, "action": action}

    issue = data["issue"]
    repo = data["repository"]

    initial_state: WorkflowState = {
        "issue_id": issue["number"],
        "repo_owner": repo["owner"]["login"],
        "repo_name": repo["name"],
        "issue_title": issue["title"],
        "issue_body": issue.get("body") or "",
        "analysis": None,
        "repo_context": None,
        "debug_result": None,
        "fix_result": None,
        "pr_url": None,
        "issue_comment_url": None,
        "errors": [],
        "current_step": "received",
        "dry_run_writes": not settings.live_writes_enabled,  # controlled by LIVE_WRITES_ENABLED in .env
    }

    background_tasks.add_task(_run_workflow, initial_state)
    logger.info("Workflow queued — issue=#%s '%s'", issue["number"], issue["title"])

    return {"status": "accepted", "issue": issue["number"]}
