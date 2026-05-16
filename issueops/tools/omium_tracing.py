"""Safe Omium observability wrapper.

All public symbols silently no-op when Omium is unavailable or the API key
is missing, so instrumentation can never break application flow.
"""

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_ready = False
_project = "issueops"
_workflow_uuid: str | None = None   # resolved once from GET /api/v1/projects


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_omium(
    api_key: str | None,
    project: str = "issueops",
    api_base_url: str | None = None,
) -> None:
    """Initialise the Omium SDK and instrument LangGraph.

    auto_trace=True is required — the SDK gates every @trace() decorator
    and the LangGraph ainvoke patch on this flag. False = silent no-ops.
    """
    global _ready, _project
    if not api_key:
        logger.info("[OMIUM] no API key configured — observability disabled")
        return
    try:
        import omium
        omium.init(
            api_key=api_key,
            project=project,
            auto_trace=True,
            auto_checkpoint=True,
            api_base_url=api_base_url or None,
        )
        omium.instrument_langgraph()
        _project = project
        _ready = True
        logger.info(
            "[OMIUM] initialized — project=%s base_url=%s",
            project, api_base_url or "default",
        )
    except Exception as exc:
        logger.warning(
            "[OMIUM] init failed (%s: %s) — observability disabled",
            type(exc).__name__, exc,
        )


# ---------------------------------------------------------------------------
# Workflow UUID resolution
# ---------------------------------------------------------------------------

async def _resolve_workflow_uuid() -> str | None:
    """Return the UUID assigned to our project by the Omium backend.

    The POST /api/v1/executions endpoint validates workflow_id as a UUID —
    passing the project name string causes a 422.  We look up the UUID once
    from GET /api/v1/projects and cache it for the process lifetime.
    """
    global _workflow_uuid

    if _workflow_uuid:
        return _workflow_uuid

    try:
        import httpx
        import omium

        config = omium.get_current_config()
        if not config:
            return None

        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                f"{config.api_base_url}/projects",
                headers={"X-API-Key": config.api_key},
            )

        if response.status_code != 200:
            logger.warning("[OMIUM] GET /projects returned %s", response.status_code)
            return None

        data = response.json()
        items = data if isinstance(data, list) else data.get("projects", [])
        for item in items:
            if item.get("name") == _project:
                _workflow_uuid = item["id"]
                logger.debug("[OMIUM] resolved workflow UUID=%s", _workflow_uuid)
                return _workflow_uuid

        logger.warning("[OMIUM] project %r not found in /projects response", _project)

    except Exception as exc:
        logger.debug("[OMIUM] _resolve_workflow_uuid failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Run registration
# ---------------------------------------------------------------------------

async def create_run(issue_id: int | str, issue_title: str = "") -> None:
    """Register a run with the Omium Execution Engine before the workflow starts.

    The dashboard "AI Systems > Runs" tab only shows executions created via
    POST /api/v1/executions — trace ingestion alone does not create run records.

    Execution flow:
      1. Resolve project UUID from GET /api/v1/projects (cached after first call).
      2. POST /api/v1/executions with workflow_id=<UUID>.
      3. Call omium.set_execution_id(returned_id) so the LangGraph ainvoke
         patch embeds it in every span sent to /traces/ingest.

    Falls back to a local correlation string if the API is unreachable.
    """
    if not _ready:
        return

    fallback_id = f"github_issue_{issue_id}"

    try:
        import httpx
        import omium

        config = omium.get_current_config()
        if not config:
            return

        workflow_uuid = await _resolve_workflow_uuid()
        if not workflow_uuid:
            logger.warning("[OMIUM] workflow UUID unavailable — skipping execution creation")
            _set_fallback(fallback_id)
            return

        payload = {
            "workflow_id": workflow_uuid,
            "agent_id": f"{_project}-agent",
            "input_data": {
                "issue_id": str(issue_id),
                "issue_title": issue_title,
            },
            "metadata": {
                "source": "github_webhook",
                "issue_id": str(issue_id),
            },
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config.api_base_url}/executions",
                json=payload,
                headers={"X-API-Key": config.api_key},
            )

        if response.status_code in (200, 201):
            execution_id = response.json().get("id") or fallback_id
            omium.set_execution_id(execution_id)
            logger.info("[OMIUM] run created — execution_id=%s", execution_id)
            return

        logger.warning(
            "[OMIUM] POST /executions returned %s: %s",
            response.status_code, response.text[:200],
        )

    except Exception as exc:
        logger.debug("[OMIUM] create_run failed (%s) — using local ID", exc)

    _set_fallback(fallback_id)


def _set_fallback(fallback_id: str) -> None:
    try:
        import omium
        omium.set_execution_id(fallback_id)
        logger.debug("[OMIUM] fallback execution_id=%s", fallback_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Span decorator
# ---------------------------------------------------------------------------

def trace(name: str | None = None) -> Callable[[F], F]:
    """Thin delegation to omium.trace().

    omium.trace() checks is_initialized() at call time, so decorating at
    module-import time (before init) is safe.
    """
    try:
        import omium
        return omium.trace(name=name)  # type: ignore[return-value]
    except Exception:
        return lambda fn: fn  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

async def checkpoint(name: str) -> None:
    """Add a named checkpoint event to the current active span.

    Uses get_current_tracer() (ContextVar) — attaches to whichever span is
    active in the current async task. Silent no-op when no tracer is active.
    """
    if not _ready:
        return
    try:
        from omium.integrations.tracer import get_current_tracer
        tracer = get_current_tracer()
        if tracer:
            tracer.add_event(f"checkpoint:{name}", {"checkpoint_name": name})
            logger.debug("[OMIUM] checkpoint=%s", name)
    except Exception as exc:
        logger.debug("[OMIUM] checkpoint %r failed: %s", name, exc)
