# IssueOps

**Autonomous GitHub Issue → Draft PR engineering agent.**

IssueOps receives a GitHub issue webhook, reasons about the codebase, diagnoses the root cause with an LLM, and — when confident enough — creates a real branch, commits a patch, opens a draft PR, and comments on the original issue. No human in the loop.

---


## Developed By

- [Chhavi](https://github.com/chhavi07-arch)
- [Sanaa Ara](https://github.com/sanaa-duhh)

---

## The Problem

Engineering teams spend hours triaging issues before a single line of code changes:

- Reading the issue, reproducing it, understanding which file owns the bug
- Searching commit history and related issues for context
- Forming a hypothesis about root cause
- Drafting a minimal fix
- Opening a PR, linking it back, explaining the reasoning

For well-scoped bugs in understood codebases, most of those steps are mechanical. IssueOps automates them.

---

## Solution

IssueOps is a webhook-driven, multi-agent workflow that turns a GitHub issue into an investigated draft PR in under two minutes.

A LangGraph pipeline coordinates five specialized agents. Each agent has a single responsibility and passes typed state to the next. The debug agent assigns a confidence score to its diagnosis. If confidence meets the threshold, the fix agent generates a patch, validates it against the real file content, and executes the GitHub write operations. If confidence is too low, the agent posts a structured investigation summary instead of touching code.

---

## Key Features

- **Real autonomous action** — creates branches, commits code, opens draft PRs, and comments on issues via the GitHub REST API
- **Confidence-gated execution** — only writes to the repo when diagnosis confidence ≥ threshold (default 0.60); escalates gracefully otherwise
- **Three-mode patch generation** — deterministic detectors → LLM find-and-replace (Mode A) → line-anchored patching (Mode B) → surgical single-file forced edit (Mode C)
- **Patch validation pipeline** — snippet matching, syntax checks, brace balance, diff size gates, and post-apply verification before any file is committed
- **Structured LLM output** — all agent outputs are Pydantic models; no free-form blobs between agents
- **Safe by default** — `LIVE_WRITES_ENABLED=false` runs the full workflow in dry-run mode, returning mock URLs without touching GitHub
- **Diagnosis-only PR mode** — when patching fails but diagnosis confidence is ≥ 0.75, optionally opens a draft PR containing the root cause analysis and suggested fix for human implementation
- **Omium observability** — full span tracing across all agents plus five workflow checkpoints, with execution registration in the Omium dashboard
- **HMAC-SHA256 webhook validation** — verifies every inbound GitHub webhook signature

---

## Architecture

```
GitHub Issue Event
       │
       ▼
┌─────────────────────┐
│  FastAPI Webhook     │  POST /webhook/github
│  (HMAC verified)     │  202 Accepted immediately
└──────────┬──────────┘
           │ background task
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Planner  │───▶│ Analyzer │───▶│   Repo Context       │  │
│  │          │    │          │    │                      │  │
│  │ Validate │    │ Classify │    │ Tree scan · File     │  │
│  │ fields   │    │ type /   │    │ fetch · Code search  │  │
│  │          │    │ severity │    │ · Commit history     │  │
│  └──────────┘    └──────────┘    └──────────┬───────────┘  │
│                                             │               │
│                                             ▼               │
│                                    ┌─────────────────┐      │
│                                    │  Debug Agent    │      │
│                                    │                 │      │
│                                    │ Root cause ·   │      │
│                                    │ Confidence ·   │      │
│                                    │ Fix strategy   │      │
│                                    └───────┬─────────┘      │
│                                            │                │
│                              ┌─────────────┴──────────┐    │
│                              │   confidence_router     │    │
│                              └────────┬────────────────┘    │
│                                       │                     │
│                          ┌────────────▼──────────────┐      │
│                          │ confidence ≥ threshold?   │      │
│                          └──────┬────────────┬───────┘      │
│                           YES   │            │  NO           │
│                                 ▼            ▼               │
│                        ┌──────────┐   ┌──────────────┐      │
│                        │ Fix PR   │   │  Escalate    │      │
│                        │ Agent    │   │  to Comment  │      │
│                        │          │   │              │      │
│                        │ Detect · │   │ Post invest- │      │
│                        │ Patch ·  │   │ igation      │      │
│                        │ Validate │   │ summary      │      │
│                        │ Commit   │   └──────────────┘      │
│                        └──────────┘                         │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
  GitHub: branch + commit + draft PR + issue comment
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Agent orchestration | LangGraph |
| LLM (primary) | OpenAI-compatible (Azure endpoint) |
| LLM (alternative) | Google Gemini 2.5 Flash |
| Structured output | Pydantic v2 |
| GitHub API | httpx (raw REST, no SDK) |
| Observability | Omium SDK |
| Deployment | Railway |
| Tunnel (local dev) | ngrok |

---

## Agent Workflow

### 1. Planner (`plan`)
Validates that all required fields are present in the incoming webhook payload (`issue_id`, `repo_owner`, `repo_name`, `issue_title`). Aborts the workflow early rather than passing incomplete state downstream.

### 2. Issue Analyzer (`analyze_issue`)
Classifies the issue using an LLM with a structured prompt. Extracts:
- Issue type (`bug` / `feature` / `docs` / `question` / `other`)
- Severity (`critical` / `high` / `medium` / `low`)
- Keywords, suspected files, stack traces, reproduction hints, duplicate likelihood

Falls back to keyword heuristics if the LLM is unavailable.

### 3. Repo Context Agent (`gather_repo_context`)
Builds targeted evidence from the repository without reading the whole codebase:
- Fetches the repo file tree and scores paths against 13 subsystem patterns (`auth`, `cache`, `service`, `repository`, `controller`, `middleware`, `validation`, and more)
- Prioritizes files explicitly named in stack traces
- Expands call chains (finds the service sibling to a matching controller, etc.)
- Fetches file contents, recent commits, and related issues

### 4. Debug Agent (`debug_root_cause`)
Reasons over the issue text, file snippets, commit history, and related issues to produce:
- Root cause statement
- Diagnosis confidence (0.0–1.0)
- Relevant files and suspected symbols
- Concrete repair strategy
- Escalation recommendation

Falls back to signal-based heuristics if the LLM is unavailable.

### 5. Confidence Router
Routes the workflow based on `debug_result.confidence` vs. `settings.confidence_threshold` (default 0.60). If the debug agent sets `escalate=True` or confidence is below threshold, the workflow routes to escalation. Otherwise it proceeds to fix generation.

### 6a. Fix PR Agent (`generate_fix_and_pr`)
Runs a staged patch generation pipeline:

| Stage | Mechanism | When used |
|---|---|---|
| **Deterministic** | Pattern-based bug detectors | Always tried first — zero LLM calls |
| **Mode A** | LLM find-and-replace with verbatim snippet copying | Primary LLM path |
| **Mode B** | LLM line-anchored patching (cites line numbers; system extracts snippet) | If Mode A produces no valid edits |
| **Mode C** | Surgical single-file forced edit | If A+B fail AND `diagnosis_confidence ≥ threshold` |

Every generated patch passes through a validation pipeline before any GitHub write:
1. Snippet presence check — `find_snippet` must exist verbatim in the target file
2. Syntax validation — balanced braces, correct indentation
3. Diff application and size gate
4. Post-apply brace balance check
5. Post-commit structural verification

### 6b. Escalate to Comment (`escalate_to_comment`)
Posts a structured investigation summary to the issue when confidence is too low to patch automatically. Includes root cause, repair strategy, relevant symbols, and files to investigate.

---

## Safety Mechanisms

### Confidence Gating
The system only executes GitHub write operations when the debug agent's diagnosis confidence meets the configured threshold. Below threshold, it posts an investigation summary and stops — no branch, no commit.

```
CONFIDENCE_THRESHOLD=0.6   # default; raise for stricter gating
```

### Dry-Run Mode
All GitHub writes are skipped by default (`LIVE_WRITES_ENABLED=false`). The full workflow still runs and mock PR/comment URLs are returned — useful for testing the pipeline without side effects. Set `LIVE_WRITES_ENABLED=true` to enable real writes.

### Patch Validation
A generated patch is rejected and the next patch mode is tried if:
- The `find_snippet` text is not found verbatim in the target file
- The replacement introduces unbalanced braces or broken syntax
- The resulting diff is empty, oversized, or a no-op
- Post-apply verification detects structural regressions

Only a patch that passes all gates reaches `_execute_writes`.

### Diagnosis-Only PR
When patch generation fails completely but diagnosis confidence is high (≥ 0.75), IssueOps can open a draft PR containing only a `.issueops/diagnosis-issue-N.md` file — root cause, repair strategy, and relevant symbols — for a human engineer to implement. Opt in with `ALLOW_DIAGNOSIS_ONLY_PR=true`.

### Webhook Signature Verification
Every inbound webhook is verified against `GITHUB_WEBHOOK_SECRET` using HMAC-SHA256. Invalid or missing signatures return HTTP 401 before any processing occurs.

---

## Omium Observability

IssueOps is instrumented with the Omium SDK for full execution tracing.

**Spans (7 total):**

| Span | Agent / function |
|---|---|
| `plan` | Planner |
| `analyze_issue` | Issue Analyzer |
| `gather_repo_context` | Repo Context Agent |
| `debug_root_cause` | Debug Agent |
| `confidence_router` | Graph router |
| `generate_fix_and_pr` | Fix PR Agent |
| `execute_writes` | GitHub write pipeline |

**Checkpoints (5 total):**

| Checkpoint | When |
|---|---|
| `after_issue_parsing` | Issue classification complete |
| `after_repo_context` | Repository evidence gathered |
| `after_debugging` | Root cause diagnosis complete |
| `before_fix_generation` | Entering patch generation |
| `before_pr_creation` | Patch validated, writes starting |

Each workflow run is registered via `POST /api/v1/executions` before `workflow.ainvoke`, so it appears as a named run in the Omium AI Systems dashboard with the full span tree and checkpoint timeline.

Set `OMIUM_API_KEY` and `OMIUM_API_URL` to enable. The integration silently no-ops when either is absent — observability never breaks application flow.

---

## Local Setup

### Prerequisites

- Python 3.11+
- A GitHub personal access token (with `repo` scope)
- An OpenAI-compatible API key **or** a Google Gemini API key
- ngrok (for exposing the local server to GitHub webhooks)

### Install

```bash
git clone <repo-url>
cd Hail_Omium

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your keys
```

---

## Environment Variables

```env
# GitHub
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# LLM provider: "openai" | "gemini"
LLM_PROVIDER=openai

# OpenAI-compatible (Azure / custom endpoint)
OPENAI_API_KEY=
OPENAI_BASE_URL=https://your-resource.services.ai.azure.com/openai/v1
OPENAI_MODEL=gpt-5.4-mini

# Google Gemini (alternative provider)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

# Omium observability (optional)
OMIUM_API_KEY=
OMIUM_API_URL=

# Workflow behaviour
CONFIDENCE_THRESHOLD=0.6
LIVE_WRITES_ENABLED=false
ALLOW_DIAGNOSIS_ONLY_PR=false

# Tuning
LLM_TIMEOUT=45
HTTP_TIMEOUT=20
LOG_LEVEL=INFO
DEFAULT_BASE_BRANCH=main
```

---

## Running Locally

### Start the server

```bash
uvicorn issueops.app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}
```

### Expose to GitHub via ngrok

```bash
ngrok http 8000
```

Copy the `https://` forwarding URL — you'll need it when configuring the GitHub webhook.

### Local dry-run test (no webhook required)

```bash
python -m issueops.tests.run_local
```

Replays `issueops/tests/test_payload.json` through the full workflow and logs the result. Always runs in dry-run mode regardless of `LIVE_WRITES_ENABLED`.

---

## Deployment on Railway

1. Push the repo to GitHub.
2. Create a new Railway project → **Deploy from GitHub repo**.
3. Set all environment variables in the Railway dashboard (same as `.env` above).
4. Set the start command:
   ```
   uvicorn issueops.app.main:app --host 0.0.0.0 --port $PORT
   ```
5. Railway assigns a public URL automatically — use it as the GitHub webhook URL.

---

## GitHub Webhook Setup

1. Go to your target repo → **Settings → Webhooks → Add webhook**.
2. **Payload URL:** `https://<your-domain>/webhook/github`
3. **Content type:** `application/json`
4. **Secret:** the value of `GITHUB_WEBHOOK_SECRET` in your environment
5. **Which events:** select **Issues** only
6. Save. GitHub sends a ping — check Recent Deliveries to confirm delivery.

IssueOps processes `issues.opened` and `issues.reopened` actions. All other events return 200 and are ignored.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — `{"status": "ok", "version": "0.1.0"}` |
| `POST` | `/webhook/github` | GitHub webhook receiver — validates HMAC, queues workflow, returns 202 |

---

## Demo Walkthrough

**Recommended setup:** a small GitHub repository with one intentional, reproducible bug.

1. **Create a GitHub issue** describing the bug (title + body with error message or stack trace).
2. GitHub delivers `issues.opened` to `POST /webhook/github`.
3. FastAPI returns `202 Accepted` in < 100ms; the workflow starts in the background.
4. **Planner** validates the webhook fields.
5. **Analyzer** classifies the issue, extracts keywords and suspected files.
6. **Repo Context** fetches the relevant source files, recent commits, and related issues.
7. **Debug Agent** reasons about the root cause and returns a confidence score.
8. If confidence ≥ 0.60:
   - **Fix PR Agent** generates a patch through the A/B/C pipeline.
   - Patch is validated against the live file.
   - Branch `issueops/fix-issue-<N>` is created.
   - Patched file is committed.
   - Draft PR is opened with root cause summary and diff.
   - Original issue receives a comment linking to the PR.
9. If confidence < 0.60:
   - Issue receives a structured comment with root cause, suspected symbols, and suggested fix for human follow-up.

**Typical end-to-end time:** 30–90 seconds.

---

## Sample Output

### Draft PR body (autonomous fix)

```markdown
## IssueOps Autonomous Fix

Resolves #42

### Root Cause
The getUser method compares boxed Long values with == instead of .equals(),
causing identity comparison to fail for values outside the JVM integer cache (-128 to 127).

### Fix Strategy
Replace == with Objects.equals() on the userId comparison in UserService.java line 87.

### Files Changed
- `src/main/java/com/example/UserService.java`

### Generated Diff
...

> **Draft PR** — generated autonomously by IssueOps. Human review required before merge.
```

### Escalation comment (low confidence)

```markdown
## IssueOps — Investigation Complete

IssueOps investigated issue #42 but patch confidence was too low to generate
an automated fix.

**Root Cause Analysis** (diagnosis confidence: 45%):
The error appears to originate in the authentication middleware, but the exact
call site could not be isolated from the available file context.

**Suggested Fix:**
Inspect the session token validation logic in AuthMiddleware. The stack trace
suggests a null dereference on an uninitialized token field.

**Relevant symbols:** `validateToken`, `SessionManager`, `AuthMiddleware`

**Files to investigate:**
- `src/middleware/AuthMiddleware.java`
- `src/service/SessionManager.java`

> Automated patch was not applied — human review required.
```

---

## Project Structure

```
issueops/
├── app/
│   └── main.py              # FastAPI app, webhook endpoint, lifespan
├── agents/
│   ├── planner.py           # Field validation, workflow entry
│   ├── analyzer.py          # Issue classification and extraction
│   ├── repo_context.py      # Repository intelligence gathering
│   ├── debug.py             # Root cause reasoning
│   └── fix_pr.py            # Patch generation, validation, GitHub writes
├── workflows/
│   ├── graph.py             # LangGraph StateGraph definition and routing
│   └── state.py             # WorkflowState TypedDict
├── tools/
│   ├── github.py            # GitHub REST API wrapper (14 functions)
│   ├── llm.py               # LLM client (OpenAI-compatible + Gemini)
│   ├── omium_tracing.py     # Omium SDK wrapper (spans + checkpoints)
│   ├── bug_detectors.py     # Deterministic pattern-based bug detectors
│   ├── patch_builder.py     # Snippet extraction and edit application
│   ├── patch_validator.py   # Edit validation and diff generation
│   ├── patch_syntax.py      # Syntax and brace balance checks
│   ├── patch_verifier.py    # Post-apply structural verification
│   ├── symbol_search.py     # Symbol definition line resolution
│   └── search.py            # Web search (Tavily)
├── schemas/
│   ├── analysis.py          # IssueAnalysis, IssueType, IssueSeverity
│   ├── debug.py             # DebugResult
│   └── fix.py               # FixResult, FileEdit, LinePatchResult, LineAnchoredEdit
├── prompts/
│   ├── analyzer.txt         # Issue classification prompt
│   ├── debug.txt            # Root cause analysis prompt
│   ├── fix_pr.txt           # Mode A: find-and-replace fix prompt
│   ├── fix_pr_anchored.txt  # Mode B: line-anchored fix prompt
│   └── fix_pr_surgical.txt  # Mode C: surgical single-file fix prompt
├── config/
│   └── settings.py          # Pydantic settings (loaded from .env)
└── tests/
    ├── run_local.py         # Local dry-run test runner
    └── test_payload.json    # Sample GitHub webhook payload
```

---

## Future Improvements

- **Test generation** — alongside the fix PR, open a second commit adding a regression test
- **Multi-file edits** — extend the patch schema and write pipeline beyond the current 2-file limit
- **Embeddings-based context retrieval** — replace tree scoring with semantic search over a repo index
- **PR feedback loop** — re-run the debug+fix pipeline when a reviewer requests changes
- **Self-hosted LLM support** — Ollama / vLLM endpoint compatibility for air-gapped deployments
- **Issue template parsing** — structured intake forms would improve analyzer accuracy on repos that use them

---



## Credits

Built for the Omium Hackathon.

| | |
|---|---|
| Observability sponsor | [Omium](https://omium.ai) |

---

*IssueOps generates draft PRs requiring human review before merge. It never auto-merges, deploys, or closes issues.*
