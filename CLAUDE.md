# CLAUDE.md

# IssueOps — Autonomous GitHub Issue-to-Draft PR Engineering Agent

## Project Overview

IssueOps is a multi-agent autonomous engineering workflow built for a hackathon sponsored track requiring:

- Multi-agent autonomy
- long-running async workflows
- webhook-driven execution
- real tool use
- deep reasoning
- external integrations
- real side effects
- optional observability bonus

The system converts a GitHub issue into an investigated draft pull request autonomously.

Core demo flow:

GitHub Issue Created
→ webhook received
→ issue analyzed
→ repository context gathered
→ root cause investigated
→ fix proposed
→ branch created
→ commit created
→ draft PR opened
→ original issue commented

This must be a real working product, not a simulation.

---

# Core Product Goal

Deliver a 5-minute hackathon demo where:

1. A judge creates a GitHub issue in the demo repo
2. The webhook triggers automatically
3. The autonomous workflow runs without human intervention
4. The system investigates the issue
5. The system creates a real draft PR
6. The original issue receives a real explanatory comment
7. The workflow is observable in logs (Omium optional later)

Primary success metric:

"GitHub issue → autonomous draft PR"

---

# Scope Boundaries

## MUST DO

The system MUST:

- receive GitHub webhook events
- validate webhook signatures
- acknowledge quickly
- process work asynchronously
- use multiple distinct agents
- use structured inter-agent communication
- perform real GitHub API reads
- perform real GitHub API writes
- use LLM reasoning
- create draft PRs
- comment on original issues
- handle failure safely
- support confidence-based escalation

---

## MUST NOT DO

Do NOT:

- auto-merge code
- deploy anything
- mutate production systems
- close issues automatically
- perform destructive actions
- execute arbitrary unsafe code
- overbuild enterprise infrastructure
- add unnecessary abstractions
- optimize prematurely

---

# Hackathon Constraints

Time available: 24 hours

Priorities:

1. demo reliability
2. visible autonomy
3. real side effects
4. judging score
5. implementation simplicity

NOT priorities:

- production-scale architecture
- perfect scalability
- exhaustive edge-case coverage
- enterprise security hardening
- microservices

When in doubt:
choose simpler implementation.

---

# Technical Stack (LOCKED)

## Language
Python 3.11+

## Backend
FastAPI

Reason:
webhook handling + async simplicity

---

## Agent Orchestration
LangGraph

Reason:
workflow is naturally a stateful branching graph

Required capabilities:

- state passing
- agent nodes
- conditional routing
- resumability if feasible

DO NOT replace with custom orchestration unless explicitly instructed.

---

## LLM
Gemini 2.5 Flash (preferred)

Fallback:
OpenRouter-compatible models

Reason:
free tier friendliness + speed

Avoid expensive models unless explicitly approved.

---

## Validation
Pydantic

All agent outputs MUST use structured validated schemas.

No free-form inter-agent blobs.

---

## GitHub Integration
Raw GitHub REST API via httpx

Avoid PyGithub unless explicitly needed.

Required operations:

- issue read
- issue comment
- code search
- file contents
- commit history
- branch creation
- commit creation
- pull request creation

---

## Search
Tavily (preferred free tier)

Fallback:
simple web search abstraction

Keep search modular.

---

## Async Execution
FastAPI background tasks or lightweight async worker

DO NOT introduce Celery/Redis unless explicitly required.

Hackathon simplicity > enterprise correctness.

---

## Tunnel
ngrok

Used for GitHub webhook exposure.

---

## Observability
Basic structured logging first

Omium integration only AFTER core product works.

---

# Architecture

System architecture:

GitHub Webhook
→ FastAPI Endpoint
→ Background Task
→ LangGraph Workflow
→ Agents
→ GitHub Actions

---

# Agent Design

Exactly 5 logical agents.

Do NOT collapse into one giant prompt.

---

## 1. Planner Agent

Responsibility:

- coordinate workflow
- manage execution order
- evaluate routing decisions

Inputs:

- webhook payload

Outputs:

- workflow state updates

Must NOT do deep issue analysis itself.

---

## 2. Issue Analyzer Agent

Responsibility:

understand issue semantics

Extract:

- issue type
- severity
- keywords
- suspected files
- stack traces
- reproduction hints
- duplicate likelihood

Output schema required.

---

## 3. Repo Context Agent

Responsibility:

gather repository intelligence

Actions:

- code search
- issue search
- commit lookup
- file reads

Must only gather evidence.

Must NOT decide fixes.

---

## 4. Debug Agent

Responsibility:

root cause reasoning

Actions:

- combine issue context
- combine repo evidence
- web search docs/errors
- form hypothesis
- assign confidence

Output:

- root cause
- confidence
- suggested fix approach
- escalate flag

---

## 5. Fix PR Agent

Responsibility:

execution

Actions:

- generate patch
- create branch
- commit code
- create draft PR
- comment issue

Must only run when confidence threshold is met.

---

# Workflow Logic

Canonical flow:

START
→ analyze_issue
→ gather_repo_context
→ debug_root_cause
→ confidence_check

if low confidence:
    escalate_to_issue_comment
    END

if medium/high confidence:
    generate_fix
    create_branch
    commit_changes
    create_draft_pr
    comment_issue
    END

---

# State Model

Use a strongly typed shared workflow state.

Suggested fields:

- issue_id
- repo_owner
- repo_name
- issue_title
- issue_body
- analysis
- repo_context
- debug_result
- fix_result
- pr_url
- issue_comment_url
- errors
- current_step

Keep state serializable.

---

# Code Organization

Required structure:

issueops/
    app/
        main.py

    agents/
        planner.py
        analyzer.py
        repo_context.py
        debug.py
        fix_pr.py

    workflows/
        graph.py
        state.py

    tools/
        github.py
        search.py
        llm.py

    schemas/
        analysis.py
        debug.py
        github.py

    config/
        settings.py

    prompts/
        analyzer.txt
        debug.txt
        fix_pr.txt

    tests/

Keep separation strict.

---

# Coding Rules

ALWAYS:

- type hints
- async where appropriate
- pydantic validation
- modular code
- environment variables for secrets
- structured logging
- explicit error handling

NEVER:

- giant monolithic files
- hidden global state
- hardcoded secrets
- silent exception swallowing
- duplicate business logic

---

# Environment Variables

Expected:

GITHUB_TOKEN=
GITHUB_WEBHOOK_SECRET=
GEMINI_API_KEY=
TAVILY_API_KEY=
OMIUM_API_KEY=

Optional values may be blank if feature disabled.

---

# Demo Strategy

Target demo repo:

small controlled GitHub repo with intentional bugs

Requirements:

- predictable bug behavior
- searchable code
- fast execution
- low token usage

DO NOT test first against large real repositories.

---

# Implementation Strategy

Build incrementally.

Required order:

1. FastAPI webhook endpoint
2. webhook signature validation
3. background async trigger
4. GitHub API wrapper
5. issue analyzer agent
6. repo context agent
7. debug agent
8. fix PR agent
9. LangGraph wiring
10. end-to-end GitHub run
11. observability

Never jump ahead.

---

# Decision Rules for Claude

If implementation choice is ambiguous:

Prefer:

- simpler
- cheaper
- faster
- more demo-reliable

Avoid architectural gold-plating.

If a feature risks timeline:
propose simplified fallback.

If a dependency is paid:
prefer free alternative first.

If code generation is uncertain:
choose safe draft PR behavior.

---

# Success Definition

Hackathon success means:

A judge creates a GitHub issue.

Within ~1–3 minutes:

the system autonomously creates a draft PR and comments on the issue.

That is the finish line.