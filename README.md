# IssueOps

Autonomous GitHub Issue → Draft PR engineering agent.

## Demo Flow

1. Judge creates a GitHub issue in the demo repo
2. Webhook triggers automatically
3. Multi-agent LangGraph workflow runs without human intervention
4. System investigates the issue and gathers repo context
5. Debug agent forms a root cause hypothesis
6. Fix PR agent creates a real draft PR
7. Original issue receives an explanatory comment

## Stack

- **FastAPI** — webhook server
- **LangGraph** — multi-agent workflow orchestration
- **Gemini 2.5 Flash** — LLM reasoning
- **Tavily** — web search for docs/errors
- **httpx** — raw GitHub REST API calls
- **Pydantic** — validated inter-agent schemas
- **ngrok** — webhook tunnel for local dev

## Setup

```bash
cp .env.example .env
# fill in .env values

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn issueops.app.main:app --reload --port 8000
```

## Webhook

Point your GitHub webhook to:
`https://<ngrok-url>/webhook/github`

Content type: `application/json`
Events: Issues
