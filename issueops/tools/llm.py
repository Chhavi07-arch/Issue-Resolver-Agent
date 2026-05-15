"""LLM abstraction — supports OpenAI-compatible and Gemini providers.

Provider is selected via settings.llm_provider ("openai" | "gemini").
All callers use generate_text() and generate_structured() — provider is invisible.
"""

import asyncio
import json
import logging
import time
from typing import Optional, Type

from pydantic import BaseModel, ValidationError

from issueops.config.settings import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


# ---------------------------------------------------------------------------
# Lazy client singletons
# ---------------------------------------------------------------------------

_gemini_client = None
_openai_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from google import genai
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    return _openai_client


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Strip markdown code fences and return the bare JSON string."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_text(prompt: str) -> str:
    """Send a prompt to the active LLM provider and return raw text.

    Raises LLMError on API failure or timeout.
    """
    provider = settings.llm_provider
    model = settings.openai_model if provider == "openai" else settings.gemini_model

    start = time.monotonic()
    logger.info(
        "LLM: request start — provider=%s model=%s prompt_chars=%d",
        provider, model, len(prompt),
    )

    if provider == "openai":
        client = _get_openai_client()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=settings.llm_timeout,
            )
        except asyncio.TimeoutError:
            raise LLMError(f"LLM request timed out after {settings.llm_timeout:.0f}s")
        except Exception as exc:
            raise LLMError(f"OpenAI API error: {type(exc).__name__}: {exc}") from exc
        text = (response.choices[0].message.content or "").strip()

    elif provider == "gemini":
        client = _get_gemini_client()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                ),
                timeout=settings.llm_timeout,
            )
        except asyncio.TimeoutError:
            raise LLMError(f"LLM request timed out after {settings.llm_timeout:.0f}s")
        except Exception as exc:
            raise LLMError(f"Gemini API error: {type(exc).__name__}: {exc}") from exc
        text = (response.text or "").strip()

    else:
        raise LLMError(
            f"Unknown LLM provider: {provider!r}. "
            "Set LLM_PROVIDER to 'openai' or 'gemini'."
        )

    elapsed = time.monotonic() - start
    logger.info(
        "LLM: response received — provider=%s model=%s latency=%.2fs response_chars=%d",
        provider, model, elapsed, len(text),
    )
    return text


async def generate_structured(
    prompt: str,
    response_model: Type[BaseModel],
) -> BaseModel:
    """Send a prompt and parse the response into a Pydantic model.

    Appends the JSON schema to the prompt automatically.
    Retries once on malformed JSON or validation error.
    Raises LLMError if both attempts fail.
    """
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)

    base_prompt = (
        f"{prompt}\n\n"
        "---\n"
        "Return ONLY a valid JSON object that matches the schema below.\n"
        "Do NOT include markdown fences, explanations, or any text outside the JSON.\n\n"
        f"Required JSON schema:\n{schema_json}"
    )

    last_exc: Optional[Exception] = None

    for attempt in range(1, 3):
        try:
            raw = await generate_text(base_prompt if attempt == 1 else base_prompt + (
                "\n\nYour previous response was not valid JSON. "
                "Return ONLY the raw JSON object — no markdown, no explanation."
            ))
            clean = _extract_json(raw)
            data = json.loads(clean)
            result = response_model.model_validate(data)
            if attempt > 1:
                logger.info("LLM: structured parse succeeded on attempt %d", attempt)
            return result

        except (json.JSONDecodeError, ValidationError) as exc:
            last_exc = exc
            logger.warning(
                "LLM: structured parse attempt %d failed — %s: %s",
                attempt, type(exc).__name__, str(exc)[:200],
            )
        except LLMError:
            raise  # propagate API/timeout errors immediately; no retry

    raise LLMError(
        f"Failed to parse structured output after 2 attempts. "
        f"Last error: {last_exc}"
    ) from last_exc
