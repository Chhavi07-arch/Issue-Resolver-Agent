"""Tavily web search abstraction."""

import logging
from typing import Any

from issueops.config.settings import settings

logger = logging.getLogger(__name__)

# TODO: initialize Tavily client
# from tavily import TavilyClient
# _client = TavilyClient(api_key=settings.tavily_api_key)


async def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web via Tavily and return a list of result dicts.

    Each result has at minimum: title, url, content.
    """
    # TODO: implement
    raise NotImplementedError


async def search_docs(error_message: str) -> list[dict[str, Any]]:
    """Convenience wrapper — searches for documentation related to an error."""
    # TODO: build a focused query and call search_web
    raise NotImplementedError
