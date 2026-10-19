"""Tavily-backed web search tool used by the research agent.

The tool returns a compact JSON string so the model can consume it reliably.
Every successful result is also recorded in a SourceRegistry, which the report
formatter uses to build the verified Sources section. This is what prevents
fabricated citations from reaching the final report.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10
MAX_SNIPPET_CHARS = 1200


class WebSearchInput(BaseModel):
    """Arguments accepted by the web search tool."""

    query: str = Field(
        description=(
            "A focused web search query. Prefer specific terms (names, dates, "
            "versions) over full sentences. One topic per query."
        ),
        min_length=1,
        max_length=400,
    )


@dataclass(frozen=True)
class SearchSource:
    """A source returned by Tavily, preserved exactly as retrieved."""

    title: str
    url: str
    snippet: str = ""
    query: str = ""


class SourceRegistry:
    """Collects unique sources across every search in one research run."""

    def __init__(self) -> None:
        self._sources: dict[str, SearchSource] = {}
        self._lock = threading.Lock()

    def add_results(
        self,
        query: str,
        results: Iterable[dict[str, Any]],
    ) -> None:
        """Record valid search results, de-duplicated by URL.

        The first title/snippet seen for a URL is kept. If a later result has a
        longer snippet, the snippet is upgraded without changing the title.
        """
        with self._lock:
            for item in results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not is_valid_http_url(url):
                    continue

                title = " ".join(str(item.get("title", "")).split()) or url
                snippet = " ".join(str(item.get("content", "")).split())
                snippet = snippet[:MAX_SNIPPET_CHARS]

                existing = self._sources.get(url)
                if existing is None:
                    self._sources[url] = SearchSource(
                        title=title,
                        url=url,
                        snippet=snippet,
                        query=query,
                    )
                elif len(snippet) > len(existing.snippet):
                    self._sources[url] = SearchSource(
                        title=existing.title,
                        url=existing.url,
                        snippet=snippet,
                        query=existing.query,
                    )

    def all(self) -> list[SearchSource]:
        """Return sources in first-seen order."""
        with self._lock:
            return list(self._sources.values())

    def as_dicts(self) -> list[dict[str, str]]:
        """Serializable form used by the UI and the report formatter."""
        return [
            {
                "title": source.title,
                "url": source.url,
                "snippet": source.snippet,
                "query": source.query,
            }
            for source in self.all()
        ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)


def is_valid_http_url(url: str) -> bool:
    """Return True only for absolute http(s) URLs with a host."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_max_results(value: int | str | None = None) -> int:
    """Clamp the configured result count to a sane range."""
    raw: int | str | None = value
    if raw is None:
        raw = os.getenv("TAVILY_MAX_RESULTS", str(DEFAULT_MAX_RESULTS))
    try:
        number = int(raw)
    except (TypeError, ValueError):
        number = DEFAULT_MAX_RESULTS
    return max(1, min(number, MAX_RESULTS_LIMIT))


def _search_error_message(exc: Exception) -> str:
    """Translate provider failures without leaking keys or raw responses."""
    text = str(exc).lower()
    name = type(exc).__name__.lower()

    if "401" in text or "unauthorized" in text or "invalid api key" in text:
        return "Tavily rejected the API key. Check TAVILY_API_KEY in your .env file."
    if "403" in text or "forbidden" in text:
        return "Tavily denied access to this search. Check the API key permissions."
    if "429" in text or "rate limit" in text or "quota" in text:
        return "Tavily rate limit or quota reached. Wait a moment and try again."
    if "timeout" in text or "timed out" in text or "connection" in text or "network" in text:
        return "Could not reach Tavily. Check your network connection and try again."
    if "plan" in text and "limit" in text:
        return "The Tavily plan limit was reached. Try fewer searches or a smaller result count."
    return f"Web search failed ({type(exc).__name__}). Try a different query."


def create_web_search_tool(
    registry: SourceRegistry | None = None,
    max_results: int | str | None = None,
) -> BaseTool:
    """Build the `web_search` tool bound to a source registry.

    Args:
        registry: Collector for verified sources. A new registry is created if
            omitted, but the caller normally passes one so the report can list
            the sources retrieved during the run.
        max_results: Number of results per query. Overrides the environment.
    """
    source_registry = registry if registry is not None else SourceRegistry()
    result_limit = resolve_max_results(max_results)

    @tool("web_search", args_schema=WebSearchInput)
    def web_search(query: str) -> str:
        """Search the public web with Tavily for current, cited information.

        Use focused queries. The result is JSON containing a query, a list of
        results with title/url/content, a result_count, and an error field.
        Only URLs returned here may be cited in the final report.
        """
        cleaned_query = " ".join(str(query).split()).strip()
        if not cleaned_query:
            return json.dumps(
                {
                    "query": "",
                    "results": [],
                    "result_count": 0,
                    "error": "The search query was empty.",
                },
                ensure_ascii=False,
            )

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return json.dumps(
                {
                    "query": cleaned_query,
                    "results": [],
                    "result_count": 0,
                    "error": (
                        "TAVILY_API_KEY is not configured. Add it to the .env "
                        "file and restart the app."
                    ),
                },
                ensure_ascii=False,
            )

        try:
            # Created lazily so a missing key never prevents the app from
            # starting, and so each run picks up the current environment.
            tavily = TavilySearch(
                max_results=result_limit,
                search_depth="basic",
                topic="general",
                include_answer=False,
                include_raw_content=False,
                handle_tool_error=False,
            )
            response = tavily.invoke({"query": cleaned_query})
            raw_results = response.get("results", []) if isinstance(response, dict) else []

            results: list[dict[str, str]] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", "")).strip()
                if not is_valid_http_url(url):
                    continue
                title = " ".join(str(item.get("title", "")).split()) or url
                snippet = " ".join(str(item.get("content", "")).split())[:MAX_SNIPPET_CHARS]
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "content": snippet,
                    }
                )

            source_registry.add_results(cleaned_query, results)
            return json.dumps(
                {
                    "query": cleaned_query,
                    "results": results,
                    "result_count": len(results),
                    "error": None,
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # Provider/network failures must not crash the app.
            return json.dumps(
                {
                    "query": cleaned_query,
                    "results": [],
                    "result_count": 0,
                    "error": _search_error_message(exc),
                },
                ensure_ascii=False,
            )

    return web_search


def collect_sources(
    results: Sequence[dict[str, Any]],
    query: str = "",
    registry: SourceRegistry | None = None,
) -> list[SearchSource]:
    """Utility for tests/scripts: register results and return the sources."""
    target = registry if registry is not None else SourceRegistry()
    target.add_results(query, results)
    return target.all()
