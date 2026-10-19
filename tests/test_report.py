"""Unit tests for report formatting and citation hygiene.

These tests are deterministic and make no network or paid API calls.
"""

from __future__ import annotations

import json

import pytest

from tools.web_search import SourceRegistry, create_web_search_tool
from utils.format_report import (
    SECTION_ORDER,
    build_sources_markdown,
    format_report,
    normalize_sources,
    strip_unverified_urls,
)

QUESTION = "What are the tradeoffs between SQL and NoSQL databases?"

SOURCES = [
    {
        "title": "SQL vs NoSQL: A Practical Comparison",
        "url": "https://example.com/sql-vs-nosql",
        "snippet": "A comparison of data models and scaling approaches.",
    },
    {
        "title": "Choosing a Database",
        "url": "https://docs.example.org/database-guide",
        "snippet": "Guidance on consistency and operational tradeoffs.",
    },
]


def _answer() -> str:
    return """\
## Research Question

What are the tradeoffs between SQL and NoSQL databases?

## Executive Summary

SQL favors consistency and structure; NoSQL favors flexible schemas and
horizontal scale. The right choice depends on access patterns.

## Key Findings

- Relational databases enforce schemas and joins
  ([SQL vs NoSQL](https://example.com/sql-vs-nosql)).
- Document stores trade some consistency guarantees for flexibility
  ([Choosing a Database](https://docs.example.org/database-guide)).

## Detailed Analysis

The tradeoff is mostly about consistency, query flexibility, and operational
complexity rather than one option being universally better.

## Limitations

The retrieved sources are introductory and do not benchmark a specific workload.
"""


def test_report_contains_required_sections_in_order() -> None:
    report = format_report(QUESTION, _answer(), SOURCES)

    positions = [report.index(f"## {title}") for title in SECTION_ORDER]
    assert positions == sorted(positions)
    assert "## Sources" in report
    assert report.endswith("\n")


def test_verified_sources_are_rendered_with_titles_and_urls() -> None:
    report = format_report(QUESTION, _answer(), SOURCES)

    assert "[SQL vs NoSQL: A Practical Comparison](https://example.com/sql-vs-nosql)" in report
    assert "[Choosing a Database](https://docs.example.org/database-guide)" in report


def test_empty_source_list_is_reported_honestly() -> None:
    report = format_report(QUESTION, _answer(), [])

    assert "## Sources" in report
    assert "No web sources were retrieved" in report
    assert "https://example.com/sql-vs-nosql" not in report
    assert "https://docs.example.org/database-guide" not in report


def test_invalid_source_urls_are_dropped() -> None:
    sources = [
        {"title": "Bad scheme", "url": "javascript:alert(1)"},
        {"title": "Not absolute", "url": "example.com/no-scheme"},
        {"title": "FTP", "url": "ftp://example.com/file"},
        {"title": "Good", "url": "https://good.example/page"},
    ]

    normalized = normalize_sources(sources)
    report = format_report(QUESTION, "No sources were used.", sources)

    assert [source["url"] for source in normalized] == ["https://good.example/page"]
    assert "javascript:alert(1)" not in report
    assert "ftp://example.com/file" not in report
    assert "https://good.example/page" in report


def test_missing_sections_are_filled() -> None:
    report = format_report(
        QUESTION,
        "## Executive Summary\n\nOnly a summary was produced.",
        SOURCES,
    )

    for title in SECTION_ORDER:
        assert f"## {title}" in report
    assert "Only a summary was produced." in report
    assert "No key findings were produced." in report
    assert "No detailed analysis was produced." in report


def test_fabricated_urls_in_the_answer_are_removed() -> None:
    answer = """\
## Executive Summary

A real claim ([real source](https://example.com/sql-vs-nosql)) and a made-up
claim ([fake source](https://fake.example/fabricated)).

A bare fabricated URL https://fake.example/bare should disappear too.
"""
    report = format_report(QUESTION, answer, SOURCES)

    assert "https://fake.example/fabricated" not in report
    assert "https://fake.example/bare" not in report
    assert "fake source" in report  # link text is kept
    assert "https://example.com/sql-vs-nosql" in report


def test_model_written_sources_section_is_replaced() -> None:
    answer = _answer() + """
## Sources

- [Made up citation](https://fake.example/made-up)
"""
    report = format_report(QUESTION, answer, SOURCES)

    assert report.count("## Sources") == 1
    assert "https://fake.example/made-up" not in report
    assert "https://example.com/sql-vs-nosql" in report


def test_duplicate_sources_are_collapsed() -> None:
    sources = [
        {"title": "Example", "url": "https://example.com/a"},
        {"title": "Example duplicate", "url": "https://example.com/a/"},
        {"title": "Example again", "url": "https://example.com/a"},
    ]

    assert len(normalize_sources(sources)) == 1
    report = format_report(QUESTION, "## Executive Summary\n\nSummary.", sources)
    assert report.count("https://example.com/a") == 1


def test_search_source_objects_are_supported() -> None:
    from tools.web_search import SearchSource

    sources = [
        SearchSource(
            title="Object source",
            url="https://objects.example/page",
            snippet="Snippet",
        )
    ]

    report = format_report(QUESTION, "## Executive Summary\n\nSummary.", sources)
    assert "[Object source](https://objects.example/page)" in report


def test_question_is_restated_when_the_model_omits_it() -> None:
    report = format_report(
        QUESTION,
        "## Executive Summary\n\nA summary without the question.",
        SOURCES,
    )

    research_question = report.split("## Research Question", 1)[1].split(
        "## Executive Summary", 1
    )[0]
    assert QUESTION in research_question


def test_unstructured_answer_becomes_the_executive_summary() -> None:
    report = format_report(QUESTION, "A plain answer with no headings.", SOURCES)

    assert "A plain answer with no headings." in report
    assert "## Executive Summary" in report


def test_empty_source_title_falls_back_to_the_domain() -> None:
    markdown = build_sources_markdown(
        [{"title": "", "url": "https://fallback.example/page"}]
    )

    assert "fallback.example" in markdown


def test_strip_unverified_urls_keeps_verified_links_only() -> None:
    text = (
        "Keep [this](https://example.com/sql-vs-nosql) but drop "
        "[that](https://fake.example/nope)."
    )
    cleaned = strip_unverified_urls(text, SOURCES)

    assert "https://example.com/sql-vs-nosql" in cleaned
    assert "https://fake.example/nope" not in cleaned
    assert "that" in cleaned


def test_citation_artifacts_are_removed() -> None:
    answer = (
        "## Executive Summary\n\n"
        "Python 3.15 is stable【1】 and this claim uses [1] style markers.\n\n"
        "A real link stays ([source](https://example.com/sql-vs-nosql)).\n"
    )
    report = format_report(QUESTION, answer, SOURCES)

    assert "【" not in report
    assert "】" not in report
    assert "[1]" not in report
    assert "https://example.com/sql-vs-nosql" in report


def test_numeric_markdown_link_labels_are_preserved() -> None:
    answer = "## Executive Summary\n\nSee [1](https://example.com/sql-vs-nosql).\n"
    report = format_report(QUESTION, answer, SOURCES)

    assert "[1](https://example.com/sql-vs-nosql)" in report


def test_web_search_tool_handles_mocked_provider_failure(monkeypatch) -> None:
    class FailingTavily:
        def __init__(self, **kwargs) -> None:
            pass

        def invoke(self, payload):
            raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("tools.web_search.TavilySearch", FailingTavily)

    registry = SourceRegistry()
    tool = create_web_search_tool(registry=registry)
    payload = json.loads(tool.invoke({"query": "database tradeoffs"}))

    assert payload["results"] == []
    assert payload["result_count"] == 0
    assert "rate limit" in payload["error"].lower()
    assert len(registry) == 0


def test_web_search_tool_preserves_mocked_results(monkeypatch) -> None:
    class FakeTavily:
        def __init__(self, **kwargs) -> None:
            pass

        def invoke(self, payload):
            return {
                "query": payload["query"],
                "results": [
                    {
                        "title": "  Real Result  ",
                        "url": "https://real.example/article",
                        "content": "  A useful snippet.  ",
                        "score": 0.9,
                    },
                    {
                        "title": "Invalid",
                        "url": "not-a-url",
                        "content": "Dropped",
                    },
                ],
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("tools.web_search.TavilySearch", FakeTavily)

    registry = SourceRegistry()
    tool = create_web_search_tool(registry=registry)
    payload = json.loads(tool.invoke({"query": "real query"}))

    assert payload["result_count"] == 1
    assert payload["results"][0] == {
        "title": "Real Result",
        "url": "https://real.example/article",
        "content": "A useful snippet.",
    }
    assert len(registry) == 1
    assert registry.all()[0].query == "real query"


def test_missing_tavily_key_returns_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    tool = create_web_search_tool(registry=SourceRegistry())
    payload = json.loads(tool.invoke({"query": "anything"}))

    assert payload["results"] == []
    assert "TAVILY_API_KEY" in payload["error"]
