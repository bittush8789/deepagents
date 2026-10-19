"""Deterministic report formatting and citation hygiene.

The LLM writes the analysis, but this module owns the final shape of the
report. It guarantees the required sections exist and that the Sources section
contains only URLs actually returned by the search tool. Any URL the model
invented is removed before the report is shown.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

SECTION_ORDER = (
    "Research Question",
    "Executive Summary",
    "Key Findings",
    "Detailed Analysis",
    "Limitations",
)

# Titles that look like a model-written source list. They are dropped and
# replaced with the registry-backed Sources section.
_SOURCE_LIKE_TITLES = frozenset(
    {
        "sources",
        "source",
        "references",
        "reference",
        "citations",
        "citation",
        "bibliography",
        "works cited",
        "source list",
    }
)

_CANONICAL_TITLES = {title.lower(): title for title in SECTION_ORDER}

_FALLBACKS = {
    "Research Question": "_No research question was provided._",
    "Executive Summary": "_No executive summary was produced._",
    "Key Findings": "_No key findings were produced._",
    "Detailed Analysis": "_No detailed analysis was produced._",
    "Limitations": (
        "_The agent did not report specific limitations. "
        "Verify important claims against the listed sources._"
    ),
}

_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"'\]\)]+", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!?*_"

# Models sometimes emit citation markers instead of Markdown links. They are
# removed so the report stays clean, while links to verified sources are kept.
_FULLWIDTH_CITATION_RE = re.compile(r"【\s*\d*\s*】")
_BRACKET_NUMBER_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\](?!\()")
_STRAY_FULLWIDTH_BRACKET_RE = re.compile(r"[【】〔〕]")


def is_valid_http_url(url: str) -> bool:
    """Return True only for absolute http(s) URLs with a host."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _url_key(url: str) -> str:
    """Normalize a URL for comparison without changing how it is displayed."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"


def _clean_url(raw: Any) -> str:
    return str(raw or "").strip().rstrip(_TRAILING_PUNCTUATION)


def normalize_sources(
    sources: Sequence[Mapping[str, Any] | Any] | None,
) -> list[dict[str, str]]:
    """Validate, de-duplicate, and normalize sources.

    Accepts mappings or objects with `title`, `url`, and optional `snippet`
    attributes (such as `tools.web_search.SearchSource`). Invalid or
    non-http(s) URLs are discarded rather than displayed.
    """
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()

    for source in sources or []:
        if isinstance(source, Mapping):
            title = source.get("title", "")
            url = source.get("url", "")
            snippet = source.get("snippet", "")
        else:
            title = getattr(source, "title", "")
            url = getattr(source, "url", "")
            snippet = getattr(source, "snippet", "")

        clean_url = _clean_url(url)
        if not is_valid_http_url(clean_url):
            continue

        key = _url_key(clean_url)
        if key in seen:
            continue

        clean_title = " ".join(str(title or "").split())
        if not clean_title:
            clean_title = urlparse(clean_url).netloc

        normalized.append(
            {
                "title": clean_title,
                "url": clean_url,
                "snippet": " ".join(str(snippet or "").split()),
            }
        )
        seen.add(key)

    return normalized


def _split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into a preamble and a list of (heading, content)."""
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return markdown.strip(), []

    preamble = markdown[: matches[0].start()].strip()
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        title = " ".join(match.group(1).strip().split())
        content = markdown[match.end() : end].strip()
        sections.append((title, content))
    return preamble, sections


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _safe_heading(title: str) -> str:
    cleaned = re.sub(r"[#\r\n]+", " ", title).strip()
    return " ".join(cleaned.split())[:120] or "Additional Notes"


def _escape_link_text(text: str) -> str:
    return (
        " ".join(text.split())
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def clean_citation_artifacts(text: str) -> str:
    """Remove bracketed/full-width citation markers the model may emit."""
    if not text:
        return ""
    text = _FULLWIDTH_CITATION_RE.sub("", text)
    text = _BRACKET_NUMBER_RE.sub("", text)
    text = _STRAY_FULLWIDTH_BRACKET_RE.sub("", text)
    return re.sub(r"[ \t]{2,}(?=[,.;:!?])", "", text)


def build_sources_markdown(sources: Sequence[Mapping[str, Any] | Any] | None) -> str:
    """Build the Sources section from verified search results only."""
    normalized = normalize_sources(sources)
    if not normalized:
        return "_No web sources were retrieved for this report._"

    lines = [
        f"{index}. [{_escape_link_text(source['title'])}]({source['url']})"
        for index, source in enumerate(normalized, start=1)
    ]
    return "\n".join(lines)


def strip_unverified_urls(
    text: str,
    allowed_sources: Sequence[Mapping[str, Any] | Any] | None,
) -> str:
    """Remove links and bare URLs that were not returned by the search tool.

    Markdown links to unverified URLs are reduced to their link text. Bare
    unverified URLs are deleted. Verified URLs are rewritten to the exact URL
    stored in the source registry so citations stay consistent.
    """
    allowed = {
        _url_key(source["url"]): source["url"]
        for source in normalize_sources(allowed_sources)
    }
    if not text:
        return ""

    def replace_markdown_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        canonical = allowed.get(_url_key(_clean_url(url)))
        if canonical:
            return f"[{label}]({canonical})"
        return label

    def replace_bare_url(match: re.Match[str]) -> str:
        raw = match.group(0)
        candidate = raw.rstrip(_TRAILING_PUNCTUATION)
        trailing = raw[len(candidate) :]
        canonical = allowed.get(_url_key(candidate))
        if canonical:
            return f"{canonical}{trailing}"
        return ""

    text = _MD_LINK_RE.sub(replace_markdown_link, text)
    text = _BARE_URL_RE.sub(replace_bare_url, text)
    return text


def _compose_sections(
    question: str,
    body: str,
    sources: Sequence[Mapping[str, Any] | Any] | None,
) -> str:
    preamble, raw_sections = _split_sections(body)

    collected: dict[str, list[str]] = {title: [] for title in SECTION_ORDER}
    extras: list[tuple[str, str]] = []

    for title, content in raw_sections:
        slug = _slug(title)
        if slug in _SOURCE_LIKE_TITLES:
            continue

        canonical = _CANONICAL_TITLES.get(slug)
        if canonical is None:
            if content:
                extras.append((title, content))
            continue

        if content:
            collected[canonical].append(content)

    # If the model returned an unstructured answer, keep it as the summary.
    if preamble and not collected["Executive Summary"]:
        collected["Executive Summary"].append(preamble)

    question_text = "\n\n".join(collected["Research Question"]).strip()
    if not question_text:
        question_text = question.strip() or _FALLBACKS["Research Question"]

    rendered: dict[str, str] = {}
    for title in SECTION_ORDER:
        content = "\n\n".join(collected[title]).strip()
        if title == "Research Question":
            content = content or question_text
        rendered[title] = content or _FALLBACKS[title]

    if extras:
        extra_markdown = "\n\n".join(
            f"### {_safe_heading(title)}\n\n{content}" for title, content in extras
        )
        rendered["Detailed Analysis"] = (
            f"{rendered['Detailed Analysis']}\n\n{extra_markdown}"
        )

    parts = [f"## {title}\n\n{rendered[title]}" for title in SECTION_ORDER]
    parts.append(f"## Sources\n\n{build_sources_markdown(sources)}")
    return "\n\n".join(parts).strip() + "\n"


def format_report(
    question: str,
    agent_answer: str,
    sources: Sequence[Mapping[str, Any] | Any] | None,
) -> str:
    """Format the agent's answer into the required, source-verified report.

    Args:
        question: The original user question.
        agent_answer: Markdown produced by the Deep Agent.
        sources: Sources returned by the search tool during the run.

    Returns:
        A Markdown report with every required section and only verified URLs.
    """
    verified = normalize_sources(sources)
    clean_body = clean_citation_artifacts(agent_answer or "")
    clean_body = strip_unverified_urls(clean_body, verified)
    return _compose_sections(question or "", clean_body, verified)
