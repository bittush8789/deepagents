"""Deep Agent orchestration for web research."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_groq import ChatGroq

from prompts.research_prompt import RESEARCH_SYSTEM_PROMPT
from tools.web_search import (
    SourceRegistry,
    create_web_search_tool,
    resolve_max_results,
)
from utils.format_report import format_report

load_dotenv()

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_STEPS = 30
MIN_MAX_STEPS = 6
MAX_MAX_STEPS = 60

DEFAULT_LANGSMITH_PROJECT = "web-search-research-agent"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


class AgentConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class AgentRunError(RuntimeError):
    """Raised when the agent cannot complete a research run."""


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration. Never expose the key values."""

    groq_api_key: str
    tavily_api_key: str
    model: str
    max_results: int
    max_steps: int


@dataclass
class ResearchResult:
    """Everything the UI needs to render one completed research run."""

    question: str
    report: str
    sources: list[dict[str, str]] = field(default_factory=list)
    activity: list[dict[str, Any]] = field(default_factory=list)
    model: str = DEFAULT_MODEL


def _read_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def get_settings() -> Settings:
    """Load and validate configuration from the environment/.env file.

    Raises:
        AgentConfigurationError: if either required API key is missing.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    missing: list[str] = []
    if not groq_key:
        missing.append("GROQ_API_KEY")
    if not tavily_key:
        missing.append("TAVILY_API_KEY")

    if missing:
        joined = " and ".join(missing)
        raise AgentConfigurationError(
            f"Missing required configuration: {joined}. "
            "Add the key(s) to the .env file in the project root, then "
            "restart Streamlit."
        )

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    max_results = resolve_max_results()
    max_steps = _read_int_env(
        "AGENT_MAX_STEPS",
        DEFAULT_MAX_STEPS,
        MIN_MAX_STEPS,
        MAX_MAX_STEPS,
    )

    return Settings(
        groq_api_key=groq_key,
        tavily_api_key=tavily_key,
        model=model,
        max_results=max_results,
        max_steps=max_steps,
    )


def get_public_config() -> dict[str, Any]:
    """Configuration safe to display in the UI (booleans, not secrets)."""
    return {
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY", "").strip()),
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "max_results": resolve_max_results(),
        "max_steps": _read_int_env(
            "AGENT_MAX_STEPS",
            DEFAULT_MAX_STEPS,
            MIN_MAX_STEPS,
            MAX_MAX_STEPS,
        ),
        "langsmith_enabled": langsmith_tracing_enabled(),
        "langsmith_project": langsmith_project(),
    }


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-ish environment variable."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def langsmith_tracing_enabled() -> bool:
    """Whether LangSmith tracing is active for this run.

    Tracing requires both an explicit opt-in (``LANGSMITH_TRACING=true``, the
    default when a key is present) and an API key. It stays off otherwise, so
    the app never emits LangSmith warnings or sends data unexpectedly.
    """
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        return False
    return _env_flag("LANGSMITH_TRACING", default=True)


def langsmith_project() -> str:
    """LangSmith project name used for traces."""
    return os.getenv("LANGSMITH_PROJECT", "").strip() or DEFAULT_LANGSMITH_PROJECT


def configure_langsmith_tracing() -> bool:
    """Publish tracing settings to LangChain through environment variables.

    LangChain reads tracing configuration from the environment when a run
    starts, so this runs once at import (after ``.env`` is loaded) and again
    whenever settings are read. The keys themselves are never logged or shown.
    """
    enabled = langsmith_tracing_enabled()

    # Always write an explicit value so a stray shell variable cannot enable
    # tracing for a run the app believes is untraced.
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_PROJECT"] = langsmith_project()
        os.environ.setdefault("LANGSMITH_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT)
    return enabled


def _build_run_config(settings: Settings, question: str) -> dict[str, Any]:
    """Build the LangGraph run config, tagging traces when tracing is on."""
    config: dict[str, Any] = {"recursion_limit": settings.max_steps}
    if langsmith_tracing_enabled():
        config["run_name"] = "web-search-research"
        config["tags"] = ["web-search-agent", "deep-agents", settings.model]
        config["metadata"] = {
            "app": "web-search-agent",
            "model": settings.model,
            "max_results": settings.max_results,
            "max_steps": settings.max_steps,
            "question_chars": len(question),
        }
    return config


def _configure_deep_agent_harness() -> None:
    """Keep this POC to a single agent and hide the shell tool.

    Deep Agents automatically adds a `task` subagent and an `execute` tool.
    The project scope explicitly excludes subagents and shell execution, so a
    provider-level harness profile disables the default subagent and withholds
    `execute` from the model. The default backend is in-memory, so filesystem
    tools never touch the local disk.
    """
    try:
        register_harness_profile(
            "groq",
            HarnessProfile(
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                excluded_tools=frozenset({"execute"}),
            ),
        )
    except Exception:  # Beta API: fail soft rather than blocking startup.
        LOGGER.warning(
            "Could not apply the Groq harness profile; default Deep Agents "
            "tools may be available for this run."
        )


_configure_deep_agent_harness()
configure_langsmith_tracing()


def build_research_agent(
    registry: SourceRegistry,
    settings: Settings | None = None,
) -> Any:
    """Create the Deep Agent with the Tavily tool registered."""
    config = settings or get_settings()
    model = ChatGroq(
        model=config.model,
        api_key=config.groq_api_key,
        temperature=0.2,
        max_retries=3,
        timeout=60,
    )
    search_tool = create_web_search_tool(
        registry=registry,
        max_results=config.max_results,
    )
    return create_deep_agent(
        model=model,
        tools=[search_tool],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )


def _friendly_agent_error(exc: Exception) -> str:
    """Convert provider/agent failures into safe, actionable messages."""
    message = str(exc).lower()

    if isinstance(exc, AgentConfigurationError):
        return str(exc)

    if isinstance(exc, AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
    if isinstance(exc, PermissionDeniedError):
        return "The Groq API key does not have access to the configured model."
    if isinstance(exc, RateLimitError):
        return (
            "Groq rate limit reached. Free-tier keys allow a limited number of "
            "tokens per minute. Wait about a minute, or lower "
            "TAVILY_MAX_RESULTS, then try again."
        )
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return "Could not reach Groq. Check your network connection and try again."
    if isinstance(exc, NotFoundError):
        return (
            "The configured Groq model was not found. Set GROQ_MODEL to a "
            "model available to your account."
        )
    if isinstance(exc, BadRequestError):
        if "tool" in message and ("fail" in message or "support" in message):
            return (
                "The model could not complete a tool call. Try again, or use "
                "a Groq model with reliable tool-calling support such as "
                "openai/gpt-oss-120b."
            )
        if "model" in message and (
            "decommission" in message
            or "not found" in message
            or "does not exist" in message
        ):
            return (
                "The configured Groq model is unavailable. Set GROQ_MODEL to "
                "a model available to your account."
            )
        return "Groq rejected the request. Check GROQ_MODEL and try again."
    if isinstance(exc, APIStatusError):
        return "Groq returned an error while processing the request. Try again."

    # LangGraph raises GraphRecursionError when the step limit is exceeded.
    if type(exc).__name__ == "GraphRecursionError":
        return (
            "The agent reached its step limit before finishing. Increase "
            "AGENT_MAX_STEPS or ask a narrower question."
        )

    return (
        f"The agent stopped unexpectedly ({type(exc).__name__}). "
        "Please try again."
    )


def message_text(content: Any) -> str:
    """Extract plain text from a LangChain message content value."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Mapping):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else json.dumps(content, default=str)

    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def extract_final_text(messages: Sequence[BaseMessage]) -> str:
    """Return the last assistant message that contains report text."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                continue
            text = message_text(message.content)
            if text:
                return text
        elif isinstance(message, Mapping) and message.get("role") == "assistant":
            if message.get("tool_calls"):
                continue
            text = message_text(message.get("content"))
            if text:
                return text
    return ""


def _clip(text: str, limit: int = 500) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _tool_output_summary(content: Any) -> str:
    text = message_text(content)
    if not text:
        return "(empty result)"
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return _clip(text)
    if isinstance(payload, dict) and "results" in payload:
        results = payload.get("results") or []
        error = payload.get("error")
        if error:
            return f"{len(results)} result(s); error: {error}"
        query = payload.get("query", "")
        return f"{len(results)} result(s)" + (f" for query: {query}" if query else "")
    return _clip(text)


def extract_activity(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Flatten tool calls/results into rows for the Streamlit inspector."""
    activity: list[dict[str, Any]] = []
    for step, message in enumerate(messages, start=1):
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                try:
                    args = json.dumps(call.get("args", {}), ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    args = str(call.get("args", {}))
                activity.append(
                    {
                        "step": step,
                        "event": "tool_call",
                        "tool": call.get("name", "unknown"),
                        "detail": _clip(args, 400),
                    }
                )
        elif isinstance(message, ToolMessage):
            activity.append(
                {
                    "step": step,
                    "event": "tool_result",
                    "tool": message.name or "unknown",
                    "detail": _tool_output_summary(message.content),
                }
            )
    return activity


def _search_events(messages: Sequence[BaseMessage]) -> tuple[int, list[str]]:
    """Count web_search calls and collect any errors returned by the tool."""
    calls = 0
    errors: list[str] = []

    for message in messages:
        if isinstance(message, AIMessage):
            calls += sum(
                1
                for call in (message.tool_calls or [])
                if call.get("name") == "web_search"
            )
        elif isinstance(message, ToolMessage) and message.name == "web_search":
            try:
                payload = json.loads(message_text(message.content))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("error"):
                errors.append(str(payload["error"]))

    return calls, errors


def _is_fatal_search_error(error: str) -> bool:
    text = error.lower()
    tokens = (
        "api key",
        "not configured",
        "permission",
        "denied",
        "rate limit",
        "quota",
        "plan limit",
        "could not reach",
    )
    return any(token in text for token in tokens)


def run_research(question: str) -> ResearchResult:
    """Run one research question through the Deep Agent.

    Raises:
        AgentConfigurationError: missing/invalid local configuration.
        AgentRunError: model, tool, or agent failure with a safe message.
    """
    cleaned_question = " ".join(str(question or "").split()).strip()
    if not cleaned_question:
        raise AgentConfigurationError("Please enter a research question.")
    if len(cleaned_question) > 2000:
        raise AgentConfigurationError(
            "The research question is too long. Keep it under 2,000 characters."
        )

    settings = get_settings()
    registry = SourceRegistry()
    configure_langsmith_tracing()

    try:
        agent = build_research_agent(registry, settings)
        state = agent.invoke(
            {"messages": [{"role": "user", "content": cleaned_question}]},
            config=_build_run_config(settings, cleaned_question),
        )
    except Exception as exc:
        raise AgentRunError(_friendly_agent_error(exc)) from exc

    if not isinstance(state, Mapping):
        raise AgentRunError("The agent returned an unexpected response format.")

    messages = state.get("messages") or []
    if not messages:
        raise AgentRunError("The agent finished without returning a response.")

    search_calls, search_errors = _search_events(messages)
    if search_calls == 0:
        raise AgentRunError(
            "The agent did not perform a web search, so the report would not "
            "be source-backed. Please try again."
        )

    sources = registry.as_dicts()
    if not sources:
        fatal_error = next(
            (error for error in search_errors if _is_fatal_search_error(error)),
            None,
        )
        if fatal_error:
            raise AgentRunError(fatal_error)

    final_text = extract_final_text(messages)
    if not final_text:
        raise AgentRunError(
            "The agent did not return a readable report. Please try again."
        )

    report = format_report(cleaned_question, final_text, sources)

    return ResearchResult(
        question=cleaned_question,
        report=report,
        sources=sources,
        activity=extract_activity(messages),
        model=settings.model,
    )
