"""Unit tests for optional LangSmith tracing configuration.

These tests are deterministic and make no network calls. They only verify how
tracing settings are read from the environment and passed to the agent run.
"""

from __future__ import annotations

import os

from agents.research_agent import (
    DEFAULT_LANGSMITH_ENDPOINT,
    DEFAULT_LANGSMITH_PROJECT,
    Settings,
    _build_run_config,
    configure_langsmith_tracing,
    get_public_config,
    langsmith_project,
    langsmith_tracing_enabled,
)

TRACING_ENV = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


def _clear_tracing_env(monkeypatch) -> None:
    for name in TRACING_ENV:
        monkeypatch.delenv(name, raising=False)


def _settings() -> Settings:
    return Settings(
        groq_api_key="test-groq-key",
        tavily_api_key="test-tavily-key",
        model="openai/gpt-oss-120b",
        max_results=5,
        max_steps=30,
    )


def test_tracing_is_off_without_an_api_key(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)

    assert langsmith_tracing_enabled() is False
    assert configure_langsmith_tracing() is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_is_on_when_key_and_flag_are_present(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    assert langsmith_tracing_enabled() is True
    assert configure_langsmith_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == DEFAULT_LANGSMITH_PROJECT
    assert os.environ["LANGSMITH_ENDPOINT"] == DEFAULT_LANGSMITH_ENDPOINT


def test_tracing_defaults_on_when_only_the_key_is_present(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")

    assert langsmith_tracing_enabled() is True


def test_tracing_can_be_explicitly_disabled(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    assert langsmith_tracing_enabled() is False


def test_custom_project_name_is_respected(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "custom-project")

    assert langsmith_project() == "custom-project"
    configure_langsmith_tracing()
    assert os.environ["LANGSMITH_PROJECT"] == "custom-project"


def test_run_config_is_minimal_when_tracing_is_off(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)

    config = _build_run_config(_settings(), "a question")

    assert config == {"recursion_limit": 30}


def test_run_config_is_tagged_when_tracing_is_on(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    config = _build_run_config(_settings(), "a question")

    assert config["recursion_limit"] == 30
    assert config["run_name"] == "web-search-research"
    assert "web-search-agent" in config["tags"]
    assert config["metadata"]["model"] == "openai/gpt-oss-120b"
    assert config["metadata"]["max_results"] == 5
    # Only the question length is recorded, never the question text.
    assert config["metadata"]["question_chars"] == len("a question")


def test_public_config_reports_tracing_without_secrets(monkeypatch) -> None:
    _clear_tracing_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "secret-groq-value")
    monkeypatch.setenv("TAVILY_API_KEY", "secret-tavily-value")
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret-langsmith-value")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    config = get_public_config()

    assert config["langsmith_enabled"] is True
    assert config["langsmith_project"] == DEFAULT_LANGSMITH_PROJECT
    dumped = repr(config)
    assert "secret-groq-value" not in dumped
    assert "secret-tavily-value" not in dumped
    assert "secret-langsmith-value" not in dumped
