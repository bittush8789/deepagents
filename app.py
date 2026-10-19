"""Streamlit UI for the Web Search Research Agent."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from agents.research_agent import (
    AgentConfigurationError,
    AgentRunError,
    get_public_config,
    run_research,
)

st.set_page_config(
    page_title="Web Search Research Agent",
    page_icon="🔎",
    layout="centered",
)

EXAMPLE_QUESTIONS = (
    "What are the main differences between the current stable versions of "
    "PostgreSQL and MySQL for a small SaaS application?",
    "What evidence exists that remote work affects team productivity?",
    "Compare the latest LangChain and LlamaIndex releases for building "
    "tool-using agents.",
)


def _render_sidebar(config: dict) -> None:
    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            f"{'✅' if config['groq_configured'] else '❌'} Groq API key"
        )
        st.markdown(
            f"{'✅' if config['tavily_configured'] else '❌'} Tavily API key"
        )

        if not (config["groq_configured"] and config["tavily_configured"]):
            st.error(
                "Add the missing keys to `.env` in the project root, then "
                "restart Streamlit."
            )

        st.divider()
        st.caption(f"Model: `{config['model']}`")
        st.caption(f"Results per search: {config['max_results']}")
        st.caption(f"Agent step limit: {config['max_steps']}")
        if config.get("langsmith_enabled"):
            st.caption(f"LangSmith tracing: ✅ `{config['langsmith_project']}`")
        else:
            st.caption("LangSmith tracing: off")
        st.divider()
        st.caption(
            "Built with Streamlit, Deep Agents, LangChain, Groq, and Tavily. "
            "API keys are never displayed."
        )


def _render_sources(sources: list[dict]) -> None:
    st.subheader("Source links")
    if not sources:
        st.info("No web sources were retrieved for this run.")
        return

    for index, source in enumerate(sources, start=1):
        title = (
            " ".join(str(source.get("title", "")).split())
            .replace("[", "\\[")
            .replace("]", "\\]")
        ) or source.get("url", "")
        url = source.get("url", "")
        st.markdown(f"**{index}. [{title}]({url})**")
        st.caption(url)


def _render_activity(activity: list[dict]) -> None:
    with st.expander("Agent & tool activity", expanded=False):
        st.caption(
            "Tool calls recorded during this run. Deep Agents may also use "
            "in-memory filesystem tools for context management."
        )
        if not activity:
            st.info("No tool activity was recorded for this run.")
            return

        rows = [
            {
                "Step": event.get("step", ""),
                "Event": event.get("event", ""),
                "Tool": event.get("tool", ""),
                "Detail": event.get("detail", ""),
            }
            for event in activity
        ]
        st.dataframe(rows, hide_index=True)


def main() -> None:
    config = get_public_config()
    _render_sidebar(config)

    st.title("Web Search Research Agent")
    st.caption(
        "Ask a question. A Deep Agent will search the web with Tavily, compare "
        "the retrieved evidence, and produce a cited research report."
    )

    configured = config["groq_configured"] and config["tavily_configured"]
    if not configured:
        st.error(
            "Configuration required: add `GROQ_API_KEY` and `TAVILY_API_KEY` "
            "to the `.env` file in the project root, then restart Streamlit."
        )

    with st.expander("Example questions", expanded=False):
        for example in EXAMPLE_QUESTIONS:
            st.markdown(f"- {example}")

    with st.form("research_form"):
        question = st.text_area(
            "Research question",
            key="question_input",
            height=110,
            placeholder=(
                "e.g. What are the tradeoffs between vector databases and "
                "keyword search for a small documentation site?"
            ),
            max_chars=2000,
        )
        submitted = st.form_submit_button(
            "Research",
            type="primary",
            width="stretch",
            disabled=not configured,
        )

    if submitted:
        st.session_state.pop("last_result", None)
        st.session_state.pop("run_error", None)

        if not (question or "").strip():
            st.session_state["run_error"] = "Please enter a research question."
        else:
            with st.spinner("Searching the web and drafting the report…"):
                try:
                    st.session_state["last_result"] = run_research(question)
                except (AgentConfigurationError, AgentRunError) as exc:
                    st.session_state["run_error"] = str(exc)
                except Exception:
                    st.session_state["run_error"] = (
                        "An unexpected error occurred while running the agent. "
                        "Please try again."
                    )

    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])

    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()
        st.subheader("Research report")
        st.caption(f"Question: {result.question}  \nModel: `{result.model}`")
        st.markdown(result.report)

        st.divider()
        _render_sources(result.sources)
        _render_activity(result.activity)

        with st.expander("About this POC", expanded=False):
            st.markdown(
                "- The agent decides which searches to run and may search "
                "more than once.\n"
                "- Only source URLs returned by Tavily can appear in the "
                "report.\n"
                "- Search snippets are not full-page verification. Check "
                "important claims yourself.\n"
                "- This is a local learning project, not a production service."
            )


if __name__ == "__main__":
    main()
