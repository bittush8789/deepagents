# 🔎 Web Search Research Agent

A small, local proof of concept. Ask a research question; a **Deep Agent**
searches the web with **Tavily**, compares the evidence, and writes a cited
report you can read in your browser.

Built with Streamlit · Deep Agents · LangChain · Groq · Tavily.

---

## ✨ Features

- Agent decides when to search and searches again when evidence is thin.
- Structured report: question, summary, findings, analysis, sources, limits.
- Sources come from real Tavily results — fabricated URLs are removed.
- Tool-activity inspector and clickable source links in the UI.
- Friendly errors for missing keys, rate limits, and network problems.
- Unit tests run without any API keys.

---

## 🔄 How it works

```text
question
  └─ Streamlit UI (app.py)
       └─ Deep Agent on Groq (agents/research_agent.py)
            └─ web_search tool → Tavily (tools/web_search.py)
                 └─ SourceRegistry keeps verified sources
       └─ report formatter adds the required sections (utils/format_report.py)
  └─ Markdown report + sources + tool activity
```

---

## 📁 Project structure

```text
web-search-agent/
├── app.py                 # Streamlit UI
├── agents/research_agent.py
├── prompts/research_prompt.py
├── tools/web_search.py
├── utils/format_report.py
├── tests/test_report.py
├── tests/test_tracing.py
├── requirements.txt
├── pytest.ini
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── .env                   # your secrets (gitignored)
├── .gitignore
└── README.md
```

---

## 🔑 Get API keys

- Groq: <https://console.groq.com/keys>
- Tavily: <https://app.tavily.com/home>

Then create your local `.env`:

```bash
cp .env.example .env      # Linux / macOS
Copy-Item .env.example .env   # Windows PowerShell
```

```dotenv
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

---

## 🐳 Run with Docker (recommended)

Needs Docker Desktop or Docker Engine with Compose v2.

```bash
docker compose up --build
```

Open <http://localhost:8501>.

Stop it with:

```bash
docker compose down
```

Notes:

- Secrets are passed at runtime via `env_file: .env`, never baked into the image.
- The container runs as a non-root user and includes a healthcheck.
- Rebuild after changing code: `docker compose up --build`.

---

## 💻 Run locally (Python)

Requires Python 3.11+ (tested on 3.13).

```powershell
# Windows
cd web-search-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

```bash
# Linux / macOS
cd web-search-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## ⚙️ Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | — | **Required.** Groq key |
| `TAVILY_API_KEY` | — | **Required.** Tavily key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model (must support tool calling) |
| `TAVILY_MAX_RESULTS` | `5` | Results per search (1–10) |
| `AGENT_MAX_STEPS` | `30` | Agent step limit (6–60) |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | — | LangSmith key (needed to trace) |
| `LANGSMITH_PROJECT` | `web-search-research-agent` | Trace project name |

Keys are never shown in the UI, logs, or reports.

---

## 🧭 Tracing (optional, off by default)

Runs can be recorded in [LangSmith](https://smith.langchain.com) for debugging
and evaluation. Add to `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=web-search-research-agent
```

Tracing stays off unless both the flag and the key are set, so nothing is sent
anywhere by default. Each run is tagged with the model and inputs; the sidebar
shows whether tracing is on. Only the question *length* is stored as metadata —
the question text itself lives in the trace, not in the run metadata.

---

## 💬 Example questions

- What are the main differences between the current stable versions of
  PostgreSQL and MySQL for a small SaaS application?
- What evidence exists that remote work affects team productivity?
- Compare the latest LangChain and LlamaIndex releases for building tool-using
  agents.

---

## 🧪 Tests

```bash
pytest -q
```

26 deterministic tests — no network and no paid API calls. Tavily is mocked.
They cover report sections, source validation, fabricated-URL removal,
search-tool success/failure, and LangSmith tracing configuration.

---

## 🛠️ Troubleshooting

| Problem | Fix |
| --- | --- |
| Missing key error | Add keys to `.env` and restart |
| “Groq rate limit reached” | Wait ~1 minute, or lower `TAVILY_MAX_RESULTS` |
| Unknown/unavailable model | Set `GROQ_MODEL` to a model on your account |
| Tool-calling failure | Try `openai/gpt-oss-120b` |
| No sources found | Rephrase the question, then retry |
| Port already in use | Change the port mapping in `docker-compose.yml` |

---

## ⚠️ Limits

- Uses Tavily **snippets**, not full pages.
- Free Groq tiers have a low tokens-per-minute limit; long runs can rate-limit.
- No streaming, no chat memory, single agent, no RAG — by design.
- Local POC only: no auth, database, or production deployment.

---

## 🔒 Security

- `.env` is gitignored and excluded from the Docker image.
- Web content is treated as untrusted; the prompt resists injection.
- The shell tool is disabled and the Deep Agent uses in-memory storage.
- The Sources section is built from real tool results, not model claims.
