# Email Assistant — Agentic RAG with CrewAI + Qdrant + Obsidian

An agentic RAG system that turns your [Obsidian](https://obsidian.md/) vault into a knowledge base, and
auto-replies to questions that land in your Gmail inbox.

When a new email arrives, the system categorizes it and — **only if it looks like a question** — searches
your knowledge base for relevant notes and sends a reply directly. If the knowledge base can't support a
faithful answer, it sends a polite "out of scope" note instead.

---

## How it works

Two independent loops run side by side:

```
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│  Obsidian vault (notes)     │        │  Gmail inbox                          │
│  .md files                  │        │  (polled every 60s)                  │
└──────────────┬──────────────┘        └──────────────┬───────────────────────┘
               │  watchdog (real-time)                 │  new unread email
               ▼                                       ▼
        chunk + contextualize                    categorize (flash)
        (flash LLM)                              QUESTION / NOTIFICATION /
               │                                 NEWSLETTER / SPAM
               ▼                                       │
        embed (fastembed, local)                       │  only if QUESTION
               │                                       ▼
               ▼                                search knowledge base
        Qdrant knowledge-base                   (QdrantHybridSearchTool)
                                                generate reply (pro LLM)
                                                       │
                                                       ▼
                                                send reply (or fallback)
```

### Two CrewAI crews

- **`KnowledgeOrganizingCrew`** — reads a Markdown note, splits it into semantic chunks, adds retrieval
  context to each chunk, embeds them locally, and stores them in Qdrant.
- **`AutoResponderCrew`** — categorizes an email thread, and conditionally (only for `QUESTION`) writes a
  reply grounded in the knowledge base.

### Model assignment

| Agent | Task | Model |
|---|---|---|
| `chunks_extractor` | semantic chunking | gateway `deepseek-v4-flash` |
| `contextualizer` | chunk contextualization | gateway `deepseek-v4-flash` |
| `categorizer` | email classification | gateway `deepseek-v4-flash` |
| `response_writer` | reply generation | gateway `deepseek-v4-pro` |

### Knowledge-base sync guarantees

- **Content-hash idempotency** — every chunk stores the SHA-256 of its source file. On startup, a file is
  re-ingested only if its content changed.
- **Orphan cleanup** — on startup, chunks whose source file no longer exists in the vault are removed, so the
  vault is the single source of truth for Qdrant.
- **Type filtering** — Excalidraw drawings, `.trash/` and `.obsidian/` files are skipped by default
  (they are not prose notes); the lists are plain `.env` entries you can edit.

### Note-quality recommendation

Prefer having a coding agent (Claude Code, Codex, OpenCode, Hermes, OpenClaw, DeepSeek harness, …) create and edit
vault notes directly, instead of copy-pasting from chat windows. Manual pasting tends to introduce dirty data —
session-scoped dead links such as `vscode-webview://...` and broken table pipes are typical. Those junk tokens end
up in the sparse (BM25) vectors after jieba segmentation: dead-link hashes occupy vocabulary dimensions that no
query will ever hit, diluting the sparse representation, and the real paths embedded inside them (which would
otherwise be valuable for lexical retrieval) no longer match cleanly.

---

## Tech stack

- [CrewAI](https://www.crewai.com/) `1.15` — agent orchestration
- [Qdrant](https://qdrant.tech/) — vector store / knowledge base, **hybrid search** (dense + sparse vectors fused by RRF, then cross-encoder rerank)
- [fastembed](https://github.com/qdrant/fastembed) — local ONNX embeddings: `BAAI/bge-small-zh-v1.5` (dense) + `Qdrant/bm25` over jieba tokens (sparse) + `BAAI/bge-reranker-base` (rerank)
- [jieba](https://github.com/fxsjy/jieba) — Chinese word segmentation feeding the BM25 sparse path
- An **OpenAI-compatible gateway** serving DeepSeek models (all LLM calls)
- Gmail API (OAuth 2.0)

No GPU required. LLM inference goes through the gateway API; embeddings, sparse vectors and reranking run locally on CPU.

> **Upgrading from the pre-hybrid version?** The collection schema changed (a named `sparse` vector was added). Delete the existing `knowledge-base` collection and restart — the vault is re-ingested automatically.

---

## Prerequisites

- Python **3.10–3.12**
- Dependency management: [uv](https://docs.astral.sh/uv/) — installs Python and dependencies in one go
- Docker (to run Qdrant locally), **or** a free [Qdrant Cloud](https://cloud.qdrant.io/) account
- An OpenAI-compatible gateway URL + API key
- Gmail API credentials (see below)

> When using VSCode, open the `emailAssistant-obsidian` folder directly as the workspace; Pylance will then automatically apply the pyright configuration in `pyproject.toml`.

---

## Configuration

Copy `.env.example` to `.env` and fill in the values:

```dotenv
# LLM gateway (OpenAI-compatible)
GATEWAY_BASE_URL=https://your-gateway.example.com/v1
GATEWAY_API_KEY=YOUR_API_KEY
GATEWAY_MAIN_MODEL=deepseek-v4-pro
GATEWAY_FAST_MODEL=deepseek-v4-flash

# Local embedding models (fastembed)
# Set HF_ENDPOINT=https://hf-mirror.com if Hugging Face is unreachable.
# Use bge-small-zh-v1.5 for Chinese-heavy vaults, bge-small-en-v1.5 for English.
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
SPARSE_MODEL_NAME=Qdrant/bm25
RERANKER_MODEL_NAME=BAAI/bge-reranker-base

# Qdrant (local Docker needs no API key)
QDRANT_LOCATION=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=obsidian-notes

# Obsidian vault path (the knowledge base source)
OBSIDIAN_VAULT_PATH=/path/to/your/vault
# Ingestion scope control: empty INCLUDE = all folders; EXCLUDE defaults
# to the trash bin (.trash) and the app config dir (.obsidian); the
# frontmatter key filter defaults to Excalidraw boards. All editable.
OBSIDIAN_EXCLUDE_FOLDERS=.trash,.obsidian
OBSIDIAN_EXCLUDE_FRONTMATTER=excalidraw-plugin

# Proxy (required behind a firewall/GFW; loaded automatically at startup,
# no shell export needed). Set the HTTP port of YOUR proxy software;
# delete or comment out on unrestricted networks.
HTTPS_PROXY=http://127.0.0.1:7890

# Disable CrewAI's built-in trace upload (no calls to app.crewai.com). Keep
# false for unattended/automated runs. Note: on a fresh machine CrewAI still
# shows a one-time "view your execution traces?" prompt (its first-run consent
# flow, unrelated to this flag) — run `crewai traces disable` once to suppress
# that prompt too.
CREWAI_TRACING_ENABLED=false

# Optional tuning
# Timeout (seconds) for every LLM call (default 180)
#LLM_TIMEOUT_SEC=180
# Optional temperature override for the reply-writing agent
#RESPONSE_TEMPERATURE=0.3
# Comma-separated substrings of gateway models that reject tool_choice
# (thinking-mode models). These fall back to JSON-mode structured output.
#NO_TOOL_CHOICE_MODELS=deepseek-v4,qwen3.8
```

### Gmail OAuth credentials

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and **enable the Gmail API**.
2. Configure the OAuth consent screen as **External**, and add your Gmail as a test user.
3. Create an **OAuth client ID** of type **Desktop app**, download the JSON, and save it as
   `credentials.json` in the project root.
4. On first run, a browser opens for you to authorize. A `token.json` is created and reused afterwards.

> If you are behind a firewall/GFW, the Python process needs a proxy to reach `www.googleapis.com`:
> configure `HTTPS_PROXY` in your `.env` (see the example above — loaded automatically at startup).

---

## Running

```bash
# any OS (uv) — Python itself is managed by uv if missing
bash scripts/run-qdrant.sh   # start local Qdrant (or use Qdrant Cloud)
uv sync                      # create .venv and install dependencies
uv run main.py               # run the assistant
```

Behind a firewall/GFW, put the proxy in `.env` — it is loaded automatically at startup:

```dotenv
HTTPS_PROXY=http://127.0.0.1:7890
```

(Use the HTTP port of your proxy software; remove the line on unrestricted networks.)

`main.py` launches two threads:

1. a filesystem watcher that syncs the Obsidian vault into Qdrant,
2. a Gmail listener that polls the inbox every 60 seconds and processes new unread threads.

Stop with `Ctrl+C` — the Gmail cursor (`last_history_id`) is saved so processing resumes where it left off.

### Run as a background task on Windows

Install a scheduled task that starts the assistant at logon and auto-restarts on crash:

```powershell
.\scripts\install-task.ps1                # install
Start-ScheduledTask -TaskName EmailAssistant
.\scripts\install-task.ps1 -Remove       # uninstall
```

### Development

```bash
uv run ruff check src tests config.py main.py   # lint
uv run pytest tests/ -q                          # unit tests + coverage
```

Both also run automatically in GitHub Actions CI (see the **Actions** tab) on every push and pull request.

### Code review (Claude Code)

A project-scoped [code-review skill](.claude/skills/code-review/SKILL.md) codifies the review workflow
for this codebase. Ask Claude Code to review a change and it loads this skill, checking project-specific
failure modes (Gmail `historyId` at-least-once, CrewAI ReAct loop caps, byte-level truncation, …) on top
of the usual correctness / security / concurrency pass.

---

## Migration to another machine

The knowledge base is **rebuilt from the vault**, so you don't need to copy Qdrant data:

1. `git clone` the repo, `uv sync`
2. copy your `.md` vault notes and point `OBSIDIAN_VAULT_PATH` at them
3. add `.env` (gateway key + vault path) and `credentials.json`
4. start an empty Qdrant and run `main.py` — the knowledge base re-ingests automatically

Note: chunking/contextualization is LLM-generated (non-deterministic), so a re-ingest is semantically
equivalent but not byte-identical. For a byte-exact copy, export a Qdrant collection snapshot instead.

---

## FAQ

**Q: An email is still unread, but the system never processes it. Why?**

A: The system deduplicates with a `last_history_id` cursor (saved in `gmail_inbox_state.json`), not the
unread/read flag. Once a thread has been processed, it won't be revisited even if it stays unread. To force a
re-scan of all unread threads, delete `gmail_inbox_state.json` and restart.

**Q: It prompts for Gmail authorization on every restart.**

A: The access token expires after 1 hour. The code now silently refreshes via the refresh token, so restarts
within ~7 days won't prompt. In Google's OAuth "Testing" mode the refresh token expires after 7 days, so you
will be re-prompted roughly weekly — a Google limitation, not a bug.

**Q: Excalidraw drawings / trash notes got into the knowledge base.**

A: Both are filtered by default: Excalidraw `.md` files (detected via the `excalidraw-plugin`
frontmatter) and the `.trash/` / `.obsidian/` folders are skipped during ingestion
(see the `OBSIDIAN_EXCLUDE_*` settings in `.env`).

**Q: Ingestion is slow.**

A: Each note costs two LLM calls (chunking + contextualization). Ingestion is a one-time cost; afterwards only
changed notes are re-processed.

**Q: Why did the system not reply to an email?**

A: It only replies to emails categorized as `QUESTION`. `NOTIFICATION`, `NEWSLETTER`, and `SPAM` are skipped
intentionally.

**Q: Are replies sent automatically (or saved as drafts)?**

A: Replies are sent automatically via the Gmail API (not saved as drafts). If you prefer manual review before
sending, change `send_message` back to a draft in `gmail/handlers.py`.

**Q: What happens when the knowledge base can't answer a question?**

A: Replies go through multiple verification layers before being sent:

1. **Source verification** — every cited source path must actually exist in the vault.
2. **Faithfulness check** — a second LLM verifies every claim in the reply is supported by the
   retrieved sources. If the model hallucinated an answer, the check fails.
3. On failure, the system sends a polite fallback note: "I'm an AI email assistant; my knowledge
   base doesn't have enough to answer this. Please wait for the owner to reply or reach out another
   way." This keeps hallucinations from reaching senders.

**Q: Do replies contain citation markers like [来源1] or [1]?**

A: No. Sources are recorded internally for verification but never shown to the recipient. Any
citation markers that leak into the generated text are stripped before sending.

**Q: First run hangs on a "Would you like to view your execution traces? [y/N] (20s timeout)" prompt.**

A: That is CrewAI's built-in first-run tracing consent prompt (not AgentOps). On a fresh machine it
blocks 20 seconds waiting for terminal input, and the prompt text lingers because CrewAI never cancels
its input thread. It happens only once — the choice is recorded in `.crewai_user.json`. For a fully
automated setup, run `crewai traces disable` once (and keep `CREWAI_TRACING_ENABLED=false` in `.env`)
so it never prompts and never uploads.

**Q: What happens if an email fails to process (e.g., the LLM returns an error)?**

A: The email is not lost. Its ID is recorded in a failure queue (persisted in `gmail_inbox_state.json`)
and retried automatically on the next poll. The `historyId` cursor still advances normally, so a failed
email never blocks or duplicates the rest. The one tradeoff: an email that fails every time (e.g., its
content always breaks the pipeline) keeps being retried once per minute and stays visible in the logs —
that is the signal to fix the underlying cause.
