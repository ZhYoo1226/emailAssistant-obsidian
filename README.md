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
        Qdrant knowledge-base                   (QdrantSearchTool)
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
- **Type filtering** — Excalidraw drawings and `.trash/` files are skipped (they are not prose notes).

---

## Tech stack

- [CrewAI](https://www.crewai.com/) `0.95` — agent orchestration
- [Qdrant](https://qdrant.tech/) — vector store / knowledge base
- [fastembed](https://github.com/qdrant/fastembed) — local ONNX embeddings (`BAAI/bge-small-en-v1.5`)
- An **OpenAI-compatible gateway** serving DeepSeek models (all LLM calls)
- Gmail API (OAuth 2.0)

No GPU required. LLM inference goes through the gateway API; embeddings run locally on CPU.

---

## Prerequisites

- Python **3.10–3.12**
- Dependency management: [uv](https://docs.astral.sh/uv/) — installs Python and dependencies in one go
- Docker (to run Qdrant locally), **or** a free [Qdrant Cloud](https://cloud.qdrant.io/) account
- An OpenAI-compatible gateway URL + API key
- Gmail API credentials (see below)

---

## Configuration

Copy `.env.example` to `.env` and fill in the values:

```dotenv
# LLM gateway (OpenAI-compatible)
GATEWAY_BASE_URL=https://your-gateway.example.com/v1
GATEWAY_API_KEY=YOUR_API_KEY
GATEWAY_MAIN_MODEL=deepseek-v4-pro
GATEWAY_FAST_MODEL=deepseek-v4-flash

# Local embedding model (fastembed)
# Set HF_ENDPOINT=https://hf-mirror.com if Hugging Face is unreachable.
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

# Qdrant (local Docker needs no API key)
QDRANT_LOCATION=http://localhost:6333
QDRANT_API_KEY=

# Obsidian vault path (the knowledge base source)
OBSIDIAN_VAULT_PATH=/path/to/your/vault
```

### Gmail OAuth credentials

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and **enable the Gmail API**.
2. Configure the OAuth consent screen as **External**, and add your Gmail as a test user.
3. Create an **OAuth client ID** of type **Desktop app**, download the JSON, and save it as
   `credentials.json` in the project root.
4. On first run, a browser opens for you to authorize. A `token.json` is created and reused afterwards.

> If you are behind a firewall/GFW, the Python process needs a proxy to reach `www.googleapis.com`:
> `export HTTPS_PROXY=http://127.0.0.1:7890` (set it in your shell profile).

---

## Running

```bash
# any OS (uv) — Python itself is managed by uv if missing
bash scripts/run-qdrant.sh   # start local Qdrant (or use Qdrant Cloud)
uv sync                      # create .venv and install dependencies
uv run main.py               # run the assistant
```

Behind a firewall/GFW, set the proxy before starting:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890   # macOS / Linux
```

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7897"; $env:HTTP_PROXY = "http://127.0.0.1:7897"   # Windows
```

`main.py` launches two threads:

1. a filesystem watcher that syncs the Obsidian vault into Qdrant,
2. a Gmail listener that polls the inbox every 60 seconds and processes new unread threads.

Stop with `Ctrl+C` — the Gmail cursor (`last_history_id`) is saved so processing resumes where it left off.

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

A: Both are filtered: Excalidraw `.md` files (detected via the `excalidraw-plugin` frontmatter) and `.trash/`
files are skipped during ingestion.

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

A: The reply is checked for faithfulness — a second LLM verifies every claim is supported by the retrieved
sources. If the check fails (e.g. the model hallucinated an answer), the system sends a polite fallback note:
"I'm an AI email assistant; my knowledge base doesn't have enough to answer this. Please wait for the owner
to reply or reach out another way." This keeps hallucinations from reaching senders.
