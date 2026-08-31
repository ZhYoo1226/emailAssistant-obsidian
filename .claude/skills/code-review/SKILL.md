---
name: code-review
description: Review code for correctness, security, and maintainability. Use when the user asks to "review", "check", "audit", or "find bugs" in code, a diff, or a PR.
---

# Code Review

Review the target code and report findings ordered by severity, each with a
concrete `file:line` reference and a suggested fix.

## When to use

- The user asks to "review", "check", "audit", or "look for bugs" in code, a diff, or a PR.

## Workflow

1. **Scope** — identify the files/diff under review. Read them in full, plus the
   callers/callees that determine their contracts, before judging.
2. **Context** — understand intent: nearby code, docstrings, config, and tests,
   so findings match the project's actual conventions rather than generic rules.
3. **Check, in this order**:
   - Correctness — logic bugs, off-by-one, wrong condition/variable, type errors.
   - Security — injection, secret exposure, auth/authorization, path traversal.
   - Concurrency — races, shared mutable state, locks, thread-safety (this project
     runs two threads: Obsidian watcher + Gmail poller).
   - Error handling — swallowed exceptions, missing retries, partial-failure state.
   - Edge cases — empty/None, multibyte (CJK), very long input, encoding/timezone.
   - Performance — accidental O(n²), N+1, redundant network/DB calls.
   - Consistency — matches surrounding idiom, naming, comment density.
4. **Verify** — for each finding, trace the actual failure path with concrete
   inputs. Drop anything that doesn't reproduce; never invent a finding to look
   thorough. If a check turns up nothing, say so.
5. **Report** — group by severity (critical / major / minor / nit). Each finding:
   `file:line`, one-sentence defect, concrete failure scenario, suggested fix.
   If you ran tests/lint, state the actual result.

## This project's specifics

- Python 3.12 · CrewAI 1.15 · Qdrant hybrid search (dense+sparse+rerank) · fastembed · Gmail API.
- Lint: `uv run ruff check src tests config.py main.py`
- Tests (with coverage): `uv run pytest tests/ -q`
- Type check: pyright via Pylance (`extraPaths = ["."]` in `pyproject.toml`).
- Conventions: Chinese docstrings/comments; `errors="ignore"` / `errors="replace"`
  for byte-level truncation; tests are pure-function only (no network/LLM/Qdrant).
- **Gmail at-least-once**: `last_history_id` must advance only AFTER a message is
  fully processed; advancing it before processing (or swallowing handler
  exceptions) silently drops emails on crash / LLM failure — check `gmail/inbox.py`.
