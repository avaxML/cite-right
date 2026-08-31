---
name: verify-multiturn-rag-citations
description: >-
  Drive Cite-Right as a multi-turn RAG citation post-processor (Python library
  plus optional Perplexity demo HTTP UI). Use when changing align_citations,
  PreparedCitationCorpus, convenience formatters, citation offsets/status, or
  any RAG follow-up citation behavior, and you need a user-path proof rather
  than only pytest.
---

# Verify multi-turn RAG citations

Cite-Right is a **Python library**. A RAG product calls `align_citations` (or `PreparedCitationCorpus.align`) after each generated turn, against **that turn's retrieved sources**. There is no conversation object and no chat CLI. The Perplexity demo at `examples/perplexity_demo/` is a **single static Q&A** HTML page; it is a visual citation UI, not a multi-turn path.

This skill is for the next agent, cold. Drive the library the way a user-facing RAG app would: retrieve → generate → cite, then a follow-up retrieve → generate → cite. Do not treat `pytest` as the user path.

Helpers live in `.cursor/skills/verify-multiturn-rag-citations/scripts/`.

## Launch

From the repo root (directory with `pyproject.toml`).

**Library (required for every proof):**

```bash
export VERIFY_RUN_ID="${VERIFY_RUN_ID:-$RANDOM}"
export VERIFY_EVIDENCE_DIR="/tmp/cite-right-verify-${VERIFY_RUN_ID}"
mkdir -p "${VERIFY_EVIDENCE_DIR}"

# Prefer the repo's uv workflow. Python 3.11+ is required.
uv venv --python 3.11
uv sync --frozen --no-install-project
uv run --no-sync maturin develop
```

If `uv` is missing, install it (`curl -LsSf https://astral.sh/uv/install.sh | sh`) or use an existing venv that already has this checkout installed editable (`pip install -e .` after numpy/pydantic; Rust wheels/extension via maturin when testing the default backend).

Ready when:

```bash
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/doctor.py
```

prints `"ok": true` and `"imported_from_checkout": true`. There is **no long-lived library server**. Each `drive_session.py` invocation is a short-lived process.

**Optional demo UI** (static citations page, port must be ours):

```bash
export CITE_RIGHT_VERIFY_PORT="${CITE_RIGHT_VERIFY_PORT:-8765}"
export CITE_RIGHT_VERIFY_DEMO_URL="http://127.0.0.1:${CITE_RIGHT_VERIFY_PORT}"
export CITE_RIGHT_VERIFY_PID_FILE="/tmp/cite-right-verify-${VERIFY_RUN_ID}.uvicorn.pid"
uv pip install -r examples/perplexity_demo/requirements.txt
uv run uvicorn examples.perplexity_demo.app:app --host 127.0.0.1 --port "${CITE_RIGHT_VERIFY_PORT}" &
echo $! > "${CITE_RIGHT_VERIFY_PID_FILE}"
# Ready when GET ${CITE_RIGHT_VERIFY_DEMO_URL}/api/citations returns 200 JSON.
```

Do not bind `:8000` if a developer demo may already own it. Do not drive a demo you did not start.

**Teardown:** run `bash .cursor/skills/verify-multiturn-rag-citations/scripts/teardown.sh`. It kills only the PID in `CITE_RIGHT_VERIFY_PID_FILE` (or `/tmp/cite-right-verify-$VERIFY_RUN_ID.uvicorn.pid`). It must not delete `${VERIFY_EVIDENCE_DIR}`.

## Doctor

Read-only. Run first whenever import, offsets, or the demo look off:

```bash
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/doctor.py
```

Pass criteria in the JSON:

- `cite_right` imports from this checkout (`imported_from_checkout`)
- `self_test.status` is `"supported"`
- `offset_invariant` and `answer_span_invariant` are true
- If `CITE_RIGHT_VERIFY_DEMO_URL` is set, `GET /api/citations` is reachable

`has_rust_core` is informational. Python fallback is a valid drive; do not fail doctor solely because Rust is missing unless the change under test is the Rust extension.

## Drive

Harness: `drive_session.py`. It is a stand-in RAG app: an in-process lexical retriever over four `SourceDocument`s, then `align_citations` on **only the retrieved set for that turn**.

```bash
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature per-turn-alignment
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature follow-up-source-isolation
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature prepared-corpus-session
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature ungrounded-follow-up
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature annotated-footnotes
uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature all
```

`--feature` selects which mapped feature to exercise. Default is `all`.

Public API to call (do not reach for private `_process_*` helpers):

- `align_citations(answer, sources)` after each turn
- `PreparedCitationCorpus.from_sources(sources)` once per session, then `corpus.align(answer)` per turn when the source set is fixed
- `annotate_answer` / `format_with_citations` / `get_citation_summary` for user-visible markers
- Status literals are exactly `"supported"`, `"partial"`, `"unsupported"` (`"partial"`, never `"partially_supported"`)

Stable handles:

- `SourceDocument.id` values in the harness: `q4_earnings`, `press_release`, `europe_sales`, `product_line`
- Turn ids: `turn1_q4_revenue`, `turn2_europe_followup`, `turn3_hallucinated_followup`, `turn4_conversational_wrapper`
- Demo routes if launched: `GET /` (HTML), `GET /api/citations` (JSON with `question`, `answer`, `spans`, `sources`)

Read the feature map in `features/` and drive every entry point listed for the feature under test. A proof that only runs `pytest tests/test_citations_api.py` is incomplete.

## Evidence

Write proofs under `${VERIFY_EVIDENCE_DIR}` (default `/tmp/cite-right-verify-$VERIFY_RUN_ID`). `drive_session.py` always writes:

- `${VERIFY_EVIDENCE_DIR}/<feature>.json` — full turn payloads, statuses, offsets, failures
- `${VERIFY_EVIDENCE_DIR}/session-summary.txt` — one line per turn

Proof standards:

- Exercise the real user path: `align_citations` / `corpus.align` / `annotate_answer` on answers a RAG app would show, not internal aligner methods and not pytest-only fixtures as the sole proof.
- Capture the action (question, retrieved `source_id`s, answer) **and** the resulting state (per-span `status`, `citations`, offset checks).
- Side effects: for library drives, the side effect is the citation payload. Confirm `source.text[char_start:char_end] == evidence` and `answer[span.char_start:span.char_end] == span.text`. For the demo, confirm `GET /api/citations` JSON matches those invariants on `example_data.SOURCES`.
- Do not mock `align_citations`. The retriever inside `drive_session.py` is the product boundary (stand-in vector store); alignment must run for real.
- A dry-run does not exist. If someone adds `--dry-run` later, observe that alignment still ran (non-empty `spans` in the JSON) rather than trusting the flag.

Keep copies of `${VERIFY_EVIDENCE_DIR}/*.json` if you need them in the agent walkthrough folder; cleanup must leave `${VERIFY_EVIDENCE_DIR}` intact.

## Cleanup

```bash
bash .cursor/skills/verify-multiturn-rag-citations/scripts/teardown.sh
```

Kill only the uvicorn PID this run recorded. Do not `pkill uvicorn` / `pkill python`. Do not delete `${VERIFY_EVIDENCE_DIR}`. Scratch: the pid file only.

Two library drives may run side by side (process-local, no shared port). Two demos may run side by side only on different `CITE_RIGHT_VERIFY_PORT` values. Never attach to an already-running `:8000` demo.

## Helpers

| Command | Role |
| --- | --- |
| `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/doctor.py` | Read-only health + offset self-test |
| `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature <id>` | Multi-turn RAG drive + evidence JSON |
| `bash .cursor/skills/verify-multiturn-rag-citations/scripts/teardown.sh` | Stop only the demo this run started |

## Feature map

See [features/README.md](features/README.md).
