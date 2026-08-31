# Multi-turn RAG citation verification map

This directory is the maintained source for verifying user-facing Cite-Right behavior in a **multi-turn RAG** product: each chat turn retrieves sources, a model generates an answer, and Cite-Right attaches character-accurate citations for that turn only.

Read this index before driving, then use the matching feature file as the recipe.

## Baseline preconditions

- Repo root has `pyproject.toml` and `src/cite_right/`.
- `VERIFY_RUN_ID` and `VERIFY_EVIDENCE_DIR=/tmp/cite-right-verify-$VERIFY_RUN_ID` are set.
- `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/doctor.py` reports `"ok": true` and `"imported_from_checkout": true`.
- Never treat a demo on port 8000 as ours unless this run started it on `CITE_RIGHT_VERIFY_PORT`.
- Never use pytest as a substitute for a mapped entry point.

## Driving conventions

- Start every recipe from a fresh `drive_session.py` process unless the feature says to reuse `PreparedCitationCorpus`.
- Call public APIs only: `align_citations`, `PreparedCitationCorpus`, `annotate_answer`, `format_with_citations`, `get_citation_summary`.
- Treat `--feature` values and `SourceDocument.id` strings as literal.
- After a mutation to alignment or status logic, re-run doctor then the affected feature file end to end.
- Cleanup stops only the optional demo PID; leave evidence JSON on disk.

## Proof and skip reporting

- Capture the user question, retrieved source ids, generated answer, and the citation payload (status + evidence + offsets).
- Library proof is the JSON under `${VERIFY_EVIDENCE_DIR}` plus a re-slice check of `char_start`/`char_end`.
- Demo proof (if used) is `GET /api/citations` body and the HTML at `GET /`. The demo is one static turn; do not report it as follow-up isolation.
- Record the feature id and `--feature` value with every artifact.
- Report an unreachable path with the command and the unmet precondition (for example, doctor failed).
- Do not report a skipped entry point as verified through pytest.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with drive_session.py` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Per-turn citation alignment](./per-turn-alignment.md) covers retrieve → generate → cite for a single RAG turn, including offset-accurate evidence.
- [Follow-up source isolation](./follow-up-source-isolation.md) covers a second turn whose citations must come from the new retrieved set, not leftover turn-1 documents.
- [Prepared corpus session](./prepared-corpus-session.md) covers reusing one prepared source set across several answers in the same session.
- [Ungrounded follow-up](./ungrounded-follow-up.md) covers hallucinated claims and conversational wrappers staying `"unsupported"`.
- [Annotated footnotes](./annotated-footnotes.md) covers user-visible `[n]` markers from `annotate_answer` / `format_with_citations`.
