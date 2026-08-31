# Per-turn citation alignment

A RAG user asks a question, the app retrieves source documents for that turn, a model answers, and Cite-Right returns per-span citations with character offsets into those sources so the UI can highlight evidence.

## Sub-features

- `retrieve-then-cite` aligns the generated answer only against this turn's retrieved `SourceDocument`s.
- `offset-highlight` exposes `char_start` / `char_end` that slice the source text exactly.
- `status-supported` marks a well-grounded revenue answer `"supported"`.

## How to get to it (user POV)

- In a RAG chat UI, send `What was Acme Q4 revenue?` and wait for an answer with source highlights.
- In an application backend, call `align_citations(answer, retrieved_docs)` after generation.
- Optionally open the Perplexity demo `GET /` after launching it, which shows one static aligned answer (not a chat).

## Driving it with drive_session.py

Preconditions:

- Doctor is green from this checkout.
- `VERIFY_EVIDENCE_DIR` exists.
- No requirement to start the demo.

- **Ask and retrieve.** The user asks about Q4 revenue. Run `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature per-turn-alignment`. The JSON `turns[0].turn_id` is `turn1_q4_revenue` and `retrieved_source_ids` includes `q4_earnings` or `press_release`.
- **Cite the answer.** The harness calls `align_citations` on `Revenue reached 5.2 billion dollars, exceeding analyst expectations.` against those sources. `turns[0].statuses` contains `"supported"` and `cited_source_ids` is a subset of `retrieved_source_ids`.
- **Check highlights.** In `turns[0].spans[*].citations[*]`, `offset_invariant` is `true` and `evidence` is a substring of the named source.
- **Proof.** Open `${VERIFY_EVIDENCE_DIR}/per-turn-alignment.json` and `${VERIFY_EVIDENCE_DIR}/session-summary.txt`. The summary line for `turn1_q4_revenue` shows `ok=True`. Top-level `"ok": true`.

## Gotchas

- Passing the full knowledge base instead of the retrieved set is not this feature; that is the prepared-corpus session.
- `retrieval_support` is not a citation. Status comes from localized `citations` and `answer_coverage`, not embedding-only hits.
- The demo `/api/citations` path proves a single canned DeepSeek mHC answer. Do not count it as `turn1_q4_revenue`.
- Status spelling is `"supported"`, never `"fully_supported"`.
