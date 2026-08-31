# Follow-up source isolation

After a first grounded answer, the user asks a follow-up. The app retrieves a **new** source set. Citations on the follow-up answer must come from that set. Leftover documents from the previous turn must not appear as `source_id`s.

## Sub-features

- `followup-retrieve` runs a second retrieve for `What about European sales?`.
- `cite-new-set` supports the Europe answer from `europe_sales`.
- `no-leak-turn1` forbids `q4_earnings` and `press_release` on the follow-up citations.

## How to get to it (user POV)

- In the same RAG chat, send a follow-up `What about European sales?` after the Q4 revenue turn.
- In a backend, retrieve again for the follow-up query, then `align_citations(followup_answer, new_docs)` — do not reuse turn-1 `sources` unless they were retrieved again.

## Driving it with drive_session.py

Preconditions:

- Doctor is green.
- You are proving isolation, not merely that turn 1 still works.

- **Turn 1.** The user already has a Q4 revenue answer. The same command drives both turns. Run `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature follow-up-source-isolation`.
- **Turn 2 retrieve.** `turns[1].turn_id` is `turn2_europe_followup`. `retrieved_source_ids` includes `europe_sales` and must not be required to include `q4_earnings`.
- **Turn 2 citations.** `turns[1].statuses` contains `"supported"`. `cited_source_ids` includes `europe_sales` and does not include `q4_earnings` or `press_release`.
- **Proof.** `${VERIFY_EVIDENCE_DIR}/follow-up-source-isolation.json` has `"ok": true`. Both `turns[0].ok` and `turns[1].ok` are true so the session still cited turn 1 correctly.

## Gotchas

- Aligning the follow-up against the union of all documents ever retrieved hides isolation bugs. The harness must pass only this turn's hits.
- A prepared corpus over the **full** library is a different feature; do not use it to claim isolation.
- Keyword retriever quality is not the product under test, but if `europe_sales` is missing from `retrieved_source_ids` the run is invalid — fix the retrieve query in the harness, do not loosen the citation check.
