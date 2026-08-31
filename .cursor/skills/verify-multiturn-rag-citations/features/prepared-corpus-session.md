# Prepared corpus session

When the source set is fixed for a session (same index, many answers), the user still gets per-answer citations, but the app prepares the corpus once and aligns each answer against it.

## Sub-features

- `prepare-once` builds `PreparedCitationCorpus.from_sources` on the session documents.
- `align-many` calls `corpus.align(answer)` for more than one answer without rebuilding.
- `same-contract` returns the same `SpanCitations` shape and offset invariants as `align_citations`.

## How to get to it (user POV)

- A RAG app with a pinned knowledge base (help-center dump, one PDF, one prepared index) scores every assistant message against that set.
- A batch job scores several candidate answers against the same sources.

## Driving it with drive_session.py

Preconditions:

- Doctor is green.
- You do not need per-turn retrieval for this feature.

- **Prepare and score.** Run `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature prepared-corpus-session`.
- **Two answers.** `prepared_corpus.turns` has two entries: Q4 revenue and Europe sales. Each has `"supported"` in `statuses`.
- **Offsets.** Every citation in those turns still satisfies the source slice invariant (failures list empty).
- **Proof.** `${VERIFY_EVIDENCE_DIR}/prepared-corpus-session.json` has `"ok": true` and `prepared_corpus.ok` true.

## Gotchas

- `from_sources` is valid with or without the Rust extension. Missing Rust is not a skip; the Python prepare path is still the user path.
- Passing a custom tokenizer/segmenter changes candidate selection; this recipe uses defaults. If your change is spaCy-only, say so and do not claim this recipe covered it.
- `corpus.align` does not take a per-call source list. If the product retrieved a **subset** this turn, use `align_citations`, not this feature.
