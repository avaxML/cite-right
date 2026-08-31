# Annotated footnotes

After citations exist, the user-facing answer shows inline markers (`[1]`, `[2]`, or `[?]`) so a reader can map sentences to sources without reading raw offsets.

## Sub-features

- `annotate-markdown` inserts `[n]` markers via `annotate_answer(..., format="markdown")`.
- `format-from-results` produces the same markers from existing `align_citations` results via `format_with_citations`.
- `unsupported-marker` can mark ungrounded spans with `[?]` when `include_unsupported` is true (default).

## How to get to it (user POV)

- A chat UI renders footnote-style citations next to the assistant message.
- A markdown report generator prints an annotated answer plus a source list.
- The Perplexity demo HTML shows footnote-style citations for its static answer (visual analogue, different payload).

## Driving it with drive_session.py

Preconditions:

- Doctor is green.
- Per-turn alignment for turn 1 is expected to succeed (markers are meaningless if nothing is supported).

- **Annotate grounded turns.** Run `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature annotated-footnotes`.
- **Markers present.** `turns[0].annotated_answer` and `turns[0].formatted_answer` contain `[1]` (or additional `[n]`) after the revenue sentence. They are not identical to the raw `answer` string.
- **Summary.** `turns[0].summary` starts with `Citation Summary:` and reports at least one fully supported span.
- **Proof.** `${VERIFY_EVIDENCE_DIR}/annotated-footnotes.json` has `"ok": true`. Save the `annotated_answer` strings in that JSON as the user-visible artifact.

## Gotchas

- `format="footnote"` uses `[^1]` and `format="superscript"` uses `^1`. This recipe asserts markdown `[n]`.
- Markers are inserted at span ends; do not grep the raw `answer` field for `[1]`.
- Demo HTML footnotes are a different numbering space (DeepSeek mHC sources). Do not require `[1]` in the demo to match `q4_earnings`.
