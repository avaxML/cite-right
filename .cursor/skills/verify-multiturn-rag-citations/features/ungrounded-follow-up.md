# Ungrounded follow-up

The user (or the model) produces a turn that is not in the sources: a hallucinated claim, or a conversational wrapper with no factual content. The UI must not attach fake highlights. Those spans are `"unsupported"` with empty `citations`.

## Sub-features

- `hallucinated-claim` marks a Mars-colonization follow-up `"unsupported"` even when Q4 docs were retrieved.
- `wrapper-sure` marks `Sure!` `"unsupported"`.
- `no-false-cite` keeps `citations` empty on those turns.

## How to get to it (user POV)

- In chat, ask `Did they colonize Mars?` after a financial RAG session and see the answer flagged as ungrounded / without source chips.
- Receive a model preface `Sure!` and see it unmarked by source ids.

## Driving it with drive_session.py

Preconditions:

- Doctor is green.
- Turn 1 sources may still be retrieved; that is the point.

- **Hallucination.** Run `uv run python .cursor/skills/verify-multiturn-rag-citations/scripts/drive_session.py --feature ungrounded-follow-up`. `turns` include `turn3_hallucinated_followup` with `statuses` all `"unsupported"` and `cited_source_ids` empty.
- **Wrapper.** `turn4_conversational_wrapper` has answer `Sure!`, status `"unsupported"`, empty citations.
- **Proof.** `${VERIFY_EVIDENCE_DIR}/ungrounded-follow-up.json` has `"ok": true`.

## Gotchas

- `"partial"` is not this feature. A contradiction of a real source (`The vaccine is not safe.` vs a safety sentence) is `"partial"` with citations. Do not assert `"unsupported"` there.
- `"unsupported"` means no localized citation survived, not a high-precision hallucination detector. Still require empty `citations` for these two canned answers.
- Embedding extras can add `retrieval_support` without changing status. Status must remain `"unsupported"` if `citations` is empty.
