---
type: concept
title: Citation status semantics
description: The rules that determine whether an answer span is marked supported, partial, or unsupported — and why retrieval_support never flips status.
tags: [citation-alignment, status, cite-right, validation]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-ad97420c2a0900f616ed0fef
    resource: repo://src/cite_right/contradiction.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

Every answer span processed by `align_citations()` returns a `SpanCitations` object with a `status` field set to one of three values: `supported`, `partial`, or `unsupported`. Status is computed by `_span_status()` in `src/cite_right/citations.py` after citation ranking, based solely on the best-ranked exact citation's `answer_coverage` component and the result of a contradiction check against the full source passage.

## The three status values

### `supported`

The span has a top-ranked citation whose `answer_coverage` is at or above `CitationConfig.supported_answer_coverage` (default `0.6`). This means the aligned evidence covers at least 60 % of the answer tokens (after stopword filtering) and no contradiction was detected.

`supported` is the only status that means the claim is well-grounded in the cited source.

### `partial`

The span has at least one exact citation, but either:

- The best citation's `answer_coverage` is below `supported_answer_coverage`; **or**
- A contradiction was detected between the answer and the candidate passage.

`partial` covers two distinct situations that share the same status code: **incomplete coverage** (the source mentions the topic but the aligned evidence is too thin) and **contradiction** (the source was found and cited, but it contradicts the claim).

There is no separate `partially_supported` or `contradicted` status. Both cases produce `partial`.

### `unsupported`

No exact citations were found for this span. Either no candidate passed the alignment quality thresholds, or the span is hallucinated/paraphrased beyond lexical or semantic recognition.

## How `_span_status()` decides

`_span_status()` (repo://src/cite_right/citations.py#L1610-L1633) implements the decision tree:

```python
def _span_status(
    citations: Sequence[Citation],
    cfg: CitationConfig,
    answer_text: str | None = None,
    candidates: Sequence[Candidate] | None = None,
) -> Literal["supported", "partial", "unsupported"]:
    if not citations:
        return "unsupported"
    best = citations[0]
    coverage = float(best.components.get("answer_coverage", 0.0))

    # Check for contradictions if answer text is provided.
    # Use the candidate passage so leftover tokens beyond truncated evidence
    # (e.g. "BC", "of which came in the first half") are visible.
    if answer_text is not None and check_contradiction(
        answer_text, _contradiction_context(best, candidates)
    ):
        # Downgrade to partial (not unsupported) if contradiction detected
        # because we have evidence, it just contradicts the claim
        return "partial"

    if coverage >= cfg.supported_answer_coverage:
        return "supported"
    return "partial"
```

The steps in order:

1. **No citations → `unsupported`** (line 1616–1617). No further checks.
2. **Contradiction check → `partial`** (lines 1621–1629). If `answer_text` is available and `check_contradiction()` returns `True`, return `partial` immediately. This takes priority over the coverage threshold.
3. **Coverage threshold → `supported` or `partial`** (lines 1631–1633). Compare the best citation's `answer_coverage` against `cfg.supported_answer_coverage` (default `0.6`). At or above → `supported`; below → `partial`.

## What does NOT affect status

### `retrieval_support` does not flip status

High embedding similarity without exact citation alignment produces a `RetrievalSupport` entry, not a `Citation`. `RetrievalSupport` is surfaced in `SpanCitations.retrieval_support` for transparency, but `_span_status()` never reads it. A span with only retrieval support (no exact citation) will still be marked `unsupported`.

This design is intentional: `retrieval_support` confirms that semantic retrieval found a relevant passage, but it does not provide localized, character-accurate evidence. Status is reserved for the quality of exact citations.

```python
# From SpanCitations docstring (repo://src/cite_right/core/results.py#L311-L313)
retrieval_support: List of retrieval-only support signals for passages
    selected during lexical and/or embedding candidate search but not
    localized into an exact citation.
```

The `CitationConfig` validator (repo://src/cite_right/core/citation_config.py#L85-L97) explicitly rejects the old options `allow_embedding_only` and `supported_embedding_similarity`, reinforcing that embedding-only results belong in `retrieval_support`, not in status:

```python
@model_validator(mode="before")
@classmethod
def _reject_removed_embedding_only_options(cls, data: object) -> object:
    if isinstance(data, dict) and {
        "allow_embedding_only",
        "supported_embedding_similarity",
    }.intersection(data):
        raise ValueError(
            "allow_embedding_only and supported_embedding_similarity were "
            "removed; inspect SpanCitations.retrieval_support for semantic "
            "retrieval signals"
        )
    return data
```

### Literal citations are `partial`, never `partially_supported`

The model only has three status values. A literal citation (exact token match) with high `answer_coverage` will be `supported`; with low coverage it is `partial`. There is no `partially_supported` enum value.

## Contradiction detection uses the full passage, not truncated evidence

Smith-Waterman local alignment returns a *truncated* evidence span — the window of the source passage that best matches the answer. This window can omit contextual words that appear beyond the aligned region.

`_contradiction_context()` (repo://src/cite_right/citations.py#L1592-L1607) prefers the full candidate passage over `citation.evidence` when looking for contradictions:

```python
def _contradiction_context(
    citation: Citation,
    candidates: Sequence[Candidate] | None,
) -> str:
    """Prefer the candidate passage over truncated Smith-Waterman evidence.

    Leftover n-grams (issue #48) attach to the wrong slot when alignment
    truncates evidence and hides the contradicting remainder of the passage.
    """
    if candidates:
        for candidate in candidates:
            if candidate.global_index == citation.candidate_index:
                passage = candidate.passage.text
                if passage:
                    return passage
    return citation.evidence
```

This ensures that contradiction signals like negation words (`not`, `n't`) that appear before the aligned window, or numbers that appear after it, are visible to the contradiction checks.

### Why contradiction → `partial`, not `unsupported`

When a contradiction is detected, `_span_status()` returns `partial` rather than `unsupported`. The reason is structural:

- `unsupported` means *no evidence was found* — the citation is discarded.
- `partial` with contradiction means *evidence was found and cited, but the claim contradicts that evidence* — the citation is retained for transparency.

Retaining the citation lets downstream consumers (e.g. a fact-checking UI) surface the contradictory passage and show the user exactly where the claim diverged from the source.

The five contradiction checks in `check_contradiction()` (repo://src/cite_right/contradiction.py#L67-L87) are: negation mismatch, number mismatch, entity swap, temporal/polarity mismatch, and number-context mismatch (the "leftover n-gram" case).

## Configuration thresholds

The primary threshold is `supported_answer_coverage`:

| Config | Default | Description |
|---|---|---|
| `supported_answer_coverage` | `0.6` | Coverage ≥ this → `supported`; below → `partial` |
| `min_answer_coverage` | `0.2` | Minimum coverage to produce a citation at all |
| `min_final_score` | `0.0` | Minimum composite score (alignment + lexical + embedding) |
| `min_alignment_score` | `0` | Minimum raw Smith-Waterman score |

The `CitationConfig` presets adjust these:

- **`strict()`**: `supported_answer_coverage=0.7`, `min_answer_coverage=0.4` — high precision
- **`permissive()`**: `supported_answer_coverage=0.4`, `min_answer_coverage=0.15` — tolerant of paraphrasing
- **`fast()`**: `top_k=1`, strict candidate limits — throughput-optimized
- **`balanced()`** / default: `supported_answer_coverage=0.6` — standard threshold

## Model invariant

`SpanCitations` carries a model validator (repo://src/cite_right/core/results.py#L339-L348) that enforces:

- `unsupported` status requires `citations` to be empty.
- `supported` or `partial` status requires at least one citation.

This means the model itself rejects the impossible combination of a non-empty citation list with `unsupported` status.
