---
type: concept
title: Convenience helpers
description: "High-level helper functions in cite-right for common RAG post-processing workflows: groundedness checks, answer annotation, citation formatting, and summary generation."
tags: [cite-right, api, hallucination, citation-alignment, rag]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-2239349d0f5307d9d0756d4c
    resource: repo://src/cite_right/convenience.py
  - id: openwiki-source-5b90716cf19f71404fb5a027
    resource: repo://src/cite_right/hallucination.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The `cite_right.convenience` module (`src/cite_right/convenience.py`) exposes six high-level helper functions designed to simplify common RAG post-processing tasks. These functions wrap the core `align_citations()` and `compute_hallucination_metrics()` pipeline into single-call APIs for quality gating, answer annotation, and human-readable reporting. All six are re-exported from the top-level `cite_right` package.

## Groundedness gate functions

### `is_grounded()`

```python
def is_grounded(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    threshold: float = 0.5,
    config: CitationConfig | None = None,
    hallucination_config: HallucinationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> bool
```

Returns `True` when `metrics.groundedness_score >= threshold`, `False` otherwise. Internally, `is_grounded()` calls `align_citations()` followed by `compute_hallucination_metrics()`. The default threshold of `0.5` means at least 50% of the answer (by character count) must be grounded in source documents.

The `groundedness_score` is a weighted average of per-span confidence values, where confidence for each span is derived from its top-ranked citation's `answer_coverage` component. A span with no citations contributes `0.0` confidence; a span with partial citations contributes the best citation's coverage value.

### `is_hallucinated()`

```python
def is_hallucinated(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    threshold: float = 0.5,
    config: CitationConfig | None = None,
    hallucination_config: HallucinationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> bool
```

Returns `True` when `metrics.hallucination_rate > threshold`, `False` otherwise. The `hallucination_rate` equals `1.0 - groundedness_score`, so the two functions are complementary inverses. The default threshold of `0.5` flags answers where more than 50% of the content (by character count) lacks grounding evidence.

### Threshold relationship to metrics

The groundedness score and hallucination rate form a closed pair:

```
groundedness_score = weighted_avg(span_confidences)       # 0.0 – 1.0
hallucination_rate = 1.0 - groundedness_score             # 0.0 – 1.0
```

- `is_grounded(threshold=t)` is equivalent to `not is_hallucinated(threshold=t)` for `t in (0, 1)`.
- The `HallucinationConfig.include_partial_in_grounded` field controls whether "partial" status spans count toward the grounded score. When `True` (the default), partial spans contribute their coverage confidence; when `False`, they contribute `0`.
- The `HallucinationConfig.weak_citation_threshold` field (default `0.4`) identifies weak citations — spans whose best citation confidence falls below this threshold but are not unsupported.

### `check_groundedness()`

```python
def check_groundedness(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: CitationConfig | None = None,
    hallucination_config: HallucinationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> HallucinationMetrics
```

Returns the full `HallucinationMetrics` object from a single call. This function is the recommended entrypoint when you need access to detailed per-span breakdowns, unsupported span lists, or ratio fields rather than just a boolean gate. It exposes the `unsupported_spans` field (list of `AnswerSpan` objects) and `weakly_supported_spans` for downstream inspection or logging.

## Answer annotation functions

### `annotate_answer()`

```python
def annotate_answer(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: CitationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
    format: Literal["markdown", "superscript", "footnote"] = "markdown",
    include_unsupported: bool = True,
) -> str
```

Performs citation alignment and returns the answer with inline citation markers appended to each answer span. It delegates to `format_with_citations()` after computing alignment results.

### `format_with_citations()`

```python
def format_with_citations(
    answer: str,
    span_citations: Sequence[SpanCitations],
    *,
    format: Literal["markdown", "superscript", "footnote"] = "markdown",
    include_unsupported: bool = True,
) -> str
```

Takes pre-computed `SpanCitations` results (e.g., from a separate call to `align_citations()`) and inserts citation markers into the answer text.

### Citation format styles

| `format` value | Marker style | Example output |
|---|---|---|
| `"markdown"` (default) | `[n]` | `Revenue grew 15%.[1] Profits doubled.[?]` |
| `"superscript"` | `^n` | `Revenue grew 15%.^1 Profits doubled.^?` |
| `"footnote"` | `[^n]` | `Revenue grew 15%.[^1] Profits doubled.[^?]` |

Markers are inserted at the end of each answer span, after stripping trailing whitespace. Spans are processed in reverse character order to preserve offsets during insertion.

### `include_unsupported` behavior

When `include_unsupported=True` (the default), spans with no citations are marked with `[?]` (or `^?` / `[^?]` depending on format). When `False`, unsupported spans receive no marker at all. This parameter lets you produce clean annotated output for downstream rendering while still preserving the alignment results.

Source numbers are assigned in the order sources first appear across all spans. A given source receives the same number throughout the formatted answer, even if it supports multiple spans.

## Citation summary function

### `get_citation_summary()`

```python
def get_citation_summary(span_citations: Sequence[SpanCitations]) -> str
```

Returns a human-readable summary string describing the citation results. The summary reports:

- Number of spans that are fully supported, partially supported, and unsupported.
- Number of unsupported spans that had **retrieval-only support candidates** — spans where `not sc.citations and sc.retrieval_support` is true. These are spans that passed lexical or embedding retrieval but failed to localize into an exact citation.
- List of unique source IDs that were cited.

Example output:

```
Citation Summary:
- 1 of 2 spans fully supported
- 1 spans unsupported
- 1 unsupported spans had retrieval-only support candidates
- Sources cited: report
```

If the input sequence is empty, the function returns `"Citation Summary: No spans to analyze"`.

## Relationship to core pipeline

```
is_grounded / is_hallucinated / check_groundedness
  └─> align_citations()          [citation alignment]
      └─> compute_hallucination_metrics()  [metrics aggregation]

annotate_answer
  └─> align_citations()
      └─> format_with_citations() [marker insertion]

format_with_citations / get_citation_summary
  └─> accept pre-computed SpanCitations directly
```

All functions accept the same underlying configuration objects (`CitationConfig`, `HallucinationConfig`, `Tokenizer`, `Embedder`) that drive the core alignment pipeline. Custom segmenters, tokenizers, and embedders can be injected at the convenience layer without touching `align_citations()` directly.
