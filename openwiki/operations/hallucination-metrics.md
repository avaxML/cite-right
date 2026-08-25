---
type: concept
title: Hallucination metrics
description: "The `compute_hallucination_metrics` function, `HallucinationConfig` knobs, and how per-span status rolls up into `groundedness_score`, `hallucination_rate`, and ratio fields."
tags: [hallucination, groundedness, metrics, cite-right]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-2239349d0f5307d9d0756d4c
    resource: repo://src/cite_right/convenience.py
  - id: openwiki-source-5b90716cf19f71404fb5a027
    resource: repo://src/cite_right/hallucination.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The hallucination metrics module (`src/cite_right/hallucination.py`) consumes the output of `align_citations()` — a sequence of `SpanCitations` objects — and produces a single `HallucinationMetrics` aggregate. This function is the standard entrypoint for RAG quality gates, monitoring pipelines, and evaluation harnesses.

The public API surface is small:

```python
from cite_right import compute_hallucination_metrics, HallucinationConfig

metrics = compute_hallucination_metrics(span_citations, config=HallucinationConfig())
```

## HallucinationConfig

`HallucinationConfig` is a frozen Pydantic model with two knobs:

| Attribute | Default | Purpose |
|-----------|---------|---------|
| `weak_citation_threshold` | `0.4` | Spans whose best citation has `answer_coverage` below this value are flagged as "weak" and appear in `weakly_supported_spans`. |
| `include_partial_in_grounded` | `True` | If `True`, `partial`-status spans count toward `groundedness_score` weighted by their citation quality. If `False`, only `supported` spans contribute; `partial` spans are excluded entirely. |

Both fields are documented in the class docstring (repo://src/cite_right/hallucination.py#L16-L30).

## How groundedness_score is computed

`groundedness_score` is a **length-weighted average** of per-span confidence values:

```
groundedness_score = weighted_confidence_sum / total_chars
```

where `weighted_confidence_sum` accumulates `confidence * span_len` only for spans that are considered "grounded" (see `include_partial_in_grounded` above), and `total_chars` is the sum of all span lengths.

This means longer unsupported or low-confidence spans pull the score down more than short ones, giving a text-length-proportional metric rather than a simple span count.

### Confidence extraction

For each `SpanCitations`, `_MetricsAccumulator._extract_confidence()` (repo://src/cite_right/hallucination.py#L189-L207) picks the **best citation** by a composite key: `(citation_confidence, citation.score)`. The confidence used is:

```python
citation.components.get("answer_coverage", 0.0)
```

This is the fraction of answer tokens matched by the exact alignment. It is 0.0 for citations that exist only via semantic/embedding similarity with no lexical match. The `best_citation_score` stored in `SpanConfidence` is the citation's raw `score` field (the weighted combination of alignment, coverage, lexical, and embedding components), not the `answer_coverage`.

The per-span `confidence` in `SpanConfidence` therefore reflects how much of the answer text is actually grounded in cited source passages, not the overall citation quality score.

### Status to groundedness mapping

| Status | Included in groundedness? | Notes |
|--------|--------------------------|-------|
| `supported` | Always | Counts with its full `confidence` weight. |
| `partial` | Conditionally | Counts only when `include_partial_in_grounded=True`. |
| `unsupported` | Never | `confidence` is 0.0 and `is_grounded` is `False`. |

This mapping is implemented in `_MetricsAccumulator._update_status_counts()` (repo://src/cite_right/hallucination.py#L214-L229).

## HallucinationMetrics output fields

### Primary aggregates

- **`groundedness_score`** — Length-weighted average confidence of grounded spans (0–1, higher is better).
- **`hallucination_rate`** — Always `1.0 - groundedness_score` (0–1, lower is better).

### Ratio fields (by character count)

- **`supported_ratio`** — Fraction of total answer characters in `supported` spans.
- **`partial_ratio`** — Fraction in `partial` spans.
- **`unsupported_ratio`** — Fraction in `unsupported` spans.

These three ratios always sum to 1.0 when `total_chars > 0`.

### Confidence statistics

- **`avg_confidence`** — Arithmetic mean of per-span `confidence` values.
- **`min_confidence`** — Minimum per-span `confidence` across all spans.

### Count fields

- **`num_spans`** — Total spans analyzed.
- **`num_supported`** — Spans with `supported` status.
- **`num_partial`** — Spans with `partial` status.
- **`num_unsupported`** — Spans with `unsupported` status.
- **`num_weak_citations`** — Spans whose best citation has `answer_coverage < weak_citation_threshold` (0.4 by default), regardless of status.

### Detail lists

- **`span_confidences`** — `list[SpanConfidence]`, one entry per input `SpanCitations`.
- **`unsupported_spans`** — Answer spans with `unsupported` status.
- **`weakly_supported_spans`** — Answer spans below `weak_citation_threshold`.

## Empty input behavior

When `span_citations` is empty, `_empty_hallucination_metrics()` (repo://src/cite_right/hallucination.py#L125-L143) returns perfect scores:

- `groundedness_score = 1.0`
- `hallucination_rate = 0.0`
- All ratios and counts zero or default.

This prevents division-by-zero and treats empty inputs as trivially grounded.

## Relationship to citation status

`HallucinationConfig` does **not** control status assignment. Status (`supported`, `partial`, `unsupported`) is determined upstream by `align_citations()` via `_span_status()` in `src/cite_right/citations.py` (lines 1610–1633). That function uses the best citation's `answer_coverage` against `CitationConfig.supported_answer_coverage` (default 0.6) to decide between `supported` and `partial`, and assigns `unsupported` when no citations exist.

See [Citation status semantics](/openwiki/concepts/status-semantics.md) for the full decision tree.

## Integration points

`compute_hallucination_metrics` is called by the convenience functions in `src/cite_right/convenience.py`:

| Function | Use case |
|----------|----------|
| `is_grounded()` | Boolean gate: `groundedness_score >= threshold` |
| `is_hallucinated()` | Boolean gate: `hallucination_rate > threshold` |
| `check_groundedness()` | Full `HallucinationMetrics` in one call |

These wrap `align_citations()` + `compute_hallucination_metrics()` for common RAG post-processing patterns.

## Example usage

```python
from cite_right import align_citations, compute_hallucination_metrics, HallucinationConfig

results = align_citations(answer, sources)
metrics = compute_hallucination_metrics(results)

print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Hallucination rate: {metrics.hallucination_rate:.1%}")
print(f"Unsupported spans: {[s.text for s in metrics.unsupported_spans]}")

# Strict mode: exclude partial spans from groundedness
strict_metrics = compute_hallucination_metrics(
    results,
    config=HallucinationConfig(include_partial_in_grounded=False)
)
```

## Key invariants

1. `groundedness_score + hallucination_rate == 1.0` always holds.
2. `supported_ratio + partial_ratio + unsupported_ratio == 1.0` when `num_spans > 0`.
3. `confidence` is always `0.0` for `unsupported` spans (no citations → no coverage).
4. A `partial` span with `answer_coverage = 0.0` (semantic-only citation) contributes `0.0` to `groundedness_score` even when `include_partial_in_grounded=True`.
5. `weakly_supported_spans` includes spans regardless of their status; a `supported` span can be weak if its `answer_coverage` is between 0.4 and the `supported_answer_coverage` threshold (0.6).
