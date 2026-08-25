---
type: concept
title: CitationConfig and weights
description: Configuration classes controlling citation alignment thresholds, scoring weights, candidate selection limits, and named presets for common use cases.
tags: [citation-alignment, configuration, cite-right, scoring]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

`CitationConfig` and `CitationWeights` are frozen Pydantic models defined in `src/cite_right/core/citation_config.py`. Both are re-exported from `cite_right.__init__` and are the primary configuration surface for `align_citations()`. `CitationConfig` is passed directly to the alignment pipeline; `CitationWeights` is a nested field that controls how individual score components combine into the final citation score.

## CitationWeights defaults

`CitationWeights` carries five weight fields that scale individual citation score components:

```python
class CitationWeights(BaseModel):
    alignment: float = 1.0
    answer_coverage: float = 1.0
    evidence_coverage: float = 0.0
    lexical: float = 0.5
    embedding: float = 0.5
```

| Field | Default | Purpose |
|---|---|---|
| `alignment` | `1.0` | Scales the normalized Smith-Waterman alignment score |
| `answer_coverage` | `1.0` | Scales the fraction of answer tokens matched in evidence |
| `evidence_coverage` | `0.0` | Scales the fraction of evidence tokens covered by the answer (currently unused) |
| `lexical` | `0.5` | Scales IDF-weighted lexical overlap between answer and source |
| `embedding` | `0.5` | Scales embedding similarity when an embedder is provided |

The final citation score is a weighted sum of these components. Weights are summed directly — they are **not** normalized — so absolute values affect the scale of the result.

## CitationConfig threshold fields

`CitationConfig` groups its fields into several logical areas:

### Output filtering

| Field | Default | Controls |
|---|---|---|
| `top_k` | `3` | Maximum citations returned per answer span |
| `min_final_score` | `0.0` | Minimum weighted citation score for inclusion |
| `min_alignment_score` | `0` | Minimum raw Smith-Waterman alignment score |
| `min_answer_coverage` | `0.2` | Minimum fraction of answer tokens matched |
| `supported_answer_coverage` | `0.6` | Coverage threshold above which a span is `supported` |
| `min_embedding_similarity` | `0.3` | Minimum embedding similarity for retrieval support |
| `max_citations_per_source` | `2` | Maximum citations from a single source per span |
| `max_retrieval_support` | `3` | Maximum retrieval-only support entries when no exact citation exists |
| `require_all_answer_tokens_in_evidence` | `False` | Whether every answer token must appear in evidence |

`supported_answer_coverage` drives the span status decision: `answer_coverage >= supported_answer_coverage` → `supported`; below → `partial`; no citations → `unsupported`. This threshold is the primary lever for precision tuning.

### Passage windowing

| Field | Default | Controls |
|---|---|---|
| `window_size_sentences` | `1` | Sentences per source passage window |
| `window_stride_sentences` | `1` | Step between consecutive windows |

Stride of `1` produces overlapping windows and maximizes recall. Larger strides reduce passage count and improve throughput at the cost of potentially missing matches.

### Candidate selection limits

| Field | Default | Controls |
|---|---|---|
| `max_candidates_lexical` | `200` | Maximum lexical (inverted-index) candidates per span |
| `max_candidates_embedding` | `200` | Maximum embedding candidates per span |
| `max_candidates_total` | `400` | Combined cap before Smith-Waterman alignment |

The pipeline selects candidates using the stronger of lexical or embedding score, then caps at `max_candidates_total` before alignment runs.

### Alignment scoring

| Field | Default | Controls |
|---|---|---|
| `match_score` | `2` | Score for matching tokens in Smith-Waterman |
| `mismatch_score` | `-1` | Penalty for non-matching tokens |
| `gap_score` | `-1` | Penalty per gap (insertion/deletion) |

Higher gap penalties produce more compact evidence spans. Lower penalties allow bridging gaps between matching regions.

### Tie-breaking

| Field | Default | Controls |
|---|---|---|
| `prefer_source_order` | `True` | Tie-breaker preference: sources first, then positions, then length |

When `False`, ties prefer earlier character positions first regardless of source order.

### Multi-span evidence

| Field | Default | Controls |
|---|---|---|
| `multi_span_evidence` | `False` | Enable non-contiguous evidence spans |
| `multi_span_merge_gap_chars` | `16` | Merge neighboring spans when gap ≤ N characters |
| `multi_span_max_spans` | `5` | Maximum spans per citation; excess falls back to single enclosing span |

When `multi_span_evidence` is `True` and alignment produces disjoint match blocks, the pipeline extracts multiple `EvidenceSpan` objects. Adjacent spans are merged if the gap between them is below `multi_span_merge_gap_chars`. If the merged count exceeds `multi_span_max_spans`, the citation falls back to a single enclosing span for backward compatibility.

## Named presets

`CitationConfig` provides four class methods that return frozen instances:

### `CitationConfig.strict()`

High-precision mode for fact-checking and high-stakes applications.

- `top_k=2`, `min_answer_coverage=0.4`, `supported_answer_coverage=0.7`
- `min_final_score=0.3`, `max_citations_per_source=1`, `max_retrieval_support=2`
- `require_all_answer_tokens_in_evidence=True`

### `CitationConfig.permissive()`

Lenient mode for paraphrased or summarized content.

- `top_k=5`, `min_answer_coverage=0.15`, `supported_answer_coverage=0.4`
- `min_embedding_similarity=0.25`, `max_citations_per_source=3`, `max_retrieval_support=5`

### `CitationConfig.fast()`

Speed-optimized configuration with reduced candidate evaluation.

- `top_k=1`, `max_candidates_lexical=50`, `max_candidates_embedding=50`
- `max_candidates_total=100`, `max_citations_per_source=1`, `max_retrieval_support=1`

### `CitationConfig.balanced()`

Default balanced configuration; functionally identical to the default constructor.

## Removed embedding-only options

`CitationConfig` includes a model validator `_reject_removed_embedding_only_options` that raises `ValueError` if initialization receives either of the deprecated keys `allow_embedding_only` or `supported_embedding_similarity`:

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

Passing either key produces a migration error directing callers to inspect `SpanCitations.retrieval_support` for semantic retrieval signals. This reflects the design principle that embedding-only results belong in `retrieval_support`, not in citation status.

## Usage example

```python
from cite_right import CitationConfig, CitationWeights, align_citations

# Use a preset
config = CitationConfig.strict()

# Or customize weights
weights = CitationWeights(
    alignment=1.0,
    answer_coverage=1.0,
    evidence_coverage=0.0,
    lexical=0.5,
    embedding=0.5,
)
config = CitationConfig(
    top_k=3,
    min_final_score=0.25,
    weights=weights,
)

results = align_citations(answer, sources, config=config)
```

## See also

- [Citation status semantics](/openwiki/concepts/status-semantics.md) — how threshold fields affect `supported` / `partial` / `unsupported` status
- [align-citations workflow](/openwiki/workflows/align-citations.md) — pipeline integration
- [high-precision tuning workflow](/openwiki/workflows/high-precision-tuning.md) — tuning guidance for strict use cases
