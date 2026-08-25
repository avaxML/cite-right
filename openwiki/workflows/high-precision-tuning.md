---
type: workflow
title: High-precision tuning
description: How to bias the citation alignment pipeline toward fewer false positives — the benchmarked high-precision configuration, the role of each filtering knob, and how to adapt the recipe for domain-specific use cases.
tags: [cite-right, citation-alignment, configuration, precision, false-positives]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-ad97420c2a0900f616ed0fef
    resource: repo://src/cite_right/contradiction.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
  - id: openwiki-source-c126bef8ff7e71bc028699de
    resource: repo://tests/test_citations_retrieval_support.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The citation alignment pipeline has multiple independent thresholds that act as sequential filters on candidate evidence. High-precision tuning means raising those thresholds so that only the strongest, most verbatim matches produce citations. The goal is to eliminate false positives — spurious citations where the answer appears to be grounded but is actually hallucinated or contradicts the source.

This page documents the benchmarked high-precision recipe from the README, explains the responsibility of each knob, and describes how to adapt the configuration for specific adversarial scenarios.

## The benchmarked high-precision recipe

The following configuration was derived via multi-dimensional grid search over a rich adversarial RAG dataset. It successfully eliminates false positives on negations, numerical updates, and entity swaps while preserving robust recall on genuinely aligned citations:

```python
from cite_right import CitationConfig, CitationWeights

# Weights balance alignment and semantic similarity without over-weighting either
high_precision_weights = CitationWeights(
    alignment=1.0,
    answer_coverage=1.0,
    evidence_coverage=0.0,
    lexical=0.5,
    embedding=0.5,
)

# High-precision configuration
high_precision_config = CitationConfig(
    top_k=1,
    min_alignment_score=0,
    min_answer_coverage=0.4,
    supported_answer_coverage=0.6,
    min_embedding_similarity=0.3,
    min_final_score=2.6,  # Key threshold for filtering adversarial and near-miss false positives
    weights=high_precision_weights,
)
```

> **Caveat**: This configuration was derived on adversarial inputs in a specific domain (RAG fact-checking). For other domains — summaries, creative rewriting, highly paraphrased answers — the same thresholds may over-reject valid citations. The knobs section below explains how to dial each threshold independently.

## How `min_final_score` filters false positives

`min_final_score` is the most impactful knob for precision because it operates on the **weighted composite score** after all individual signals have been computed. A citation must pass:

1. Lexical prefilter (IDF-weighted token overlap)
2. Candidate ranking and selection limits
3. Smith-Waterman alignment quality (`_should_use_alignment()`)
4. The composite score gate (`_build_exact_citation()`)

The composite score formula in `_compute_final_score()` (repo://src/cite_right/citations.py#L808-L821) is:

```python
def _compute_final_score(
    metrics: dict[str, float],
    lexical_score: float,
    embed_score: float,
    cfg: CitationConfig,
) -> float:
    return (
        cfg.weights.alignment * metrics["normalized_alignment"]
        + cfg.weights.answer_coverage * metrics["answer_coverage"]
        + cfg.weights.evidence_coverage * metrics["evidence_coverage"]
        + cfg.weights.lexical * lexical_score
        + cfg.weights.embedding * max(0.0, embed_score)
    )
```

With the high-precision weights (`alignment=1.0`, `answer_coverage=1.0`, `lexical=0.5`, `embedding=0.5`), the maximum achievable score is:

```
1.0 * 1.0 + 1.0 * 1.0 + 0.0 * 1.0 + 0.5 * 1.0 + 0.5 * 1.0 = 3.0
```

Setting `min_final_score=2.6` requires the citation to achieve ~87% of the theoretical maximum. This eliminates:

- **Near-miss paraphrases**: High lexical but low alignment score
- **Embedding-only matches**: High semantic similarity but poor token overlap
- **Partial numeric matches**: `answer_coverage` below the required threshold

## The role of each knob

### Coverage thresholds

| Knob | Default | High-precision | Effect |
|---|---|---|---|
| `min_answer_coverage` | `0.2` | `0.4` | Fraction of answer tokens that must appear in aligned evidence. Raising from 0.2 to 0.4 rejects citations with significant paraphrasing or missing tokens. |
| `supported_answer_coverage` | `0.6` | `0.6` | Coverage threshold for `supported` status. Below this → `partial`. Keeping it at 0.6 means a citation must cover at least 60% of answer tokens to be considered "supported". |
| `require_all_answer_tokens_in_evidence` | `False` | Not set (False) | When `True`, **every** answer token must appear in the evidence (checked via `_answer_tokens_match_evidence()`). This is an even stricter filter than `min_answer_coverage`. Use it for strict negation and number-mismatch guarding. |

The relationship between coverage and status:

```python
# From _span_status() (repo://src/cite_right/citations.py#L1610-L1633)
if coverage >= cfg.supported_answer_coverage:
    return "supported"
return "partial"
```

A citation with `answer_coverage=0.45` and `supported_answer_coverage=0.6` gets `partial` status — the citation is retained but flagged. This is intentional: it surfaces the evidence for human review rather than silently discarding it.

### Score gates

| Knob | Default | High-precision | Effect |
|---|---|---|---|
| `min_final_score` | `0.0` | `2.6` | Minimum composite weighted score. The README recipe sets this to filter near-miss and adversarial false positives. Raising it eliminates weaker candidates at the cost of potentially dropping valid paraphrased citations. |
| `min_alignment_score` | `0` | `0` | Minimum raw Smith-Waterman score. Kept at 0 because `min_answer_coverage` and `min_final_score` subsume this check. |
| `min_embedding_similarity` | `0.3` | `0.3` | Minimum embedding similarity for retrieval support entries (not for citations themselves). Kept at 0.3 to allow semantic expansion of candidates without letting pure embedding matches become citations. |

### Output limits

| Knob | Default | High-precision | Effect |
|---|---|---|---|
| `top_k` | `3` | `1` | Maximum citations returned per answer span. Reducing to 1 focuses on the single best citation and reduces noise in downstream processing. |
| `max_citations_per_source` | `2` | `2` (default) | Maximum citations from a single source. The README recipe does not override this; the `CitationConfig.strict()` preset uses `1`. |
| `max_retrieval_support` | `3` | `3` (default) | Maximum retrieval-only support entries. High-precision keeps this at the default; reducing it further (e.g., to `1`) suppresses semantic-only signals. |

### Candidate selection limits

| Knob | Default | High-precision | Effect |
|---|---|---|---|
| `max_candidates_lexical` | `200` | `200` (default) | Maximum lexical candidates per span. The recipe uses defaults, relying on `min_final_score` for filtering rather than pre-alignment pruning. |
| `max_candidates_embedding` | `200` | `200` (default) | Maximum embedding candidates. |
| `max_candidates_total` | `400` | `400` (default) | Combined cap before alignment. |

## Multi-span evidence settings

When an answer's evidence is distributed across multiple non-contiguous regions in the source, the pipeline can return multiple `EvidenceSpan` objects per citation. The multi-span settings control this behavior:

| Knob | Default | Effect |
|---|---|---|
| `multi_span_evidence` | `False` | Enable non-contiguous evidence extraction. When `True` and alignment produces disjoint match blocks, the pipeline extracts separate `EvidenceSpan` objects (repo://src/cite_right/citations.py#L1384-L1402). |
| `multi_span_merge_gap_chars` | `16` | Merge neighboring spans when the gap between them is ≤ N characters in the source. Lower values (e.g., `5`) produce more granular spans; higher values (e.g., `50`) merge loosely related segments. |
| `multi_span_max_spans` | `5` | Maximum spans per citation. If merging produces more than this many spans, the citation falls back to a single enclosing span for backward compatibility. |

For high-precision use cases, keeping `multi_span_evidence=False` (the default) is recommended because:

1. **Simpler output**: Single contiguous spans are easier to render and verify
2. **Reduced surface for errors**: Multi-span extraction multiplies the opportunities for offset miscalculation
3. **Paraphrase insensitivity**: Multi-span evidence typically indicates the answer synthesized information from multiple locations, which is more likely to be paraphrased than verbatim

## Contradiction detection as a precision layer

Even with high `min_final_score`, some false positives survive: a sentence that looks like a good alignment but contains a negated verb or a wrong number. Contradiction detection runs as a final check in `_span_status()` (repo://src/cite_right/citations.py#L1621-L1629) against the full candidate passage (not the truncated alignment evidence).

The five contradiction checks in `check_contradiction()` (repo://src/cite_right/contradiction.py#L67-L87):

1. **Negation mismatch**: Answer has negation but source doesn't, or vice versa (e.g., "shall make every" vs "shall make no")
2. **Number mismatch**: Different numeric values in similar positions (e.g., "125 days" vs "124 days")
3. **Entity swap**: Key entity tokens appear in different positions
4. **Temporal/polarity mismatch**: Conflicting temporal markers (e.g., "BC" vs "ago") or polarity (e.g., "support" vs "oppose")
5. **Number context mismatch**: Same number but appears to modify a different quantity (the "leftover n-gram" case)

When a contradiction is detected, the span receives `partial` status — the citation is retained for transparency rather than discarded, so downstream consumers can surface the contradicting passage.

## `CitationConfig.strict()` vs README high-precision recipe

`CitationConfig.strict()` (repo://src/cite_right/core/citation_config.py#L100-L126) and the README high-precision recipe serve similar goals but differ in key parameters:

| Parameter | `strict()` | README high-precision |
|---|---|---|
| `top_k` | `2` | `1` |
| `min_answer_coverage` | `0.4` | `0.4` |
| `supported_answer_coverage` | `0.7` | `0.6` |
| `min_final_score` | `0.3` | `2.6` |
| `max_citations_per_source` | `1` | `2` (default) |
| `max_retrieval_support` | `2` | `3` (default) |
| `require_all_answer_tokens_in_evidence` | `True` | `False` |

Key differences:

- **`strict()` sets `require_all_answer_tokens_in_evidence=True`**: This is a hard gate — every answer token must appear in evidence. It catches number mismatches and negation flips that `min_final_score` alone might miss.
- **README recipe sets `supported_answer_coverage=0.6`** instead of `strict()`'s `0.7`, meaning slightly more citations reach `supported` status.
- **README recipe's `min_final_score=2.6`** is the product of grid-search optimization. `strict()` relies on `require_all_answer_tokens_in_evidence` rather than a composite score gate.

For maximum precision on adversarial inputs, combine both approaches:

```python
from cite_right import CitationConfig, CitationWeights

# Combines strict()'s require_all_answer_tokens with README's grid-searched min_final_score
max_precision_config = CitationConfig(
    top_k=1,
    min_answer_coverage=0.4,
    supported_answer_coverage=0.6,
    min_final_score=2.6,
    require_all_answer_tokens_in_evidence=True,
    max_citations_per_source=1,
    max_retrieval_support=1,
    weights=CitationWeights(
        alignment=1.0,
        answer_coverage=1.0,
        evidence_coverage=0.0,
        lexical=0.5,
        embedding=0.5,
    ),
)
```

## Tuning guidance by adversarial scenario

### Negation flips

**Scenario**: Answer says "X is not Y" but source says "X is Y".

**Guard**: Set `require_all_answer_tokens_in_evidence=True`. This forces every answer token (including "not") to appear in the aligned evidence. A source without the negation token cannot satisfy this check.

From the test `test_strict_exact_citation_rejects_negation_token_mismatch()` (repo://tests/test_citations_retrieval_support.py#L169-L180), the Constitution example ("shall make **no** law" vs "shall make **every** law") is rejected when `require_all_answer_tokens_in_evidence=True`.

### Numeric updates

**Scenario**: Answer states "Revenue grew 15%" but source says "12%".

**Guards**: 
1. `require_all_answer_tokens_in_evidence=True` forces the exact number token to appear
2. `min_final_score=2.6` requires ~87% of the theoretical maximum score, which near-miss numbers typically fail
3. Contradiction detection's `has_number_mismatch()` catches same-sentence number conflicts

From `test_strict_exact_citation_rejects_numeric_token_mismatch()` (repo://tests/test_citations_retrieval_support.py#L131-L166), "125 days" vs "124 days" produces `unsupported` status with the high-precision config.

### Entity swaps

**Scenario**: Answer attributes a finding to the wrong study or person.

**Guards**:
1. `min_answer_coverage=0.4` ensures the entity name tokens are part of the aligned evidence
2. `supported_answer_coverage=0.6` requires the entity to be in the core supported region
3. `has_entity_swap()` in contradiction detection flags when key entity tokens appear in swapped positions

### Highly paraphrased content

**Trade-off**: The high-precision configuration is aggressive about rejecting paraphrases. For use cases that require tolerant paraphrase handling (e.g., summarization evaluation), consider:

```python
# Relaxed high-precision variant for paraphrased sources
permissive_high_precision_config = CitationConfig(
    top_k=2,
    min_answer_coverage=0.25,  # Lower from 0.4
    supported_answer_coverage=0.5,  # Lower from 0.6
    min_final_score=1.8,  # Lower from 2.6
    max_retrieval_support=3,
    weights=CitationWeights(
        alignment=0.8,  # Slightly lower
        answer_coverage=0.8,
        lexical=0.7,  # Higher - favor exact word overlap
        embedding=0.7,
    ),
)
```

The key insight: **lower `min_answer_coverage` for recall, raise `min_final_score` for precision**. These knobs are not redundant — coverage controls what fraction of the answer must be present, while `min_final_score` controls the overall composite quality gate.

## Verifying precision on your dataset

Use the `on_metrics` callback to instrument the pipeline and identify where false positives originate:

```python
from cite_right import align_citations, CitationConfig, CitationWeights

all_metrics = []

results = align_citations(
    answer,
    sources,
    config=high_precision_config,
    on_metrics=lambda m: all_metrics.append(m),
)

# Inspect per-span metrics
for span_result, metrics in zip(results, all_metrics):
    print(f"Span: {span_result.answer_span.text[:50]}...")
    print(f"  Status: {span_result.status}")
    print(f"  Citations: {len(span_result.citations)}")
    print(f"  Candidates evaluated: {metrics.num_candidates}")
    print(f"  Alignments run: {metrics.num_alignments}")
    for cit in span_result.citations:
        print(f"    Score={cit.score:.3f}, coverage={cit.components.get('answer_coverage', 0):.2f}")
```

A high ratio of `num_candidates` to `num_alignments` with low final scores indicates the candidate selection is too permissive. Raising `max_candidates_lexical` or adding `require_all_answer_tokens_in_evidence=True` helps.

## See also

- [CitationConfig and weights](/openwiki/operations/citation-config.md) — full reference for all configuration fields
- [Citation status semantics](/openwiki/concepts/status-semantics.md) — how `supported`, `partial`, and `unsupported` are determined
- [align_citations workflow](/openwiki/workflows/align-citations.md) — pipeline integration and return shape
