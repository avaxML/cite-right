---
type: workflow
title: Groundedness checks workflow
description: Patterns for using span-level hallucination checks and claim-level fact verification in RAG post-processing pipelines.
tags: [cite-right, hallucination, fact-verification, groundedness, rag, quality-gate]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-2239349d0f5307d9d0756d4c
    resource: repo://src/cite_right/convenience.py
  - id: openwiki-source-9046040d0bcf7862617b852f
    resource: repo://src/cite_right/fact_verification.py
  - id: openwiki-source-5b90716cf19f71404fb5a027
    resource: repo://src/cite_right/hallucination.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

Groundedness checking in cite-right operates at two distinct granularity levels, each suited to different RAG post-processing scenarios. The **span-level** approach (`is_grounded`, `is_hallucinated`, `check_groundedness`) evaluates whole answer segments against sources. The **claim-level** approach (`verify_facts`) decomposes answers into atomic claims and verifies each independently.

## Span-level: Hallucination metrics

Span-level groundedness checks operate on answer segments (typically sentences) as atomic units. These functions live in `src/cite_right/convenience.py` and use `compute_hallucination_metrics()` internally.

### Boolean gate functions

`is_grounded()` and `is_hallucinated()` provide simple boolean quality gates for RAG pipelines:

```python
from cite_right import is_grounded, is_hallucinated

if is_grounded(answer, sources, threshold=0.5):
    print("Answer meets minimum groundedness")
```

```python
if is_hallucinated(answer, sources, threshold=0.5):
    print("Answer exceeds hallucination tolerance")
```

Both functions share the same default `threshold=0.5`, meaning at least 50% of the answer (by character count) must be grounded. They are complementary inverses: `is_grounded(t)` equals `not is_hallucinated(t)` for any threshold `t` in `(0, 1)`.

### Detailed metrics

`check_groundedness()` returns the full `HallucinationMetrics` object when you need per-span breakdowns:

```python
from cite_right import check_groundedness

metrics = check_groundedness(answer, sources)

print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Unsupported: {[s.text for s in metrics.unsupported_spans]}")
```

### Output structure

`HallucinationMetrics` provides:

| Field | Description |
|-------|-------------|
| `groundedness_score` | Length-weighted average of per-span confidence (0–1, higher is better) |
| `hallucination_rate` | Always `1.0 - groundedness_score` (lower is better) |
| `supported_ratio` | Fraction of answer characters in "supported" spans |
| `partial_ratio` | Fraction in "partial" spans |
| `unsupported_ratio` | Fraction in "unsupported" spans |
| `num_spans` | Total answer segments analyzed |
| `unsupported_spans` | `AnswerSpan` objects with no citation support |

### Span status classification

The status of each span is determined upstream by `align_citations()` based on the best citation's `answer_coverage` component:

| Status | Condition | Included in groundedness? |
|--------|-----------|--------------------------|
| `supported` | `answer_coverage >= 0.6` | Always |
| `partial` | `0.0 < answer_coverage < 0.6` | When `include_partial_in_grounded=True` (default) |
| `unsupported` | No citations or `answer_coverage = 0` | Never |

### Confidence extraction

For each span, confidence is extracted from the best citation's `answer_coverage` component:

```python
citation.components.get("answer_coverage", 0.0)
```

This means a span supported only by semantic/embedding similarity (no lexical match) contributes `0.0` to the groundedness score, even if the citation score itself is high.

### Configuration options

`HallucinationConfig` controls metric behavior:

```python
from cite_right import HallucinationConfig, check_groundedness

# Strict mode: only "supported" spans count as grounded
strict_metrics = check_groundedness(
    answer, sources,
    hallucination_config=HallucinationConfig(include_partial_in_grounded=False)
)
```

| Attribute | Default | Effect |
|-----------|--------|--------|
| `weak_citation_threshold` | `0.4` | Spans below this threshold appear in `weakly_supported_spans` |
| `include_partial_in_grounded` | `True` | Whether "partial" spans contribute to groundedness |

## Claim-level: Fact verification

Fact verification operates on atomic claims decomposed from answer sentences, enabling fine-grained attribution at the conjunction level. The primary entrypoint is `verify_facts()` in `src/cite_right/fact_verification.py`.

### Basic usage

```python
from cite_right import verify_facts, FactVerificationConfig

metrics = verify_facts(answer, sources)
print(f"Verification rate: {metrics.verification_rate:.1%}")
print(f"Verified: {[c.text for c in metrics.verified_claims]}")
```

### Output structure

`FactVerificationMetrics` provides claim-level aggregates:

| Field | Description |
|-------|-------------|
| `num_claims` | Total atomic claims analyzed |
| `num_verified` | Claims with strong source support |
| `num_partial` | Claims with partial support |
| `num_unverified` | Claims with no support |
| `verification_rate` | Proportion of verified claims (0–1) |
| `avg_confidence` | Average confidence across all claims |
| `claim_verifications` | Per-claim `ClaimVerification` objects |
| `verified_claims` / `partial_claims` / `unverified_claims` | Categorized claim lists |

### Claim status classification

Unlike span-level, claim-level verification uses explicit coverage thresholds:

```python
class FactVerificationConfig:
    verified_coverage_threshold: float = 0.6  # >= 60% coverage = verified
    partial_coverage_threshold: float = 0.3   # >= 30% coverage = partial
```

| Status | Condition |
|--------|-----------|
| `verified` | `answer_coverage >= 0.6` |
| `partial` | `0.3 <= answer_coverage < 0.6` |
| `unverified` | `answer_coverage < 0.3` or no citations |

### Claim decomposition

By default, `verify_facts()` uses `SimpleClaimDecomposer`, which treats each answer span as a single claim. For conjunction-aware splitting, use `SpacyClaimDecomposer`:

```python
from cite_right import verify_facts, SpacyClaimDecomposer

metrics = verify_facts(
    answer, sources,
    claim_decomposer=SpacyClaimDecomposer()
)
```

This decomposes compound sentences at conjunction boundaries:

```
Input:  "Revenue grew and profits increased."
Output: ["Revenue grew", "profits increased"]
```

Each claim is then verified independently against sources, enabling granular attribution for complex sentences.

## Threshold comparison

The two approaches use different default thresholds:

| Approach | Function | Default Threshold | Meaning |
|----------|----------|-------------------|---------|
| Span-level | `is_grounded` | `0.5` | 50% of answer (by chars) must be grounded |
| Span-level | `is_hallucinated` | `0.5` | Flag when >50% of answer is unsupported |
| Claim-level | `verify_facts` | `0.6` for verified, `0.3` for partial | 60% coverage for verified, 30% for partial |

The claim-level thresholds are more granular:
- **Verified** at `>= 0.6` requires stronger evidence than span-level's default `0.5`
- **Partial** captures claims with moderate support (`>= 0.3`) that fall between verified and unverified

## Choosing between approaches

### Use span-level when:

- You need a simple pass/fail quality gate
- Answer segments are self-contained units
- Performance is critical (single alignment pass)
- You want per-span detail lists for error inspection

```python
if is_grounded(answer, sources, threshold=0.6):
    return answer
else:
    return regenerate_or_flag()
```

### Use claim-level when:

- Compound sentences need granular attribution
- You need to identify exactly which facts are unverified
- Downstream systems need per-claim confidence scores
- Regulatory or audit requirements demand claim-level evidence

```python
metrics = verify_facts(answer, sources)
if metrics.verification_rate < 0.8:
    for claim in metrics.unverified_claims:
        log_warning(f"Unverified claim: {claim.text}")
```

### Combining both

For comprehensive quality assurance, use both approaches:

```python
from cite_right import is_grounded, verify_facts

# Fast gate check first
if not is_grounded(answer, sources, threshold=0.5):
    return "LOW_GROUNDEDNESS"

# Detailed claim analysis for acceptable answers
metrics = verify_facts(answer, sources)
if metrics.verification_rate < 0.7:
    return "UNVERIFIED_CLAIMS"

return answer
```

## Example workflow

```python
from cite_right import (
    check_groundedness,
    verify_facts,
    SourceDocument,
    FactVerificationConfig,
)

# Prepare sources
sources = [
    SourceDocument(id="report", text="Annual report: Revenue grew 15% in Q4 2024."),
]

# Candidate answer
answer = "Revenue grew 15%. The company announced plans to colonize Mars."

# Span-level: Quick groundedness check
metrics = check_groundedness(answer, sources)
print(f"Groundedness: {metrics.groundedness_score:.1%}")
# Output: Groundedness: 25.0%

# Claim-level: Detailed fact verification
fv_metrics = verify_facts(answer, sources)
print(f"Verified: {fv_metrics.num_verified}/{fv_metrics.num_claims}")
# Output: Verified: 1/2

for cv in fv_metrics.claim_verifications:
    print(f"  [{cv.status}] {cv.claim.text}")
# Output:
#   [verified] Revenue grew 15%.
#   [unverified] The company announced plans to colonize Mars.
```

## Integration with RAG pipelines

### LangChain integration

```python
from cite_right import is_grounded, from_langchain_documents

# After generating answer from RAG chain
docs = from_langchain_documents(retriever.invoke(query))
if not is_grounded(answer, docs, threshold=0.6):
    raise ValueError("Generated answer insufficiently grounded")
```

### LlamaIndex integration

```python
from cite_right import check_groundedness, from_llamaindex_nodes

nodes = from_llamaindex_nodes(query_engine.retrieve(query))
metrics = check_groundedness(answer, nodes)
if metrics.hallucination_rate > 0.3:
    # Trigger regeneration or human review
    pass
```

## Key invariants

1. `groundedness_score + hallucination_rate == 1.0` always holds
2. `supported_ratio + partial_ratio + unsupported_ratio == 1.0` when spans exist
3. For span-level: a semantic-only citation contributes `0.0` confidence
4. For claim-level: a claim with no citations is always `unverified`
5. `verification_rate = num_verified / num_claims` (0.0–1.0, higher is better)
