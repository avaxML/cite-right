---
type: operation
title: Fact-level verification
description: The verify_facts function decomposes RAG answers into atomic claims and verifies each claim independently against source documents using citation alignment.
tags: [fact-verification, cite-right, claim-decomposition, citation-alignment]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-c2258de3abef6fc5c0b37b70
    resource: repo://src/cite_right/claims.py
  - id: openwiki-source-9046040d0bcf7862617b852f
    resource: repo://src/cite_right/fact_verification.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

Fact-level verification in cite-right operates on a pipeline that first decomposes a generated answer into atomic claims, then verifies each claim independently against the source documents. The primary entrypoint is `verify_facts()` in `src/cite_right/fact_verification.py`, which returns an aggregate `FactVerificationMetrics` object containing per-claim verification results.

Unlike sentence-level citation alignment (which treats each sentence as a unit), fact-level verification enables fine-grained attribution by splitting compound sentences at conjunction boundaries. A single sentence like "Revenue grew and profits increased" is split into two separate claims that can each be verified independently.

## Core API

### verify_facts

```python
def verify_facts(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: FactVerificationConfig | None = None,
    claim_decomposer: ClaimDecomposer | None = None,
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> FactVerificationMetrics
```

**Parameters:**

- `answer`: The generated text to verify.
- `sources`: Source documents or chunks to verify against. Accepts raw strings, `SourceDocument`, or `SourceChunk` objects.
- `config`: Optional `FactVerificationConfig` controlling thresholds and citation behavior.
- `claim_decomposer`: Strategy for splitting answer text into claims. Defaults to `SimpleClaimDecomposer` (no splitting).
- `answer_segmenter`: Strategy for splitting the answer into initial spans before decomposition. Defaults to `SimpleAnswerSegmenter`.
- `source_segmenter`, `tokenizer`, `embedder`, `backend`: Passed through to the underlying `align_citations` call per claim.

**Returns:** `FactVerificationMetrics` containing aggregate statistics and per-claim `ClaimVerification` objects.

### FactVerificationConfig

```python
class FactVerificationConfig(BaseModel):
    verified_coverage_threshold: float = 0.6
    partial_coverage_threshold: float = 0.3
    citation_config: CitationConfig | None = None
```

**Attributes:**

- `verified_coverage_threshold` (default `0.6`): Minimum `answer_coverage` score for a claim to be marked **verified**. When `answer_coverage >= 0.6`, the claim status is `"verified"`.
- `partial_coverage_threshold` (default `0.3`): Minimum `answer_coverage` for a claim to be marked **partial**. When `0.3 <= answer_coverage < 0.6`, the status is `"partial"`. Below `0.3`, the status is `"unverified"`.
- `citation_config`: Optional `CitationConfig` passed to `align_citations` per claim. If `None`, a default config is derived with `top_k=3`, `min_answer_coverage=0.2`, and `supported_answer_coverage` set to the same value as `verified_coverage_threshold`.

### FactVerificationMetrics

```python
class FactVerificationMetrics(BaseModel):
    num_claims: int
    num_verified: int
    num_partial: int
    num_unverified: int
    verification_rate: float  # proportion of verified claims
    avg_confidence: float
    min_confidence: float
    claim_verifications: list[ClaimVerification]
    verified_claims: list[Claim]
    partial_claims: list[Claim]
    unverified_claims: list[Claim]
```

### ClaimVerification

```python
class ClaimVerification(BaseModel):
    claim: Claim
    status: Literal["verified", "partial", "unverified"]
    confidence: float
    best_citation: Citation | None
    all_citations: list[Citation]
    source_ids: list[str]
```

## Claim Decomposition Pipeline

### SimpleClaimDecomposer

`SimpleClaimDecomposer` is a fallback decomposer used when spaCy is unavailable. It treats each answer span as a single atomic claim without further splitting.

```python
class SimpleClaimDecomposer:
    def decompose(self, span: AnswerSpan) -> list[Claim]:
        return [
            Claim(
                text=span.text,
                char_start=span.char_start,
                char_end=span.char_end,
                source_span=span,
                claim_index=0,
            )
        ]
```

### SpacyClaimDecomposer

`SpacyClaimDecomposer` uses spaCy's dependency parsing to identify coordinated clauses joined by conjunctions. Each conjunction boundary becomes a split point, enabling independent verification of each clause.

```python
class SpacyClaimDecomposer:
    def __init__(
        self,
        model: str = "en_core_web_sm",
        *,
        min_claim_tokens: int = 2,
    ) -> None:
```

**Initialization parameters:**

- `model`: spaCy model name (default `"en_core_web_sm"`).
- `min_claim_tokens`: Minimum token count for a valid claim. Claims below this threshold are merged back into neighboring claims.

### Conjunction Boundary Detection

The `_find_claim_boundaries` method in `SpacyClaimDecomposer` identifies split points by scanning for `conj` (conjunction) dependency relations:

```python
def _find_claim_boundaries(self, doc: Any) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []

    for token in doc:
        if token.dep_ != "conj":
            continue

        boundary = self._get_boundary_for_conj(token, doc, boundaries)
        if boundary is not None:
            boundaries.append(boundary)

    return sorted(set(boundaries))
```

**How conjunction splitting works:**

1. **Token iteration**: For each token in the parsed document, check if its dependency label is `"conj"` (conjunct).
2. **Coordinating conjunction lookup**: For each `conj` token, `_get_boundary_for_conj` finds the associated coordinating conjunction (`cc`) by looking at the token's head's children.
3. **Boundary extraction**: Two strategies determine the split point:
   - `_boundary_from_cc`: When a `cc` token is found (e.g., "and", "or"), the split occurs after the coordinating conjunction. For "Revenue grew **and** profits increased", the boundary is positioned after "and ".
   - `_boundary_from_separator`: When no `cc` is found but a comma or semicolon precedes the conjoined token, the boundary is placed after the separator.
4. **Claim extraction**: `_extract_claims` uses the computed boundaries to slice the answer text into separate claim strings, each wrapped in a `Claim` object with accurate character offsets.

**Example decomposition:**

```
Input: "Revenue grew and profits increased."
Output:
  - Claim 1: "Revenue grew"
  - Claim 2: "profits increased"
```

## Per-Claim Verification Flow

`verify_facts` verifies claims sequentially through `_verify_all_claims`:

```python
def _verify_all_claims(
    claims: list[Claim],
    sources: Sequence[str | SourceDocument | SourceChunk],
    cfg: FactVerificationConfig,
    citation_config: CitationConfig,
    ...
) -> FactVerificationMetrics:
```

For each claim, `_verify_claim` invokes `align_citations()` with the claim text as the answer:

```python
def _verify_claim(
    claim: Claim,
    sources,
    config: FactVerificationConfig,
    citation_config: CitationConfig,
    ...
) -> ClaimVerification:
    results = align_citations(
        answer=claim.text,
        sources=sources,
        config=citation_config,
        ...
    )

    all_citations: list[Citation] = []
    for span_result in results:
        all_citations.extend(span_result.citations)

    if not all_citations:
        return ClaimVerification(claim=claim, status="unverified", confidence=0.0, ...)

    best_citation = max(all_citations, key=lambda c: c.score)
    answer_coverage = float(best_citation.components.get("answer_coverage", 0.0))

    if answer_coverage >= config.verified_coverage_threshold:
        status = "verified"
    elif answer_coverage >= config.partial_coverage_threshold:
        status = "partial"
    else:
        status = "unverified"

    return ClaimVerification(
        claim=claim,
        status=status,
        confidence=answer_coverage,
        best_citation=best_citation,
        all_citations=all_citations,
        source_ids=list({c.source_id for c in all_citations}),
    )
```

**Key behaviors:**

- `align_citations` is called once per claim, not once per answer.
- The `citation_config` passed to `align_citations` is derived from `FactVerificationConfig.citation_config`, or defaults to `CitationConfig(top_k=3, min_answer_coverage=0.2, supported_answer_coverage=cfg.verified_coverage_threshold)`.
- The confidence score is the `answer_coverage` component from the best citation.
- `source_ids` deduplicates the supporting sources across all citations for the claim.

## Status Determination

The verification status is determined solely by `answer_coverage` from the best citation:

| `answer_coverage` range | Status |
|-------------------------|--------|
| `>= 0.6`                | `verified` |
| `>= 0.3` and `< 0.6`   | `partial` |
| `< 0.3`                 | `unverified` |

This is a different threshold system than sentence-level citation alignment, which also considers contradiction detection. Fact-level verification focuses on coverage alone.

## Aggregate Metrics Computation

After verifying all claims, `_verify_all_claims` aggregates results:

```python
verification_rate = len(verified) / len(claims) if claims else 1.0
avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 1.0
min_confidence = min(confidence_values) if confidence_values else 1.0
```

Empty input (no claims) returns a full set of optimistic defaults: `verification_rate=1.0`, `avg_confidence=1.0`, `min_confidence=1.0`.

## Usage Example

```python
from cite_right import verify_facts, FactVerificationConfig, SpacyClaimDecomposer

config = FactVerificationConfig(
    verified_coverage_threshold=0.7,
    partial_coverage_threshold=0.4,
)

decomposer = SpacyClaimDecomposer()

metrics = verify_facts(
    answer="The company reported revenue of $5.2B and profits grew 15% year-over-year.",
    sources=["Annual Report 2023: Revenue was $5.2 billion. Net profit increased by 15%."],
    config=config,
    claim_decomposer=decomposer,
)

print(f"Verification rate: {metrics.verification_rate:.1%}")
print(f"Verified claims: {[c.text for c in metrics.verified_claims]}")
print(f"Unverified claims: {[c.text for c in metrics.unverified_claims]}")
```

## Relationship to Citation Alignment

Fact-level verification builds on the citation alignment pipeline (`align_citations`). Each claim is verified by calling `align_citations(answer=claim.text, sources=sources, config=citation_config)`, which performs:

1. Source text segmentation
2. Candidate generation via lexical and embedding similarity
3. Smith-Waterman sequence alignment between claim tokens and source candidates
4. Citation ranking by weighted score components

The `answer_coverage` component from the resulting `Citation.components` dictionary is the primary signal for claim verification status.

## Default Dependencies

When optional dependencies are absent:

- **spaCy**: If `SpacyClaimDecomposer` is requested but spaCy is not installed, a `RuntimeError` is raised with installation instructions.
- **embedder**: If not provided, `align_citations` falls back to lexical-only matching.
- **tokenizer**: If not provided, uses the default tokenizer configured in `align_citations`.

## Invariants and Failure Modes

- `FactVerificationConfig` is a frozen Pydantic model; configuration cannot be modified after creation.
- `Claim` objects are frozen; claims are immutable once created.
- If a claim has no citations (`all_citations` is empty), the claim is always marked `unverified` with `confidence=0.0`.
- Character offsets in `Claim` objects are absolute offsets in the original answer string and are preserved from the source `AnswerSpan`.
- The `SpacyClaimDecomposer` requires spaCy models to be pre-downloaded. The default model is `en_core_web_sm`.

## Related Concepts

- **[Citation status semantics](/openwiki/concepts/status-semantics.md)**: Documents the three-tier status system used in sentence-level citation alignment. Fact-level verification uses a similar threshold model but applies it per-claim rather than per-sentence.
