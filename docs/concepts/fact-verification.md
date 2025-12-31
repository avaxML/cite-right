# Fact Verification

While hallucination detection operates at the sentence level, fact verification provides finer-grained analysis by decomposing sentences into individual claims. This approach catches situations where a sentence combines accurate information with unsupported assertions.

## The Need for Claim-Level Analysis

Consider the sentence "Revenue grew 15% in Q4 and the company announced a stock split." This sentence contains two distinct claims. If the source only supports the revenue claim, sentence-level analysis might mark the entire sentence as partially supported without identifying which part is problematic.

Fact verification addresses this by splitting the sentence into atomic claims, verifying each independently, and reporting which specific claims lack source support.

## Using verify_facts

The `verify_facts` function performs claim-level verification. This function is defined in `src/cite_right/fact_verification.py`.

```python
from cite_right import SourceDocument, verify_facts

answer = "The product launched in March and sales exceeded 10 million units."
sources = [
    SourceDocument(
        id="press_release",
        text="The new product line was introduced to the market in March 2024."
    )
]

result = verify_facts(answer, sources)

print(f"Total claims: {result.total_claims}")
print(f"Verified: {result.num_verified}")
print(f"Partial: {result.num_partial}")
print(f"Unverified: {result.num_unverified}")
```

In this example, the claim about the March launch should verify against the source, while the sales figure claim will be flagged as unverified.

## Claim Decomposition

Before verification, sentences are decomposed into atomic claims. Cite-Right provides two decomposition approaches.

The `SimpleClaimDecomposer` returns each sentence as a single claim without further splitting. This provides a baseline that matches sentence-level behavior.

The `SpacyClaimDecomposer` uses spaCy's dependency parsing to identify coordinated clauses. Sentences connected by "and" or "but" are split into separate claims. This approach requires the spacy optional dependency.

```python
from cite_right import verify_facts
from cite_right.claims import SpacyClaimDecomposer

decomposer = SpacyClaimDecomposer()
result = verify_facts(answer, sources, claim_decomposer=decomposer)
```

Custom decomposers can be created by implementing the `ClaimDecomposer` protocol. This allows integration with more sophisticated claim extraction techniques such as those based on large language models.

## FactVerificationResult

The `verify_facts` function returns a `FactVerificationResult` object containing claim-level details.

### Aggregate Metrics

The `total_claims` field reports the number of claims identified in the answer.

The `num_verified` field counts claims with full source support.

The `num_partial` field counts claims with partial support.

The `num_unverified` field counts claims without adequate source support.

The `verification_rate` provides the proportion of claims that are verified or partially verified.

### Claim Details

The `claims` field contains a list of `ClaimVerification` objects with information about each claim.

```python
for claim in result.claims:
    print(f"Claim: {claim.text}")
    print(f"Status: {claim.status}")

    if claim.citations:
        best = claim.citations[0]
        print(f"Evidence: {best.evidence}")
```

Each `ClaimVerification` includes the claim text, its verification status, and any citations that support it.

## Verification Status

Claims receive one of three status values.

A status of "verified" indicates the claim has strong source support. The citation aligns well with the claim text.

A status of "partial" indicates some support exists but it may not fully cover the claim. This often occurs with paraphrased content or when only part of a compound claim is supported.

A status of "unverified" indicates no adequate source support was found. This claim may be hallucinated or derived from knowledge outside the provided sources.

## Integration with Hallucination Detection

Fact verification complements rather than replaces hallucination detection. The two approaches serve different purposes.

Use hallucination detection when you need a quick assessment of overall answer quality. The metrics provide aggregate scores suitable for quality gates and monitoring.

Use fact verification when you need to identify specific problematic claims for manual review or correction. The claim-level output enables precise feedback to users or downstream systems.

```python
from cite_right import (
    check_groundedness,
    verify_facts,
)
from cite_right.claims import SpacyClaimDecomposer

# Quick quality check
metrics = check_groundedness(answer, sources)

if metrics.hallucination_rate > 0.2:
    # Detailed analysis of problematic content
    result = verify_facts(
        answer,
        sources,
        claim_decomposer=SpacyClaimDecomposer()
    )

    for claim in result.claims:
        if claim.status == "unverified":
            print(f"Unsupported claim: {claim.text}")
```

## Configuration

The `verify_facts` function accepts the same configuration options as `align_citations`, including tokenizer, segmenter, embedder, and backend selection.

```python
from cite_right import CitationConfig, verify_facts

config = CitationConfig(
    top_k=3,
    min_score_threshold=0.3
)

result = verify_facts(answer, sources, config=config)
```

The configuration affects how claims are aligned to sources. Stricter thresholds produce more conservative verification, flagging more claims as unverified when alignment quality is marginal.

## Performance Considerations

Claim decomposition adds a preprocessing step but does not significantly impact performance for typical answer lengths. The SpaCy decomposer requires loading a language model, which adds startup time but processes quickly once loaded.

For high-throughput applications, consider reusing the decomposer instance across calls to avoid repeated model loading.

```python
decomposer = SpacyClaimDecomposer()  # Load once

for answer in answers:
    result = verify_facts(answer, sources, claim_decomposer=decomposer)
```

## Limitations

Claim decomposition relies on syntactic patterns and may not correctly split all compound sentences. Complex sentences with multiple nested clauses may be handled as single claims.

The verification only checks explicit textual support. Valid logical inferences from the source text will be marked as unverified if they are not stated directly.

Numerical claims require exact or near-exact matches. A source stating "approximately 15%" will not verify a claim stating "exactly 15%" even though the information is consistent.
