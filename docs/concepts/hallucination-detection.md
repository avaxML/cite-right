# Hallucination Detection

Language models sometimes generate content that is not grounded in the provided source material. Cite-Right provides tools to detect and quantify this phenomenon, helping you build more trustworthy AI applications.

## Understanding Hallucination

In the context of retrieval-augmented generation, hallucination refers to generated content that cannot be traced back to the retrieved sources. This may occur when the model draws on its parametric knowledge rather than the provided context, when it makes logical leaps not supported by the text, or when it simply generates plausible-sounding but unfounded claims.

Hallucination detection in Cite-Right works by analyzing the citation alignment results. Answer spans that align well with sources are considered grounded. Those without adequate source support are flagged as potentially hallucinated.

## Computing Hallucination Metrics

The `compute_hallucination_metrics` function analyzes alignment results and produces aggregate statistics. This function is defined in `src/cite_right/hallucination.py`.

```python
from cite_right import SourceDocument, align_citations, compute_hallucination_metrics

answer = """The company reported record profits in Q4.
They announced plans to expand into Asia.
The CEO will retire next month."""

sources = [
    SourceDocument(
        id="earnings",
        text="Fourth quarter profits reached an all-time high, beating analyst expectations."
    )
]

results = align_citations(answer, sources)
metrics = compute_hallucination_metrics(results)

print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Hallucination rate: {metrics.hallucination_rate:.1%}")
```

In this example, the first sentence about record profits should align with the source. The second and third sentences have no source support and will contribute to the hallucination rate.

## HallucinationMetrics

The `compute_hallucination_metrics` function returns a `HallucinationMetrics` object containing comprehensive statistics.

### Aggregate Scores

The `groundedness_score` is a weighted confidence score between 0 and 1. Higher values indicate better grounding in sources. This score considers both the number of supported spans and the quality of their citations.

The `hallucination_rate` is the proportion of content that lacks source support, also between 0 and 1. Lower values indicate less hallucination. This metric complements the groundedness score by focusing on problematic content.

```python
if metrics.groundedness_score > 0.8:
    print("Answer is well-grounded")
elif metrics.hallucination_rate > 0.3:
    print("Warning: Significant hallucination detected")
```

### Span Ratios

Three ratio metrics describe how answer content distributes across support levels, weighted by character count.

The `supported_ratio` indicates what proportion of the answer text is fully supported by sources.

The `partial_ratio` indicates what proportion has partial support, meaning some citation was found but it may not fully cover the claim.

The `unsupported_ratio` indicates what proportion has no adequate source support.

These ratios sum to 1.0 and provide a quick overview of answer composition.

### Span Counts

Three count metrics provide raw numbers rather than ratios.

The `num_supported` field counts answer spans with full source support.

The `num_partial` field counts spans with partial support.

The `num_unsupported` field counts spans without adequate support.

These counts are useful for understanding the structure of the answer independent of span length.

### Confidence Statistics

The `avg_confidence` field reports the average confidence score across all spans.

The `min_confidence` field reports the lowest confidence score, identifying the weakest point in the answer.

### Weak Citation Tracking

The `num_weak_citations` field counts spans where a citation was found but the quality is below a configurable threshold. These represent borderline cases that may warrant manual review.

### Problem Span Identification

The `unsupported_spans` field contains a list of `AnswerSpan` objects that received no adequate citations. These are the specific pieces of text most likely to be hallucinated.

The `weakly_supported_spans` field contains spans with low-quality citations that may be unreliable.

```python
if metrics.unsupported_spans:
    print("Potentially hallucinated content:")
    for span in metrics.unsupported_spans:
        print(f"  '{span.text}'")
```

### Per-Span Details

The `span_confidences` field provides a list of `SpanConfidence` objects with detailed information about each answer span.

```python
for conf in metrics.span_confidences:
    print(f"Text: {conf.span.text}")
    print(f"Confidence: {conf.confidence:.2f}")
    print(f"Status: {conf.status}")
    print(f"Top source: {conf.top_source_id}")
```

Each `SpanConfidence` includes the span text, its confidence score, status, and the identifier of the best matching source if any.

## Configuration

The `HallucinationConfig` class provides control over how metrics are computed.

```python
from cite_right import HallucinationConfig, compute_hallucination_metrics

config = HallucinationConfig(
    weak_citation_threshold=0.4,
    include_partial_in_grounded=True
)

metrics = compute_hallucination_metrics(results, config=config)
```

The `weak_citation_threshold` parameter sets the minimum answer coverage score for a citation to be considered adequate. Citations below this threshold are counted as weak.

The `include_partial_in_grounded` parameter controls whether partial matches contribute to the groundedness score. Setting this to False produces a stricter groundedness metric that only counts fully supported spans.

## Convenience Functions

For common use cases, high-level convenience functions provide quick answers.

### is_grounded

The `is_grounded` function returns a boolean indicating whether the answer meets a groundedness threshold.

```python
from cite_right import is_grounded

if is_grounded(answer, sources, threshold=0.6):
    # Proceed with the response
    pass
else:
    # Request clarification or regenerate
    pass
```

This function internally calls `align_citations` and `compute_hallucination_metrics`, making it a one-step check suitable for quality gates.

### is_hallucinated

The `is_hallucinated` function checks whether the hallucination rate exceeds a threshold.

```python
from cite_right import is_hallucinated

if is_hallucinated(answer, sources, threshold=0.3):
    print("Warning: Answer may contain hallucinations")
```

This provides the inverse perspective from `is_grounded`, focusing on problem content.

### check_groundedness

The `check_groundedness` function combines alignment and metrics computation in a single call, returning the full `HallucinationMetrics` object.

```python
from cite_right import check_groundedness

metrics = check_groundedness(answer, sources)
print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Problematic spans: {len(metrics.unsupported_spans)}")
```

This is useful when you need both the boolean decision and the detailed metrics for logging or analysis.

## Integration Patterns

### Quality Gate

Use hallucination detection as a quality gate before presenting responses to users.

```python
def generate_with_verification(query, sources):
    answer = generate_answer(query, sources)
    metrics = check_groundedness(answer, sources)

    if metrics.groundedness_score < 0.5:
        return regenerate_with_emphasis_on_sources(query, sources)

    return answer
```

### User Interface Indicators

Display confidence indicators in the user interface based on hallucination metrics.

```python
def get_confidence_indicator(metrics):
    if metrics.groundedness_score > 0.8:
        return "high_confidence"
    elif metrics.groundedness_score > 0.5:
        return "moderate_confidence"
    else:
        return "low_confidence"
```

### Logging and Monitoring

Track hallucination rates over time to identify model or prompt degradation.

```python
import logging

def log_hallucination_metrics(query, answer, metrics):
    logging.info(
        "hallucination_check",
        extra={
            "query_hash": hash(query),
            "groundedness": metrics.groundedness_score,
            "hallucination_rate": metrics.hallucination_rate,
            "unsupported_count": metrics.num_unsupported
        }
    )
```

## Limitations

Hallucination detection identifies answer content that lacks source support. It does not verify factual accuracy beyond the provided sources. If a source document itself contains errors, the detection will still mark content derived from it as grounded.

The detection is also limited to explicit textual alignment. Logical inferences that are correct but not stated verbatim in sources will be marked as unsupported. Applications requiring inference verification need additional techniques beyond citation alignment.
