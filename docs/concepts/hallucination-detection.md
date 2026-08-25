# Hallucination Detection

Cite-Right is a groundedness and citation tagger, not a clean hallucination detector. It marks whether each answer span has localized source support. That is useful for highlighting, quality gates, and aggregate groundedness scores. It is not a substitute for a dedicated hallucination classifier.

On RAGTruth test (2,675 answers), 0.4.0 quality matched 0.3.1. False-supported on gold hallucinations is about 1.6%. Unsupported precision is about 14%. The library overflags: many spans tagged `unsupported` are not gold hallucinations. If `partial` counts as not fully supported, gold hallucinations are rarely blessed as `supported`.

## Understanding Hallucination

In the context of retrieval-augmented generation, hallucination usually means generated content that cannot be traced back to the retrieved sources. That may occur when the model draws on parametric knowledge rather than the provided context, when it makes logical leaps not supported by the text, or when it generates plausible-sounding but unfounded claims.

Cite-Right does not classify those cases with a trained detector. It analyzes citation alignment results. Answer spans that produce citations are `"supported"` or `"partial"`. Spans with no surviving citation are `"unsupported"`. Cheap contradiction (negation, number, leftover n-gram slot, entity swap) downgrades to `"partial"`, never `"unsupported"`. The status literal is `"partial"`, never `"partially_supported"`.

Treat `unsupported` as "no localized citation survived," not as a high-precision hallucination label.

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
        text="Fourth quarter profits reached an all-time high, beating analyst expectations.",
    )
]

results = align_citations(answer, sources)
metrics = compute_hallucination_metrics(results)

print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Hallucination rate: {metrics.hallucination_rate:.1%}")
```

In this example, the first sentence about record profits should align with the source. The second and third sentences have no source support and will contribute to the hallucination rate. Because unsupported precision is low, do not treat that rate as a hallucination probability.

## HallucinationMetrics

The `compute_hallucination_metrics` function returns a `HallucinationMetrics` object containing comprehensive statistics.

### Aggregate Scores

The `groundedness_score` is a weighted confidence score between 0 and 1. Higher values indicate better grounding in sources. This score considers both the number of supported spans and the quality of their citations.

The `hallucination_rate` is `1 - groundedness_score`. It is the proportion of content that the tagger did not count as grounded, also between 0 and 1. With default `HallucinationConfig`, `partial` spans can contribute to groundedness. If you need a stricter reading, set `include_partial_in_grounded=False` so only `"supported"` counts.

```python
if metrics.groundedness_score > 0.8:
    print("Answer is well-grounded")
elif metrics.hallucination_rate > 0.3:
    print("Warning: Significant ungrounded content")
```

Thresholds like these are application policy, not validated hallucination cutoffs.

### Span Ratios

Three ratio metrics describe how answer content distributes across support levels, weighted by character count.

The `supported_ratio` indicates what proportion of the answer text is fully supported by sources.

The `partial_ratio` indicates what proportion has partial support, meaning some citation was found but it may not fully cover the claim, or contradiction downgraded the span.

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

The `num_weak_citations` field counts spans where a citation was found but answer coverage is below a configurable threshold. These represent borderline cases that may warrant manual review.

### Problem Span Identification

The `unsupported_spans` field contains a list of `AnswerSpan` objects that received no adequate citations. These are the pieces of text with no localized citation, which may or may not be gold hallucinations.

The `weakly_supported_spans` field contains spans with low-quality citations that may be unreliable.

```python
if metrics.unsupported_spans:
    print("Spans with no localized citation:")
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
    print(f"Sources: {conf.source_ids}")
```

Each `SpanConfidence` includes the span text, its confidence score, status, and the identifier of the best matching source if any.

## Configuration

The `HallucinationConfig` class provides control over how metrics are computed.

```python
from cite_right import HallucinationConfig, compute_hallucination_metrics

config = HallucinationConfig(
    weak_citation_threshold=0.4, include_partial_in_grounded=True
)

metrics = compute_hallucination_metrics(results, config=config)
```

The `weak_citation_threshold` parameter sets the minimum answer coverage for a citation to be considered adequate. Citations below this threshold are counted as weak.

The `include_partial_in_grounded` parameter controls whether `"partial"` matches contribute to the groundedness score. Setting this to `False` produces a stricter groundedness metric that only counts fully supported spans. If `partial` is excluded, gold hallucinations are rarely counted as `supported`.

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

This function internally calls `align_citations` and `compute_hallucination_metrics`, making it a one-step check suitable for quality gates. It inherits the same overflag behavior as the metrics above.

### is_hallucinated

The `is_hallucinated` function checks whether the hallucination rate exceeds a threshold.

```python
from cite_right import is_hallucinated

if is_hallucinated(answer, sources, threshold=0.3):
    print("Warning: Answer may contain ungrounded content")
```

This is the inverse of `is_grounded`. It is still a groundedness check, not a high-precision hallucination detector.

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

Use citation status as a quality gate before presenting responses to users. Prefer inspecting `"supported"` versus `"partial"` versus `"unsupported"` over treating hallucination rate as a calibrated probability.

```python
def generate_with_verification(query, sources):
    answer = generate_answer(query, sources)
    metrics = check_groundedness(answer, sources)

    if metrics.groundedness_score < 0.5:
        return regenerate_with_emphasis_on_sources(query, sources)

    return answer
```

### User Interface Indicators

Display confidence indicators in the user interface based on citation metrics, and show the actual evidence offsets rather than a red/green hallucination badge.

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

Track groundedness and unsupported rates over time to identify model or prompt degradation. Log span status counts, not just a single hallucination rate, so overflag does not look like a sudden hallucination spike.

```python
import logging


def log_hallucination_metrics(query, answer, metrics):
    logging.info(
        "hallucination_check",
        extra={
            "query_hash": hash(query),
            "groundedness": metrics.groundedness_score,
            "hallucination_rate": metrics.hallucination_rate,
            "unsupported_count": metrics.num_unsupported,
            "partial_count": metrics.num_partial,
        },
    )
```

## Limitations

Hallucination detection identifies answer content that lacks localized source support. It does not verify factual accuracy beyond the provided sources. If a source document itself contains errors, content derived from it is still marked as grounded.

Unsupported precision is about 14% on RAGTruth test. Grounded paraphrases are less overflagged than in 0.3.1 (content-word overlap can emit a citation when sequential Smith-Waterman coverage is low), but the tagger still overflags relative to gold hallucination labels.

The detection is limited to explicit textual alignment plus the cheap contradiction checks above. Logical inferences that are correct but not stated in sources will often be marked as unsupported. Applications requiring inference verification need additional techniques beyond citation alignment.
