"""Hallucination detection metrics for RAG responses.

This module provides aggregate metrics to measure how well a generated answer
is grounded in source documents, based on citation alignment results.
"""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from cite_right.core.results import AnswerSpan, SpanCitations


class HallucinationConfig(BaseModel):
    """Configuration for hallucination metric computation.

    Attributes:
        weak_citation_threshold: Citations with answer_coverage below this
            value are considered "weak" evidence. Default 0.4.
        include_partial_in_grounded: If True, "partial" status spans count
            toward the grounded score (weighted by their best citation quality).
            If False, only "supported" spans count as grounded. Default True.
    """

    model_config = ConfigDict(frozen=True)

    weak_citation_threshold: float = 0.4
    include_partial_in_grounded: bool = True


class SpanConfidence(BaseModel):
    """Confidence assessment for a single answer span.

    Attributes:
        span: The answer span being assessed.
        status: The citation status ("supported", "partial", "unsupported").
        confidence: Confidence score for this span (0-1). Based on best
            citation's answer_coverage, or 0 if unsupported.
        is_grounded: Whether this span is considered grounded in sources.
        best_citation_score: Score of the best citation, or None if unsupported.
        source_ids: List of source IDs that support this span.
    """

    model_config = ConfigDict(frozen=True)

    span: AnswerSpan
    status: Literal["supported", "partial", "unsupported"]
    confidence: float
    is_grounded: bool
    best_citation_score: float | None = None
    source_ids: list[str] = Field(default_factory=list)


class HallucinationMetrics(BaseModel):
    """Aggregate hallucination metrics for a generated answer.

    These metrics quantify how well the answer is grounded in source documents
    based on citation alignment results.

    Attributes:
        groundedness_score: Overall score of how well the answer is grounded
            in sources (0-1). Higher is better. Computed as weighted average
            of span confidence scores by character length.
        hallucination_rate: Proportion of the answer that is not grounded
            (0-1). Lower is better. Equals 1 - groundedness_score.
        supported_ratio: Proportion of spans (by char count) that are
            fully "supported".
        partial_ratio: Proportion of spans (by char count) that are "partial".
        unsupported_ratio: Proportion of spans (by char count) that are
            "unsupported".
        avg_confidence: Average confidence score across all spans.
        min_confidence: Minimum confidence score across all spans.
        num_spans: Total number of answer spans analyzed.
        num_supported: Number of spans with "supported" status.
        num_partial: Number of spans with "partial" status.
        num_unsupported: Number of spans with "unsupported" status.
        num_weak_citations: Number of spans with weak citations (low coverage
            but not unsupported).
        span_confidences: Per-span confidence details.
        unsupported_spans: List of answer spans that are unsupported.
        weakly_supported_spans: List of answer spans with weak evidence.
    """

    model_config = ConfigDict(frozen=True)

    groundedness_score: float
    hallucination_rate: float
    supported_ratio: float
    partial_ratio: float
    unsupported_ratio: float
    avg_confidence: float
    min_confidence: float
    num_spans: int
    num_supported: int
    num_partial: int
    num_unsupported: int
    num_weak_citations: int
    span_confidences: list[SpanConfidence] = Field(default_factory=list)
    unsupported_spans: list[AnswerSpan] = Field(default_factory=list)
    weakly_supported_spans: list[AnswerSpan] = Field(default_factory=list)


def compute_hallucination_metrics(
    span_citations: Sequence[SpanCitations],
    *,
    config: HallucinationConfig | None = None,
) -> HallucinationMetrics:
    """Compute hallucination metrics from citation alignment results.

    This function analyzes the output of `align_citations()` to produce
    aggregate metrics measuring how well the generated answer is grounded
    in source documents.

    Args:
        span_citations: List of SpanCitations from `align_citations()`.
        config: Optional configuration for metric computation.

    Returns:
        HallucinationMetrics with aggregate and per-span confidence data.

    Example:
        >>> from cite_right import align_citations
        >>> from cite_right.hallucination import compute_hallucination_metrics
        >>>
        >>> results = align_citations(answer, sources)
        >>> metrics = compute_hallucination_metrics(results)
        >>> print(f"Groundedness: {metrics.groundedness_score:.1%}")
        >>> print(f"Hallucination rate: {metrics.hallucination_rate:.1%}")
        >>> for span in metrics.unsupported_spans:
        ...     print(f"  Unsupported: {span.text!r}")
    """
    cfg = config or HallucinationConfig()

    if not span_citations:
        return HallucinationMetrics(
            groundedness_score=1.0,
            hallucination_rate=0.0,
            supported_ratio=1.0,
            partial_ratio=0.0,
            unsupported_ratio=0.0,
            avg_confidence=1.0,
            min_confidence=1.0,
            num_spans=0,
            num_supported=0,
            num_partial=0,
            num_unsupported=0,
            num_weak_citations=0,
            span_confidences=[],
            unsupported_spans=[],
            weakly_supported_spans=[],
        )

    span_confidences: list[SpanConfidence] = []
    unsupported_spans: list[AnswerSpan] = []
    weakly_supported_spans: list[AnswerSpan] = []

    total_chars = 0
    supported_chars = 0
    partial_chars = 0
    unsupported_chars = 0

    num_supported = 0
    num_partial = 0
    num_unsupported = 0
    num_weak = 0

    confidence_values: list[float] = []
    weighted_confidence_sum = 0.0

    for sc in span_citations:
        span = sc.answer_span
        span_len = len(span.text)
        total_chars += span_len

        # Determine confidence from best citation
        if sc.citations:
            best = sc.citations[0]
            answer_coverage = float(best.components.get("answer_coverage", 0.0))
            confidence = answer_coverage
            best_score = best.score
            source_ids = list({c.source_id for c in sc.citations})

            # Check for weak citation
            if answer_coverage < cfg.weak_citation_threshold:
                num_weak += 1
                weakly_supported_spans.append(span)
        else:
            confidence = 0.0
            best_score = None
            source_ids = []

        confidence_values.append(confidence)

        # Determine grounded status based on config
        if sc.status == "supported":
            is_grounded = True
            num_supported += 1
            supported_chars += span_len
        elif sc.status == "partial":
            is_grounded = cfg.include_partial_in_grounded
            num_partial += 1
            partial_chars += span_len
        else:  # unsupported
            is_grounded = False
            num_unsupported += 1
            unsupported_chars += span_len
            unsupported_spans.append(span)

        # Weight confidence by character length for grounded score
        if is_grounded:
            weighted_confidence_sum += confidence * span_len
        # Unsupported spans contribute 0 to weighted sum

        span_confidences.append(
            SpanConfidence(
                span=span,
                status=sc.status,
                confidence=confidence,
                is_grounded=is_grounded,
                best_citation_score=best_score,
                source_ids=source_ids,
            )
        )

    # Compute aggregate metrics
    if total_chars > 0:
        groundedness_score = weighted_confidence_sum / total_chars
        supported_ratio = supported_chars / total_chars
        partial_ratio = partial_chars / total_chars
        unsupported_ratio = unsupported_chars / total_chars
    else:
        groundedness_score = 1.0
        supported_ratio = 1.0
        partial_ratio = 0.0
        unsupported_ratio = 0.0

    hallucination_rate = 1.0 - groundedness_score

    if confidence_values:
        avg_confidence = sum(confidence_values) / len(confidence_values)
        min_confidence = min(confidence_values)
    else:
        avg_confidence = 1.0
        min_confidence = 1.0

    return HallucinationMetrics(
        groundedness_score=groundedness_score,
        hallucination_rate=hallucination_rate,
        supported_ratio=supported_ratio,
        partial_ratio=partial_ratio,
        unsupported_ratio=unsupported_ratio,
        avg_confidence=avg_confidence,
        min_confidence=min_confidence,
        num_spans=len(span_citations),
        num_supported=num_supported,
        num_partial=num_partial,
        num_unsupported=num_unsupported,
        num_weak_citations=num_weak,
        span_confidences=span_confidences,
        unsupported_spans=unsupported_spans,
        weakly_supported_spans=weakly_supported_spans,
    )
