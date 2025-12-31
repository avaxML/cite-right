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
    """Compute hallucination metrics from citation alignment results."""
    cfg = config or HallucinationConfig()

    if not span_citations:
        return _empty_hallucination_metrics()

    accumulator = _MetricsAccumulator()

    for sc in span_citations:
        accumulator.process_span(sc, cfg)

    return accumulator.build_metrics(len(span_citations))


def _empty_hallucination_metrics() -> HallucinationMetrics:
    """Return metrics for empty input."""
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


class _MetricsAccumulator:
    """Accumulator for computing hallucination metrics across spans."""

    def __init__(self) -> None:
        self.span_confidences: list[SpanConfidence] = []
        self.unsupported_spans: list[AnswerSpan] = []
        self.weakly_supported_spans: list[AnswerSpan] = []
        self.confidence_values: list[float] = []
        self.weighted_confidence_sum = 0.0
        self.total_chars = 0
        self.supported_chars = 0
        self.partial_chars = 0
        self.unsupported_chars = 0
        self.num_supported = 0
        self.num_partial = 0
        self.num_unsupported = 0
        self.num_weak = 0

    def process_span(self, sc: SpanCitations, cfg: HallucinationConfig) -> None:
        """Process a single span citation."""
        span = sc.answer_span
        span_len = len(span.text)
        self.total_chars += span_len

        confidence, best_score, source_ids = self._extract_confidence(sc, cfg)
        self.confidence_values.append(confidence)

        is_grounded = self._update_status_counts(sc, span_len, cfg)

        if is_grounded:
            self.weighted_confidence_sum += confidence * span_len

        self.span_confidences.append(
            SpanConfidence(
                span=span,
                status=sc.status,
                confidence=confidence,
                is_grounded=is_grounded,
                best_citation_score=best_score,
                source_ids=source_ids,
            )
        )

    def _extract_confidence(
        self, sc: SpanCitations, cfg: HallucinationConfig
    ) -> tuple[float, float | None, list[str]]:
        """Extract confidence info from citations."""
        if not sc.citations:
            return 0.0, None, []

        best = sc.citations[0]
        answer_coverage = float(best.components.get("answer_coverage", 0.0))
        source_ids = list({c.source_id for c in sc.citations})

        if answer_coverage < cfg.weak_citation_threshold:
            self.num_weak += 1
            self.weakly_supported_spans.append(sc.answer_span)

        return answer_coverage, best.score, source_ids

    def _update_status_counts(
        self, sc: SpanCitations, span_len: int, cfg: HallucinationConfig
    ) -> bool:
        """Update status counts and return whether span is grounded."""
        if sc.status == "supported":
            self.num_supported += 1
            self.supported_chars += span_len
            return True
        if sc.status == "partial":
            self.num_partial += 1
            self.partial_chars += span_len
            return cfg.include_partial_in_grounded
        self.num_unsupported += 1
        self.unsupported_chars += span_len
        self.unsupported_spans.append(sc.answer_span)
        return False

    def build_metrics(self, num_spans: int) -> HallucinationMetrics:
        """Build final metrics from accumulated data."""
        if self.total_chars > 0:
            groundedness_score = self.weighted_confidence_sum / self.total_chars
            supported_ratio = self.supported_chars / self.total_chars
            partial_ratio = self.partial_chars / self.total_chars
            unsupported_ratio = self.unsupported_chars / self.total_chars
        else:
            groundedness_score, supported_ratio = 1.0, 1.0
            partial_ratio, unsupported_ratio = 0.0, 0.0

        avg_confidence = (
            sum(self.confidence_values) / len(self.confidence_values)
            if self.confidence_values
            else 1.0
        )
        min_confidence = min(self.confidence_values) if self.confidence_values else 1.0

        return HallucinationMetrics(
            groundedness_score=groundedness_score,
            hallucination_rate=1.0 - groundedness_score,
            supported_ratio=supported_ratio,
            partial_ratio=partial_ratio,
            unsupported_ratio=unsupported_ratio,
            avg_confidence=avg_confidence,
            min_confidence=min_confidence,
            num_spans=num_spans,
            num_supported=self.num_supported,
            num_partial=self.num_partial,
            num_unsupported=self.num_unsupported,
            num_weak_citations=self.num_weak,
            span_confidences=self.span_confidences,
            unsupported_spans=self.unsupported_spans,
            weakly_supported_spans=self.weakly_supported_spans,
        )
