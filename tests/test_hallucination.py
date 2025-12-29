"""Tests for hallucination detection metrics."""

import pytest
from pydantic import ValidationError

from cite_right import (
    HallucinationConfig,
    SourceDocument,
    SpanConfidence,
    align_citations,
    compute_hallucination_metrics,
)
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.results import AnswerSpan, Citation, SpanCitations


class TestComputeHallucinationMetricsEmpty:
    """Tests for empty input handling."""

    def test_empty_input_returns_perfect_scores(self) -> None:
        metrics = compute_hallucination_metrics([])

        assert metrics.groundedness_score == 1.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.supported_ratio == 1.0
        assert metrics.partial_ratio == 0.0
        assert metrics.unsupported_ratio == 0.0
        assert metrics.avg_confidence == 1.0
        assert metrics.min_confidence == 1.0
        assert metrics.num_spans == 0
        assert metrics.num_supported == 0
        assert metrics.num_partial == 0
        assert metrics.num_unsupported == 0
        assert metrics.num_weak_citations == 0
        assert metrics.span_confidences == []
        assert metrics.unsupported_spans == []
        assert metrics.weakly_supported_spans == []


class TestComputeHallucinationMetricsSupported:
    """Tests for fully supported answers."""

    def test_single_fully_supported_span(self) -> None:
        span = AnswerSpan(
            text="The sky is blue.",
            char_start=0,
            char_end=16,
        )
        citation = Citation(
            score=2.5,
            source_id="doc1",
            source_index=0,
            candidate_index=0,
            char_start=10,
            char_end=26,
            evidence="The sky is blue.",
            components={
                "answer_coverage": 0.9,
                "normalized_alignment": 0.85,
            },
        )
        span_citations = [
            SpanCitations(
                answer_span=span,
                citations=[citation],
                status="supported",
            )
        ]

        metrics = compute_hallucination_metrics(span_citations)

        assert metrics.groundedness_score == pytest.approx(0.9, rel=0.01)
        assert metrics.hallucination_rate == pytest.approx(0.1, rel=0.01)
        assert metrics.supported_ratio == 1.0
        assert metrics.partial_ratio == 0.0
        assert metrics.unsupported_ratio == 0.0
        assert metrics.num_spans == 1
        assert metrics.num_supported == 1
        assert metrics.num_partial == 0
        assert metrics.num_unsupported == 0
        assert metrics.unsupported_spans == []
        assert len(metrics.span_confidences) == 1
        assert metrics.span_confidences[0].is_grounded is True
        assert metrics.span_confidences[0].source_ids == ["doc1"]


class TestComputeHallucinationMetricsUnsupported:
    """Tests for fully unsupported answers."""

    def test_single_unsupported_span(self) -> None:
        span = AnswerSpan(
            text="Aliens built the pyramids.",
            char_start=0,
            char_end=26,
        )
        span_citations = [
            SpanCitations(
                answer_span=span,
                citations=[],
                status="unsupported",
            )
        ]

        metrics = compute_hallucination_metrics(span_citations)

        assert metrics.groundedness_score == 0.0
        assert metrics.hallucination_rate == 1.0
        assert metrics.supported_ratio == 0.0
        assert metrics.unsupported_ratio == 1.0
        assert metrics.num_unsupported == 1
        assert len(metrics.unsupported_spans) == 1
        assert metrics.unsupported_spans[0].text == "Aliens built the pyramids."
        assert metrics.span_confidences[0].is_grounded is False
        assert metrics.span_confidences[0].confidence == 0.0

    def test_multiple_unsupported_spans(self) -> None:
        spans = [
            SpanCitations(
                answer_span=AnswerSpan(text="Claim one.", char_start=0, char_end=10),
                citations=[],
                status="unsupported",
            ),
            SpanCitations(
                answer_span=AnswerSpan(text="Claim two.", char_start=11, char_end=21),
                citations=[],
                status="unsupported",
            ),
        ]

        metrics = compute_hallucination_metrics(spans)

        assert metrics.groundedness_score == 0.0
        assert metrics.hallucination_rate == 1.0
        assert metrics.num_unsupported == 2
        assert len(metrics.unsupported_spans) == 2


class TestComputeHallucinationMetricsMixed:
    """Tests for mixed supported/partial/unsupported answers."""

    def test_mixed_supported_and_unsupported(self) -> None:
        # 20 chars supported, 20 chars unsupported
        supported_span = AnswerSpan(
            text="This is supported!!",
            char_start=0,
            char_end=19,
        )
        unsupported_span = AnswerSpan(
            text="This is not found!!",
            char_start=20,
            char_end=39,
        )
        citation = Citation(
            score=2.0,
            source_id="doc1",
            source_index=0,
            candidate_index=0,
            char_start=0,
            char_end=19,
            evidence="This is supported!!",
            components={"answer_coverage": 0.8},
        )

        span_citations = [
            SpanCitations(
                answer_span=supported_span,
                citations=[citation],
                status="supported",
            ),
            SpanCitations(
                answer_span=unsupported_span,
                citations=[],
                status="unsupported",
            ),
        ]

        metrics = compute_hallucination_metrics(span_citations)

        # Groundedness: (0.8 * 19 + 0 * 19) / 38 = 15.2 / 38 ≈ 0.4
        assert metrics.groundedness_score == pytest.approx(0.4, rel=0.05)
        assert metrics.hallucination_rate == pytest.approx(0.6, rel=0.05)
        assert metrics.supported_ratio == pytest.approx(0.5, rel=0.05)
        assert metrics.unsupported_ratio == pytest.approx(0.5, rel=0.05)
        assert metrics.num_supported == 1
        assert metrics.num_unsupported == 1
        assert len(metrics.unsupported_spans) == 1

    def test_partial_spans_included_in_grounded_by_default(self) -> None:
        partial_span = AnswerSpan(
            text="Partial match here.",
            char_start=0,
            char_end=19,
        )
        citation = Citation(
            score=1.5,
            source_id="doc1",
            source_index=0,
            candidate_index=0,
            char_start=0,
            char_end=10,
            evidence="Partial match",
            components={"answer_coverage": 0.5},
        )

        span_citations = [
            SpanCitations(
                answer_span=partial_span,
                citations=[citation],
                status="partial",
            )
        ]

        metrics = compute_hallucination_metrics(span_citations)

        assert metrics.partial_ratio == 1.0
        assert metrics.num_partial == 1
        # Partial is grounded by default
        assert metrics.groundedness_score == pytest.approx(0.5, rel=0.01)
        assert metrics.span_confidences[0].is_grounded is True

    def test_partial_spans_excluded_when_configured(self) -> None:
        partial_span = AnswerSpan(
            text="Partial match here.",
            char_start=0,
            char_end=19,
        )
        citation = Citation(
            score=1.5,
            source_id="doc1",
            source_index=0,
            candidate_index=0,
            char_start=0,
            char_end=10,
            evidence="Partial match",
            components={"answer_coverage": 0.5},
        )

        span_citations = [
            SpanCitations(
                answer_span=partial_span,
                citations=[citation],
                status="partial",
            )
        ]

        config = HallucinationConfig(include_partial_in_grounded=False)
        metrics = compute_hallucination_metrics(span_citations, config=config)

        # Partial not counted as grounded
        assert metrics.groundedness_score == 0.0
        assert metrics.span_confidences[0].is_grounded is False


class TestComputeHallucinationMetricsWeakCitations:
    """Tests for weak citation detection."""

    def test_weak_citations_detected(self) -> None:
        span = AnswerSpan(
            text="Weakly supported claim.",
            char_start=0,
            char_end=23,
        )
        citation = Citation(
            score=1.0,
            source_id="doc1",
            source_index=0,
            candidate_index=0,
            char_start=0,
            char_end=10,
            evidence="Weakly",
            components={"answer_coverage": 0.3},  # Below default 0.4 threshold
        )

        span_citations = [
            SpanCitations(
                answer_span=span,
                citations=[citation],
                status="partial",
            )
        ]

        metrics = compute_hallucination_metrics(span_citations)

        assert metrics.num_weak_citations == 1
        assert len(metrics.weakly_supported_spans) == 1
        assert metrics.weakly_supported_spans[0].text == "Weakly supported claim."

    def test_custom_weak_threshold(self) -> None:
        span = AnswerSpan(
            text="Maybe weak claim.",
            char_start=0,
            char_end=17,
        )
        citation = Citation(
            score=1.5,
            source_id="doc1",
            source_index=0,
            candidate_index=0,
            char_start=0,
            char_end=10,
            evidence="Maybe weak",
            components={"answer_coverage": 0.5},
        )

        span_citations = [
            SpanCitations(
                answer_span=span,
                citations=[citation],
                status="partial",
            )
        ]

        # With default threshold (0.4), this is not weak
        metrics_default = compute_hallucination_metrics(span_citations)
        assert metrics_default.num_weak_citations == 0

        # With higher threshold (0.6), this is weak
        config = HallucinationConfig(weak_citation_threshold=0.6)
        metrics_strict = compute_hallucination_metrics(span_citations, config=config)
        assert metrics_strict.num_weak_citations == 1


class TestComputeHallucinationMetricsConfidenceStats:
    """Tests for confidence statistics."""

    def test_avg_and_min_confidence(self) -> None:
        spans = [
            SpanCitations(
                answer_span=AnswerSpan(text="High conf.", char_start=0, char_end=10),
                citations=[
                    Citation(
                        score=2.0,
                        source_id="d1",
                        source_index=0,
                        candidate_index=0,
                        char_start=0,
                        char_end=10,
                        evidence="High conf.",
                        components={"answer_coverage": 0.9},
                    )
                ],
                status="supported",
            ),
            SpanCitations(
                answer_span=AnswerSpan(text="Low conf..", char_start=11, char_end=21),
                citations=[
                    Citation(
                        score=1.0,
                        source_id="d2",
                        source_index=1,
                        candidate_index=0,
                        char_start=0,
                        char_end=5,
                        evidence="Low",
                        components={"answer_coverage": 0.3},
                    )
                ],
                status="partial",
            ),
        ]

        metrics = compute_hallucination_metrics(spans)

        assert metrics.avg_confidence == pytest.approx(0.6, rel=0.01)
        assert metrics.min_confidence == pytest.approx(0.3, rel=0.01)


class TestComputeHallucinationMetricsMultipleSources:
    """Tests for multiple source handling."""

    def test_source_ids_collected(self) -> None:
        span = AnswerSpan(text="Multi-source claim.", char_start=0, char_end=19)
        citations = [
            Citation(
                score=2.0,
                source_id="doc1",
                source_index=0,
                candidate_index=0,
                char_start=0,
                char_end=10,
                evidence="Multi-source",
                components={"answer_coverage": 0.8},
            ),
            Citation(
                score=1.5,
                source_id="doc2",
                source_index=1,
                candidate_index=0,
                char_start=5,
                char_end=15,
                evidence="source claim",
                components={"answer_coverage": 0.6},
            ),
        ]

        span_citations = [
            SpanCitations(
                answer_span=span,
                citations=citations,
                status="supported",
            )
        ]

        metrics = compute_hallucination_metrics(span_citations)

        assert set(metrics.span_confidences[0].source_ids) == {"doc1", "doc2"}


class TestHallucinationMetricsIntegration:
    """Integration tests with align_citations."""

    def test_integration_with_align_citations_supported(self) -> None:
        fact = "The climate policy reduces carbon emissions by 40 percent"
        answer = f"{fact}."
        source = f"Research shows that {fact}. This was verified by multiple studies."

        config = CitationConfig(
            top_k=1,
            min_alignment_score=10,
            min_answer_coverage=0.6,
            supported_answer_coverage=0.6,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(answer, [source], config=config)
        metrics = compute_hallucination_metrics(results)

        assert metrics.num_spans == 1
        assert metrics.num_supported == 1
        assert metrics.groundedness_score > 0.5
        assert metrics.hallucination_rate < 0.5
        assert len(metrics.unsupported_spans) == 0

    def test_integration_with_align_citations_unsupported(self) -> None:
        answer = "Aliens definitely built the ancient pyramids in Egypt."
        source = (
            "The pyramids were built by skilled Egyptian workers over many decades."
        )

        config = CitationConfig(
            top_k=1,
            min_alignment_score=10,
            min_answer_coverage=0.6,
            supported_answer_coverage=0.6,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(answer, [source], config=config)
        metrics = compute_hallucination_metrics(results)

        assert metrics.num_spans == 1
        assert metrics.num_unsupported == 1
        assert metrics.groundedness_score == 0.0
        assert metrics.hallucination_rate == 1.0
        assert len(metrics.unsupported_spans) == 1

    def test_integration_mixed_answer(self) -> None:
        fact1 = "Acme Corp reported revenue of 5.2 billion dollars"
        hallucinated = "They also announced plans to colonize Mars"
        answer = f"{fact1}. {hallucinated}."

        sources = [
            SourceDocument(
                id="financial",
                text=f"In the annual report, {fact1} for fiscal year 2023.",
            ),
            SourceDocument(
                id="irrelevant",
                text="Unrelated content about weather patterns.",
            ),
        ]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=10,
            min_answer_coverage=0.5,
            supported_answer_coverage=0.6,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(answer, sources, config=config)
        metrics = compute_hallucination_metrics(results)

        assert metrics.num_spans == 2
        # First span should be supported, second unsupported
        assert metrics.num_unsupported >= 1
        assert 0.0 < metrics.groundedness_score < 1.0
        assert 0.0 < metrics.hallucination_rate < 1.0
        assert len(metrics.unsupported_spans) >= 1


class TestSpanConfidenceModel:
    """Tests for SpanConfidence model."""

    def test_span_confidence_fields(self) -> None:
        span = AnswerSpan(text="Test.", char_start=0, char_end=5)
        conf = SpanConfidence(
            span=span,
            status="supported",
            confidence=0.85,
            is_grounded=True,
            best_citation_score=2.5,
            source_ids=["doc1", "doc2"],
        )

        assert conf.span == span
        assert conf.status == "supported"
        assert conf.confidence == 0.85
        assert conf.is_grounded is True
        assert conf.best_citation_score == 2.5
        assert conf.source_ids == ["doc1", "doc2"]

    def test_span_confidence_unsupported_defaults(self) -> None:
        span = AnswerSpan(text="Test.", char_start=0, char_end=5)
        conf = SpanConfidence(
            span=span,
            status="unsupported",
            confidence=0.0,
            is_grounded=False,
        )

        assert conf.best_citation_score is None
        assert conf.source_ids == []


class TestHallucinationConfigModel:
    """Tests for HallucinationConfig model."""

    def test_default_values(self) -> None:
        config = HallucinationConfig()

        assert config.weak_citation_threshold == 0.4
        assert config.include_partial_in_grounded is True

    def test_custom_values(self) -> None:
        config = HallucinationConfig(
            weak_citation_threshold=0.5,
            include_partial_in_grounded=False,
        )

        assert config.weak_citation_threshold == 0.5
        assert config.include_partial_in_grounded is False

    def test_config_is_frozen(self) -> None:
        config = HallucinationConfig()

        with pytest.raises(ValidationError):
            config.weak_citation_threshold = 0.6  # type: ignore


class TestHallucinationMetricsModel:
    """Tests for HallucinationMetrics model."""

    def test_metrics_is_frozen(self) -> None:
        metrics = compute_hallucination_metrics([])

        with pytest.raises(ValidationError):
            metrics.groundedness_score = 0.5  # type: ignore
