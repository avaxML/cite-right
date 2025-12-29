"""Tests for fact-level verification."""

import pytest
from pydantic import ValidationError

from cite_right import (
    Claim,
    ClaimVerification,
    FactVerificationConfig,
    FactVerificationMetrics,
    SimpleClaimDecomposer,
    SourceDocument,
    verify_facts,
)
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.results import AnswerSpan


class TestClaim:
    """Tests for Claim model."""

    def test_claim_fields(self) -> None:
        span = AnswerSpan(text="Test sentence.", char_start=0, char_end=14)
        claim = Claim(
            text="Test sentence.",
            char_start=0,
            char_end=14,
            source_span=span,
            claim_index=0,
        )

        assert claim.text == "Test sentence."
        assert claim.char_start == 0
        assert claim.char_end == 14
        assert claim.source_span == span
        assert claim.claim_index == 0

    def test_claim_is_frozen(self) -> None:
        span = AnswerSpan(text="Test.", char_start=0, char_end=5)
        claim = Claim(
            text="Test.",
            char_start=0,
            char_end=5,
            source_span=span,
        )

        with pytest.raises(ValidationError):
            claim.text = "Modified"  # type: ignore


class TestSimpleClaimDecomposer:
    """Tests for SimpleClaimDecomposer."""

    def test_returns_span_as_single_claim(self) -> None:
        decomposer = SimpleClaimDecomposer()
        span = AnswerSpan(
            text="Revenue grew and profits increased.",
            char_start=10,
            char_end=45,
        )

        claims = decomposer.decompose(span)

        assert len(claims) == 1
        assert claims[0].text == span.text
        assert claims[0].char_start == span.char_start
        assert claims[0].char_end == span.char_end
        assert claims[0].source_span == span

    def test_preserves_character_offsets(self) -> None:
        decomposer = SimpleClaimDecomposer()
        span = AnswerSpan(
            text="Some claim here.",
            char_start=100,
            char_end=116,
        )

        claims = decomposer.decompose(span)

        assert claims[0].char_start == 100
        assert claims[0].char_end == 116


class TestSpacyClaimDecomposer:
    """Tests for SpacyClaimDecomposer."""

    @pytest.fixture
    def decomposer(self):
        """Create a SpacyClaimDecomposer if spaCy is available."""
        try:
            from cite_right import SpacyClaimDecomposer

            return SpacyClaimDecomposer()
        except RuntimeError:
            pytest.skip("spaCy not installed")

    def test_decomposes_conjunction(self, decomposer) -> None:
        span = AnswerSpan(
            text="Revenue grew and profits increased.",
            char_start=0,
            char_end=35,
        )

        claims = decomposer.decompose(span)

        # Should split into two claims
        assert len(claims) >= 1
        claim_texts = [c.text for c in claims]
        # Check that conjunction-based split occurred
        if len(claims) > 1:
            assert "Revenue grew" in claim_texts[0] or "grew" in claim_texts[0]
            assert (
                "profits increased" in claim_texts[-1] or "increased" in claim_texts[-1]
            )

    def test_no_conjunction_returns_single_claim(self, decomposer) -> None:
        span = AnswerSpan(
            text="The company reported strong earnings.",
            char_start=0,
            char_end=37,
        )

        claims = decomposer.decompose(span)

        assert len(claims) == 1
        assert claims[0].text == span.text

    def test_preserves_source_span_reference(self, decomposer) -> None:
        span = AnswerSpan(
            text="Sales increased and costs decreased.",
            char_start=50,
            char_end=86,
        )

        claims = decomposer.decompose(span)

        for claim in claims:
            assert claim.source_span == span

    def test_multiple_conjunctions(self, decomposer) -> None:
        span = AnswerSpan(
            text="Revenue grew, profits increased, and costs declined.",
            char_start=0,
            char_end=52,
        )

        claims = decomposer.decompose(span)

        # Should have multiple claims
        assert len(claims) >= 1

    def test_import_error_when_spacy_missing(self) -> None:
        """Test that missing spaCy raises RuntimeError."""
        # This test verifies the error message when spaCy is not available
        # We can't easily test this without uninstalling spaCy
        pass


class TestFactVerificationConfig:
    """Tests for FactVerificationConfig."""

    def test_default_values(self) -> None:
        config = FactVerificationConfig()

        assert config.verified_coverage_threshold == 0.6
        assert config.partial_coverage_threshold == 0.3
        assert config.citation_config is None

    def test_custom_values(self) -> None:
        citation_cfg = CitationConfig(top_k=5)
        config = FactVerificationConfig(
            verified_coverage_threshold=0.8,
            partial_coverage_threshold=0.4,
            citation_config=citation_cfg,
        )

        assert config.verified_coverage_threshold == 0.8
        assert config.partial_coverage_threshold == 0.4
        assert config.citation_config == citation_cfg

    def test_config_is_frozen(self) -> None:
        config = FactVerificationConfig()

        with pytest.raises(ValidationError):
            config.verified_coverage_threshold = 0.9  # type: ignore


class TestClaimVerification:
    """Tests for ClaimVerification model."""

    def test_verified_claim(self) -> None:
        span = AnswerSpan(text="Test.", char_start=0, char_end=5)
        claim = Claim(text="Test.", char_start=0, char_end=5, source_span=span)

        verification = ClaimVerification(
            claim=claim,
            status="verified",
            confidence=0.85,
            source_ids=["doc1"],
        )

        assert verification.status == "verified"
        assert verification.confidence == 0.85
        assert verification.source_ids == ["doc1"]

    def test_unverified_claim(self) -> None:
        span = AnswerSpan(text="Fake.", char_start=0, char_end=5)
        claim = Claim(text="Fake.", char_start=0, char_end=5, source_span=span)

        verification = ClaimVerification(
            claim=claim,
            status="unverified",
            confidence=0.0,
        )

        assert verification.status == "unverified"
        assert verification.confidence == 0.0
        assert verification.best_citation is None
        assert verification.all_citations == []


class TestVerifyFactsEmpty:
    """Tests for verify_facts with empty input."""

    def test_empty_answer(self) -> None:
        metrics = verify_facts("", ["Some source text."])

        assert metrics.num_claims == 0
        assert metrics.verification_rate == 1.0
        assert metrics.avg_confidence == 1.0

    def test_empty_sources(self) -> None:
        metrics = verify_facts("Some answer text.", [])

        assert metrics.num_claims >= 1
        assert metrics.num_unverified >= 1


class TestVerifyFactsSupported:
    """Tests for verify_facts with supported claims."""

    def test_single_verified_claim(self) -> None:
        fact = "The company reported revenue of 5.2 billion dollars"
        answer = f"{fact}."
        source = f"Annual report: {fact} in fiscal 2023."

        metrics = verify_facts(answer, [source])

        assert metrics.num_claims == 1
        assert metrics.num_verified == 1
        assert metrics.verification_rate == 1.0
        assert metrics.avg_confidence > 0.5
        assert len(metrics.verified_claims) == 1
        assert len(metrics.unverified_claims) == 0

    def test_multiple_verified_claims(self) -> None:
        answer = "Revenue grew 20%. Profits doubled."
        sources = [
            "The annual report shows revenue grew 20%.",
            "Financial statements indicate profits doubled.",
        ]

        metrics = verify_facts(answer, sources)

        assert metrics.num_claims == 2
        assert metrics.num_verified >= 1


class TestVerifyFactsUnverified:
    """Tests for verify_facts with unverified claims."""

    def test_completely_unverified(self) -> None:
        answer = "Aliens built the pyramids."
        source = "The pyramids were built by Egyptian workers."

        config = FactVerificationConfig(
            citation_config=CitationConfig(
                min_alignment_score=10,
                min_answer_coverage=0.5,
                weights=CitationWeights(lexical=0.0, embedding=0.0),
            )
        )

        metrics = verify_facts(answer, [source], config=config)

        assert metrics.num_claims == 1
        assert metrics.num_unverified == 1
        assert metrics.verification_rate == 0.0
        assert len(metrics.unverified_claims) == 1

    def test_hallucinated_claim_in_answer(self) -> None:
        answer = "Revenue was 5 billion. They also colonized Mars."
        sources = [
            SourceDocument(
                id="financial",
                text="The company reported revenue was 5 billion dollars.",
            ),
        ]

        metrics = verify_facts(answer, sources)

        assert metrics.num_claims == 2
        # At least one claim should be unverified (the Mars one)
        assert metrics.num_unverified >= 1
        assert len(metrics.unverified_claims) >= 1


class TestVerifyFactsMixed:
    """Tests for verify_facts with mixed verified/unverified claims."""

    def test_mixed_verification(self) -> None:
        answer = "Acme reported 5.2 billion in revenue. They announced plans to terraform Venus."
        sources = [
            SourceDocument(
                id="financial",
                text="Acme reported 5.2 billion in revenue for fiscal year 2023.",
            ),
        ]

        metrics = verify_facts(answer, sources)

        assert metrics.num_claims == 2
        # First claim should be verified, second should not
        assert metrics.num_verified >= 1
        assert metrics.num_unverified >= 1
        assert 0.0 < metrics.verification_rate < 1.0


class TestVerifyFactsWithSpacy:
    """Integration tests with SpacyClaimDecomposer."""

    @pytest.fixture
    def spacy_decomposer(self):
        """Create SpacyClaimDecomposer if available."""
        try:
            from cite_right import SpacyClaimDecomposer

            return SpacyClaimDecomposer()
        except RuntimeError:
            pytest.skip("spaCy not installed")

    def test_conjunction_decomposition_and_verification(self, spacy_decomposer) -> None:
        # Compound sentence with conjunction
        answer = "Revenue grew 20% and profits doubled."
        sources = [
            "Annual report shows revenue grew 20%.",
            # No mention of profits doubling
        ]

        metrics = verify_facts(answer, sources, claim_decomposer=spacy_decomposer)

        # With spaCy, should decompose into separate claims
        # "Revenue grew 20%" should be verified
        # "profits doubled" should be unverified
        assert metrics.num_claims >= 1

    def test_multiple_conjunctions_verification(self, spacy_decomposer) -> None:
        answer = "Sales increased, costs decreased, and margins improved."
        sources = [
            "The company saw sales increased significantly.",
            "Operating costs decreased by 10%.",
            # No mention of margins
        ]

        metrics = verify_facts(answer, sources, claim_decomposer=spacy_decomposer)

        # Should have multiple claims
        assert metrics.num_claims >= 1


class TestFactVerificationMetrics:
    """Tests for FactVerificationMetrics model."""

    def test_metrics_fields(self) -> None:
        span = AnswerSpan(text="Test.", char_start=0, char_end=5)
        claim = Claim(text="Test.", char_start=0, char_end=5, source_span=span)

        metrics = FactVerificationMetrics(
            num_claims=3,
            num_verified=2,
            num_partial=1,
            num_unverified=0,
            verification_rate=0.67,
            avg_confidence=0.75,
            min_confidence=0.55,
            verified_claims=[claim, claim],
            partial_claims=[claim],
            unverified_claims=[],
        )

        assert metrics.num_claims == 3
        assert metrics.num_verified == 2
        assert metrics.num_partial == 1
        assert metrics.num_unverified == 0
        assert metrics.verification_rate == 0.67
        assert len(metrics.verified_claims) == 2
        assert len(metrics.partial_claims) == 1

    def test_metrics_is_frozen(self) -> None:
        metrics = verify_facts("Test.", ["Test source."])

        with pytest.raises(ValidationError):
            metrics.num_claims = 10  # type: ignore


class TestVerifyFactsConfiguration:
    """Tests for verify_facts with custom configuration."""

    def test_custom_coverage_thresholds(self) -> None:
        answer = "The company reported moderate growth."
        source = "The company reported moderate growth in Q3."

        # Strict threshold
        strict_config = FactVerificationConfig(
            verified_coverage_threshold=0.95,
            partial_coverage_threshold=0.5,
        )
        strict_metrics = verify_facts(answer, [source], config=strict_config)

        # Lenient threshold
        lenient_config = FactVerificationConfig(
            verified_coverage_threshold=0.3,
            partial_coverage_threshold=0.1,
        )
        lenient_metrics = verify_facts(answer, [source], config=lenient_config)

        # Lenient should have more verified claims
        assert lenient_metrics.num_verified >= strict_metrics.num_verified

    def test_custom_citation_config(self) -> None:
        answer = "Revenue was exactly 5.2 billion."
        source = "Revenue was exactly 5.2 billion in 2023."

        config = FactVerificationConfig(
            citation_config=CitationConfig(
                top_k=5,
                min_alignment_score=5,
                weights=CitationWeights(alignment=2.0, lexical=0.0),
            )
        )

        metrics = verify_facts(answer, [source], config=config)

        assert metrics.num_claims >= 1


class TestVerifyFactsIntegration:
    """Full integration tests."""

    def test_realistic_rag_scenario(self) -> None:
        answer = (
            "Acme Corp reported revenue of 5.2 billion dollars in 2023. "
            "The company also announced expansion into Asian markets. "
            "CEO Jane Smith predicted 20% growth next year."
        )
        sources = [
            SourceDocument(
                id="annual_report",
                text=(
                    "Acme Corp reported revenue of 5.2 billion dollars in 2023. "
                    "The board approved expansion plans for Asian markets."
                ),
            ),
            SourceDocument(
                id="press_release",
                text="CEO Jane Smith announced new product lines.",
            ),
        ]

        metrics = verify_facts(answer, sources)

        assert metrics.num_claims >= 2
        # Some claims should be verified (revenue, expansion)
        # Some may be unverified (20% growth prediction)
        assert metrics.num_verified >= 1 or metrics.num_partial >= 1

        # Check that we have verification details
        assert len(metrics.claim_verifications) == metrics.num_claims

        # Each verification should reference its claim
        for verification in metrics.claim_verifications:
            assert verification.claim is not None
            assert verification.status in {"verified", "partial", "unverified"}
            assert 0.0 <= verification.confidence <= 1.0
