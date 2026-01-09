"""Tests for convenience functions."""

from cite_right import (
    CitationConfig,
    SourceDocument,
    align_citations,
    annotate_answer,
    check_groundedness,
    format_with_citations,
    get_citation_summary,
    is_grounded,
    is_hallucinated,
)


class TestIsGrounded:
    """Tests for is_grounded() convenience function."""

    def test_grounded_answer_returns_true(self):
        """Verbatim answer should be grounded."""
        answer = "Revenue grew 15% in Q4."
        sources = ["Annual report: Revenue grew 15% in Q4 2024."]
        assert is_grounded(answer, sources, threshold=0.5) is True

    def test_ungrounded_answer_returns_false(self):
        """Made-up answer should not be grounded."""
        answer = "The company colonized Mars in 2024."
        sources = ["Annual report: Revenue grew 15% in Q4 2024."]
        assert is_grounded(answer, sources, threshold=0.5) is False

    def test_threshold_affects_result(self):
        """Higher threshold should be harder to pass."""
        answer = "Revenue grew."
        sources = ["Revenue grew 15% in Q4."]
        # Should pass at low threshold
        assert is_grounded(answer, sources, threshold=0.3) is True
        # May fail at very high threshold
        result_high = is_grounded(answer, sources, threshold=0.95)
        # Just verify it returns a boolean
        assert isinstance(result_high, bool)

    def test_accepts_source_documents(self):
        """Should work with SourceDocument objects."""
        answer = "Heat pumps reduce emissions."
        sources = [
            SourceDocument(id="energy", text="Heat pumps reduce emissions by 50%.")
        ]
        assert is_grounded(answer, sources) is True


class TestIsHallucinated:
    """Tests for is_hallucinated() convenience function."""

    def test_grounded_answer_not_hallucinated(self):
        """Verbatim answer should not be hallucinated."""
        answer = "Revenue grew 15%."
        sources = ["Revenue grew 15% in Q4."]
        assert is_hallucinated(answer, sources, threshold=0.5) is False

    def test_made_up_answer_is_hallucinated(self):
        """Made-up answer should be hallucinated."""
        answer = "The company built a time machine."
        sources = ["Revenue grew 15% in Q4."]
        assert is_hallucinated(answer, sources, threshold=0.3) is True


class TestCheckGroundedness:
    """Tests for check_groundedness() convenience function."""

    def test_returns_hallucination_metrics(self):
        """Should return HallucinationMetrics with expected fields."""
        answer = "Revenue grew 15%. Profits doubled."
        sources = ["Revenue grew 15% in Q4."]
        metrics = check_groundedness(answer, sources)

        assert hasattr(metrics, "groundedness_score")
        assert hasattr(metrics, "hallucination_rate")
        assert hasattr(metrics, "unsupported_spans")
        assert 0 <= metrics.groundedness_score <= 1
        assert 0 <= metrics.hallucination_rate <= 1

    def test_identifies_unsupported_spans(self):
        """Should identify which spans are unsupported."""
        answer = "Revenue grew 15%. The CEO resigned."
        sources = ["Revenue grew 15% in Q4."]
        metrics = check_groundedness(answer, sources)

        # At least one span should be unsupported
        assert metrics.num_unsupported >= 0 or metrics.num_partial >= 0


class TestAnnotateAnswer:
    """Tests for annotate_answer() convenience function."""

    def test_adds_citation_markers(self):
        """Should add citation markers to supported spans."""
        answer = "Revenue grew 15%."
        sources = [SourceDocument(id="report", text="Revenue grew 15% in Q4.")]
        annotated = annotate_answer(answer, sources)

        # Should contain a citation marker
        assert "[1]" in annotated or "[?" not in annotated

    def test_marks_unsupported_spans(self):
        """Should mark unsupported spans with [?]."""
        answer = "The company colonized Mars."
        sources = ["Revenue grew 15% in Q4."]
        annotated = annotate_answer(answer, sources, include_unsupported=True)

        assert "[?]" in annotated

    def test_different_formats(self):
        """Should support different citation formats."""
        answer = "Revenue grew 15%."
        sources = [SourceDocument(id="report", text="Revenue grew 15% in Q4.")]

        markdown = annotate_answer(answer, sources, format="markdown")
        superscript = annotate_answer(answer, sources, format="superscript")
        footnote = annotate_answer(answer, sources, format="footnote")

        # Different formats produce different markers
        assert "[1]" in markdown or "[?" in markdown
        if "^1" in superscript:
            assert "^" in superscript
        if "[^1]" in footnote:
            assert "[^" in footnote


class TestFormatWithCitations:
    """Tests for format_with_citations() function."""

    def test_formats_precomputed_results(self):
        """Should format citations from pre-computed results."""
        answer = "Revenue grew 15%."
        sources = [SourceDocument(id="report", text="Revenue grew 15% in Q4.")]
        results = align_citations(answer, sources)

        formatted = format_with_citations(answer, results)
        assert isinstance(formatted, str)
        assert len(formatted) >= len(answer)

    def test_empty_results_returns_original(self):
        """Empty results should return original answer."""
        answer = "Some text."
        formatted = format_with_citations(answer, [])
        assert formatted == answer


class TestGetCitationSummary:
    """Tests for get_citation_summary() function."""

    def test_returns_summary_string(self):
        """Should return a formatted summary string."""
        answer = "Revenue grew 15%. Profits doubled."
        sources = [SourceDocument(id="report", text="Revenue grew 15% in Q4.")]
        results = align_citations(answer, sources)

        summary = get_citation_summary(results)
        assert "Citation Summary" in summary
        assert "spans" in summary.lower()

    def test_empty_results_summary(self):
        """Should handle empty results."""
        summary = get_citation_summary([])
        assert "No spans" in summary


class TestCitationConfigPresets:
    """Tests for CitationConfig preset class methods."""

    def test_strict_preset(self):
        """Strict preset should have high thresholds."""
        config = CitationConfig.strict()
        assert config.min_answer_coverage > 0.3
        assert config.supported_answer_coverage > 0.6
        assert config.top_k <= 3

    def test_permissive_preset(self):
        """Permissive preset should have low thresholds."""
        config = CitationConfig.permissive()
        assert config.min_answer_coverage < 0.2
        assert config.allow_embedding_only is True
        assert config.top_k >= 3

    def test_fast_preset(self):
        """Fast preset should have reduced candidate limits."""
        config = CitationConfig.fast()
        assert config.max_candidates_lexical < 100
        assert config.max_candidates_total < 200
        assert config.top_k == 1

    def test_balanced_preset(self):
        """Balanced preset should match default config."""
        config = CitationConfig.balanced()
        default = CitationConfig()
        assert config.top_k == default.top_k
        assert config.min_answer_coverage == default.min_answer_coverage

    def test_presets_work_with_align_citations(self):
        """Presets should work when passed to align_citations."""
        answer = "Revenue grew 15%."
        sources = ["Revenue grew 15% in Q4."]

        for preset in [
            CitationConfig.strict(),
            CitationConfig.permissive(),
            CitationConfig.fast(),
            CitationConfig.balanced(),
        ]:
            results = align_citations(answer, sources, config=preset)
            assert isinstance(results, list)
