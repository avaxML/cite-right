"""Tests for error conditions and edge cases in cite-right."""

import pytest

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.text.segmenter_simple import SimpleSegmenter
from cite_right.text.tokenizer import SimpleTokenizer


class TestAlignCitationsErrorConditions:
    """Test error handling in align_citations function."""

    def test_empty_sources_returns_unsupported(self) -> None:
        """Verify empty source list returns unsupported status."""
        answer = "Some answer text."
        results = align_citations(answer, [])

        assert len(results) == 1, "Expected one result for empty sources"
        assert results[0].status == "unsupported", (
            "Empty sources should result in unsupported"
        )
        assert results[0].citations == [], "Empty sources should have no citations"

    def test_empty_answer_returns_empty_results(self) -> None:
        """Verify empty answer returns empty results."""
        sources = [SourceDocument(id="doc", text="Some source text.")]
        results = align_citations("", sources)

        assert results == [], "Empty answer should return empty results"

    def test_whitespace_only_answer_returns_empty_results(self) -> None:
        """Verify whitespace-only answer returns empty results."""
        sources = [SourceDocument(id="doc", text="Some source text.")]
        results = align_citations("   \n\t  ", sources)

        assert results == [], "Whitespace-only answer should return empty results"

    def test_source_with_empty_text_is_skipped(self) -> None:
        """Verify sources with empty text don't cause errors."""
        answer = "The answer text."
        sources = [
            SourceDocument(id="empty", text=""),
            SourceDocument(id="valid", text="The answer text."),
        ]

        results = align_citations(answer, sources)

        assert len(results) == 1
        # Should find citation in valid source, not empty one
        if results[0].citations:
            assert results[0].citations[0].source_id == "valid"

    def test_whitespace_only_source_is_handled(self) -> None:
        """Verify sources with only whitespace don't cause errors."""
        answer = "Test answer."
        sources = [
            SourceDocument(id="whitespace", text="   \n\t  "),
            SourceDocument(id="valid", text="Test answer."),
        ]

        results = align_citations(answer, sources)

        assert len(results) >= 1
        # Should not crash and should handle gracefully

    def test_very_short_answer_handled(self) -> None:
        """Verify very short answers (single word) are handled."""
        answer = "Hi"
        sources = [SourceDocument(id="doc", text="Hi there.")]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        # Should not raise exception
        results = align_citations(answer, sources, config=config)
        assert isinstance(results, list)

    def test_unicode_answer_and_sources(self) -> None:
        """Verify unicode text is handled correctly."""
        answer = "日本語テスト 中文测试 한국어테스트"
        sources = [
            SourceDocument(id="unicode", text="日本語テスト 中文测试 한국어테스트")
        ]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        # Should not raise exception
        results = align_citations(answer, sources, config=config)
        assert isinstance(results, list)

    def test_special_characters_in_text(self) -> None:
        """Verify special characters don't cause issues."""
        answer = "Price is $100.00 (50% off!) & free shipping."
        sources = [
            SourceDocument(
                id="special", text="Price is $100.00 (50% off!) & free shipping."
            )
        ]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        # Should not raise exception
        results = align_citations(answer, sources, config=config)
        assert isinstance(results, list)

    def test_very_long_text_handled(self) -> None:
        """Verify very long texts don't cause memory issues."""
        # Create a moderately long text (not too long to slow down tests)
        long_text = "This is a sentence. " * 100
        answer = long_text
        sources = [SourceDocument(id="long", text=long_text)]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.1,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        # Should complete without memory issues
        results = align_citations(answer, sources, config=config)
        assert isinstance(results, list)

    def test_invalid_backend_raises_error(self) -> None:
        """Verify invalid backend parameter raises appropriate error."""
        answer = "Test answer."
        sources = [SourceDocument(id="doc", text="Test answer.")]

        with pytest.raises((ValueError, KeyError)):
            align_citations(answer, sources, backend="invalid_backend")  # type: ignore


class TestConfigValidation:
    """Test configuration parameter validation."""

    def test_config_with_zero_top_k(self) -> None:
        """Verify top_k=0 is handled gracefully."""
        answer = "Test answer."
        sources = [SourceDocument(id="doc", text="Test answer.")]

        config = CitationConfig(
            top_k=0,  # Edge case
            min_alignment_score=1,
            min_answer_coverage=0.5,
        )

        # Should not crash
        results = align_citations(answer, sources, config=config)
        assert isinstance(results, list)

    def test_config_with_high_min_alignment_score(self) -> None:
        """Verify very high min_alignment_score results in unsupported."""
        answer = "Test answer."
        sources = [SourceDocument(id="doc", text="Test answer.")]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=999999,  # Impossibly high
            min_answer_coverage=0.5,
        )

        results = align_citations(answer, sources, config=config)
        assert len(results) == 1
        # With such a high threshold, nothing should be supported
        assert results[0].status == "unsupported"


class TestSimpleTokenizerEdgeCases:
    """Test edge cases in SimpleTokenizer."""

    def test_tokenize_empty_string(self) -> None:
        """Verify empty string tokenization."""
        tokenizer = SimpleTokenizer()
        result = tokenizer.tokenize("")

        assert result.text == ""
        assert result.token_ids == []
        assert result.token_spans == []

    def test_tokenize_whitespace_only(self) -> None:
        """Verify whitespace-only string tokenization."""
        tokenizer = SimpleTokenizer()
        result = tokenizer.tokenize("   \t\n  ")

        assert result.token_ids == []
        assert result.token_spans == []

    def test_tokenize_single_character(self) -> None:
        """Verify single character tokenization."""
        tokenizer = SimpleTokenizer()
        result = tokenizer.tokenize("a")

        assert len(result.token_ids) == 1
        assert result.token_spans == [(0, 1)]

    def test_tokenize_only_punctuation(self) -> None:
        """Verify punctuation-only string tokenization."""
        tokenizer = SimpleTokenizer()
        result = tokenizer.tokenize(".,!?")

        # Punctuation should be stripped, resulting in no tokens
        assert result.token_ids == []


class TestSimpleSegmenterEdgeCases:
    """Test edge cases in SimpleSegmenter."""

    def test_segment_empty_string(self) -> None:
        """Verify empty string segmentation."""
        segmenter = SimpleSegmenter()
        segments = segmenter.segment("")

        assert segments == []

    def test_segment_no_sentence_boundary(self) -> None:
        """Verify text without sentence boundaries."""
        segmenter = SimpleSegmenter()
        text = "No sentence boundary here"
        segments = segmenter.segment(text)

        # Should return the whole text as one segment
        assert len(segments) == 1
        assert segments[0].text == text

    def test_segment_multiple_newlines(self) -> None:
        """Verify multiple newlines are handled."""
        segmenter = SimpleSegmenter()
        text = "First.\n\n\n\nSecond."
        segments = segmenter.segment(text)

        assert len(segments) == 2
        assert segments[0].text == "First."
        assert segments[1].text == "Second."

    def test_segment_preserves_offsets(self) -> None:
        """Verify segment offsets correctly map back to original text."""
        segmenter = SimpleSegmenter()
        text = "First sentence. Second sentence."
        segments = segmenter.segment(text)

        for segment in segments:
            extracted = text[segment.doc_char_start : segment.doc_char_end]
            assert extracted == segment.text, (
                f"Offset mismatch: expected '{segment.text}' but got '{extracted}'"
            )


class TestSourceDocumentValidation:
    """Test SourceDocument edge cases."""

    def test_source_document_with_none_id(self) -> None:
        """Verify SourceDocument handles None-like id gracefully."""
        # ID should be a string, test with empty string
        doc = SourceDocument(id="", text="Some text.")
        assert doc.id == ""
        assert doc.text == "Some text."

    def test_source_document_text_types(self) -> None:
        """Verify SourceDocument text attribute works correctly."""
        doc = SourceDocument(id="test", text="Normal text.")
        assert doc.text == "Normal text."

        # Test with unicode
        doc_unicode = SourceDocument(id="unicode", text="日本語テスト")
        assert doc_unicode.text == "日本語テスト"
