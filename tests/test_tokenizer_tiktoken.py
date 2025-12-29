"""Tests for TiktokenTokenizer."""

import pytest

tiktoken = pytest.importorskip("tiktoken")

from cite_right.text.tokenizer_tiktoken import TiktokenTokenizer


def _get_encoding_or_skip(encoding_name: str = "cl100k_base"):
    """Helper to get tiktoken encoding, skipping test if unavailable."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        pytest.skip(f"Unable to load tiktoken encoding (network issue?): {e}")


@pytest.fixture
def encoding():
    """Fixture providing tiktoken encoding."""
    return _get_encoding_or_skip("cl100k_base")


@pytest.fixture
def tokenizer(encoding):
    """Fixture providing a TiktokenTokenizer with preloaded encoding."""
    return TiktokenTokenizer(encoding=encoding)


class TestTiktokenTokenizer:
    """Test suite for TiktokenTokenizer."""

    def test_basic_tokenization(self, tokenizer):
        """Test basic tokenization produces valid results."""
        result = tokenizer.tokenize("Hello, world!")

        assert result.text == "Hello, world!"
        assert len(result.token_ids) > 0
        assert len(result.token_ids) == len(result.token_spans)

    def test_empty_string(self, tokenizer):
        """Test tokenization of empty string."""
        result = tokenizer.tokenize("")

        assert result.text == ""
        assert result.token_ids == []
        assert result.token_spans == []

    def test_token_spans_cover_text(self, tokenizer, encoding):
        """Test that token spans correctly map back to original text."""
        text = "The quick brown fox jumps over the lazy dog."
        result = tokenizer.tokenize(text)

        # Verify each span extracts the correct substring
        for i, (start, end) in enumerate(result.token_spans):
            token_text = text[start:end]
            assert len(token_text) > 0, f"Token {i} has empty span"
            # Decode the token and verify it matches
            decoded = encoding.decode([result.token_ids[i]])
            assert decoded == token_text, f"Token {i}: {decoded!r} != {token_text!r}"

    def test_unicode_text(self, tokenizer):
        """Test tokenization of unicode text."""
        text = "Hello 世界! 🎉 Café"
        result = tokenizer.tokenize(text)

        assert result.text == text
        assert len(result.token_ids) > 0
        # Verify spans are valid character indices
        for start, end in result.token_spans:
            assert 0 <= start < len(text)
            assert 0 < end <= len(text)
            assert start < end

    def test_different_encodings(self, encoding):
        """Test using different tiktoken encodings."""
        text = "Hello, world!"

        # Test cl100k_base (GPT-4)
        tokenizer_cl100k = TiktokenTokenizer(encoding=encoding)
        result_cl100k = tokenizer_cl100k.tokenize(text)
        assert tokenizer_cl100k.encoding_name == "cl100k_base"

        # Test p50k_base (Codex) - may fail if not cached
        try:
            p50k_encoding = tiktoken.get_encoding("p50k_base")
            tokenizer_p50k = TiktokenTokenizer(encoding=p50k_encoding)
            result_p50k = tokenizer_p50k.tokenize(text)
            assert tokenizer_p50k.encoding_name == "p50k_base"
            assert len(result_p50k.token_ids) > 0
        except Exception:
            pass  # Skip p50k_base if not available

        # Different encodings may produce different token counts
        assert len(result_cl100k.token_ids) > 0

    def test_custom_encoding_object(self, encoding):
        """Test passing a pre-initialized encoding object."""
        tokenizer = TiktokenTokenizer(encoding=encoding)

        result = tokenizer.tokenize("Test text")
        assert len(result.token_ids) > 0
        assert tokenizer.encoding_name == "cl100k_base"

    def test_special_characters(self, tokenizer):
        """Test tokenization of text with special characters."""
        text = "Price: $100.50 (20% off)"
        result = tokenizer.tokenize(text)

        # Reconstruct text from spans
        reconstructed = "".join(text[s:e] for s, e in result.token_spans)
        assert reconstructed == text

    def test_multiline_text(self, tokenizer):
        """Test tokenization of multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        result = tokenizer.tokenize(text)

        assert result.text == text
        # Reconstruct and verify
        reconstructed = "".join(text[s:e] for s, e in result.token_spans)
        assert reconstructed == text

    def test_whitespace_handling(self, tokenizer):
        """Test that whitespace is preserved in spans."""
        text = "word1  word2   word3"
        result = tokenizer.tokenize(text)

        # Full text should be reconstructable from spans
        reconstructed = "".join(text[s:e] for s, e in result.token_spans)
        assert reconstructed == text

    def test_consistent_tokenization(self, tokenizer):
        """Test that same text produces same tokens."""
        text = "Consistency test"

        result1 = tokenizer.tokenize(text)
        result2 = tokenizer.tokenize(text)

        assert result1.token_ids == result2.token_ids
        assert result1.token_spans == result2.token_spans

    def test_spans_are_non_overlapping(self, tokenizer):
        """Test that token spans don't overlap."""
        text = "The quick brown fox"
        result = tokenizer.tokenize(text)

        for i in range(len(result.token_spans) - 1):
            _, end1 = result.token_spans[i]
            start2, _ = result.token_spans[i + 1]
            assert end1 <= start2, f"Spans overlap at index {i}"

    def test_spans_cover_full_text(self, tokenizer):
        """Test that spans cover the entire input text."""
        text = "Complete coverage test"
        result = tokenizer.tokenize(text)

        # First span should start at 0
        assert result.token_spans[0][0] == 0
        # Last span should end at len(text)
        assert result.token_spans[-1][1] == len(text)
