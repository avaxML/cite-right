"""Tests for HuggingFaceTokenizer."""

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from cite_right.text.tokenizer_huggingface import HuggingFaceTokenizer


def _load_bert_tokenizer_or_skip():
    """Load BERT tokenizer, skipping if network unavailable."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        pytest.skip(f"Unable to load pretrained tokenizer (network issue?): {e}")


@pytest.fixture
def bert_tokenizer():
    """Create a BERT tokenizer for testing."""
    return _load_bert_tokenizer_or_skip()


@pytest.fixture
def tokenizer(bert_tokenizer):
    """Create a HuggingFaceTokenizer wrapper."""
    return HuggingFaceTokenizer(bert_tokenizer, add_special_tokens=False)


class TestHuggingFaceTokenizerWithTransformers:
    """Test suite for HuggingFaceTokenizer with transformers library."""

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

    def test_token_spans_map_to_text(self, tokenizer):
        """Test that token spans correctly map back to original text."""
        text = "The quick brown fox"
        result = tokenizer.tokenize(text)

        for start, end in result.token_spans:
            token_text = text[start:end]
            assert len(token_text) > 0

    def test_unicode_text(self, tokenizer):
        """Test tokenization of unicode text."""
        text = "Hello 世界!"
        result = tokenizer.tokenize(text)

        assert result.text == text
        assert len(result.token_ids) > 0
        for start, end in result.token_spans:
            assert 0 <= start <= len(text)
            assert 0 <= end <= len(text)

    def test_with_special_tokens(self, bert_tokenizer):
        """Test tokenization with special tokens enabled."""
        tokenizer = HuggingFaceTokenizer(bert_tokenizer, add_special_tokens=True)
        result = tokenizer.tokenize("Hello world")

        # BERT adds [CLS] and [SEP] tokens
        assert len(result.token_ids) >= 2

    def test_subword_tokenization(self, tokenizer):
        """Test that subword tokenization works correctly."""
        text = "unbelievable"
        result = tokenizer.tokenize(text)

        # BERT should split this into subwords
        assert len(result.token_ids) >= 1
        # Verify spans are valid
        for start, end in result.token_spans:
            assert text[start:end]

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


class TestHuggingFaceTokenizerFromPretrained:
    """Test suite for from_pretrained class method."""

    def test_from_pretrained_basic(self):
        """Test loading tokenizer with from_pretrained."""
        try:
            tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            pytest.skip(f"Unable to load pretrained tokenizer: {e}")

        result = tokenizer.tokenize("Hello world")

        assert len(result.token_ids) > 0
        assert len(result.token_spans) == len(result.token_ids)

    def test_from_pretrained_with_options(self):
        """Test from_pretrained with custom options."""
        try:
            tokenizer = HuggingFaceTokenizer.from_pretrained(
                "bert-base-uncased",
                add_special_tokens=True,
                use_fast=True,
            )
        except Exception as e:
            pytest.skip(f"Unable to load pretrained tokenizer: {e}")

        result = tokenizer.tokenize("Test")
        assert len(result.token_ids) > 0


class TestHuggingFaceTokenizerWithTokenizers:
    """Test suite for HuggingFaceTokenizer with tokenizers library."""

    @pytest.fixture
    def fast_tokenizer(self):
        """Create a tokenizer using the tokenizers library."""
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import WordPieceTrainer

        # Create a simple WordPiece tokenizer
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # Train on some sample text
        trainer = WordPieceTrainer(
            vocab_size=1000,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )
        tokenizer.train_from_iterator(
            [
                "hello world",
                "the quick brown fox",
                "jumps over the lazy dog",
                "testing tokenization",
            ],
            trainer=trainer,
        )

        return tokenizer

    def test_basic_tokenization(self, fast_tokenizer):
        """Test basic tokenization with tokenizers library."""
        tokenizer = HuggingFaceTokenizer(fast_tokenizer)
        result = tokenizer.tokenize("hello world")

        assert len(result.token_ids) > 0
        assert len(result.token_ids) == len(result.token_spans)

    def test_empty_string(self, fast_tokenizer):
        """Test tokenization of empty string."""
        tokenizer = HuggingFaceTokenizer(fast_tokenizer)
        result = tokenizer.tokenize("")

        assert result.text == ""
        assert result.token_ids == []
        assert result.token_spans == []

    def test_spans_map_correctly(self, fast_tokenizer):
        """Test that spans map to the correct text."""
        tokenizer = HuggingFaceTokenizer(fast_tokenizer)
        text = "hello world"
        result = tokenizer.tokenize(text)

        for start, end in result.token_spans:
            token_text = text[start:end]
            assert len(token_text) > 0


class TestHuggingFaceTokenizerErrors:
    """Test error handling in HuggingFaceTokenizer."""

    def test_invalid_tokenizer_type(self):
        """Test that invalid tokenizer types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported tokenizer type"):
            HuggingFaceTokenizer("not a tokenizer")  # type: ignore

    def test_invalid_tokenizer_object(self):
        """Test that arbitrary objects raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported tokenizer type"):
            HuggingFaceTokenizer({"key": "value"})  # type: ignore
