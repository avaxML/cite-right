"""Tests for SimpleTokenizer."""

from cite_right.text.tokenizer import SimpleTokenizer


def test_tokenizer_spans_and_ids() -> None:
    """Verify tokenizer produces correct spans and assigns consistent IDs."""
    text = "Hello, WORLD! Hello"
    tokenizer = SimpleTokenizer()
    tokenized = tokenizer.tokenize(text)

    assert tokenized.token_spans == [
        (0, 5),
        (7, 12),
        (14, 19),
    ], f"Unexpected spans: {tokenized.token_spans}"
    assert [text[start:end] for start, end in tokenized.token_spans] == [
        "Hello",
        "WORLD",
        "Hello",
    ], "Span text extraction mismatch"
    assert tokenized.token_ids[0] == tokenized.token_ids[2], (
        "Same words should have same IDs (case-insensitive)"
    )
    assert tokenized.token_ids[0] != tokenized.token_ids[1], (
        "Different words should have different IDs"
    )


def test_tokenizer_repeated_tokens() -> None:
    """Verify repeated tokens (case-insensitive) get the same ID."""
    text = "hi Hi HI"
    tokenizer = SimpleTokenizer()
    tokenized = tokenizer.tokenize(text)

    assert len(set(tokenized.token_ids)) == 1, (
        "All case variations should map to same ID"
    )


def test_tokenizer_normalizes_percent_and_numbers() -> None:
    """Verify tokenizer normalizes percent symbols and number formats."""
    tokenizer = SimpleTokenizer()

    # % should be normalized to 'percent'
    left = tokenizer.tokenize("34%")
    right = tokenizer.tokenize("34 percent")
    assert left.token_ids[0] == right.token_ids[0], "Number should match across formats"
    assert left.token_ids[1] == right.token_ids[1], (
        "'%' and 'percent' should have same ID"
    )

    # Commas in numbers should be normalized
    with_commas = tokenizer.tokenize("1,200")
    plain = tokenizer.tokenize("1200")
    assert with_commas.token_ids == plain.token_ids, (
        "Numbers with/without commas should match"
    )


def test_tokenizer_keeps_hyphens_and_apostrophes_inside_tokens() -> None:
    """Verify tokenizer preserves hyphens and apostrophes within words."""
    tokenizer = SimpleTokenizer()
    text = "State-of-the-art company's device"
    tokenized = tokenizer.tokenize(text)
    tokens = [text[start:end] for start, end in tokenized.token_spans]

    assert "State-of-the-art" in tokens, "Hyphenated compound should be single token"
    assert "company's" in tokens, "Possessive should be single token"


def test_tokenizer_handles_unicode() -> None:
    """Verify tokenizer handles unicode text correctly."""
    tokenizer = SimpleTokenizer()
    text = "café résumé naïve"
    tokenized = tokenizer.tokenize(text)

    assert len(tokenized.token_ids) == 3, (
        f"Expected 3 tokens, got {len(tokenized.token_ids)}"
    )
    # Verify spans map back correctly
    for start, end in tokenized.token_spans:
        extracted = text[start:end]
        assert len(extracted) > 0, "Token span should not be empty"


def test_tokenizer_handles_mixed_punctuation() -> None:
    """Verify tokenizer handles various punctuation correctly."""
    tokenizer = SimpleTokenizer()
    text = "Hello! World? Yes... No—maybe."
    tokenized = tokenizer.tokenize(text)

    tokens = [text[start:end] for start, end in tokenized.token_spans]
    assert "Hello" in tokens
    assert "World" in tokens
    assert "Yes" in tokens
    assert "No" in tokens
    assert "maybe" in tokens
