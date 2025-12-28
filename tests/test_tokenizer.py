from cite_right.text.tokenizer import SimpleTokenizer


def test_tokenizer_spans_and_ids() -> None:
    text = "Hello, WORLD! Hello"
    tokenizer = SimpleTokenizer()
    tokenized = tokenizer.tokenize(text)

    assert tokenized.token_spans == [(0, 5), (7, 12), (14, 19)]
    assert [text[start:end] for start, end in tokenized.token_spans] == [
        "Hello",
        "WORLD",
        "Hello",
    ]
    assert tokenized.token_ids[0] == tokenized.token_ids[2]
    assert tokenized.token_ids[0] != tokenized.token_ids[1]


def test_tokenizer_repeated_tokens() -> None:
    text = "hi Hi HI"
    tokenizer = SimpleTokenizer()
    tokenized = tokenizer.tokenize(text)
    assert len(set(tokenized.token_ids)) == 1


def test_tokenizer_normalizes_percent_and_numbers() -> None:
    tokenizer = SimpleTokenizer()
    left = tokenizer.tokenize("34%")
    right = tokenizer.tokenize("34 percent")
    assert left.token_ids[0] == right.token_ids[0]
    assert left.token_ids[1] == right.token_ids[1]

    with_commas = tokenizer.tokenize("1,200")
    plain = tokenizer.tokenize("1200")
    assert with_commas.token_ids == plain.token_ids


def test_tokenizer_keeps_hyphens_and_apostrophes_inside_tokens() -> None:
    tokenizer = SimpleTokenizer()
    text = "State-of-the-art company’s device"
    tokenized = tokenizer.tokenize(text)
    tokens = [text[start:end] for start, end in tokenized.token_spans]
    assert "State-of-the-art" in tokens
    assert "company’s" in tokens
