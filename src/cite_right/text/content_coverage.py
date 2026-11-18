"""Content-word overlap used to keep paraphrases from falling out of citations.

Smith-Waterman still localizes evidence. Sequential match count alone treats a
restatement as unsupported when shared content words are scattered or reordered.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Iterable, Mapping, Sequence

# Function words only. Polarity markers (not, no, never) stay as content.
from cite_right.text.term_aliases import TermAliases, canonical_token_id

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "of",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
    }
)


def stopword_token_ids(vocab: Mapping[str, int] | None) -> frozenset[int]:
    """Return tokenizer IDs for stopwords that appear in ``vocab``."""
    if not vocab:
        return frozenset()
    return frozenset(int(vocab[word]) for word in STOPWORDS if word in vocab)


def query_stopword_token_ids(
    text: str,
    token_ids: Sequence[int],
    token_spans: Sequence[tuple[int, int]],
    aliases: TermAliases | None = None,
) -> frozenset[int]:
    """Classify query stopwords and exclude alias groups that contain stopwords."""
    stopwords = frozenset(
        token_id
        for token_id, (start, end) in zip(token_ids, token_spans, strict=True)
        if _is_non_content_token(text[start:end])
    )
    return stopwords | aliases.stopword_ids if aliases else stopwords


def _is_non_content_token(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip().casefold()
    if normalized in STOPWORDS:
        return True
    return all(_is_separator(char) for char in normalized)


def _is_separator(char: str) -> bool:
    if char.isspace():
        return True
    category = unicodedata.category(char)
    if category == "Mn":
        return "VARIATION SELECTOR" in unicodedata.name(char, "")
    # Ratio signs carry quantity information even in punctuation categories.
    return category.startswith("P") and not unicodedata.name(char, "").endswith(
        ("PERCENT SIGN", "PER MILLE SIGN", "PER TEN THOUSAND SIGN")
    )


def content_token_coverage(
    answer_tokens: Iterable[int],
    passage_tokens: Iterable[int],
    stopword_ids: frozenset[int],
    aliases: Mapping[int, int] | None = None,
) -> float:
    """Fraction of non-stop answer tokens that occur in the passage (with multiplicity)."""
    answer_content = list(_content_ids(answer_tokens, stopword_ids, aliases))
    if not answer_content:
        return 0.0
    available = Counter(_content_ids(passage_tokens, stopword_ids, aliases))
    hits = 0
    for token in answer_content:
        remaining = available.get(token, 0)
        if remaining <= 0:
            continue
        hits += 1
        available[token] = remaining - 1
    return hits / len(answer_content)


def _content_ids(
    tokens: Iterable[int],
    stopword_ids: frozenset[int],
    aliases: Mapping[int, int] | None,
) -> Iterable[int]:
    content = (token for token in tokens if token not in stopword_ids)
    if not aliases:
        return content
    return (canonical_token_id(token, aliases) for token in content)
