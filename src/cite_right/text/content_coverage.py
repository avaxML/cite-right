"""Content-word overlap used to keep paraphrases from falling out of citations.

Smith-Waterman still localizes evidence. Sequential match count alone treats a
restatement as unsupported when shared content words are scattered or reordered.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

# Function words only. Polarity markers (not, no, never) stay as content.
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


def content_token_coverage(
    answer_tokens: Iterable[int],
    passage_tokens: Iterable[int],
    stopword_ids: frozenset[int],
) -> float:
    """Fraction of non-stop answer tokens that occur in the passage (with multiplicity)."""
    answer_content = [token for token in answer_tokens if token not in stopword_ids]
    if not answer_content:
        return 0.0
    available = Counter(token for token in passage_tokens if token not in stopword_ids)
    hits = 0
    for token in answer_content:
        remaining = available.get(token, 0)
        if remaining <= 0:
            continue
        hits += 1
        available[token] = remaining - 1
    return hits / len(answer_content)
