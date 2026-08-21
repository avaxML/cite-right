from __future__ import annotations

from typing import Sequence

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.models.base import Embedder


class KeywordEmbedder:
    """A deterministic embedder that keys off a substring.

    This is used to test the embedding-only citation path without external
    dependencies.
    """

    def __init__(self, keyword: str) -> None:
        self._keyword = keyword.casefold()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if self._keyword in text.casefold():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_align_citations_embedding_only_returns_retrieval_support_only() -> None:
    embedder: Embedder = KeywordEmbedder("assertions")

    sources = [
        SourceDocument(
            id="noise",
            text="Weather report: storms are likely this weekend.",
        ),
        SourceDocument(
            id="target",
            text=(
                "We propose LM Assertions, expressed as boolean conditions, and integrate them "
                "into DSPy."
            ),
        ),
    ]
    answer = "LM Assertions are boolean conditions that improve reliability."

    results = align_citations(
        answer,
        sources,
        embedder=embedder,
        config=CitationConfig(
            top_k=1,
            max_candidates_lexical=0,
            max_candidates_embedding=10,
            max_candidates_total=10,
            min_embedding_similarity=0.5,
            min_alignment_score=10_000,
            min_answer_coverage=1.0,
            weights=CitationWeights(
                alignment=0.0,
                answer_coverage=0.0,
                lexical=0.0,
                embedding=1.0,
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []
    assert len(results[0].retrieval_support) == 1

    support = results[0].retrieval_support[0]
    assert support.source_id == "target"
    assert support.source_index == 1
    assert support.passage_text == sources[1].text
    assert support.passage_char_start == 0
    assert support.passage_char_end == len(sources[1].text)
    assert support.embedding_score >= 0.5
    assert support.lexical_score == 0.0


def test_align_citations_embedding_support_does_not_upgrade_exact_status() -> None:
    embedder: Embedder = KeywordEmbedder("earnings")

    results = align_citations(
        "The firm posted robust earnings.",
        [
            SourceDocument(id="noise", text="Weather report: storms are likely."),
            SourceDocument(
                id="finance", text="Robust earnings were reported this quarter."
            ),
        ],
        embedder=embedder,
        config=CitationConfig(
            top_k=1,
            max_candidates_lexical=0,
            max_candidates_embedding=10,
            max_candidates_total=10,
            min_embedding_similarity=0.5,
            min_alignment_score=10_000,
            min_answer_coverage=1.0,
            weights=CitationWeights(
                alignment=0.0,
                answer_coverage=0.0,
                lexical=0.0,
                embedding=1.0,
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []
    assert [support.source_id for support in results[0].retrieval_support] == [
        "finance"
    ]


def test_align_citations_retrieval_support_respects_own_limit() -> None:
    embedder: Embedder = KeywordEmbedder("target")

    sources = [
        SourceDocument(id=f"doc-{idx}", text=f"target passage {idx}")
        for idx in range(5)
    ]

    results = align_citations(
        "target",
        sources,
        embedder=embedder,
        config=CitationConfig(
            top_k=1,
            max_candidates_lexical=0,
            max_candidates_embedding=10,
            max_candidates_total=10,
            max_retrieval_support=2,
            min_embedding_similarity=0.5,
            min_alignment_score=10_000,
            min_answer_coverage=1.0,
            weights=CitationWeights(
                alignment=0.0,
                answer_coverage=0.0,
                lexical=0.0,
                embedding=1.0,
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []
    assert len(results[0].retrieval_support) == 2
