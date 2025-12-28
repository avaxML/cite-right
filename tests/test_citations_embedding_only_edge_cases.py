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


def test_align_citations_embedding_only_populates_evidence_spans() -> None:
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
            allow_embedding_only=True,
            min_embedding_similarity=0.5,
            supported_embedding_similarity=0.5,
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
    assert results[0].status == "supported"
    assert results[0].citations

    citation = results[0].citations[0]
    assert citation.source_id == "target"
    assert citation.components.get("embedding_only") == 1.0
    assert citation.evidence == sources[1].text
    assert len(citation.evidence_spans) == 1
    assert citation.evidence_spans[0].evidence == citation.evidence
    assert citation.evidence_spans[0].char_start == citation.char_start
    assert citation.evidence_spans[0].char_end == citation.char_end
