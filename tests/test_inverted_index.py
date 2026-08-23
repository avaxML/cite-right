"""Tests for inverted index retrieval."""

import pytest

from cite_right import PreparedCitationCorpus, align_citations, CitationConfig
from cite_right.text.tokenizer import SimpleTokenizer
from cite_right.text.segmenter_simple import SimpleSegmenter

from .conftest import requires_rust


@requires_rust
def test_inverted_index_is_built_with_rust_prepare() -> None:
    """Verify that the inverted index is built during Rust preparation."""
    sources = [
        "Revenue grew 15% in Q4.",
        "Costs decreased by 10%.",
        "Profit margins improved significantly.",
    ]
    
    corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=CitationConfig(),
        use_rust=True,
    )
    
    # Check that inverted index was created
    assert corpus.inverted_index is not None
    assert len(corpus.inverted_index) > 0
    
    # Check structure: should be dict[int, list[tuple[int, int, int, int]]]
    for token_id, postings in corpus.inverted_index.items():
        assert isinstance(token_id, int)
        assert isinstance(postings, list)
        if postings:
            candidate_idx, token_pos, char_start, char_end = postings[0]
            assert isinstance(candidate_idx, int)
            assert isinstance(token_pos, int)
            assert isinstance(char_start, int)
            assert isinstance(char_end, int)


@requires_rust
def test_inverted_index_improves_retrieval() -> None:
    """Verify that inverted index-based retrieval works correctly."""
    sources = ["Revenue grew 15% in Q4 2024."] * 100  # Many similar candidates
    answer = "Revenue grew 15% in Q4."
    
    results = align_citations(answer, sources, config=CitationConfig(top_k=1))
    
    assert len(results) == 1
    assert results[0].status == "supported"
    assert len(results[0].citations) > 0


def test_python_fallback_without_index() -> None:
    """Verify that Python fallback works without inverted index."""
    sources = ["Revenue grew 15% in Q4 2024."]
    answer = "Revenue grew 15% in Q4."
    
    # Force Python path
    corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=CitationConfig(),
        use_rust=False,
    )
    
    # Python path should not have inverted index
    assert corpus.inverted_index is None
    
    # But should still work
    results = corpus.align(answer)
    assert len(results) == 1
    assert results[0].status == "supported"


@requires_rust
def test_inverted_index_with_embedder() -> None:
    """Verify that inverted index is not built when embedder is provided."""
    from cite_right.models.base import Embedder
    
    class SimpleEmbedder(Embedder):
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]
    
    sources = ["Revenue grew 15% in Q4 2024."]
    
    corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=CitationConfig(),
        embedder=SimpleEmbedder(),
        use_rust=True,
    )
    
    # Currently, Rust prepare is skipped when embedder is provided
    # So inverted_index should be None
    # (This could be changed in the future to build both)
    assert corpus.inverted_index is None
