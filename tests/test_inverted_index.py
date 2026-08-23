"""Tests for inverted index retrieval."""


from cite_right import CitationConfig, PreparedCitationCorpus, align_citations

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
    
    # Check that inverted index was created (Rust object, opaque to Python)
    assert corpus.inverted_index is not None
    
    # Verify it's a Rust InvertedIndex object by checking it has a query method
    assert hasattr(corpus.inverted_index, "query")
    
    # Test that we can query it successfully
    test_tokens = [1, 2, 3]  # Some token IDs
    result = corpus.inverted_index.query(test_tokens, 10)
    assert isinstance(result, list)
    # Result should be a list of candidate indices
    for idx in result:
        assert isinstance(idx, int)
        assert 0 <= idx < len(corpus.candidates)


@requires_rust
def test_inverted_index_stays_in_rust() -> None:
    """Verify that the inverted index is not rehydrated from Python on each query."""
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
    
    # The index should be the same object across queries
    index1 = corpus.inverted_index
    index2 = corpus.inverted_index
    assert index1 is index2  # Same Python object reference
    
    # Query multiple times - index stays in Rust
    test_tokens = [1, 2, 3]
    result1 = corpus.inverted_index.query(test_tokens, 10)
    result2 = corpus.inverted_index.query(test_tokens, 10)
    
    # Results should be deterministic
    assert result1 == result2


@requires_rust
def test_inverted_index_improves_retrieval() -> None:
    """Verify that inverted index-based retrieval works correctly."""
    sources = ["Revenue grew 15% in Q4 2024."] * 100  # Many similar candidates
    answer = "Revenue grew 15% in Q4."
    
    results = align_citations(answer, sources, config=CitationConfig(top_k=1))
    
    assert len(results) == 1
    assert results[0].status == "supported"
    assert len(results[0].citations) > 0


@requires_rust
def test_inverted_index_never_returns_empty_when_tokens_exist() -> None:
    """Verify that inverted index never returns empty seeds when tokens exist."""
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
    
    # Query with tokens that exist in the corpus
    answer = "Revenue grew in Q4."
    tokenized = corpus.tokenizer.tokenize(answer)
    
    # Should never get empty seeds when tokens exist
    seed_candidates = corpus.inverted_index.query(tokenized.token_ids, 100)
    assert len(seed_candidates) > 0, "Should not return empty seeds when query tokens exist"
    
    # Should find the relevant candidate
    revenue_candidates = [c.global_index for c in corpus.candidates if c.source.source_index == 0]
    assert any(idx in seed_candidates for idx in revenue_candidates)


@requires_rust
def test_inverted_index_uses_intersection() -> None:
    """Verify that inverted index uses intersection with rare tokens."""
    # Create sources where a unique token appears in only one passage
    sources = [
        "The company reported strong growth.",  # Common words
        "Revenue increased significantly.",      # Common words
        "Xylophone sales doubled.",             # Unique word "Xylophone"
    ]
    
    corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=CitationConfig(),
        use_rust=True,
    )
    
    # Tokenize a query with the unique word
    answer = "Xylophone sales doubled."
    tokenized = corpus.tokenizer.tokenize(answer)
    
    # Query the index
    seed_candidates = corpus.inverted_index.query(tokenized.token_ids, 10)
    
    # Should only seed the passage containing "Xylophone" (or very few passages)
    # With intersection, we should get much fewer than all candidates
    assert len(seed_candidates) < len(corpus.candidates)
    
    # The passage with "Xylophone" should be in the seeds
    # (candidate 2 corresponds to source index 2)
    xylophone_candidates = [c.global_index for c in corpus.candidates if c.source.source_index == 2]
    if xylophone_candidates:
        # At least one xylophone candidate should be in seeds
        assert any(idx in seed_candidates for idx in xylophone_candidates)


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
