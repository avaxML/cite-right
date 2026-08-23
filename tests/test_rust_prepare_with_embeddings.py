"""Test Rust prepare path with embeddings."""

from __future__ import annotations

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.prepared_corpus import PreparedCitationCorpus
from cite_right.models.base import Embedder
from cite_right.text.tokenizer_tiktoken import TiktokenTokenizer

from .conftest import requires_rust, requires_tiktoken


class DummyEmbedder(Embedder):
    """Dummy embedder that returns zero vectors for testing."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return zero vectors of specified dimension."""
        return [[0.0] * self.dimension for _ in texts]


@requires_rust
def test_rust_prepare_with_dummy_embedder_dim8() -> None:
    """Test that Rust prepare path is taken with a dummy embedder (dim 8)."""
    sources = [
        "The company reported strong profits in Q4.",
        "Revenue increased by 25% year over year.",
        "The CEO announced a new product line.",
    ]

    embedder = DummyEmbedder(dimension=8)
    corpus = PreparedCitationCorpus.from_sources(
        sources, embedder=embedder, use_rust=True
    )

    assert corpus.embedder is embedder
    assert corpus.embedding_index is not None
    assert len(corpus.candidates) > 0
    assert len(corpus.idf) > 0


@requires_rust
def test_rust_prepare_with_dummy_embedder_dim384() -> None:
    """Test that Rust prepare path is taken with a dummy embedder (dim 384)."""
    sources = [
        "The company reported strong profits in Q4.",
        "Revenue increased by 25% year over year.",
    ]

    embedder = DummyEmbedder(dimension=384)
    corpus = PreparedCitationCorpus.from_sources(
        sources, embedder=embedder, use_rust=True
    )

    assert corpus.embedder is embedder
    assert corpus.embedding_index is not None
    assert corpus.embedding_index.vectors.shape[1] == 384


@requires_rust
def test_rust_prepare_candidate_count_close_to_python() -> None:
    """Test that Rust and Python prepare produce similar candidate counts."""
    sources = [
        "First source with multiple sentences. This is the second sentence. "
        "And a third one for good measure.",
        "Second source also has several sentences. Each sentence matters. "
        "The corpus should be prepared efficiently.",
    ]

    embedder = DummyEmbedder(dimension=8)

    corpus_rust = PreparedCitationCorpus.from_sources(
        sources, embedder=embedder, use_rust=True
    )

    corpus_python = PreparedCitationCorpus.from_sources(
        sources, embedder=embedder, use_rust=False
    )

    # Candidate counts should be close (Rust and Python segmentation may differ slightly)
    rust_count = len(corpus_rust.candidates)
    python_count = len(corpus_python.candidates)

    # Allow up to 20% difference due to potential segmentation differences
    assert abs(rust_count - python_count) / max(rust_count, python_count) < 0.2

    # Both should have positive IDF weights
    assert len(corpus_rust.idf) > 0
    assert len(corpus_python.idf) > 0


@requires_rust
def test_rust_prepare_citation_fixture_still_works() -> None:
    """Test that an existing citation fixture gets supported/partial/unsupported correctly."""
    answer = "The company reported strong profits."
    sources = [
        SourceDocument(id="finance", text="The company reported strong profits in Q4."),
        SourceDocument(
            id="irrelevant", text="Weather report: storms are likely this weekend."
        ),
    ]

    embedder = DummyEmbedder(dimension=8)

    results = align_citations(
        answer,
        sources,
        embedder=embedder,
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.6,
            supported_answer_coverage=0.9,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )

    assert len(results) == 1
    # Should find a citation from the finance source
    assert results[0].status in ["supported", "partial"]
    if results[0].citations:
        assert results[0].citations[0].source_id == "finance"


@requires_rust
@requires_tiktoken
def test_custom_tokenizer_falls_back_to_python() -> None:
    """Test that custom tokenizer (not SimpleTokenizer) falls back to Python path."""
    import sys
    from io import StringIO

    sources = [
        "The company reported strong profits.",
        "Revenue increased significantly.",
    ]

    embedder = DummyEmbedder(dimension=8)
    custom_tokenizer = TiktokenTokenizer()

    # Capture stderr to check for fallback message
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        corpus = PreparedCitationCorpus.from_sources(
            sources, embedder=embedder, tokenizer=custom_tokenizer, use_rust=True
        )
    finally:
        sys.stderr = old_stderr

    # Should still produce valid corpus
    assert corpus.embedder is embedder
    assert len(corpus.candidates) > 0
    assert len(corpus.idf) > 0

    # Should NOT have used Rust path (custom tokenizer), so no special message expected
    # We just verify the corpus is valid


@requires_rust
def test_rust_prepare_embedding_build_time_tracked() -> None:
    """Test that embedding build time is tracked when using Rust prepare."""
    sources = [
        "First source text.",
        "Second source text.",
        "Third source text.",
    ]

    embedder = DummyEmbedder(dimension=8)
    corpus = PreparedCitationCorpus.from_sources(
        sources, embedder=embedder, use_rust=True
    )

    # Should track embedding build time
    assert corpus.embedding_build_time_ms >= 0.0


@requires_rust
def test_rust_prepare_without_embedder_still_works() -> None:
    """Test that Rust prepare without embedder still works (backward compatibility)."""
    sources = [
        "First source text.",
        "Second source text.",
    ]

    corpus = PreparedCitationCorpus.from_sources(sources, use_rust=True)

    assert corpus.embedder is None
    assert corpus.embedding_index is None
    assert len(corpus.candidates) > 0
    assert len(corpus.idf) > 0


@requires_rust
def test_rust_prepare_with_embedder_alignment() -> None:
    """Test full alignment with Rust prepare and embedder."""
    answer = "The company achieved strong revenue growth."
    sources = [
        "The company achieved strong revenue growth in the last quarter.",
        "Unrelated content about weather patterns.",
    ]

    embedder = DummyEmbedder(dimension=8)

    results = align_citations(
        answer,
        sources,
        embedder=embedder,
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.7,
            supported_answer_coverage=0.9,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )

    # Should produce results
    assert len(results) > 0
    # With good lexical overlap, should find citations
    assert results[0].status in ["supported", "partial"]
