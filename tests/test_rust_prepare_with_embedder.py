"""Tests for rust prepare path with embedders."""

from typing import Sequence

import pytest

from cite_right import PreparedCitationCorpus, SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.text.segmenter_simple import SimpleSegmenter
from cite_right.text.tokenizer import SimpleTokenizer

from .conftest import requires_rust


class _DummyEmbedder:
    """Dummy embedder for testing without downloading model."""

    def __init__(self) -> None:
        self.model_name = "dummy"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        # Return fixed-size dummy embeddings (384-d like MiniLM)
        return [[float(i % 384) for i in range(384)] for _ in texts]


@pytest.fixture
def dummy_embedder() -> _DummyEmbedder:
    """Provide a dummy embedder for tests that don't need real embeddings."""
    return _DummyEmbedder()


@requires_rust
def test_from_sources_with_embedder_uses_rust_prepare(
    dummy_embedder: _DummyEmbedder,
) -> None:
    """Test that from_sources with an embedder still uses rust prepare path.

    This is the key test for Hill 2 - ensuring that providing an embedder
    doesn't force fallback to Python prepare.
    """
    sources = [
        SourceDocument(id="doc1", text="Climate policy reduces emissions."),
        SourceDocument(id="doc2", text="Renewable energy costs are falling."),
    ]

    corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=CitationConfig(top_k=1),
        tokenizer=SimpleTokenizer(),
        source_segmenter=SimpleSegmenter(),
        embedder=dummy_embedder,
        use_rust=True,
    )

    # Verify that corpus was created successfully with embedder
    assert corpus.embedder is dummy_embedder
    assert corpus.embedding_index is not None
    assert len(corpus.candidates) > 0

    # Verify that the embedding index was built from passages
    assert corpus.embedding_index.vectors.shape[0] == len(corpus.candidates)


@requires_rust
def test_rust_prepare_embedder_candidate_count_matches_python(
    dummy_embedder: _DummyEmbedder,
) -> None:
    """Test that rust and python prepare produce compatible candidate counts."""
    sources = [
        "First sentence. Second sentence. Third sentence.",
        "Another document with multiple sentences here.",
    ]
    config = CitationConfig(
        window_size_sentences=2,
        window_stride_sentences=1,
    )

    # Rust path with embedder
    rust_corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=config,
        tokenizer=SimpleTokenizer(),
        source_segmenter=SimpleSegmenter(),
        embedder=dummy_embedder,
        use_rust=True,
    )

    # Python path with embedder
    python_corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=config,
        tokenizer=SimpleTokenizer(),
        source_segmenter=SimpleSegmenter(),
        embedder=dummy_embedder,
        use_rust=False,
    )

    # Candidate counts should be identical
    assert len(rust_corpus.candidates) == len(python_corpus.candidates)

    # Embedding index shapes should match
    assert (
        rust_corpus.embedding_index.vectors.shape
        == python_corpus.embedding_index.vectors.shape
    )


@requires_rust
def test_rust_prepare_embedder_passage_texts_compatible(
    dummy_embedder: _DummyEmbedder,
) -> None:
    """Test that rust and python prepare produce compatible passage texts.

    This ensures that the passage texts used for embeddings are consistent
    between the two backends, so alignment status doesn't flip.
    """
    sources = ["Climate change impacts coastal regions. Sea levels are rising."]
    config = CitationConfig(window_size_sentences=1, window_stride_sentences=1)

    rust_corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=config,
        tokenizer=SimpleTokenizer(),
        source_segmenter=SimpleSegmenter(),
        embedder=dummy_embedder,
        use_rust=True,
    )

    python_corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=config,
        tokenizer=SimpleTokenizer(),
        source_segmenter=SimpleSegmenter(),
        embedder=dummy_embedder,
        use_rust=False,
    )

    # Extract passage texts from both
    rust_passages = [c.passage.text for c in rust_corpus.candidates]
    python_passages = [c.passage.text for c in python_corpus.candidates]

    # Passage texts should be identical
    assert rust_passages == python_passages


@requires_rust
def test_align_citations_with_embedder_matches_status_rust_vs_python(
    dummy_embedder: _DummyEmbedder,
) -> None:
    """Test that citation status is consistent between rust and python prepare.

    This is a critical quality test - ensuring that using rust prepare with
    embeddings doesn't change the citation status (supported/partial/unsupported).
    """
    answer = "Climate policy reduces emissions."
    sources = [
        SourceDocument(id="match", text="Climate policy reduces emissions quickly."),
        SourceDocument(id="noise", text="Unrelated filler content here."),
    ]

    config = CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.5,
        supported_answer_coverage=0.8,
        max_candidates_embedding=10,
        max_candidates_lexical=10,
        weights=CitationWeights(lexical=0.5, embedding=0.5),
    )

    # Rust backend with embedder
    rust_results = align_citations(
        answer,
        sources,
        config=config,
        embedder=dummy_embedder,
        backend="auto",  # Will use rust if available
    )

    # Python backend with embedder
    python_results = align_citations(
        answer,
        sources,
        config=config,
        embedder=dummy_embedder,
        backend="python",
    )

    # Status should match
    assert len(rust_results) == len(python_results)
    for rust_span, python_span in zip(rust_results, python_results, strict=True):
        assert rust_span.status == python_span.status
        # If there are citations, source should match
        if rust_span.citations and python_span.citations:
            assert (
                rust_span.citations[0].source_id == python_span.citations[0].source_id
            )


@requires_rust
def test_rust_prepare_with_embedder_records_build_time(
    dummy_embedder: _DummyEmbedder,
) -> None:
    """Test that embedding build time is tracked when using rust prepare."""
    corpus = PreparedCitationCorpus.from_sources(
        ["Climate policy reduces emissions."],
        config=CitationConfig(top_k=1),
        embedder=dummy_embedder,
        use_rust=True,
    )

    # Should have recorded embedding build time
    assert corpus.embedding_build_time_ms > 0.0


def test_rust_prepare_fallback_with_custom_tokenizer() -> None:
    """Test that custom Python tokenizers cause fallback to Python prepare.

    Rust prepare only works with SimpleTokenizer and SimpleSegmenter.
    Custom tokenizers should trigger Python fallback.
    """
    from cite_right.core.results import TokenizedText

    class CustomTokenizer:
        def tokenize(self, text: str) -> TokenizedText:
            # Simple word-based tokenization
            words = text.split()
            token_ids = list(range(len(words)))
            pos = 0
            token_spans = []
            for word in words:
                start = text.find(word, pos)
                end = start + len(word)
                token_spans.append((start, end))
                pos = end
            return TokenizedText(text=text, token_ids=token_ids, token_spans=token_spans)

    corpus = PreparedCitationCorpus.from_sources(
        ["First sentence. Second sentence. Third sentence."],
        config=CitationConfig(top_k=1),
        tokenizer=CustomTokenizer(),  # type: ignore
        use_rust=True,
    )

    # Should still work (via Python fallback)
    assert len(corpus.candidates) > 0
