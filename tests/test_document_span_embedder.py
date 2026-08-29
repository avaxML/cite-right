"""Tests for document-level span embedder."""

import os

import numpy as np
import pytest

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.prepared_corpus import PreparedCitationCorpus
from cite_right.text.passage import Passage

_SMALL_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

if os.environ.get("CITE_RIGHT_RUN_EMBEDDINGS_TESTS") != "1":
    pytest.skip(
        "Set CITE_RIGHT_RUN_EMBEDDINGS_TESTS=1 to run embeddings tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def document_span_embedder():
    pytest.importorskip("sentence_transformers")
    from cite_right.models.document_span_embedder import DocumentSpanEmbedder

    try:
        return DocumentSpanEmbedder(_SMALL_MODEL)
    except OSError as exc:
        pytest.skip(f"Embedding model is not available offline: {exc}")


def test_document_span_embedder_short_source_single_passage(
    document_span_embedder,
) -> None:
    """Test encoding a single passage from a short document."""
    source_text = "The company reported strong profits."
    passage = Passage(
        doc_char_start=0,
        doc_char_end=len(source_text),
        segment_start=0,
        segment_end=1,
        source_text=source_text,
    )

    embeddings = document_span_embedder.encode_document_spans(source_text, [passage])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384  # MiniLM-L3-v2 embedding dimension
    assert all(np.isfinite(embeddings[0]))
    assert not all(v == 0.0 for v in embeddings[0])  # Non-zero embedding


def test_document_span_embedder_multiple_passages(
    document_span_embedder,
) -> None:
    """Test encoding multiple passages from a single document."""
    source_text = "First sentence here. Second sentence here. Third sentence here."
    passages = [
        Passage(
            doc_char_start=0,
            doc_char_end=20,
            segment_start=0,
            segment_end=1,
            source_text=source_text,
        ),
        Passage(
            doc_char_start=21,
            doc_char_end=44,
            segment_start=1,
            segment_end=2,
            source_text=source_text,
        ),
    ]

    embeddings = document_span_embedder.encode_document_spans(source_text, passages)

    assert len(embeddings) == 2
    assert all(len(emb) == 384 for emb in embeddings)
    assert all(all(np.isfinite(emb)) for emb in embeddings)

    # Different passages should have different embeddings
    assert embeddings[0] != embeddings[1]


def test_document_span_embedder_long_source_chunking(
    document_span_embedder,
) -> None:
    """Test that long documents requiring chunking work correctly."""
    # Create a document that exceeds max_seq_length (256 tokens for MiniLM)
    # Approximately 4-5 words per token, so ~1000+ words should require chunking
    sentence = "The quick brown fox jumps over the lazy dog. "
    long_source = sentence * 100  # ~450 words, should require chunking

    # Create passages spanning different parts of the document
    passages = [
        Passage(
            doc_char_start=0,
            doc_char_end=len(sentence),
            segment_start=0,
            segment_end=1,
            source_text=long_source,
        ),
        Passage(
            doc_char_start=len(sentence) * 50,
            doc_char_end=len(sentence) * 51,
            segment_start=50,
            segment_end=51,
            source_text=long_source,
        ),
        Passage(
            doc_char_start=len(sentence) * 99,
            doc_char_end=len(long_source),
            segment_start=99,
            segment_end=100,
            source_text=long_source,
        ),
    ]

    embeddings = document_span_embedder.encode_document_spans(long_source, passages)

    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)
    assert all(all(np.isfinite(emb)) for emb in embeddings)


def test_document_span_embedder_retrieval_with_prepared_corpus(
    document_span_embedder,
) -> None:
    """Test that document span embedder works with PreparedCitationCorpus."""
    sources = [
        "Unrelated noise text here.",
        "The company reported strong profits in the quarterly earnings.",
    ]

    corpus = PreparedCitationCorpus.from_sources(
        sources,
        embedder=document_span_embedder,
        config=CitationConfig(
            window_size_sentences=1,
            window_stride_sentences=1,
        ),
    )

    assert corpus.embedding_index is not None
    assert len(corpus.embedding_index.vectors) == len(corpus.candidates)

    # Align and verify retrieval works
    results = corpus.align(
        "The firm posted robust earnings.",
        backend="python",
    )

    # Should retrieve the finance passage via embeddings
    assert len(results) > 0


def test_document_span_embedder_fixture_status_labels(
    document_span_embedder,
) -> None:
    """Test that status labels (supported/partial/unsupported) are preserved."""
    # Use a fixture where we expect supported status
    answer = "The company reported strong profits."
    sources = [SourceDocument(id="finance", text=answer)]

    results = align_citations(
        answer,
        sources,
        embedder=document_span_embedder,
        config=CitationConfig(
            top_k=1,
            max_candidates_lexical=10,
            max_candidates_embedding=10,
            max_candidates_total=10,
            min_alignment_score=1,
            min_answer_coverage=0.7,
            supported_answer_coverage=0.8,
            weights=CitationWeights(
                alignment=1.0,
                answer_coverage=1.0,
                lexical=0.5,
                embedding=0.5,
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "supported"
    assert len(results[0].citations) > 0


def test_document_span_vs_per_passage_similarity(
    document_span_embedder,
) -> None:
    """Test that document-span pooling gives similar results to per-passage encoding."""
    from cite_right.models.sbert_embedder import SentenceTransformerEmbedder

    per_passage_embedder = SentenceTransformerEmbedder(_SMALL_MODEL)

    source_text = "The company reported strong profits."
    passage = Passage(
        doc_char_start=0,
        doc_char_end=len(source_text),
        segment_start=0,
        segment_end=1,
        source_text=source_text,
    )

    # Get document-span embedding
    span_embedding = document_span_embedder.encode_document_spans(
        source_text, [passage]
    )[0]

    # Get per-passage embedding
    per_passage_embedding = per_passage_embedder.encode([passage.text])[0]

    # Compute cosine similarity
    span_vec = np.array(span_embedding)
    per_passage_vec = np.array(per_passage_embedding)

    cosine_sim = np.dot(span_vec, per_passage_vec) / (
        np.linalg.norm(span_vec) * np.linalg.norm(per_passage_vec)
    )

    # Cosine similarity should be high (>0.9) for short sentences
    assert cosine_sim > 0.9, f"Cosine similarity too low: {cosine_sim}"


def test_document_span_embedder_empty_passages() -> None:
    """Test handling of empty passage list."""
    pytest.importorskip("sentence_transformers")
    from cite_right.models.document_span_embedder import DocumentSpanEmbedder

    try:
        embedder = DocumentSpanEmbedder(_SMALL_MODEL)
    except OSError as exc:
        pytest.skip(f"Embedding model is not available offline: {exc}")

    embeddings = embedder.encode_document_spans("Some text", [])
    assert embeddings == []


def test_supports_span_pooling(document_span_embedder) -> None:
    """Test that DocumentSpanEmbedder reports span pooling support."""
    assert document_span_embedder.supports_span_pooling()
