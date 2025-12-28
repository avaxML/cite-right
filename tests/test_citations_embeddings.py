import os

import pytest

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.models.sbert_embedder import SentenceTransformerEmbedder

_SMALL_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

if os.environ.get("CITE_RIGHT_RUN_EMBEDDINGS_TESTS") != "1":
    pytest.skip(
        "Set CITE_RIGHT_RUN_EMBEDDINGS_TESTS=1 to run embeddings tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    pytest.importorskip("sentence_transformers")
    try:
        return SentenceTransformerEmbedder(_SMALL_MODEL)
    except OSError as exc:
        pytest.skip(f"Embedding model is not available offline: {exc}")


def test_align_citations_embeddings_paraphrase_embedding_only(
    embedder: SentenceTransformerEmbedder,
) -> None:
    answer = "The firm posted robust earnings."
    sources = [
        SourceDocument(
            id="noise", text="Weather report: storms are likely this weekend."
        ),
        SourceDocument(id="finance", text="The company reported strong profits."),
    ]

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
            min_embedding_similarity=0.2,
            supported_embedding_similarity=0.2,
            min_alignment_score=1000,
            min_answer_coverage=1.0,
            weights=CitationWeights(
                alignment=0.0, answer_coverage=0.0, lexical=0.0, embedding=1.0
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "supported"
    assert results[0].citations

    citation = results[0].citations[0]
    assert citation.source_id == "finance"
    assert citation.source_index == 1
    assert citation.evidence == sources[1].text
    assert citation.components.get("embedding_only") == 1.0
    assert float(citation.components.get("embedding_score", 0.0)) >= 0.2


def test_align_citations_embeddings_retrieves_candidate_with_lexical_disabled(
    embedder: SentenceTransformerEmbedder,
) -> None:
    phrase = "Acme reported revenue of 5.2 billion dollars in 2020."
    answer = "In 2020, Acme reported revenue of 5.2 billion dollars."
    sources = [
        SourceDocument(id="noise0", text="Completely unrelated filler."),
        SourceDocument(id="noise1", text="More unrelated text."),
        SourceDocument(id="finance", text=f"Intro. {phrase} Outro."),
    ]

    results = align_citations(
        answer,
        sources,
        embedder=embedder,
        config=CitationConfig(
            top_k=1,
            max_candidates_lexical=0,
            max_candidates_embedding=10,
            max_candidates_total=10,
            min_alignment_score=1,
            min_answer_coverage=0.5,
            supported_answer_coverage=0.8,
            weights=CitationWeights(lexical=0.0, embedding=0.2),
        ),
    )

    assert len(results) == 1
    assert results[0].citations
    assert results[0].citations[0].source_id == "finance"
