from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights


def test_align_citations_lexical_only_returns_retrieval_support_when_alignment_fails() -> (
    None
):
    results = align_citations(
        "The firm posted robust earnings.",
        [
            SourceDocument(id="noise", text="Weather report: storms are likely."),
            SourceDocument(
                id="finance",
                text="Robust earnings were reported this quarter.",
            ),
        ],
        config=CitationConfig(
            top_k=1,
            max_candidates_lexical=10,
            max_candidates_embedding=0,
            max_candidates_total=10,
            min_alignment_score=10_000,
            min_answer_coverage=1.0,
            weights=CitationWeights(
                alignment=0.0,
                answer_coverage=0.0,
                lexical=1.0,
                embedding=0.0,
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []
    assert len(results[0].retrieval_support) == 1

    support = results[0].retrieval_support[0]
    assert support.source_id == "finance"
    assert support.source_index == 1
    assert support.passage_text == "Robust earnings were reported this quarter."
    assert support.embedding_score == 0.0
    assert support.lexical_score > 0.0
