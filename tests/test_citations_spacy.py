import pytest

from cite_right import SpacyAnswerSegmenter, SpacySegmenter, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights


def test_align_citations_spacy_clause_segmentation_cites_each_clause() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model not installed")

    answer = "Apple revenue is up and stocks are down."
    sources = [
        "Intro filler. Apple revenue is up. Outro filler.",
        "stocks are down. Extra filler follows.",
    ]

    results = align_citations(
        answer,
        sources,
        answer_segmenter=SpacyAnswerSegmenter(split_clauses=True),
        source_segmenter=SpacySegmenter(),
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.8,
            supported_answer_coverage=0.8,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )

    assert [item.answer_span.text for item in results] == [
        "Apple revenue is up",
        "stocks are down.",
    ]
    assert [item.citations[0].evidence for item in results] == [
        "Apple revenue is up",
        "stocks are down",
    ]
    assert [item.citations[0].source_index for item in results] == [0, 1]


def test_align_citations_spacy_does_not_split_lists() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model not installed")

    answer = "Apples, oranges, and pears are tasty."
    sources = [answer]

    results = align_citations(
        answer,
        sources,
        answer_segmenter=SpacyAnswerSegmenter(split_clauses=True),
        source_segmenter=SpacySegmenter(),
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.8,
            supported_answer_coverage=0.8,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )
    assert len(results) == 1
    assert results[0].answer_span.text == answer
    assert results[0].citations[0].evidence == answer[:-1]
