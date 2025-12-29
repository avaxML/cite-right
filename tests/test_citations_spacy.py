"""Tests for SpaCy-based segmentation in citation alignment."""

from cite_right import SpacyAnswerSegmenter, SpacySegmenter, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights

from .conftest import requires_spacy_model


@requires_spacy_model
def test_align_citations_spacy_clause_segmentation_cites_each_clause() -> None:
    """Verify SpaCy clause segmentation produces separate citations per clause."""
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
    ], "Answer spans don't match expected clauses"
    assert [item.citations[0].evidence for item in results] == [
        "Apple revenue is up",
        "stocks are down",
    ], "Citation evidence doesn't match expected"
    assert [item.citations[0].source_index for item in results] == [
        0,
        1,
    ], "Source indices don't match expected"


@requires_spacy_model
def test_align_citations_spacy_does_not_split_lists() -> None:
    """Verify SpaCy doesn't incorrectly split comma-separated lists."""
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
    assert len(results) == 1, "Expected single result for list sentence"
    assert results[0].answer_span.text == answer
    assert results[0].citations[0].evidence == answer[:-1], (
        "Evidence should exclude trailing punctuation"
    )
