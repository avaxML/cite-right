"""Tests for the main align_citations API."""

import pytest

from cite_right import SourceChunk, SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights

from .conftest import requires_rust


@pytest.mark.parametrize("source_count", [5, 10, 20, 40, 50])
def test_align_citations_many_sources_is_deterministic(source_count: int) -> None:
    phrase = "climate policy reduces emissions quickly"
    answer = f"{phrase}."

    match_idx = source_count // 2
    sources = [f"Filler source {idx} with no overlap." for idx in range(source_count)]
    sources[match_idx] = f"Intro sentence. {phrase}. Trailing sentence."

    config = CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.5,
        supported_answer_coverage=0.9,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )

    results = align_citations(answer, sources, config=config)
    assert len(results) == 1

    span = results[0]
    assert (
        answer[span.answer_span.char_start : span.answer_span.char_end]
        == span.answer_span.text
    )
    assert span.citations

    citation = span.citations[0]
    assert citation.source_index == match_idx
    assert citation.evidence == phrase
    assert sources[match_idx][citation.char_start : citation.char_end] == phrase

    assert align_citations(answer, sources, config=config) == results


@pytest.mark.parametrize("source_count", [5, 10, 20, 40, 50])
def test_align_citations_multi_sentence_across_many_sources(source_count: int) -> None:
    phrase_a = "battery storage lowers peak demand"
    phrase_b = "hydrogen infrastructure remains expensive"
    phrase_c = "heat pumps cut household emissions"

    answer = f"{phrase_a}. {phrase_b}.\n\n{phrase_c}."

    sources = [
        f"Filler {idx} with irrelevant content only." for idx in range(source_count)
    ]
    sources[0] = f"Intro. {phrase_a}. Outro."
    mid = source_count // 2
    sources[mid] = f"{phrase_b}."
    sources[-1] = f"More filler. {phrase_c}."

    config = CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.8,
        supported_answer_coverage=0.8,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )

    results = align_citations(answer, sources, config=config)
    assert len(results) == 3
    assert [item.citations[0].evidence for item in results if item.citations] == [
        phrase_a,
        phrase_b,
        phrase_c,
    ]
    assert results[0].citations[0].source_index == 0
    assert results[1].citations[0].source_index == mid
    assert results[2].citations[0].source_index == source_count - 1

    assert align_citations(answer, sources, config=config) == results


def test_align_citations_multi_paragraph_answer_aligns_partials_and_offsets() -> None:
    fact_1 = "Acme Corp reported revenue of 5.2 billion dollars in 2020"
    fact_2 = (
        "The Falcon X chip delivers 18 percent higher efficiency under sustained load"
    )
    fact_3 = "found a 34 percent reduction in symptoms after eight weeks"

    answer = (
        f"{fact_1}, while analysts debated expansion to Antarctica and Mars.\n"
        "zzunsupported claim about a secret Mars office.\n\n"
        f"{fact_2}.\n\n"
        f"Researchers {fact_3}."
    )

    doc_1 = (
        "Executive summary with unrelated material. "
        f"{fact_1}. "
        "More text that is not used in the generated answer."
    )
    doc_2_full = (
        "Long report with unrelated background. "
        f"{fact_2}. "
        "Extra paragraphs follow that are not cited."
    )
    doc_3 = (
        "Clinical appendix with extensive discussion. "
        f"A randomized trial {fact_3} compared with placebo. "
        "Additional notes about secondary endpoints are omitted."
    )

    fact_2_start = doc_2_full.find(fact_2)
    assert fact_2_start != -1
    fact_2_end = fact_2_start + len(fact_2)
    chunk_2 = SourceChunk(
        source_id="hardware",
        text=doc_2_full[fact_2_start:fact_2_end],
        doc_char_start=fact_2_start,
        doc_char_end=fact_2_end,
        document_text=doc_2_full,
    )

    sources = [
        SourceDocument(id="finance", text=doc_1),
        chunk_2,
        SourceDocument(id="clinical", text=doc_3),
        SourceDocument(id="irrelevant", text="Completely unrelated filler."),
    ]

    config = CitationConfig(
        top_k=1,
        min_alignment_score=16,
        min_answer_coverage=0.2,
        supported_answer_coverage=0.6,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )

    results = align_citations(answer, sources, config=config)
    assert len(results) == 4

    for item in results:
        span = item.answer_span
        assert answer[span.char_start : span.char_end] == span.text

    first = results[0]
    assert first.status == "partial"
    assert first.citations
    cite1 = first.citations[0]
    assert cite1.source_id == "finance"
    assert cite1.evidence == fact_1
    assert doc_1[cite1.char_start : cite1.char_end] == cite1.evidence

    second = results[1]
    assert second.status == "unsupported"
    assert second.citations == []

    third = results[2]
    assert third.status == "supported"
    assert third.citations
    cite2 = third.citations[0]
    assert cite2.source_id == "hardware"
    assert cite2.char_start == fact_2_start
    assert cite2.char_end == fact_2_end
    assert doc_2_full[cite2.char_start : cite2.char_end] == fact_2
    assert cite2.evidence == fact_2

    fourth = results[3]
    assert fourth.status == "supported"
    assert fourth.citations
    cite3 = fourth.citations[0]
    assert cite3.source_id == "clinical"

    expected_fact3 = f"{fact_3}"
    start3 = doc_3.find(expected_fact3)
    assert start3 != -1
    assert cite3.char_start == start3
    assert cite3.char_end == start3 + len(expected_fact3)
    assert cite3.evidence == expected_fact3
    assert doc_3[cite3.char_start : cite3.char_end] == cite3.evidence

    assert align_citations(answer, sources, config=config) == results


def test_align_citations_windowing_enables_cross_sentence_evidence() -> None:
    answer = (
        "The Falcon X chip uses a 7 nanometer process and it delivers 18 percent higher "
        "efficiency under sustained load."
    )
    source = (
        "The Falcon X chip uses a 7 nanometer process. "
        "And it delivers 18 percent higher efficiency under sustained load."
    )

    strict = CitationConfig(
        top_k=1,
        min_alignment_score=10,
        min_answer_coverage=0.8,
        supported_answer_coverage=0.8,
        window_size_sentences=1,
        window_stride_sentences=1,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )
    without_window = align_citations(answer, [source], config=strict)
    assert len(without_window) == 1
    assert without_window[0].status == "unsupported"
    assert without_window[0].citations == []

    windowed = CitationConfig(
        top_k=1,
        min_alignment_score=10,
        min_answer_coverage=0.8,
        supported_answer_coverage=0.8,
        window_size_sentences=2,
        window_stride_sentences=1,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )
    with_window = align_citations(answer, [source], config=windowed)
    assert len(with_window) == 1
    assert with_window[0].status == "supported"
    assert with_window[0].citations

    citation = with_window[0].citations[0]
    assert source[citation.char_start : citation.char_end] == citation.evidence
    assert "7 nanometer process" in citation.evidence
    assert "18 percent higher efficiency" in citation.evidence


@requires_rust
def test_align_citations_python_and_rust_backends_match() -> None:
    """Verify Python and Rust backends produce identical citation results."""
    phrase = "climate policy reduces emissions quickly"
    answer = f"{phrase}."
    sources = [
        SourceDocument(id="a", text=f"Intro. {phrase}. Outro."),
        SourceDocument(id="b", text="Completely unrelated filler."),
    ]

    config = CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.5,
        supported_answer_coverage=0.9,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )

    python = align_citations(answer, sources, config=config, backend="python")
    rust = align_citations(answer, sources, config=config, backend="rust")
    assert rust == python
