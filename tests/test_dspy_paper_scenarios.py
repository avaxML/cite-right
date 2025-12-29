"""Tests using real-world DSPy paper scenarios for citation alignment."""

import pytest

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights

from .conftest import (
    DSPY_ASSERTIONS,
    DSPY_COMPILER,
    DSPY_MODEL,
    IRRELEVANT_SOURCES,
)


def _paper_scenario_config(*, multi_span_evidence: bool = True) -> CitationConfig:
    """Return a config tuned for deterministic, paper-style scenarios."""
    return CitationConfig(
        top_k=1,
        min_alignment_score=6,
        min_answer_coverage=0.6,
        supported_answer_coverage=0.75,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
        window_size_sentences=2,
        window_stride_sentences=1,
        multi_span_evidence=multi_span_evidence,
        multi_span_merge_gap_chars=0,
    )


def _build_dspy_sources(
    num_sources: int,
    *,
    include_assertions: bool,
) -> list[SourceDocument]:
    """Build a fixed set of DSPy excerpts plus irrelevant sources.

    Args:
        num_sources: Total number of sources to return (must be >= 3).
        include_assertions: Whether to include the assertions excerpt.

    Returns:
        A deterministic list of `SourceDocument` with irrelevant sources first.
    """
    if num_sources < 3:
        raise ValueError("num_sources must be >= 3")

    relevant: list[SourceDocument] = [
        SourceDocument(id="dspy", text=DSPY_MODEL),
        SourceDocument(id="compiler", text=DSPY_COMPILER),
    ]
    if include_assertions:
        relevant.append(SourceDocument(id="assertions", text=DSPY_ASSERTIONS))

    num_irrelevant = num_sources - len(relevant)
    irrelevant = [
        SourceDocument(
            id=f"noise-{idx}", text=IRRELEVANT_SOURCES[idx % len(IRRELEVANT_SOURCES)]
        )
        for idx in range(num_irrelevant)
    ]
    return irrelevant + relevant


def test_dspy_paper_style_multi_source_multi_paragraph_scenario() -> None:
    """Test multi-source multi-paragraph scenario with DSPy paper excerpts."""
    sources = [
        SourceDocument(id="dspy", text=DSPY_MODEL),
        SourceDocument(id="compiler", text=DSPY_COMPILER),
        SourceDocument(id="assertions", text=DSPY_ASSERTIONS),
    ]

    answer = (
        "DSPy abstracts LM pipelines as text transformation graphs, and LMs are invoked through "
        "declarative modules. "
        "Compiling relies on a teleprompter; the compiler first finds all unique Predict "
        "modules."
        "\n\n"
        "We propose LM Assertions as boolean conditions and integrate them into DSPy with hard "
        "Assertions and soft Suggestions. These methods guarantee a 200% improvement on every "
        "benchmark."
    )

    results = align_citations(answer, sources, config=_paper_scenario_config())
    assert len(results) == 5, "Expected 5 answer segments"

    # First segment: should cite dspy source
    first = results[0]
    assert first.status == "supported"
    assert first.citations, "Expected citations for first segment"
    cite0 = first.citations[0]
    assert cite0.source_id == "dspy"
    assert cite0.evidence_spans
    assert any(
        "text transformation graphs" in span.evidence for span in cite0.evidence_spans
    ), "Expected 'text transformation graphs' in evidence spans"
    assert any(
        "declarative modules" in span.evidence for span in cite0.evidence_spans
    ), "Expected 'declarative modules' in evidence spans"
    assert DSPY_MODEL[cite0.char_start : cite0.char_end] == cite0.evidence
    for span in cite0.evidence_spans:
        assert DSPY_MODEL[span.char_start : span.char_end] == span.evidence

    # Second segment: should cite compiler source (teleprompter)
    second = results[1]
    assert second.status == "supported"
    assert second.citations
    cite1 = second.citations[0]
    assert cite1.source_id == "compiler"
    assert "teleprompter" in cite1.evidence
    assert DSPY_COMPILER[cite1.char_start : cite1.char_end] == cite1.evidence
    for span in cite1.evidence_spans:
        assert DSPY_COMPILER[span.char_start : span.char_end] == span.evidence

    # Third segment: should cite compiler source (Predict)
    third = results[2]
    assert third.status == "supported"
    assert third.citations
    cite2 = third.citations[0]
    assert cite2.source_id == "compiler"
    assert "Predict" in cite2.evidence
    assert DSPY_COMPILER[cite2.char_start : cite2.char_end] == cite2.evidence
    for span in cite2.evidence_spans:
        assert DSPY_COMPILER[span.char_start : span.char_end] == span.evidence

    # Fourth segment: should cite assertions source
    fourth = results[3]
    assert fourth.status == "supported"
    assert fourth.citations
    cite3 = fourth.citations[0]
    assert cite3.source_id == "assertions"
    assert "boolean conditions" in cite3.evidence
    assert "Assertions" in cite3.evidence
    assert "Suggestions" in cite3.evidence
    assert DSPY_ASSERTIONS[cite3.char_start : cite3.char_end] == cite3.evidence
    for span in cite3.evidence_spans:
        assert DSPY_ASSERTIONS[span.char_start : span.char_end] == span.evidence

    # Fifth segment: fabricated claim, should be unsupported
    fifth = results[4]
    assert fifth.status == "unsupported"
    assert fifth.citations == [], "Fabricated claim should have no citations"


def test_dspy_paper_style_percent_normalization_matches_percent_symbol() -> None:
    """Verify percent normalization matches '25%' to '25 percent'."""
    source = (
        "Within minutes of compiling, a few lines of DSPy allow pipelines that outperform "
        "standard few-shot prompting (generally by over 25% and 65%, respectively)."
    )
    answer = (
        "A few lines of DSPy allow pipelines that outperform standard few-shot prompting, "
        "generally by over 25 percent and 65 percent."
    )

    results = align_citations(
        answer,
        [SourceDocument(id="dspy", text=source)],
        config=_paper_scenario_config(multi_span_evidence=False),
    )
    assert len(results) == 1
    assert results[0].status == "supported"
    assert results[0].citations
    citation = results[0].citations[0]
    assert citation.source_id == "dspy"
    assert source[citation.char_start : citation.char_end] == citation.evidence, (
        "Citation offsets don't match evidence text"
    )


@pytest.mark.parametrize("num_sources", [3, 5, 10, 15])
def test_dspy_paper_style_irrelevant_sources_do_not_break_alignment(
    num_sources: int,
) -> None:
    """Verify irrelevant sources don't break alignment to relevant sources."""
    include_assertions = num_sources != 3
    sources = _build_dspy_sources(num_sources, include_assertions=include_assertions)
    assert len(sources) == num_sources
    assert any(doc.id.startswith("noise-") for doc in sources), (
        "Expected noise sources in list"
    )

    source_text_by_id = {doc.id: doc.text for doc in sources}

    answer = (
        "DSPy abstracts LM pipelines as text transformation graphs, where LMs are invoked "
        "through declarative modules. "
        "Compiling relies on a teleprompter, and the compiler first finds all unique Predict "
        "modules in a program. "
        "We propose LM Assertions, expressed as boolean conditions, and integrate them into "
        "DSPy with hard Assertions and soft Suggestions."
        "\n\n"
        "This achieves a 200% improvement in all domains."
    )

    results = align_citations(answer, sources, config=_paper_scenario_config())
    assert len(results) == 4

    # First segment
    first = results[0]
    assert first.status == "supported"
    cite0 = first.citations[0]
    assert cite0.source_id == "dspy"
    assert "text transformation graphs" in cite0.evidence
    assert "declarative modules" in cite0.evidence
    text0 = source_text_by_id[cite0.source_id]
    assert text0[cite0.char_start : cite0.char_end] == cite0.evidence
    for span in cite0.evidence_spans:
        assert text0[span.char_start : span.char_end] == span.evidence

    # Second segment
    second = results[1]
    assert second.status == "supported"
    cite1 = second.citations[0]
    assert cite1.source_id == "compiler"
    assert "teleprompter" in cite1.evidence
    assert "Predict" in cite1.evidence
    text1 = source_text_by_id[cite1.source_id]
    assert text1[cite1.char_start : cite1.char_end] == cite1.evidence
    for span in cite1.evidence_spans:
        assert text1[span.char_start : span.char_end] == span.evidence

    # Third segment (depends on whether assertions source is included)
    third = results[2]
    if include_assertions:
        assert third.status == "supported"
        cite2 = third.citations[0]
        assert cite2.source_id == "assertions"
        assert "boolean conditions" in cite2.evidence
        assert "hard Assertions" in cite2.evidence
        assert "soft Suggestions" in cite2.evidence
        text2 = source_text_by_id[cite2.source_id]
        assert text2[cite2.char_start : cite2.char_end] == cite2.evidence
        for span in cite2.evidence_spans:
            assert text2[span.char_start : span.char_end] == span.evidence
    else:
        assert third.status == "unsupported"
        assert third.citations == []

    # Fourth segment (fabricated claim)
    fourth = results[3]
    assert fourth.status == "unsupported"
    assert fourth.citations == []
