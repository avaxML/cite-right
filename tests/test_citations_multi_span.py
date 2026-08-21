"""Tests for multi-span evidence extraction in citations."""

from cite_right import SourceChunk, SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights

from .conftest import requires_rust_blocks


def _multi_span_config(
    *,
    merge_gap_chars: int = 0,
    max_spans: int = 5,
) -> CitationConfig:
    """Return a config enabling multi-span evidence for deterministic tests."""
    return CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.8,
        supported_answer_coverage=0.8,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
        multi_span_evidence=True,
        multi_span_merge_gap_chars=merge_gap_chars,
        multi_span_max_spans=max_spans,
    )


def test_align_citations_multi_span_evidence_splits_disjoint_matches() -> None:
    """Verify multi-span extracts disjoint matching segments."""
    answer = "alpha beta gamma delta."
    source = "alpha beta X Y gamma delta."

    results = align_citations(
        answer,
        [source],
        config=_multi_span_config(),
        backend="python",
    )
    assert len(results) == 1, "Expected exactly one result"
    assert results[0].citations, "Expected citations to be present"

    citation = results[0].citations[0]
    assert [span.evidence for span in citation.evidence_spans] == [
        "alpha beta",
        "gamma delta",
    ], "Evidence spans don't match expected"
    assert source[citation.char_start : citation.char_end] == citation.evidence, (
        "Citation evidence doesn't match source slice"
    )
    assert citation.evidence == "alpha beta X Y gamma delta"
    assert citation.exact_evidence == "alpha beta ... gamma delta"

    for span in citation.evidence_spans:
        assert source[span.char_start : span.char_end] == span.evidence, (
            f"Span evidence mismatch: expected '{span.evidence}'"
        )


def test_align_citations_multi_span_uses_best_equal_score_traceback() -> None:
    """Verify citations use the optimal equal-score traceback for coverage."""
    answer = "alpha beta alpha."
    source = "alpha x beta beta alpha."

    results = align_citations(
        answer,
        [source],
        config=_multi_span_config(),
        backend="python",
    )
    assert len(results) == 1
    assert results[0].status == "supported"
    assert results[0].citations

    citation = results[0].citations[0]
    assert citation.components["matches"] == 3.0
    assert citation.components["answer_coverage"] == 1.0
    assert [span.evidence for span in citation.evidence_spans] == [
        "alpha",
        "beta",
        "alpha",
    ]


def test_align_citations_multi_span_evidence_respects_sourcechunk_offsets() -> None:
    """Verify multi-span respects SourceChunk document offsets."""
    answer = "alpha beta gamma delta."
    core_text = "alpha beta X Y gamma delta."
    full_doc = f"Intro: {core_text} Outro."

    start = full_doc.find(core_text)
    assert start != -1, "Core text not found in full_doc"
    end = start + len(core_text)

    chunk = SourceChunk(
        source_id="doc",
        text=core_text,
        doc_char_start=start,
        doc_char_end=end,
        document_text=full_doc,
    )

    results = align_citations(
        answer,
        [chunk],
        config=_multi_span_config(),
        backend="python",
    )
    assert len(results) == 1
    assert results[0].citations

    citation = results[0].citations[0]
    assert citation.source_id == "doc"
    assert [span.evidence for span in citation.evidence_spans] == [
        "alpha beta",
        "gamma delta",
    ]
    assert full_doc[citation.char_start : citation.char_end] == citation.evidence, (
        "Citation offsets don't map correctly to full document"
    )

    for span in citation.evidence_spans:
        assert full_doc[span.char_start : span.char_end] == span.evidence, (
            f"Span offsets don't map correctly: expected '{span.evidence}'"
        )


@requires_rust_blocks
def test_align_citations_multi_span_python_and_rust_backends_match() -> None:
    """Verify Python and Rust backends produce identical multi-span results."""
    answer = "alpha beta gamma delta."
    source = SourceDocument(id="doc", text="alpha beta X Y gamma delta.")

    config = _multi_span_config()
    python = align_citations(answer, [source], config=config, backend="python")
    rust = align_citations(answer, [source], config=config, backend="rust")
    assert rust == python, "Rust and Python backends produced different results"


def test_align_citations_multi_span_merge_gap_chars_merges_spans() -> None:
    """Verify merge_gap_chars combines adjacent spans within threshold."""
    answer = "alpha beta gamma delta."
    source = "alpha beta X gamma delta."

    results = align_citations(
        answer,
        [source],
        config=_multi_span_config(merge_gap_chars=3),
        backend="python",
    )
    assert len(results) == 1
    assert results[0].citations

    citation = results[0].citations[0]
    assert citation.evidence_spans
    assert len(citation.evidence_spans) == 1, "Expected spans to be merged"
    assert citation.evidence_spans[0].evidence == citation.evidence
    assert citation.evidence == "alpha beta X gamma delta"
    assert citation.exact_evidence == citation.evidence


def test_align_citations_multi_span_max_spans_falls_back_to_contiguous_span() -> None:
    """Verify max_spans falls back to a single enclosing exact evidence span."""
    answer = "alpha beta gamma delta."
    source = "alpha X beta Y gamma Z delta."

    many_spans = align_citations(
        answer,
        [source],
        config=_multi_span_config(merge_gap_chars=0, max_spans=10),
        backend="python",
    )
    citation_many = many_spans[0].citations[0]
    assert [span.evidence for span in citation_many.evidence_spans] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]

    fallback = align_citations(
        answer,
        [source],
        config=_multi_span_config(merge_gap_chars=0, max_spans=2),
        backend="python",
    )
    assert fallback[0].status == "supported"
    assert fallback[0].retrieval_support == []
    assert len(fallback[0].citations) == 1

    citation = fallback[0].citations[0]
    assert len(citation.evidence_spans) == 1
    assert citation.evidence_spans[0].evidence == citation.evidence
    assert citation.exact_evidence == citation.evidence
    assert source[citation.char_start : citation.char_end] == citation.evidence
    assert citation.evidence == "alpha X beta Y gamma Z delta"
    assert citation.char_start == 0
    assert citation.char_end == len(citation.evidence)


def test_citation_exact_evidence_falls_back_to_legacy_evidence_when_spans_absent() -> (
    None
):
    """Verify exact_evidence stays safe for defensive empty-span cases."""
    answer = "alpha beta gamma."
    source = "alpha beta gamma."

    results = align_citations(
        answer,
        [source],
        config=CitationConfig(top_k=1),
        backend="python",
    )

    assert len(results) == 1
    assert results[0].citations

    citation = results[0].citations[0].model_copy(update={"evidence_spans": []})
    assert citation.exact_evidence == citation.evidence


def test_align_citations_sourcechunk_without_document_text_slices_locally() -> None:
    """Verify SourceChunk without document_text uses local slicing."""
    answer = "alpha beta gamma delta."
    chunk_text = "alpha beta X Y gamma delta."
    base = 123
    chunk = SourceChunk(
        source_id="chunk",
        text=chunk_text,
        doc_char_start=base,
        doc_char_end=base + len(chunk_text),
        document_text=None,
    )

    results = align_citations(
        answer,
        [chunk],
        config=_multi_span_config(),
        backend="python",
    )
    assert len(results) == 1
    assert results[0].citations

    citation = results[0].citations[0]
    assert citation.source_id == "chunk"

    local_start = citation.char_start - base
    local_end = citation.char_end - base
    assert chunk_text[local_start:local_end] == citation.evidence, (
        "Local slicing produced incorrect evidence"
    )

    for span in citation.evidence_spans:
        local_start = span.char_start - base
        local_end = span.char_end - base
        assert chunk_text[local_start:local_end] == span.evidence, (
            f"Local span slicing failed for '{span.evidence}'"
        )


def test_align_citations_multi_span_is_deterministic() -> None:
    """Verify multi-span results are deterministic across runs."""
    answer = "alpha beta gamma delta."
    source = "alpha beta X Y gamma delta."
    config = _multi_span_config()

    first = align_citations(answer, [source], config=config, backend="python")
    second = align_citations(answer, [source], config=config, backend="python")
    assert second == first, "Multi-span results are not deterministic"
