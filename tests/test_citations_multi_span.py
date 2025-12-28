import pytest

from cite_right import SourceChunk, SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights


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
    answer = "alpha beta gamma delta."
    source = "alpha beta X Y gamma delta."

    results = align_citations(
        answer,
        [source],
        config=_multi_span_config(),
        backend="python",
    )
    assert len(results) == 1
    assert results[0].citations

    citation = results[0].citations[0]
    assert [span.evidence for span in citation.evidence_spans] == [
        "alpha beta",
        "gamma delta",
    ]
    assert source[citation.char_start : citation.char_end] == citation.evidence
    assert citation.evidence == "alpha beta X Y gamma delta"

    for span in citation.evidence_spans:
        assert source[span.char_start : span.char_end] == span.evidence


def test_align_citations_multi_span_evidence_respects_sourcechunk_offsets() -> None:
    answer = "alpha beta gamma delta."
    core_text = "alpha beta X Y gamma delta."
    full_doc = f"Intro: {core_text} Outro."

    start = full_doc.find(core_text)
    assert start != -1
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
    assert full_doc[citation.char_start : citation.char_end] == citation.evidence

    for span in citation.evidence_spans:
        assert full_doc[span.char_start : span.char_end] == span.evidence


def test_align_citations_multi_span_python_and_rust_backends_match() -> None:
    try:
        from cite_right import _core  # noqa: F401
    except ImportError:
        pytest.skip("Rust extension not built")
    if not hasattr(_core, "align_pair_blocks_details"):
        pytest.skip(
            "Rust extension is missing align_pair_blocks_details (rebuild required)"
        )

    answer = "alpha beta gamma delta."
    source = SourceDocument(id="doc", text="alpha beta X Y gamma delta.")

    config = _multi_span_config()
    python = align_citations(answer, [source], config=config, backend="python")
    rust = align_citations(answer, [source], config=config, backend="rust")
    assert rust == python


def test_align_citations_multi_span_merge_gap_chars_merges_spans() -> None:
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
    assert len(citation.evidence_spans) == 1
    assert citation.evidence_spans[0].evidence == citation.evidence
    assert citation.evidence == "alpha beta X gamma delta"


def test_align_citations_multi_span_max_spans_falls_back_to_contiguous() -> None:
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
    citation_fallback = fallback[0].citations[0]
    assert len(citation_fallback.evidence_spans) == 1
    assert citation_fallback.evidence_spans[0].evidence == citation_fallback.evidence
    assert citation_fallback.evidence == "alpha X beta Y gamma Z delta"
    assert citation_fallback.components.get("num_evidence_spans") == 1.0


def test_align_citations_sourcechunk_without_document_text_slices_locally() -> None:
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
    assert chunk_text[local_start:local_end] == citation.evidence

    for span in citation.evidence_spans:
        local_start = span.char_start - base
        local_end = span.char_end - base
        assert chunk_text[local_start:local_end] == span.evidence


def test_align_citations_multi_span_is_deterministic() -> None:
    answer = "alpha beta gamma delta."
    source = "alpha beta X Y gamma delta."
    config = _multi_span_config()

    first = align_citations(answer, [source], config=config, backend="python")
    second = align_citations(answer, [source], config=config, backend="python")
    assert second == first
