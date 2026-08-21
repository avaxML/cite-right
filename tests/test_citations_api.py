"""Tests for the main align_citations API."""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

import cite_right
import cite_right.citations as citations_module
from cite_right import (
    PreparedCitationCorpus,
    SourceChunk,
    SourceDocument,
    align_citations,
)
from cite_right.citations import _rank_and_limit_citations
from cite_right.core.aligner_py import SmithWatermanAligner
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.prepared_corpus import (
    build_candidates,
    build_source_passages,
    normalize_sources,
)
from cite_right.core.results import Alignment, Citation, EvidenceSpan, TokenizedText
from cite_right.text.segmenter_simple import SimpleSegmenter
from cite_right.text.tokenizer import SimpleTokenizer

from .conftest import requires_rust


def test_how_it_works_describes_status_using_answer_coverage() -> None:
    docs_path = Path(__file__).resolve().parents[1] / "docs/concepts/how-it-works.md"
    docs_text = docs_path.read_text(encoding="utf-8")

    assert "best citation score" not in docs_text
    assert "best citation's answer coverage" in docs_text


class CountingBatchAligner:
    """Aligner test double that tracks batch vs single-call usage."""

    def __init__(self) -> None:
        self.align_calls = 0
        self.align_batch_calls = 0

    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment:
        self.align_calls += 1
        return Alignment(score=0, token_start=0, token_end=0)

    def align_batch(
        self, seq1: Sequence[int], seqs: Sequence[Sequence[int]]
    ) -> list[Alignment]:
        self.align_batch_calls += 1
        return [self.align(seq1, seq2) for seq2 in seqs]


class LegacyAligner:
    """Custom aligner implementing the pre-batch public protocol."""

    def __init__(self) -> None:
        self._delegate = SmithWatermanAligner()

    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment:
        return self._delegate.align(seq1, seq2)


class TimedEmbedder:
    """Deterministic embedder that records time spent in encode calls."""

    def __init__(self) -> None:
        self.elapsed_ms = 0.0

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        started = time.perf_counter()
        time.sleep(0.01)
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        return [[1.0, 0.0] for _ in texts]


def _test_citation(
    *,
    source_id: str,
    source_index: int,
    candidate_index: int,
    char_start: int,
    char_end: int,
    score: float = 1.0,
    evidence_spans: list[tuple[int, int]] | None = None,
) -> Citation:
    spans = (
        [
            EvidenceSpan(
                char_start=span_start,
                char_end=span_end,
                evidence=f"{source_id}:{span_start}-{span_end}",
            )
            for span_start, span_end in evidence_spans
        ]
        if evidence_spans is not None
        else [
            EvidenceSpan(
                char_start=char_start,
                char_end=char_end,
                evidence=f"{source_id}:{char_start}-{char_end}",
            )
        ]
    )
    return Citation(
        score=score,
        source_id=source_id,
        source_index=source_index,
        candidate_index=candidate_index,
        char_start=char_start,
        char_end=char_end,
        evidence=f"{source_id}:{char_start}-{char_end}",
        evidence_spans=spans,
        components={},
    )


def test_align_citations_auto_falls_back_when_rust_core_lacks_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = "alpha beta gamma."
    source = "alpha x gamma."
    config = CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.2,
        supported_answer_coverage=0.6,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )

    old_core = SimpleNamespace(align_pair=lambda *_args: (3, 0, 3))
    monkeypatch.setattr(cite_right, "_core", old_core, raising=False)

    python_results = align_citations(answer, [source], config=config, backend="python")
    auto_results = align_citations(answer, [source], config=config, backend="auto")

    assert python_results[0].status == "supported"
    assert auto_results == python_results


def test_align_citations_rust_backend_requires_detailed_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_core = SimpleNamespace(align_pair=lambda *_args: (3, 0, 3))
    monkeypatch.setattr(cite_right, "_core", old_core, raising=False)

    with pytest.raises(RuntimeError, match="detailed alignment"):
        align_citations(
            "alpha beta gamma.",
            ["alpha x gamma."],
            config=CitationConfig(top_k=1),
            backend="rust",
        )


def test_align_citations_uses_batch_alignment_api() -> None:
    answer = "alpha beta gamma."
    sources = [
        SourceDocument(id="match", text="alpha beta gamma."),
        SourceDocument(id="near", text="alpha beta delta."),
    ]
    aligner = CountingBatchAligner()

    align_citations(
        answer,
        sources,
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
        backend="python",
        aligner=aligner,
    )

    assert aligner.align_batch_calls == 1
    assert aligner.align_calls <= 2


def test_align_citations_accepts_legacy_single_alignment_api() -> None:
    results = align_citations(
        "alpha beta gamma.",
        [SourceDocument(id="match", text="alpha beta gamma.")],
        config=CitationConfig(top_k=1),
        aligner=LegacyAligner(),
    )

    assert results[0].citations[0].source_id == "match"


def test_prepared_corpus_align_resolves_default_aligner_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = "alpha beta gamma. delta epsilon zeta."
    corpus = PreparedCitationCorpus.from_sources(
        ["alpha beta gamma. delta epsilon zeta."],
        config=CitationConfig(top_k=1),
    )
    aligner_calls = 0

    def counting_default_aligner(
        cfg: CitationConfig, *, backend: str
    ) -> SmithWatermanAligner:
        nonlocal aligner_calls
        aligner_calls += 1
        return SmithWatermanAligner(
            match_score=cfg.match_score,
            mismatch_score=cfg.mismatch_score,
            gap_score=cfg.gap_score,
            return_match_blocks=cfg.multi_span_evidence,
        )

    monkeypatch.setattr(citations_module, "_default_aligner", counting_default_aligner)

    results = corpus.align(answer, backend="python")

    assert len(results) == 2
    assert aligner_calls == 1


def test_rank_and_limit_citations_prefers_source_order_in_equal_score_ties() -> None:
    cfg = CitationConfig(top_k=3, prefer_source_order=True)

    citations = [
        _test_citation(
            source_id="later-source",
            source_index=1,
            candidate_index=0,
            char_start=0,
            char_end=10,
        ),
        _test_citation(
            source_id="earlier-source",
            source_index=0,
            candidate_index=1,
            char_start=50,
            char_end=60,
        ),
    ]

    ranked = _rank_and_limit_citations(citations, cfg)

    assert [citation.source_id for citation in ranked] == [
        "earlier-source",
        "later-source",
    ]


def test_rank_and_limit_citations_prefers_earlier_position_when_source_order_disabled() -> (
    None
):
    cfg = CitationConfig(top_k=3, prefer_source_order=False)

    citations = [
        _test_citation(
            source_id="earlier-source",
            source_index=0,
            candidate_index=0,
            char_start=50,
            char_end=60,
        ),
        _test_citation(
            source_id="later-source",
            source_index=1,
            candidate_index=1,
            char_start=5,
            char_end=15,
        ),
    ]

    ranked = _rank_and_limit_citations(citations, cfg)

    assert [citation.source_id for citation in ranked] == [
        "later-source",
        "earlier-source",
    ]


def test_rank_and_limit_citations_dedupes_by_source_and_evidence_span_tuple() -> None:
    cfg = CitationConfig(top_k=3)

    citations = [
        _test_citation(
            source_id="source-a",
            source_index=0,
            candidate_index=2,
            char_start=0,
            char_end=12,
            score=0.9,
            evidence_spans=[(0, 5), (8, 12)],
        ),
        _test_citation(
            source_id="source-a",
            source_index=0,
            candidate_index=1,
            char_start=0,
            char_end=12,
            score=1.0,
            evidence_spans=[(0, 5), (8, 12)],
        ),
        _test_citation(
            source_id="source-a",
            source_index=0,
            candidate_index=3,
            char_start=0,
            char_end=12,
            score=0.8,
            evidence_spans=[(0, 4), (8, 12)],
        ),
        _test_citation(
            source_id="source-b",
            source_index=1,
            candidate_index=4,
            char_start=0,
            char_end=12,
            score=0.7,
            evidence_spans=[(0, 5), (8, 12)],
        ),
    ]

    ranked = _rank_and_limit_citations(citations, cfg)

    assert [(citation.source_id, citation.candidate_index) for citation in ranked] == [
        ("source-a", 1),
        ("source-a", 3),
        ("source-b", 4),
    ]


def test_rank_and_limit_citations_applies_per_source_cap_after_deduping() -> None:
    cfg = CitationConfig(top_k=3, max_citations_per_source=1)

    citations = [
        _test_citation(
            source_id="source-a",
            source_index=0,
            candidate_index=3,
            char_start=0,
            char_end=10,
            score=1.0,
            evidence_spans=[(0, 5), (8, 10)],
        ),
        _test_citation(
            source_id="source-a",
            source_index=0,
            candidate_index=2,
            char_start=0,
            char_end=10,
            score=0.95,
            evidence_spans=[(0, 5), (8, 10)],
        ),
        _test_citation(
            source_id="source-a",
            source_index=0,
            candidate_index=1,
            char_start=20,
            char_end=30,
            score=0.9,
            evidence_spans=[(20, 30)],
        ),
        _test_citation(
            source_id="source-b",
            source_index=1,
            candidate_index=4,
            char_start=5,
            char_end=15,
            score=0.85,
            evidence_spans=[(5, 15)],
        ),
    ]

    ranked = _rank_and_limit_citations(citations, cfg)

    assert [(citation.source_id, citation.candidate_index) for citation in ranked] == [
        ("source-a", 3),
        ("source-b", 4),
    ]


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


def test_align_citations_preserves_fullwidth_source_offsets_after_nfkc_matching() -> (
    None
):
    answer = "ABC 123 percent."
    source = "Prefix ＡＢＣ １２３％ suffix"

    results = align_citations(
        answer,
        [source],
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=1.0,
            supported_answer_coverage=1.0,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "supported"
    assert results[0].citations

    citation = results[0].citations[0]
    expected = "ＡＢＣ １２３％"
    start = source.index(expected)

    assert citation.char_start == start
    assert citation.char_end == start + len(expected)
    assert citation.evidence == expected
    assert source[citation.char_start : citation.char_end] == citation.evidence


def test_align_citations_preserves_combining_mark_offsets_after_normalized_match() -> (
    None
):
    answer = "café noir."
    source = "Prefix cafe\u0301 noir suffix"

    results = align_citations(
        answer,
        [source],
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=1.0,
            supported_answer_coverage=1.0,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "supported"
    assert results[0].citations

    citation = results[0].citations[0]
    expected = "cafe\u0301 noir"
    start = source.index(expected)

    assert citation.char_start == start
    assert citation.char_end == start + len(expected)
    assert citation.evidence == expected
    assert source[citation.char_start : citation.char_end] == citation.evidence


def test_align_citations_preserves_dash_variant_offsets_after_normalized_match() -> (
    None
):
    answer = "state-of-the-art device."
    source = "Prefix state–of–the–art device suffix"

    results = align_citations(
        answer,
        [source],
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=1.0,
            supported_answer_coverage=1.0,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "supported"
    assert results[0].citations

    citation = results[0].citations[0]
    expected = "state–of–the–art device"
    start = source.index(expected)

    assert citation.char_start == start
    assert citation.char_end == start + len(expected)
    assert citation.evidence == expected
    assert source[citation.char_start : citation.char_end] == citation.evidence


def test_align_citations_preserves_curly_apostrophe_offsets_after_normalized_match() -> (
    None
):
    answer = "company's revenue grew."
    source = "Prefix company’s revenue grew suffix"

    results = align_citations(
        answer,
        [source],
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=1.0,
            supported_answer_coverage=1.0,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "supported"
    assert results[0].citations

    citation = results[0].citations[0]
    expected = "company’s revenue grew"
    start = source.index(expected)

    assert citation.char_start == start
    assert citation.char_end == start + len(expected)
    assert citation.evidence == expected
    assert source[citation.char_start : citation.char_end] == citation.evidence


def test_build_candidates_tokenizes_each_source_once_for_overlapping_windows() -> None:
    source_text = "Alpha beta. Beta gamma. Gamma delta."
    config = CitationConfig(window_size_sentences=2, window_stride_sentences=1)
    source_passages = build_source_passages(
        normalize_sources([source_text]),
        SimpleSegmenter(),
        config,
    )

    class CountingTokenizer:
        def __init__(self) -> None:
            self._delegate = SimpleTokenizer()
            self.calls: list[str] = []

        def tokenize(self, text: str) -> TokenizedText:
            self.calls.append(text)
            return self._delegate.tokenize(text)

    tokenizer = CountingTokenizer()
    candidates = build_candidates(source_passages, tokenizer)

    assert len(source_passages) == 1
    assert len(source_passages[0][1]) == 2
    assert len(candidates) == 2
    assert tokenizer.calls == [source_text]


def test_slice_tokenized_text_clips_token_spans_to_passage_bounds() -> None:
    source_text = "Alpha beta. Beta gamma."
    config = CitationConfig(window_size_sentences=1, window_stride_sentences=1)
    source_passages = build_source_passages(
        normalize_sources([source_text]),
        SimpleSegmenter(),
        config,
    )

    class BoundaryTokenizer:
        def tokenize(self, text: str) -> TokenizedText:
            if text == source_text:
                return TokenizedText(
                    text=text,
                    token_ids=[1],
                    token_spans=[(0, len(text))],
                )
            raise AssertionError(f"Unexpected tokenize call for {text!r}")

    candidates = build_candidates(source_passages, BoundaryTokenizer())

    assert len(candidates) == 2
    first = candidates[0]
    second = candidates[1]
    assert first.token_spans == [(0, len(first.passage.text))]
    assert second.token_spans == [(0, len(second.passage.text))]


def test_build_candidates_indexes_token_boundaries_for_overlapping_windows() -> None:
    source_text = "A1 A2. B1 B2. C1 C2. D1 D2. E1 E2. F1 F2. G1 G2. H1 H2."
    config = CitationConfig(window_size_sentences=2, window_stride_sentences=1)
    source_passages = build_source_passages(
        normalize_sources([source_text]),
        SimpleSegmenter(),
        config,
    )

    class CountingSpans(list[tuple[int, int]]):
        def __init__(self, spans: list[tuple[int, int]]) -> None:
            super().__init__(spans)
            self.iteration_count = 0

        def __iter__(self):
            for span in super().__iter__():
                self.iteration_count += 1
                yield span

    base = SimpleTokenizer().tokenize(source_text)
    source_spans = CountingSpans(base.token_spans)

    class BoundaryCountingTokenizer:
        def tokenize(self, text: str) -> TokenizedText:
            if text != source_text:
                raise AssertionError(f"Unexpected tokenize call for {text!r}")
            return TokenizedText.model_construct(
                text=text,
                token_ids=list(base.token_ids),
                token_spans=source_spans,
            )

    candidates = build_candidates(source_passages, BoundaryCountingTokenizer())

    assert len(candidates) == len(source_passages[0][1]) == 7
    assert source_spans.iteration_count <= len(base.token_spans) * 2


def test_prepared_citation_corpus_matches_align_citations() -> None:
    answer = "Heat pumps reduce emissions."
    sources = [SourceDocument(id="energy", text="Heat pumps reduce emissions by 50%.")]
    config = CitationConfig(top_k=1)

    corpus = PreparedCitationCorpus.from_sources(sources, config=config)

    assert corpus.align(answer) == align_citations(answer, sources, config=config)


def test_prepared_citation_corpus_reuses_source_preparation() -> None:
    corpus = PreparedCitationCorpus.from_sources(
        [SourceDocument(id="energy", text="Heat pumps reduce emissions by 50%.")],
        config=CitationConfig(top_k=1),
    )

    first = corpus.align("Heat pumps reduce emissions.")
    second = corpus.align("Heat pumps reduce emissions by 50%.")

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].citations[0].source_id == "energy"
    assert second[0].citations[0].source_id == "energy"


def test_prepared_citation_corpus_is_exported() -> None:
    assert PreparedCitationCorpus is not None


def test_align_citations_wrapper_reports_metrics() -> None:
    captured = []

    align_citations(
        "Heat pumps reduce emissions.",
        [SourceDocument(id="energy", text="Heat pumps reduce emissions by 50%.")],
        on_metrics=captured.append,
    )

    assert len(captured) == 1
    assert captured[0].total_time_ms >= 0.0


def test_align_citations_metrics_include_source_and_query_embeddings() -> None:
    captured = []
    embedder = TimedEmbedder()

    align_citations(
        "Heat pumps reduce emissions.",
        [SourceDocument(id="energy", text="Heat pumps reduce emissions by 50%.")],
        embedder=embedder,
        on_metrics=captured.append,
    )

    assert captured[0].embedding_time_ms >= embedder.elapsed_ms * 0.9


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
