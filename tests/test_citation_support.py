from __future__ import annotations

from cite_right.citation_support import (
    citation_is_supported,
    demote_unsupported_secondaries,
)
from cite_right.citations import _build_retrieval_support, _span_status
from cite_right.core.citation_config import CitationConfig
from cite_right.core.prepared_corpus import Candidate, NormalizedSource
from cite_right.core.results import Citation, EvidenceSpan, RetrievalSupport
from cite_right.text.passage import Passage

ANSWER = "Backups are rotated every 90 days."
BEST_TEXT = "Backups are rotated every 90 days."
LOOSE_TEXT = "Backups are encrypted at rest."
SWAPPED_TEXT = "Backups are rotated every 30 days."


def _candidate(index: int, source_index: int, text: str) -> Candidate:
    source = NormalizedSource(
        source_id=f"doc-{source_index}",
        source_index=source_index,
        text=text,
        base_doc_offset=0,
        full_text=None,
    )
    passage = Passage(
        doc_char_start=0,
        doc_char_end=len(text),
        segment_start=0,
        segment_end=1,
        source_text=text,
    )
    return Candidate(
        global_index=index,
        source=source,
        passage=passage,
        token_ids=[],
        token_spans=[],
        token_set=frozenset(),
    )


def _citation(
    candidate: Candidate,
    *,
    score: float,
    coverage: float | None,
    spans: list[tuple[int, int]] | None = None,
) -> Citation:
    text = candidate.source.text
    ranges = spans or [(0, len(text))]
    components = {} if coverage is None else {"answer_coverage": coverage}
    components["lexical_score"] = 0.25
    return Citation(
        score=score,
        source_id=candidate.source.source_id,
        source_index=candidate.source.source_index,
        candidate_index=candidate.global_index,
        char_start=min(start for start, _ in ranges),
        char_end=max(end for _, end in ranges),
        evidence=text,
        evidence_spans=[
            EvidenceSpan(char_start=s, char_end=e, evidence=text[s:e])
            for s, e in ranges
        ],
        components=components,
    )


def _demote(
    citations: list[Citation],
    candidates: list[Candidate],
    *,
    cfg: CitationConfig | None = None,
    status: str = "supported",
    retrieval_support: list[RetrievalSupport] | None = None,
) -> tuple[list[Citation], list[RetrievalSupport]]:
    return demote_unsupported_secondaries(
        citations,
        retrieval_support or [],
        status=status,
        cfg=cfg or CitationConfig(),
        answer_text=ANSWER,
        candidates=candidates,
        build_support=_build_retrieval_support,
    )


def test_cross_source_low_coverage_secondary_is_demoted() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    loose = _candidate(1, 1, LOOSE_TEXT)
    citations = [
        _citation(best, score=1.2, coverage=1.0),
        _citation(loose, score=0.6, coverage=0.5),
    ]

    kept, support = _demote(citations, [best, loose])

    assert kept == [citations[0]]
    assert [entry.candidate_index for entry in support] == [1]
    assert support[0].source_id == "doc-1"
    assert support[0].passage_text == LOOSE_TEXT
    assert support[0].retrieval_score == 0.6
    assert support[0].lexical_score == 0.25


def test_secondary_at_exact_supported_coverage_is_kept() -> None:
    cfg = CitationConfig()
    best = _candidate(0, 0, BEST_TEXT)
    other = _candidate(1, 1, BEST_TEXT)
    citations = [
        _citation(best, score=1.2, coverage=1.0),
        _citation(other, score=0.9, coverage=cfg.supported_answer_coverage),
    ]

    kept, support = _demote(citations, [best, other], cfg=cfg)

    assert kept == citations
    assert support == []


def test_contradicted_full_coverage_secondary_is_demoted() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    swapped = _candidate(1, 1, SWAPPED_TEXT)
    citations = [
        _citation(best, score=1.2, coverage=1.0),
        _citation(swapped, score=1.1, coverage=1.0),
    ]

    kept, support = _demote(citations, [best, swapped])

    assert kept == [citations[0]]
    assert [entry.candidate_index for entry in support] == [1]


def test_same_source_low_coverage_secondary_is_demoted() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    sibling = _candidate(1, 0, LOOSE_TEXT)
    citations = [
        _citation(best, score=1.2, coverage=1.0),
        _citation(sibling, score=0.5, coverage=0.4),
    ]

    kept, support = _demote(citations, [best, sibling])

    assert kept == [citations[0]]
    assert [entry.source_id for entry in support] == ["doc-0"]


def test_partial_span_keeps_citations_unchanged() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    loose = _candidate(1, 1, LOOSE_TEXT)
    citations = [
        _citation(best, score=0.8, coverage=0.5),
        _citation(loose, score=0.6, coverage=0.3),
    ]

    kept, support = _demote(citations, [best, loose], status="partial")

    assert kept == citations
    assert support == []


def test_demoted_candidate_already_in_retrieval_support_is_not_duplicated() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    loose = _candidate(1, 1, LOOSE_TEXT)
    existing = _build_retrieval_support(loose, 0.9, 0.3, 0.0)
    citations = [
        _citation(best, score=1.2, coverage=1.0),
        _citation(loose, score=0.6, coverage=0.5),
    ]

    kept, support = _demote(citations, [best, loose], retrieval_support=[existing])

    assert kept == [citations[0]]
    assert support == [existing]


def test_kept_secondaries_keep_their_order_when_one_is_demoted() -> None:
    cands = [_candidate(i, i, BEST_TEXT) for i in range(4)]
    citations = [
        _citation(cands[0], score=1.3, coverage=1.0),
        _citation(cands[1], score=0.9, coverage=0.9),
        _citation(cands[2], score=0.9, coverage=0.2),
        _citation(cands[3], score=0.9, coverage=0.8),
    ]

    kept, support = _demote(citations, cands)

    assert [c.candidate_index for c in kept] == [0, 1, 3]
    assert [s.candidate_index for s in support] == [2]


def test_secondary_without_answer_coverage_is_demoted_without_error() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    bare = _candidate(1, 1, LOOSE_TEXT)
    citations = [
        _citation(best, score=1.2, coverage=1.0),
        _citation(bare, score=0.6, coverage=None),
    ]

    kept, support = _demote(citations, [best, bare])

    assert kept == [citations[0]]
    assert [entry.candidate_index for entry in support] == [1]


def test_multi_span_secondary_with_enough_coverage_is_kept() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    multi = _candidate(1, 1, "Backups are rotated every 90 days and logs are archived.")
    citations = [
        _citation(best, score=1.2, coverage=1.0),
        _citation(multi, score=0.9, coverage=0.8, spans=[(0, 15), (16, 29)]),
    ]

    kept, support = _demote(citations, [best, multi])

    assert kept == citations
    assert support == []


def test_best_citation_is_never_removed() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    citations = [_citation(best, score=1.2, coverage=1.0)]

    kept, support = _demote(citations, [best])

    assert kept == citations
    assert support == []


def test_citation_is_supported_matches_span_status() -> None:
    cfg = CitationConfig()
    best = _candidate(0, 0, BEST_TEXT)
    swapped = _candidate(1, 1, SWAPPED_TEXT)
    good = _citation(best, score=1.0, coverage=1.0)
    weak = _citation(best, score=1.0, coverage=0.1)
    contradicted = _citation(swapped, score=1.0, coverage=1.0)

    for citation in (good, weak, contradicted):
        expected = (
            "supported"
            if citation_is_supported(citation, cfg, ANSWER, [best, swapped])
            else "partial"
        )
        assert (
            _span_status([citation], cfg, ANSWER, candidates=[best, swapped])
            == expected
        )
    assert citation_is_supported(good, cfg, ANSWER, [best, swapped])
    assert not citation_is_supported(weak, cfg, ANSWER, [best, swapped])
    assert not citation_is_supported(contradicted, cfg, ANSWER, [best, swapped])


def test_max_retrieval_support_zero_drops_demoted_entries_via_align() -> None:
    from cite_right import SourceDocument, align_citations

    cfg = CitationConfig(max_retrieval_support=0)
    result = align_citations(
        ANSWER,
        [
            SourceDocument(id="retention", text="Backups are rotated every 90 days."),
            SourceDocument(
                id="security",
                text="Backups are stored offsite. Passwords rotate quarterly.",
            ),
        ],
        config=cfg,
    )

    assert any(span.status == "supported" for span in result)
    for span in result:
        assert span.retrieval_support == []
        if span.status == "supported":
            assert all(
                c.components["answer_coverage"] >= cfg.supported_answer_coverage
                for c in span.citations
            )


def test_demoted_citation_on_an_already_cited_candidate_adds_no_support() -> None:
    best = _candidate(0, 0, BEST_TEXT)
    citations = [
        _citation(best, score=1.2, coverage=1.0, spans=[(0, 15)]),
        _citation(best, score=0.5, coverage=0.2, spans=[(16, 34)]),
    ]

    kept, support = _demote(citations, [best])

    assert kept == [citations[0]]
    assert support == []
