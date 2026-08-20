from collections.abc import Iterator
from unittest.mock import patch

from cite_right import SourceDocument, align_citations
from cite_right.citations import _answer_tokens_match_evidence
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.results import Alignment


class LyingAligner:
    def align(self, seq1, seq2) -> Alignment:
        return Alignment(
            score=100,
            token_start=0,
            token_end=1,
            query_start=0,
            query_end=len(seq1),
            matches=len(seq1),
        )


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


def test_answer_token_guard_stops_after_all_required_tokens_are_found() -> None:
    def evidence_tokens() -> Iterator[int]:
        yield 7
        yield 7
        raise AssertionError("guard scanned evidence after the answer was covered")

    assert _answer_tokens_match_evidence(
        answer_tokens=[7, 7],
        evidence_tokens=evidence_tokens(),
    )


def test_answer_token_guard_exact_sequence_avoids_frequency_map() -> None:
    with patch(
        "cite_right.citations.Counter",
        side_effect=AssertionError("exact sequence should not allocate a Counter"),
    ):
        assert _answer_tokens_match_evidence(
            answer_tokens=[1, 2, 2, 3],
            evidence_tokens=[1, 2, 2, 3],
        )


def test_answer_token_guard_exact_lists_use_native_comparison() -> None:
    class NoPythonIterationList(list[int]):
        def __iter__(self):
            raise AssertionError("exact token lists should use native comparison")

    assert _answer_tokens_match_evidence(
        answer_tokens=[1, 2, 3],
        evidence_tokens=NoPythonIterationList([1, 2, 3]),
    )


def test_answer_token_guard_trusts_complete_exact_alignment() -> None:
    with patch(
        "cite_right.citations._answer_tokens_match_evidence",
        side_effect=AssertionError("complete exact alignment needs no token rescan"),
    ):
        results = align_citations(
            "Heat pumps reduce emissions.",
            [
                SourceDocument(
                    id="energy",
                    text="Heat pumps reduce emissions.",
                )
            ],
            config=CitationConfig.strict().model_copy(
                update={"require_all_answer_tokens_in_evidence": True}
            ),
        )

    assert results[0].citations[0].source_id == "energy"


def test_answer_token_guard_does_not_trust_custom_match_count() -> None:
    results = align_citations(
        "Heat pumps reduce emissions.",
        [SourceDocument(id="energy", text="Heat unrelated unrelated unrelated.")],
        config=CitationConfig.strict().model_copy(
            update={"require_all_answer_tokens_in_evidence": True}
        ),
        aligner=LyingAligner(),
    )

    assert results[0].citations == []


def test_strict_exact_citation_rejects_numeric_token_mismatch() -> None:
    results = align_citations(
        "Ceres completes one orbit every 125 days.",
        [
            SourceDocument(
                id="science",
                text="Ceres completes one orbit every 124 days.",
            )
        ],
        config=CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.4,
            supported_answer_coverage=0.7,
            min_final_score=0.3,
            max_candidates_lexical=10,
            max_candidates_embedding=0,
            max_candidates_total=10,
            max_retrieval_support=2,
            max_citations_per_source=1,
            require_all_answer_tokens_in_evidence=True,
            weights=CitationWeights(
                alignment=1.0,
                answer_coverage=1.0,
                evidence_coverage=0.0,
                lexical=0.5,
                embedding=0.0,
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []
    assert len(results[0].retrieval_support) == 1
    assert results[0].retrieval_support[0].source_id == "science"


def test_strict_exact_citation_rejects_negation_token_mismatch() -> None:
    results = align_citations(
        "Congress shall make every law respecting an establishment of religion.",
        [
            SourceDocument(
                id="constitution",
                text="Congress shall make no law respecting an establishment of religion",
            )
        ],
        config=CitationConfig.strict().model_copy(
            update={"require_all_answer_tokens_in_evidence": True}
        ),
    )

    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []
    assert len(results[0].retrieval_support) == 1
    assert results[0].retrieval_support[0].source_id == "constitution"


def test_strict_exact_citation_does_not_split_u_s_abbreviation_into_supported_stub() -> (
    None
):
    results = align_citations(
        "CDC is the lead U.S. government agency for agricultural exports.",
        [
            SourceDocument(
                id="cdc",
                text="CDC is the lead U.S. government agency for public health.",
            )
        ],
        config=CitationConfig.strict().model_copy(
            update={"require_all_answer_tokens_in_evidence": True}
        ),
    )

    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []
    assert len(results[0].retrieval_support) == 1
    assert results[0].retrieval_support[0].source_id == "cdc"
