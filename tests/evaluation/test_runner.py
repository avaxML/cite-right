from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any, Literal, get_args

import pytest

from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.results import (
    AnswerSpan,
    Citation,
    EvidenceSpan,
    RetrievalSupport,
    SpanCitations,
)
from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase


def test_runner_models_are_frozen_and_backend_literal_is_python_or_rust() -> None:
    runner = _runner_module()

    assert get_args(runner.Backend) == ("python", "rust")
    assert get_args(runner.RunErrorCode) == (
        "exception",
        "timeout",
        "invalid_answer_span",
        "ambiguous_answer_span",
        "unmappable_answer_span",
    )

    config = runner.canonicalize_config(CitationConfig())
    run = runner.CaseRun(
        case_id="case-1",
        backend="python",
        config=config,
        outputs=(),
        output_unit_ids=(),
        duration_ns=17,
        error=None,
    )

    with pytest.raises((TypeError, AttributeError)):
        run.backend = "rust"  # type: ignore[misc]

    with pytest.raises((TypeError, AttributeError)):
        config.sha256 = "mutated"  # type: ignore[misc]


def test_execute_case_records_supported_exact_output_and_single_unit_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    exact = _span_citations(
        text="Alpha section",
        start=0,
        end=13,
        citations=(
            _citation("source-alpha", 0, 13, "Alpha support"),
        ),
        status="supported",
    )
    _patch_align_citations(monkeypatch, runner, outputs=[exact])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(5, 23),
    )

    assert run.case_id == case.case_id
    assert run.backend == "python"
    assert run.outputs == (exact,)
    assert run.output_unit_ids == (("unit-alpha",),)
    assert run.duration_ns == 18
    assert run.error is None


def test_execute_case_maps_boundary_crossing_answer_spans_to_the_union_of_touched_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_adjacent_units()
    crossing = _span_citations(
        text="AlphaBeta",
        start=0,
        end=9,
        citations=(
            _citation("source-shared", 0, 9, "AlphaBeta"),
        ),
        status="supported",
    )
    _patch_align_citations(monkeypatch, runner, outputs=[crossing])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(11, 29),
    )

    assert run.outputs == (crossing,)
    assert run.output_unit_ids == (("unit-alpha", "unit-beta"),)
    assert run.error is None


def test_execute_case_reports_gap_only_answer_spans_as_unmappable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    gap_span = _span_citations(
        text=" // ",
        start=13,
        end=17,
        status="unsupported",
    )
    _patch_align_citations(monkeypatch, runner, outputs=[gap_span])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(100, 104),
    )

    assert run.outputs == (gap_span,)
    assert run.output_unit_ids == ()
    assert run.error is not None
    assert run.error.code == "unmappable_answer_span"
    assert run.error.output_index == 0


def test_execute_case_reports_answer_text_mismatches_as_ambiguous_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    mismatched = _span_citations(
        text="Alpha mismatch",
        start=0,
        end=13,
        citations=(
            _citation("source-alpha", 0, 13, "Alpha support"),
        ),
        status="supported",
    )
    _patch_align_citations(monkeypatch, runner, outputs=[mismatched])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(200, 211),
    )

    assert run.error is not None
    assert run.error.code == "ambiguous_answer_span"
    assert run.error.output_index == 0


def test_execute_case_reports_invalid_answer_offsets_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    invalid = _span_citations(
        text="Alpha section",
        start=-1,
        end=13,
        citations=(
            _citation("source-alpha", 0, 13, "Alpha support"),
        ),
        status="supported",
    )
    _patch_align_citations(monkeypatch, runner, outputs=[invalid])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(300, 306),
    )

    assert run.error is not None
    assert run.error.code == "invalid_answer_span"
    assert run.error.output_index == 0


def test_execute_case_preserves_retrieval_only_support_separately_from_exact_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    retrieval_only = _span_citations(
        text="Beta section",
        start=17,
        end=29,
        retrieval_support=(
            RetrievalSupport(
                retrieval_score=0.9,
                source_id="source-beta",
                source_index=0,
                candidate_index=0,
                passage_char_start=14,
                passage_char_end=26,
                passage_text="Beta support",
                embedding_score=0.0,
                lexical_score=0.9,
            ),
        ),
        status="unsupported",
    )
    _patch_align_citations(monkeypatch, runner, outputs=[retrieval_only])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(400, 420),
    )

    assert run.error is None
    assert run.outputs[0].citations == []
    assert run.outputs[0].retrieval_support == list(retrieval_only.retrieval_support)
    assert run.output_unit_ids == (("unit-beta",),)


def test_execute_case_captures_runtime_exceptions_as_run_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()

    def boom(*args: Any, **kwargs: Any) -> list[SpanCitations]:
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(runner, "align_citations", boom)

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(500, 509),
    )

    assert run.outputs == ()
    assert run.output_unit_ids == ()
    assert run.error is not None
    assert run.error.code == "exception"
    assert run.error.exception_type == "RuntimeError"
    assert "backend exploded" in run.error.message
    assert run.duration_ns == 9


def test_execute_case_captures_timeouts_with_an_injected_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    exact = _span_citations(
        text="Alpha section",
        start=0,
        end=13,
        citations=(
            _citation("source-alpha", 0, 13, "Alpha support"),
        ),
        status="supported",
    )
    _patch_align_citations(monkeypatch, runner, outputs=[exact])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(1_000, 1_250),
        timeout_ns=200,
    )

    assert run.outputs == ()
    assert run.output_unit_ids == ()
    assert run.error is not None
    assert run.error.code == "timeout"
    assert run.duration_ns == 250


def test_execute_case_allows_empty_outputs_without_promoting_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    _patch_align_citations(monkeypatch, runner, outputs=[])

    run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(700, 700),
    )

    assert run.outputs == ()
    assert run.output_unit_ids == ()
    assert run.error is None
    assert run.duration_ns == 0


def test_canonicalize_config_ignores_input_order_when_serializing_and_hashing() -> None:
    runner = _runner_module()
    first = {
        "top_k": 1,
        "weights": {
            "embedding": 0.0,
            "lexical": 0.0,
        },
        "min_alignment_score": 5,
    }
    second = {
        "min_alignment_score": 5,
        "weights": {
            "lexical": 0.0,
            "embedding": 0.0,
        },
        "top_k": 1,
    }

    left = runner.canonicalize_config(first)
    right = runner.canonicalize_config(second)

    assert left == right
    assert left.canonical_json == canonical_json_bytes(first).decode("utf-8")
    assert left.sha256 == sha256_hex(canonical_json_bytes(first))


def test_execute_case_records_backend_specific_runs_in_a_comparable_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_module()
    case = _case_with_gap_between_units()
    outputs = [
        _span_citations(
            text="Alpha section",
            start=0,
            end=13,
            citations=(
                _citation("source-alpha", 0, 13, "Alpha support"),
            ),
            status="supported",
        )
    ]
    _patch_align_citations(monkeypatch, runner, outputs=outputs)

    python_run = runner.execute_case(
        case=case,
        backend="python",
        config=_strict_config(),
        clock=_clock(0, 10),
    )
    rust_run = runner.execute_case(
        case=case,
        backend="rust",
        config=_strict_config(),
        clock=_clock(0, 10),
    )

    assert python_run.backend == "python"
    assert rust_run.backend == "rust"
    assert python_run.case_id == rust_run.case_id == case.case_id
    assert python_run.config == rust_run.config
    assert python_run.outputs == rust_run.outputs
    assert python_run.output_unit_ids == rust_run.output_unit_ids
    assert python_run.error == rust_run.error is None


def _runner_module() -> Any:
    return importlib.import_module("evaluation.runner")


def _patch_align_citations(
    monkeypatch: pytest.MonkeyPatch,
    runner: Any,
    *,
    outputs: Sequence[SpanCitations],
) -> None:
    def fake_align_citations(*args: Any, **kwargs: Any) -> list[SpanCitations]:
        return list(outputs)

    monkeypatch.setattr(runner, "align_citations", fake_align_citations)


def _strict_config() -> CitationConfig:
    return CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.8,
        supported_answer_coverage=0.8,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )


def _clock(*ticks: int) -> Callable[[], int]:
    values = iter(ticks)
    last = ticks[-1]

    def fake_clock() -> int:
        nonlocal last
        try:
            last = next(values)
        except StopIteration:
            return last
        return last

    return fake_clock


def _span_citations(
    *,
    text: str,
    start: int,
    end: int,
    citations: Sequence[Citation] = (),
    retrieval_support: Sequence[RetrievalSupport] = (),
    status: Literal["supported", "partial", "unsupported"],
) -> SpanCitations:
    return SpanCitations(
        answer_span=AnswerSpan(
            text=text,
            char_start=start,
            char_end=end,
            kind="sentence",
            paragraph_index=0,
            sentence_index=0,
        ),
        citations=list(citations),
        retrieval_support=list(retrieval_support),
        status=status,
    )


def _citation(
    source_id: str,
    char_start: int,
    char_end: int,
    evidence: str,
) -> Citation:
    return Citation(
        score=1.0,
        source_id=source_id,
        source_index=0,
        candidate_index=0,
        char_start=char_start,
        char_end=char_end,
        evidence=evidence,
        evidence_spans=[
            EvidenceSpan(
                char_start=char_start,
                char_end=char_end,
                evidence=evidence,
            )
        ],
        components={
            "answer_coverage": 1.0,
            "alignment_score": 1.0,
        },
    )


def _case_with_gap_between_units() -> EvaluationCase:
    answer = "Alpha section // Beta section"
    source_alpha = "Alpha support"
    source_beta = "Beta support"
    return EvaluationCase.model_validate(
        {
            "case_id": "case-gap",
            "dataset_version": "1.0.0",
            "split": "dev",
            "document_family_id": "family-gap",
            "transformation_family_id": "transformation-gap",
            "provenance": {"kind": "authored"},
            "sources": (
                {"source_id": "source-alpha", "text": source_alpha},
                {"source_id": "source-beta", "text": source_beta},
            ),
            "answer": answer,
            "evaluation_units": (
                {
                    "unit_id": "unit-alpha",
                    "answer_span": {"start": 0, "end": 13},
                    "text": "Alpha section",
                    "claims": (
                        {
                            "claim_id": "claim-alpha",
                            "answer_span": {"start": 0, "end": 13},
                            "text": "Alpha section",
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "req-alpha",
                                    "alternatives": (
                                        {
                                            "source_id": "source-alpha",
                                            "spans": (
                                                {
                                                    "start": 0,
                                                    "end": len(source_alpha),
                                                },
                                            ),
                                        },
                                    ),
                                },
                            ),
                        },
                    ),
                },
                {
                    "unit_id": "unit-beta",
                    "answer_span": {"start": 17, "end": 29},
                    "text": "Beta section",
                    "claims": (
                        {
                            "claim_id": "claim-beta",
                            "answer_span": {"start": 17, "end": 29},
                            "text": "Beta section",
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "req-beta",
                                    "alternatives": (
                                        {
                                            "source_id": "source-beta",
                                            "spans": (
                                                {
                                                    "start": 0,
                                                    "end": len(source_beta),
                                                },
                                            ),
                                        },
                                    ),
                                },
                            ),
                            "acceptable_retrieval_source_ids": ("source-beta",),
                        },
                    ),
                },
            ),
        }
    )


def _case_with_adjacent_units() -> EvaluationCase:
    answer = "AlphaBeta"
    source_text = "AlphaBeta evidence"
    return EvaluationCase.model_validate(
        {
            "case_id": "case-adjacent",
            "dataset_version": "1.0.0",
            "split": "dev",
            "document_family_id": "family-adjacent",
            "transformation_family_id": "transformation-adjacent",
            "provenance": {"kind": "authored"},
            "sources": (
                {"source_id": "source-shared", "text": source_text},
            ),
            "answer": answer,
            "evaluation_units": (
                {
                    "unit_id": "unit-alpha",
                    "answer_span": {"start": 0, "end": 5},
                    "text": "Alpha",
                    "claims": (
                        {
                            "claim_id": "claim-alpha",
                            "answer_span": {"start": 0, "end": 5},
                            "text": "Alpha",
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "req-alpha",
                                    "alternatives": (
                                        {
                                            "source_id": "source-shared",
                                            "spans": ({"start": 0, "end": 5},),
                                        },
                                    ),
                                },
                            ),
                        },
                    ),
                },
                {
                    "unit_id": "unit-beta",
                    "answer_span": {"start": 5, "end": 9},
                    "text": "Beta",
                    "claims": (
                        {
                            "claim_id": "claim-beta",
                            "answer_span": {"start": 5, "end": 9},
                            "text": "Beta",
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "req-beta",
                                    "alternatives": (
                                        {
                                            "source_id": "source-shared",
                                            "spans": ({"start": 5, "end": 9},),
                                        },
                                    ),
                                },
                            ),
                        },
                    ),
                },
            ),
        }
    )
