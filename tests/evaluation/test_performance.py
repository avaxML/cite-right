from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase


def test_performance_module_exists_to_anchor_the_contract_suite() -> None:
    importlib.import_module("evaluation.performance")


def test_performance_models_are_frozen_and_literal_contracts_are_stable() -> None:
    performance = _performance_module_or_skip()

    assert get_args(performance.Backend) == ("python", "rust")
    assert get_args(performance.FailureStage) == ("prepare_corpus", "answer", "worker")

    sample = _sample_measurement(
        performance,
        case_id="case-1",
        document_family_id="family-a",
        split="dev",
        prepared_corpus_duration_ns=11,
        answer_duration_ns=19,
        total_duration_ns=30,
        peak_memory_bytes=256,
    )
    environment = performance.EnvironmentMetadata(
        python_implementation="CPython",
        python_version="3.11.9",
        platform="macOS-15.0-arm64",
        package_version="0.1.0",
        backend="python",
        config_sha256="a" * 64,
        workload_sha256="b" * 64,
    )

    with pytest.raises((ValidationError, TypeError, AttributeError)):
        sample.total_duration_ns = 1  # type: ignore[misc]

    with pytest.raises((ValidationError, TypeError, AttributeError)):
        environment.backend = "rust"  # type: ignore[misc]


def test_duration_models_reject_negative_counts_and_durations() -> None:
    performance = _performance_module_or_skip()

    with pytest.raises(ValidationError, match="duration and count fields must be non-negative"):
        performance.DurationSummary(
            sample_count=-1,
            total_duration_ns=0,
            median_duration_ns=0,
            p95_duration_ns=0,
        )

    with pytest.raises(ValidationError, match="duration and count fields must be non-negative"):
        _sample_measurement(
            performance,
            case_id="case-1",
            document_family_id="family-a",
            split="dev",
            prepared_corpus_duration_ns=-1,
            answer_duration_ns=1,
            total_duration_ns=0,
        )

    with pytest.raises(ValidationError, match="cache snapshot fields must be non-negative"):
        performance.CacheSnapshot(entries=-1, bytes=0)


def test_summarize_measurements_excludes_warmups_from_measured_sample_count() -> None:
    performance = _performance_module_or_skip()

    report = performance.summarize_measurements(
        samples=(
            _sample_measurement(
                performance,
                case_id="warmup",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=5,
                answer_duration_ns=15,
                total_duration_ns=20,
            ),
            _sample_measurement(
                performance,
                case_id="case-1",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=7,
                answer_duration_ns=23,
                total_duration_ns=30,
            ),
            _sample_measurement(
                performance,
                case_id="case-2",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=11,
                answer_duration_ns=29,
                total_duration_ns=40,
            ),
        ),
        warmup_count=1,
        backend="python",
        workload=_workload_selection(performance, selected_case_ids=("warmup", "case-1", "case-2")),
        environment=_environment_metadata(performance, backend="python"),
    )

    assert report.warmup_count == 1
    assert report.measured_sample_count == 2


def test_summarize_measurements_excludes_warmup_failures_from_measured_report() -> None:
    performance = _performance_module_or_skip()

    report = performance.summarize_measurements(
        samples=(
            _sample_measurement(
                performance,
                case_id="warmup-fail",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=5,
                answer_duration_ns=7,
                total_duration_ns=12,
                failure=_failure_record(
                    performance,
                    case_id="warmup-fail",
                    document_family_id="family-a",
                    split="dev",
                    stage="answer",
                    error_type="RuntimeError",
                    message="warmup boom",
                ),
            ),
            _sample_measurement(
                performance,
                case_id="case-1",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=11,
                answer_duration_ns=19,
                total_duration_ns=30,
            ),
        ),
        warmup_count=1,
        backend="python",
        workload=_workload_selection(
            performance, selected_case_ids=("warmup-fail", "case-1")
        ),
        environment=_environment_metadata(performance, backend="python"),
    )

    assert report.warmup_count == 1
    assert report.measured_sample_count == 1
    assert report.failure_count == 0
    assert report.failures == ()
    assert report.prepared_corpus.total_duration_ns == 11
    assert report.answer.total_duration_ns == 19
    assert report.end_to_end.total_duration_ns == 30


def test_summarize_measurements_uses_median_for_measured_end_to_end_samples() -> None:
    performance = _performance_module_or_skip()

    report = _summarize_with_measured_totals(
        performance,
        totals=(10, 30, 70),
        prepared=(1, 3, 7),
        answer=(9, 27, 63),
    )

    assert report.end_to_end.median_duration_ns == 30


def test_summarize_measurements_uses_nearest_rank_p95_with_ceil_indexing() -> None:
    performance = _performance_module_or_skip()

    report = _summarize_with_measured_totals(
        performance,
        totals=(10, 20, 30, 40, 50, 60),
        prepared=(1, 2, 3, 4, 5, 6),
        answer=(9, 18, 27, 36, 45, 54),
    )

    assert report.end_to_end.p95_duration_ns == 60


def test_summarize_measurements_derives_throughput_from_total_measured_duration() -> None:
    performance = _performance_module_or_skip()

    report = _summarize_with_measured_totals(
        performance,
        totals=(10, 20, 30),
        prepared=(1, 2, 3),
        answer=(9, 18, 27),
    )

    assert report.throughput_cases_per_second == pytest.approx(50_000_000.0)


def test_summarize_measurements_keeps_prepare_and_answer_durations_separate() -> None:
    performance = _performance_module_or_skip()

    report = _summarize_with_measured_totals(
        performance,
        totals=(10, 20),
        prepared=(1, 4),
        answer=(9, 16),
    )

    assert report.prepared_corpus.total_duration_ns == 5
    assert report.answer.total_duration_ns == 25
    assert report.end_to_end.total_duration_ns == 30


def test_summarize_measurements_keeps_measured_failures_in_denominators() -> None:
    performance = _performance_module_or_skip()

    report = performance.summarize_measurements(
        samples=(
            _sample_measurement(
                performance,
                case_id="case-success",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=10,
                answer_duration_ns=20,
                total_duration_ns=30,
            ),
            _sample_measurement(
                performance,
                case_id="case-answer-fail",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=11,
                answer_duration_ns=22,
                total_duration_ns=33,
                failure=_failure_record(
                    performance,
                    case_id="case-answer-fail",
                    document_family_id="family-a",
                    split="dev",
                    stage="answer",
                    error_type="RuntimeError",
                    message="boom",
                ),
            ),
            _sample_measurement(
                performance,
                case_id="case-prepare-fail",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=5,
                answer_duration_ns=0,
                total_duration_ns=5,
                failure=_failure_record(
                    performance,
                    case_id="case-prepare-fail",
                    document_family_id="family-a",
                    split="dev",
                    stage="prepare_corpus",
                    error_type="RuntimeError",
                    message="broken prepare",
                ),
            ),
        ),
        warmup_count=0,
        backend="python",
        workload=_workload_selection(
            performance,
            selected_case_ids=(
                "case-success",
                "case-answer-fail",
                "case-prepare-fail",
            ),
        ),
        environment=_environment_metadata(performance, backend="python"),
    )

    assert report.measured_sample_count == 3
    assert report.failure_count == 2
    assert report.prepared_corpus.sample_count == 3
    assert report.prepared_corpus.total_duration_ns == 26
    assert report.answer.sample_count == 2
    assert report.answer.total_duration_ns == 42
    assert report.end_to_end.sample_count == 3
    assert report.end_to_end.total_duration_ns == 68
    assert report.throughput_cases_per_second == pytest.approx(
        3 * 1_000_000_000 / 68
    )


def test_summarize_measurements_reports_retained_cache_when_observable() -> None:
    performance = _performance_module_or_skip()

    report = performance.summarize_measurements(
        samples=(
            _sample_measurement(
                performance,
                case_id="case-1",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=5,
                answer_duration_ns=15,
                total_duration_ns=20,
                cache_before=performance.CacheSnapshot(entries=9, bytes=900),
                cache_after=performance.CacheSnapshot(entries=6, bytes=600),
            ),
            _sample_measurement(
                performance,
                case_id="case-2",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=5,
                answer_duration_ns=15,
                total_duration_ns=20,
                cache_before=performance.CacheSnapshot(entries=6, bytes=600),
                cache_after=performance.CacheSnapshot(entries=4, bytes=400),
            ),
        ),
        warmup_count=0,
        backend="python",
        workload=_workload_selection(performance, selected_case_ids=("case-1", "case-2")),
        environment=_environment_metadata(performance, backend="python"),
    )

    assert report.cache_retention is not None
    assert report.cache_retention.before.entries == 9
    assert report.cache_retention.after.entries == 4
    assert report.cache_retention.retained_entries == 4
    assert report.cache_retention.retained_bytes == 400


def test_summarize_measurements_reports_peak_memory_from_the_measured_maximum() -> None:
    performance = _performance_module_or_skip()

    report = performance.summarize_measurements(
        samples=(
            _sample_measurement(
                performance,
                case_id="case-1",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=5,
                answer_duration_ns=15,
                total_duration_ns=20,
                peak_memory_bytes=100,
            ),
            _sample_measurement(
                performance,
                case_id="case-2",
                document_family_id="family-a",
                split="dev",
                prepared_corpus_duration_ns=5,
                answer_duration_ns=15,
                total_duration_ns=20,
                peak_memory_bytes=250,
            ),
        ),
        warmup_count=0,
        backend="python",
        workload=_workload_selection(performance, selected_case_ids=("case-1", "case-2")),
        environment=_environment_metadata(performance, backend="python"),
    )

    assert report.peak_memory_bytes == 250


def test_build_environment_metadata_captures_runtime_package_backend_and_hashes() -> None:
    performance = _performance_module_or_skip()
    workload = _workload_selection(
        performance,
        seed=17,
        selected_case_ids=("case-1", "case-2"),
        selected_document_family_ids=("family-a",),
    )
    config = {"beta": {"delta": 4, "charlie": 3}, "alpha": 1}

    metadata = performance.build_environment_metadata(
        backend="python",
        config=config,
        workload=workload,
    )

    assert metadata.python_implementation == platform.python_implementation()
    assert metadata.python_version == platform.python_version()
    assert metadata.platform == platform.platform()
    assert metadata.package_version == "0.1.0"
    assert metadata.backend == "python"
    assert metadata.config_sha256 == sha256_hex(canonical_json_bytes(config))
    assert metadata.workload_sha256 == sha256_hex(
        canonical_json_bytes(workload.model_dump(mode="python"))
    )


def test_select_workload_is_deterministic_for_seed_and_family_filter_and_excludes_holdout() -> None:
    performance = _performance_module_or_skip()
    cases = (
        _case(case_id="train-a", split="train", document_family_id="family-a"),
        _case(case_id="dev-a", split="dev", document_family_id="family-a"),
        _case(case_id="holdout-a", split="holdout", document_family_id="family-a"),
        _case(case_id="train-b", split="train", document_family_id="family-b"),
    )

    first = performance.select_workload(cases, seed=7, family_filter=("family-a",))
    second = performance.select_workload(cases, seed=7, family_filter=("family-a",))

    assert first.seed == 7
    assert first.family_filter == ("family-a",)
    assert first.selected_case_ids == second.selected_case_ids
    assert set(first.selected_case_ids) == {"train-a", "dev-a"}
    assert first.selected_document_family_ids == ("family-a",)
    assert "holdout-a" not in first.selected_case_ids


def test_run_benchmark_uses_fake_clock_for_duration_math_and_reports_failures_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance_module_or_skip()
    cases = (
        _case(case_id="warmup", split="dev", document_family_id="family-a"),
        _case(case_id="case-ok", split="dev", document_family_id="family-a"),
        _case(case_id="case-fail", split="dev", document_family_id="family-a"),
    )
    clock = _fake_clock(
        0,
        5,
        5,
        17,
        100,
        110,
        110,
        140,
        200,
        203,
        203,
        211,
    )

    def prepare_corpus(case: EvaluationCase) -> str:
        return f"prepared:{case.case_id}"

    def answer_case(case: EvaluationCase, prepared_corpus: str) -> str:
        assert prepared_corpus == f"prepared:{case.case_id}"
        if case.case_id == "case-fail":
            raise RuntimeError("boom")
        return f"answer:{case.case_id}"

    monkeypatch.setattr(
        performance,
        "select_workload",
        lambda _cases, *, seed, family_filter=(): performance.WorkloadSelection(
            seed=seed,
            family_filter=tuple(family_filter),
            selected_case_ids=("warmup", "case-ok", "case-fail"),
            selected_document_family_ids=("family-a",),
        ),
    )

    report = performance.run_benchmark(
        cases=cases,
        backend="python",
        config={"strict": True},
        seed=13,
        warmup_count=1,
        clock=clock,
        prepare_corpus=prepare_corpus,
        answer_case=answer_case,
    )

    assert report.measured_sample_count == 2
    assert report.failure_count == 1
    assert report.prepared_corpus.total_duration_ns == 13
    assert report.prepared_corpus.sample_count == 2
    assert report.answer.total_duration_ns == 38
    assert report.answer.sample_count == 2
    assert report.end_to_end.total_duration_ns == 51
    assert report.end_to_end.sample_count == 2
    assert report.throughput_cases_per_second == pytest.approx(
        2 * 1_000_000_000 / 51
    )
    assert len(report.failures) == 1
    assert report.failures[0].case_id == "case-fail"
    assert report.failures[0].stage == "answer"
    assert report.failures[0].error_type == "RuntimeError"
    assert "boom" in report.failures[0].message


def test_run_benchmark_uses_workload_selection_order_even_when_input_is_reversed() -> None:
    performance = _performance_module_or_skip()
    cases = (
        _case(case_id="case-a", split="dev", document_family_id="family-b"),
        _case(case_id="case-b", split="train", document_family_id="family-a"),
        _case(case_id="case-c", split="dev", document_family_id="family-a"),
    )

    forward_report = performance.run_benchmark(
        cases=cases,
        backend="python",
        config={"strict": True},
        seed=13,
        warmup_count=1,
        clock=_fake_clock(0, 1, 1, 2, 10, 11, 11, 12, 20, 21, 21, 22),
        prepare_corpus=lambda case: case.case_id,
        answer_case=lambda case, prepared_corpus: prepared_corpus,
    )
    reversed_report = performance.run_benchmark(
        cases=tuple(reversed(cases)),
        backend="python",
        config={"strict": True},
        seed=13,
        warmup_count=1,
        clock=_fake_clock(0, 1, 1, 2, 10, 11, 11, 12, 20, 21, 21, 22),
        prepare_corpus=lambda case: case.case_id,
        answer_case=lambda case, prepared_corpus: prepared_corpus,
    )

    assert tuple(sample.case_id for sample in forward_report.samples) == tuple(
        sample.case_id for sample in reversed_report.samples
    )
    assert forward_report.samples[0].case_id == reversed_report.samples[0].case_id
    assert forward_report.measured_sample_count == reversed_report.measured_sample_count
    assert forward_report.end_to_end.total_duration_ns == (
        reversed_report.end_to_end.total_duration_ns
    )


def test_run_benchmark_rejects_duplicate_input_case_ids() -> None:
    performance = _performance_module_or_skip()
    cases = (
        _case(case_id="case-1", split="dev", document_family_id="family-a"),
        _case(case_id="case-1", split="train", document_family_id="family-b"),
    )

    original_select_workload = performance.select_workload
    performance.select_workload = (  # type: ignore[method-assign]
        lambda _cases, *, seed, family_filter=(): performance.WorkloadSelection(
            seed=seed,
            family_filter=tuple(family_filter),
            selected_case_ids=("case-1",),
            selected_document_family_ids=("family-a", "family-b"),
        )
    )
    with pytest.raises(ValueError, match="duplicate case ids in benchmark input"):
        try:
            performance.run_benchmark(
                cases=cases,
                backend="python",
                config={"strict": True},
                seed=7,
                prepare_corpus=lambda case: case.case_id,
                answer_case=lambda case, prepared_corpus: prepared_corpus,
            )
        finally:
            performance.select_workload = original_select_workload  # type: ignore[method-assign]


def test_run_benchmark_rejects_missing_selected_case_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance_module_or_skip()
    cases = (
        _case(case_id="case-1", split="dev", document_family_id="family-a"),
        _case(case_id="case-2", split="dev", document_family_id="family-a"),
    )

    def fake_select_workload(
        _cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
        *,
        seed: int,
        family_filter: tuple[str, ...] | list[str] = (),
    ) -> Any:
        del _cases, seed, family_filter
        return performance.WorkloadSelection(
            seed=7,
            family_filter=(),
            selected_case_ids=("missing-case",),
            selected_document_family_ids=("family-a",),
        )

    monkeypatch.setattr(performance, "select_workload", fake_select_workload)

    with pytest.raises(ValueError, match="missing case ids from benchmark input"):
        performance.run_benchmark(
            cases=cases,
            backend="python",
            config={"strict": True},
            seed=7,
            prepare_corpus=lambda case: case.case_id,
            answer_case=lambda case, prepared_corpus: prepared_corpus,
        )


def test_benchmark_report_payload_has_canonical_serialization() -> None:
    performance = _performance_module_or_skip()

    report = _summarize_with_measured_totals(
        performance,
        totals=(20, 40),
        prepared=(5, 7),
        answer=(15, 33),
    )
    payload = report.model_dump(mode="python")
    reordered = {
        "workload": payload["workload"],
        "samples": payload["samples"],
        "prepared_corpus": payload["prepared_corpus"],
        "peak_memory_bytes": payload["peak_memory_bytes"],
        "measured_sample_count": payload["measured_sample_count"],
        "failures": payload["failures"],
        "environment": payload["environment"],
        "end_to_end": payload["end_to_end"],
        "cache_retention": payload["cache_retention"],
        "backend": payload["backend"],
        "answer": payload["answer"],
        "warmup_count": payload["warmup_count"],
        "throughput_cases_per_second": payload["throughput_cases_per_second"],
        "failure_count": payload["failure_count"],
    }

    assert canonical_json_bytes(payload) == canonical_json_bytes(reordered)


def test_smoke_worker_runs_in_a_bounded_isolated_process_without_downloads() -> None:
    _performance_module_or_skip()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CITE_RIGHT_DISABLE_MODEL_DOWNLOADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.performance", "smoke-worker"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["backend"] == "python"
    assert payload["measured_sample_count"] >= 1
    assert payload["failures"] == []


def _performance_module_or_skip() -> Any:
    return pytest.importorskip(
        "evaluation.performance",
        reason="evaluation.performance is not implemented yet",
    )


def _sample_measurement(
    performance: Any,
    *,
    case_id: str,
    document_family_id: str,
    split: str,
    prepared_corpus_duration_ns: int,
    answer_duration_ns: int,
    total_duration_ns: int,
    peak_memory_bytes: int | None = None,
    cache_before: Any | None = None,
    cache_after: Any | None = None,
    failure: Any | None = None,
) -> Any:
    return performance.SampleMeasurement(
        case_id=case_id,
        document_family_id=document_family_id,
        split=split,
        prepared_corpus_duration_ns=prepared_corpus_duration_ns,
        answer_duration_ns=answer_duration_ns,
        total_duration_ns=total_duration_ns,
        peak_memory_bytes=peak_memory_bytes,
        cache_before=cache_before,
        cache_after=cache_after,
        failure=failure,
    )


def _environment_metadata(performance: Any, *, backend: str) -> Any:
    return performance.EnvironmentMetadata(
        python_implementation="CPython",
        python_version="3.11.9",
        platform="macOS-15.0-arm64",
        package_version="0.1.0",
        backend=backend,
        config_sha256="a" * 64,
        workload_sha256="b" * 64,
    )


def _failure_record(
    performance: Any,
    *,
    case_id: str,
    document_family_id: str,
    split: str,
    stage: str,
    error_type: str,
    message: str,
) -> Any:
    return performance.FailureRecord(
        case_id=case_id,
        document_family_id=document_family_id,
        split=split,
        stage=stage,
        error_type=error_type,
        message=message,
    )


def _workload_selection(
    performance: Any,
    *,
    seed: int = 7,
    family_filter: tuple[str, ...] = (),
    selected_case_ids: tuple[str, ...] = ("case-1",),
    selected_document_family_ids: tuple[str, ...] = ("family-a",),
) -> Any:
    return performance.WorkloadSelection(
        seed=seed,
        family_filter=family_filter,
        selected_case_ids=selected_case_ids,
        selected_document_family_ids=selected_document_family_ids,
    )


def _summarize_with_measured_totals(
    performance: Any,
    *,
    totals: tuple[int, ...],
    prepared: tuple[int, ...],
    answer: tuple[int, ...],
) -> Any:
    samples = tuple(
        _sample_measurement(
            performance,
            case_id=f"case-{index}",
            document_family_id="family-a",
            split="dev",
            prepared_corpus_duration_ns=prepared_duration,
            answer_duration_ns=answer_duration,
            total_duration_ns=total_duration,
        )
        for index, (prepared_duration, answer_duration, total_duration) in enumerate(
            zip(prepared, answer, totals, strict=True),
            start=1,
        )
    )
    return performance.summarize_measurements(
        samples=samples,
        warmup_count=0,
        backend="python",
        workload=_workload_selection(
            performance,
            selected_case_ids=tuple(sample.case_id for sample in samples),
        ),
        environment=_environment_metadata(performance, backend="python"),
    )


def _fake_clock(*values: int):
    iterator = iter(values)

    def _clock() -> int:
        return next(iterator)

    return _clock


def _case(*, case_id: str, split: str, document_family_id: str) -> EvaluationCase:
    source_text = "Paris is in France."
    answer = "Paris is in France."

    return EvaluationCase.model_validate(
        {
            "case_id": case_id,
            "dataset_version": "1.0.0",
            "split": split,
            "document_family_id": document_family_id,
            "transformation_family_id": "summary",
            "provenance": {
                "kind": "authored",
                "title": "Paris facts",
                "origin": "internal",
                "publisher": "Cite-Right",
                "license": "permissive",
                "retrieval_date": date(2026, 7, 17),
                "snapshot_hash": f"snapshot-{case_id}",
            },
            "sources": (
                {
                    "source_id": "source-paris",
                    "text": source_text,
                    "chunk_id": "chunk-1",
                    "chunk_char_start": 0,
                    "chunk_char_end": len(source_text),
                },
            ),
            "answer": answer,
            "evaluation_units": (
                {
                    "unit_id": "unit-1",
                    "answer_span": {"start": 0, "end": len(answer)},
                    "text": answer,
                    "claims": (
                        {
                            "claim_id": "claim-1",
                            "answer_span": {"start": 0, "end": len(answer)},
                            "text": answer,
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "req-1",
                                    "alternatives": (
                                        {
                                            "source_id": "source-paris",
                                            "spans": (
                                                {"start": 0, "end": len(source_text)},
                                            ),
                                        },
                                    ),
                                },
                            ),
                            "acceptable_retrieval_source_ids": ("source-paris",),
                        },
                    ),
                },
            ),
        }
    )
