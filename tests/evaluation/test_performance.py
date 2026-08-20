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

    with pytest.raises(
        ValidationError, match="duration and count fields must be non-negative"
    ):
        performance.DurationSummary(
            sample_count=-1,
            total_duration_ns=0,
            median_duration_ns=0,
            p95_duration_ns=0,
        )

    with pytest.raises(
        ValidationError, match="duration and count fields must be non-negative"
    ):
        _sample_measurement(
            performance,
            case_id="case-1",
            document_family_id="family-a",
            split="dev",
            prepared_corpus_duration_ns=-1,
            answer_duration_ns=1,
            total_duration_ns=0,
        )

    with pytest.raises(
        ValidationError, match="cache snapshot fields must be non-negative"
    ):
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
        workload=_workload_selection(
            performance, selected_case_ids=("warmup", "case-1", "case-2")
        ),
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


def test_summarize_measurements_derives_throughput_from_total_measured_duration() -> (
    None
):
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
    assert report.throughput_cases_per_second == pytest.approx(3 * 1_000_000_000 / 68)


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
        workload=_workload_selection(
            performance, selected_case_ids=("case-1", "case-2")
        ),
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
        workload=_workload_selection(
            performance, selected_case_ids=("case-1", "case-2")
        ),
        environment=_environment_metadata(performance, backend="python"),
    )

    assert report.peak_memory_bytes == 250


def test_build_environment_metadata_captures_runtime_package_backend_and_hashes() -> (
    None
):
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


def test_select_workload_is_deterministic_for_seed_and_family_filter_and_excludes_holdout() -> (
    None
):
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
    assert report.throughput_cases_per_second == pytest.approx(2 * 1_000_000_000 / 51)
    assert len(report.failures) == 1
    assert report.failures[0].case_id == "case-fail"
    assert report.failures[0].stage == "answer"
    assert report.failures[0].error_type == "RuntimeError"
    assert "boom" in report.failures[0].message


def test_run_benchmark_uses_workload_selection_order_even_when_input_is_reversed() -> (
    None
):
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


def test_performance_smoke_artifact_records_real_scenario_measurements(
    tmp_path: Path,
) -> None:
    performance = _performance_module_or_skip()
    output_path = tmp_path / "smoke.json"

    performance.run_performance_smoke(output_path=output_path)

    payload = json.loads(output_path.read_bytes())
    assert len(payload["dataset_hash"]) == 64
    assert payload["config"] == performance.CitationConfig().model_dump(mode="json")
    assert payload["config_sha256"] == sha256_hex(
        canonical_json_bytes(payload["config"])
    )
    scenarios = payload["scenarios"]
    assert scenarios
    assert payload["measurement_iterations"] >= 25
    assert {scenario["execution_path"] for scenario in scenarios} == {
        "one-shot",
        "prepared",
    }
    assert {scenario["embeddings"] for scenario in scenarios} == {"off", "on"}
    assert {scenario["candidate_bucket"] for scenario in scenarios} == {
        "small",
        "medium",
        "large",
    }
    assert {scenario["source_length"] for scenario in scenarios} == {"short", "long"}
    assert {scenario["answer_shape"] for scenario in scenarios} == {"single", "multi"}
    assert "python" in {scenario["backend"] for scenario in scenarios}
    for execution_path in ("one-shot", "prepared"):
        path_scenarios = [
            scenario
            for scenario in scenarios
            if scenario["execution_path"] == execution_path
            and scenario["backend"] == "python"
        ]
        assert {scenario["embeddings"] for scenario in path_scenarios} == {"off", "on"}
        assert {scenario["candidate_bucket"] for scenario in path_scenarios} == {
            "small",
            "medium",
            "large",
        }
        assert {scenario["source_length"] for scenario in path_scenarios} == {
            "short",
            "long",
        }
        assert {scenario["answer_shape"] for scenario in path_scenarios} == {
            "single",
            "multi",
        }
    for scenario in scenarios:
        assert len(scenario["correctness_hash"]) == 64
        assert len(scenario["raw_samples_ns"]) == payload["trial_count"]
        assert len(scenario["raw_prepared_samples_ns"]) == payload["trial_count"]
        assert len(scenario["raw_end_to_end_samples_ns"]) == payload["trial_count"]
        assert scenario["prepared_corpus"]["sample_count"] == payload["trial_count"]
        assert scenario["answer"]["sample_count"] == payload["trial_count"]
        assert scenario["end_to_end"]["sample_count"] == payload["trial_count"]
        assert scenario["throughput_cases_per_second"] >= 0
        assert (
            scenario["peak_memory_bytes"] is None or scenario["peak_memory_bytes"] >= 0
        )
    assert payload["raw_samples_ns"] == [
        sum(scenario["raw_end_to_end_samples_ns"][index] for scenario in scenarios)
        for index in range(payload["trial_count"])
    ]


def test_smoke_scenarios_execute_real_one_shot_prepared_and_embedding_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance_module_or_skip()
    calls: list[tuple[str, bool, str, int]] = []

    monkeypatch.setattr(
        performance,
        "_execute_one_shot",
        lambda *, case, backend, config, embedder: (
            calls.append(("one-shot", embedder is not None, backend, config.top_k))
            or [{"case_id": case.case_id}]
        ),
    )
    monkeypatch.setattr(
        performance,
        "_execute_prepared",
        lambda *, case, backend, config, embedder: (
            calls.append(("prepared", embedder is not None, backend, config.top_k))
            or [{"case_id": case.case_id}]
        ),
    )

    scenarios = performance._smoke_scenarios()
    config = performance.CitationConfig.strict()
    for scenario in scenarios:
        performance._execute_smoke_scenario(scenario, config=config)

    expected = {
        (
            scenario.execution_path,
            scenario.embeddings == "on",
            scenario.backend,
            config.top_k,
        )
        for scenario in scenarios
    }
    assert set(calls) == expected
    assert ("one-shot", False, "python", config.top_k) in calls
    assert ("prepared", True, "python", config.top_k) in calls
    if performance._rust_backend_supported():
        assert any(backend == "rust" for _, _, backend, _ in calls)
    else:
        assert all(backend == "python" for _, _, backend, _ in calls)


def test_selected_smoke_workload_hash_matches_candidate_measurement_subset() -> None:
    performance = _performance_module_or_skip()
    assert performance.SMOKE_TRIAL_COUNT >= 40

    expected = performance.selected_smoke_workload_hash(
        backend="python",
        embeddings="off",
    )
    measured = performance.measure_candidate_smoke(
        backend="python",
        embeddings="off",
        config=performance.CitationConfig.strict(),
    )

    assert measured["workload_hash"] == expected
    raw_samples = measured["raw_end_to_end_samples_ns"]
    assert isinstance(raw_samples, dict)
    assert raw_samples
    assert all(
        len(samples) == performance.SMOKE_TRIAL_COUNT
        for samples in raw_samples.values()
    )


def test_smoke_trial_replaces_import_paths_strips_holdout_keys_and_sets_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    performance = _performance_module_or_skip()
    request = performance._default_smoke_worker_request()
    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del args
        captured.update(kwargs)
        response = performance.SmokeWorkerSuccessResponse(
            ok=True,
            backend="python",
            warmup_count=1,
            measured_sample_count=1,
            failures=[],
            prepared_total_ns=1,
            answer_total_ns=2,
            end_to_end_total_ns=3,
            raw_sample_ns=2,
        )
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=canonical_json_bytes(response),
            stderr=b"",
        )

    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted")
    monkeypatch.setenv("CITE_RIGHT_HOLDOUT_KEY_FILE", "/tmp/holdout.key")
    monkeypatch.setenv("CITE_RIGHT_ATTESTATION_KEY_FILE", "/tmp/attestation.key")
    monkeypatch.setattr(performance.subprocess, "run", fake_run)

    performance._run_smoke_trial(request)

    child_env = captured["env"]
    assert child_env["PYTHONPATH"] == str(
        Path(performance.__file__).resolve().parents[1]
    )
    assert child_env["PYTHONSAFEPATH"] == "1"
    assert "CITE_RIGHT_HOLDOUT_KEY_FILE" not in child_env
    assert "CITE_RIGHT_ATTESTATION_KEY_FILE" not in child_env
    assert isinstance(captured["timeout"], (int, float))
    assert captured["timeout"] > 0


def test_compare_smoke_command_reports_canonical_deltas_for_matching_artifacts(
    tmp_path: Path,
) -> None:
    _performance_module_or_skip()

    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_payload = _smoke_artifact_payload(
        backend="python",
        correctness_hash="c" * 64,
        protocol_hash="p" * 64,
        workload_hash="w" * 64,
        raw_samples_ns=[100, 120, 140],
    )
    right_payload = _smoke_artifact_payload(
        backend="python",
        correctness_hash="c" * 64,
        protocol_hash="p" * 64,
        workload_hash="w" * 64,
        raw_samples_ns=[80, 100, 160],
    )
    left_path.write_bytes(canonical_json_bytes(left_payload))
    right_path.write_bytes(canonical_json_bytes(right_payload))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.performance",
            "compare-smoke",
            str(left_path),
            str(right_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=_offline_subprocess_env(),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["left"] == str(left_path)
    assert payload["right"] == str(right_path)
    assert payload["correctness_hash"] == left_payload["correctness_hash"]
    assert payload["protocol_hash"] == left_payload["protocol_hash"]
    assert payload["workload_hash"] == left_payload["workload_hash"]
    assert payload["backends"] == ["python"]
    assert payload["raw_samples_ns"]["left"] == [100, 120, 140]
    assert payload["raw_samples_ns"]["right"] == [80, 100, 160]
    scenario_id = "python:embeddings-off:one-shot:small:short:single"
    assert set(payload["scenario_timing"][scenario_id]) == {
        "prepared_corpus",
        "answer",
        "end_to_end",
    }
    assert payload["timing"]["median_delta_ns"] == -20
    assert payload["timing"]["median_ratio"] == pytest.approx(100 / 120)
    assert payload["timing"]["mean_delta_ns"] == pytest.approx((-20) / 3)
    assert payload["timing"]["mean_ratio"] == pytest.approx((340 / 3) / 120)
    # Population variance: right = 10400 / 9, left = 800 / 3.
    assert payload["timing"]["variance_delta_ns"] == pytest.approx(
        (10400 / 9) - (800 / 3)
    )
    assert canonical_json_bytes(payload) == result.stdout.encode("utf-8")


def test_compare_smoke_command_exits_one_with_structured_stderr_for_mismatched_hashes(
    tmp_path: Path,
) -> None:
    _performance_module_or_skip()

    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_bytes(
        canonical_json_bytes(
            _smoke_artifact_payload(
                backend="python",
                correctness_hash="c" * 64,
                protocol_hash="p" * 64,
                workload_hash="w" * 64,
                raw_samples_ns=[100, 120, 140],
            )
        )
    )
    right_path.write_bytes(
        canonical_json_bytes(
            _smoke_artifact_payload(
                backend="python",
                correctness_hash="d" * 64,
                protocol_hash="p" * 64,
                workload_hash="w" * 64,
                raw_samples_ns=[100, 120, 140],
            )
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.performance",
            "compare-smoke",
            str(left_path),
            str(right_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=_offline_subprocess_env(),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["error"]["type"] in {"ValueError", "RuntimeError"}
    assert "correctness_hash" in payload["error"]["message"]


def test_compare_smoke_command_exits_one_with_structured_stderr_for_malformed_artifact(
    tmp_path: Path,
) -> None:
    _performance_module_or_skip()

    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text("{not-json", encoding="utf-8")
    right_path.write_bytes(
        canonical_json_bytes(
            _smoke_artifact_payload(
                backend="python",
                correctness_hash="c" * 64,
                protocol_hash="p" * 64,
                workload_hash="w" * 64,
                raw_samples_ns=[100, 120, 140],
            )
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.performance",
            "compare-smoke",
            str(left_path),
            str(right_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=_offline_subprocess_env(),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["error"]["type"] in {"JSONDecodeError", "ValueError"}
    assert "traceback" not in payload["error"]["message"].lower()


def _performance_module_or_skip() -> Any:
    return pytest.importorskip(
        "evaluation.performance",
        reason="evaluation.performance is not implemented yet",
    )


def _offline_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CITE_RIGHT_DISABLE_MODEL_DOWNLOADS"] = "1"
    return env


def _smoke_artifact_payload(
    *,
    backend: str,
    correctness_hash: str,
    protocol_hash: str,
    workload_hash: str,
    raw_samples_ns: list[int],
) -> dict[str, Any]:
    ordered_samples = sorted(raw_samples_ns)
    sample_count = len(raw_samples_ns)
    median = float(ordered_samples[sample_count // 2])
    duration_summary = {
        "sample_count": sample_count,
        "total_duration_ns": sum(raw_samples_ns),
        "median_duration_ns": median,
        "p95_duration_ns": ordered_samples[-1],
    }
    return {
        "backends": [backend],
        "dataset_hash": "d" * 64,
        "config": {"top_k": 2},
        "config_sha256": "a" * 64,
        "correctness_hash": correctness_hash,
        "protocol_hash": protocol_hash,
        "workload_hash": workload_hash,
        "warmup_count": 1,
        "trial_count": len(raw_samples_ns),
        "measurement_iterations": 25,
        "raw_samples_ns": raw_samples_ns,
        "failures": [],
        "scenarios": [
            {
                "scenario_id": f"{backend}:embeddings-off:one-shot:small:short:single",
                "backend": backend,
                "execution_path": "one-shot",
                "embeddings": "off",
                "candidate_bucket": "small",
                "source_length": "short",
                "answer_shape": "single",
                "correctness_hash": "e" * 64,
                "raw_samples_ns": raw_samples_ns,
                "raw_prepared_samples_ns": [0] * sample_count,
                "raw_end_to_end_samples_ns": raw_samples_ns,
                "prepared_corpus": {
                    "sample_count": sample_count,
                    "total_duration_ns": 0,
                    "median_duration_ns": 0.0,
                    "p95_duration_ns": 0,
                },
                "answer": duration_summary,
                "end_to_end": duration_summary,
                "throughput_cases_per_second": sample_count
                * 1_000_000_000
                / sum(raw_samples_ns),
                "peak_memory_bytes": 1024,
            }
        ],
        "workload": {
            "strata": [
                "one_shot",
                "prepared",
                "small_candidates",
                "medium_candidates",
                "large_candidates",
                "short_sources",
                "long_sources",
                "single_sentence_answers",
                "multi_sentence_answers",
                "embeddings_off",
                "embeddings_on",
            ],
            "selected_case_ids": [
                "python:embeddings-off:one-shot:small:short:single",
                "python:embeddings-on:prepared:medium:long:multi",
            ],
            "selected_backend_ids": [backend],
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "git_revision": "f" * 40,
            "dependencies": {"pydantic": "2.0.0"},
        },
    }


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
