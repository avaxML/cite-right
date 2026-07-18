"""Reproducible performance measurement for evaluation workloads."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata as importlib_metadata
from math import ceil
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from cite_right import PreparedCitationCorpus, align_citations
from cite_right.core.citation_config import CitationConfig
from cite_right.core.results import SourceChunk, SourceDocument
from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase, Split

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
Backend: TypeAlias = Literal["python", "rust"]
FailureStage: TypeAlias = Literal["prepare_corpus", "answer", "worker"]
Clock: TypeAlias = Callable[[], int]
PeakMemoryReader: TypeAlias = Callable[[], int | None]
CacheSnapshotReader: TypeAlias = Callable[
    [Literal["before", "after"], EvaluationCase, object | None], "CacheSnapshot | None"
]
PrepareCorpusFn: TypeAlias = Callable[[EvaluationCase], object]
AnswerCaseFn: TypeAlias = Callable[[EvaluationCase, object], object]
MAX_EXCEPTION_MESSAGE_CODEPOINTS = 256
TRUNCATION_MARKER = "... [truncated]"
SMOKE_PROTOCOL_VERSION = "evaluation.performance.smoke.v1"
SMOKE_TRIAL_COUNT = 20
SMOKE_WARMUP_COUNT = 1
SMOKE_MEASUREMENT_ITERATIONS = 25
SMOKE_WORKER_TIMEOUT_SECONDS = 20
SMOKE_STRATA = (
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
)
_DEPENDENCY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+")


class _FrozenModel(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__class__.__pydantic_fields__:
            raise TypeError(f"{self.__class__.__name__} is immutable")
        super().__setattr__(name, value)


class CacheSnapshot(_FrozenModel):
    entries: int
    bytes: int

    @model_validator(mode="after")
    def _validate_counts(self) -> CacheSnapshot:
        if self.entries < 0 or self.bytes < 0:
            raise ValueError("cache snapshot fields must be non-negative")
        return self


class CacheRetention(_FrozenModel):
    before: CacheSnapshot
    after: CacheSnapshot
    retained_entries: int
    retained_bytes: int
    delta_entries: int
    delta_bytes: int

    @model_validator(mode="after")
    def _validate_counts(self) -> CacheRetention:
        if self.retained_entries < 0 or self.retained_bytes < 0:
            raise ValueError("cache retention fields must be non-negative")
        return self


class DurationSummary(_FrozenModel):
    sample_count: int
    total_duration_ns: int
    median_duration_ns: float
    p95_duration_ns: int

    @model_validator(mode="after")
    def _validate_counts(self) -> DurationSummary:
        if (
            self.sample_count < 0
            or self.total_duration_ns < 0
            or self.median_duration_ns < 0
            or self.p95_duration_ns < 0
        ):
            raise ValueError("duration and count fields must be non-negative")
        if self.sample_count == 0 and (
            self.total_duration_ns != 0
            or self.median_duration_ns != 0
            or self.p95_duration_ns != 0
        ):
            raise ValueError("empty duration summaries must use zero-valued durations")
        return self


class FailureRecord(_FrozenModel):
    case_id: str
    document_family_id: str
    split: Split
    stage: FailureStage
    error_type: str
    message: str


class SampleMeasurement(_FrozenModel):
    case_id: str
    document_family_id: str
    split: Split
    prepared_corpus_duration_ns: int
    answer_duration_ns: int
    total_duration_ns: int
    peak_memory_bytes: int | None = None
    cache_before: CacheSnapshot | None = None
    cache_after: CacheSnapshot | None = None
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def _validate_sample(self) -> SampleMeasurement:
        values = (
            self.prepared_corpus_duration_ns,
            self.answer_duration_ns,
            self.total_duration_ns,
        )
        if any(value < 0 for value in values) or (
            self.peak_memory_bytes is not None and self.peak_memory_bytes < 0
        ):
            raise ValueError("duration and count fields must be non-negative")
        if self.total_duration_ns != (
            self.prepared_corpus_duration_ns + self.answer_duration_ns
        ):
            raise ValueError(
                "sample durations must satisfy total_duration_ns == prepared_corpus_duration_ns + answer_duration_ns"
            )
        return self


class WorkloadSelection(_FrozenModel):
    seed: int
    family_filter: tuple[str, ...] = ()
    selected_case_ids: tuple[str, ...]
    selected_document_family_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_selection(self) -> WorkloadSelection:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if len(set(self.family_filter)) != len(self.family_filter):
            raise ValueError("family_filter must not contain duplicates")
        if len(set(self.selected_case_ids)) != len(self.selected_case_ids):
            raise ValueError("selected_case_ids must not contain duplicates")
        if len(set(self.selected_document_family_ids)) != len(
            self.selected_document_family_ids
        ):
            raise ValueError("selected_document_family_ids must not contain duplicates")
        return self


class EnvironmentMetadata(_FrozenModel):
    python_implementation: str
    python_version: str
    platform: str
    package_version: str
    backend: Backend
    config_sha256: str
    workload_sha256: str

    @model_validator(mode="after")
    def _validate_hashes(self) -> EnvironmentMetadata:
        for field_name in ("config_sha256", "workload_sha256"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(
                    f"{field_name} must be a 64-character lowercase hex digest"
                )
        return self


class BenchmarkReport(_FrozenModel):
    backend: Backend
    workload: WorkloadSelection
    environment: EnvironmentMetadata
    warmup_count: int
    measured_sample_count: int
    failure_count: int
    prepared_corpus: DurationSummary
    answer: DurationSummary
    end_to_end: DurationSummary
    throughput_cases_per_second: float
    peak_memory_bytes: int | None = None
    cache_retention: CacheRetention | None = None
    failures: tuple[FailureRecord, ...] = ()
    samples: tuple[SampleMeasurement, ...] = ()

    @model_validator(mode="after")
    def _validate_report(self) -> BenchmarkReport:
        if (
            self.warmup_count < 0
            or self.measured_sample_count < 0
            or self.failure_count < 0
        ):
            raise ValueError("duration and count fields must be non-negative")
        if self.throughput_cases_per_second < 0:
            raise ValueError("throughput must be non-negative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak memory must be non-negative")
        if self.prepared_corpus.sample_count != self.measured_sample_count:
            raise ValueError(
                "prepared_corpus summary must use the measured sample count"
            )
        if self.end_to_end.sample_count != self.measured_sample_count:
            raise ValueError("end_to_end summary must use the measured sample count")
        if self.failure_count != len(self.failures):
            raise ValueError("failure_count must equal len(failures)")
        if len(self.failures) > len(self.samples):
            raise ValueError("failure count cannot exceed sample count")
        return self


class SmokeEnvironment(_FrozenModel):
    python_version: str
    platform: str
    cpu: str
    git_revision: str
    dependencies: dict[str, str]


class SmokeWorkload(_FrozenModel):
    strata: list[str]
    selected_case_ids: list[str]
    selected_backend_ids: list[Backend]

    @model_validator(mode="after")
    def _validate_selection(self) -> SmokeWorkload:
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("strata must not contain duplicates")
        if tuple(self.selected_case_ids) != tuple(sorted(self.selected_case_ids)):
            raise ValueError("selected_case_ids must be sorted")
        if len(set(self.selected_case_ids)) != len(self.selected_case_ids):
            raise ValueError("selected_case_ids must not contain duplicates")
        if len(set(self.selected_backend_ids)) != len(self.selected_backend_ids):
            raise ValueError("selected_backend_ids must not contain duplicates")
        return self


class SmokeScenario(_FrozenModel):
    scenario_id: str
    backend: Backend
    execution_path: Literal["one-shot", "prepared"]
    embeddings: Literal["off", "on"]
    candidate_bucket: Literal["small", "medium", "large"]
    source_length: Literal["short", "long"]
    answer_shape: Literal["single", "multi"]
    case: EvaluationCase


class SmokeScenarioReport(_FrozenModel):
    scenario_id: str
    backend: Backend
    execution_path: Literal["one-shot", "prepared"]
    embeddings: Literal["off", "on"]
    candidate_bucket: Literal["small", "medium", "large"]
    source_length: Literal["short", "long"]
    answer_shape: Literal["single", "multi"]
    correctness_hash: str
    raw_samples_ns: list[int]
    raw_prepared_samples_ns: list[int]
    raw_end_to_end_samples_ns: list[int]
    prepared_corpus: DurationSummary
    answer: DurationSummary
    end_to_end: DurationSummary
    throughput_cases_per_second: float
    peak_memory_bytes: int | None

    @model_validator(mode="after")
    def _validate_report(self) -> SmokeScenarioReport:
        if len(self.correctness_hash) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.correctness_hash
        ):
            raise ValueError("correctness_hash must be a 64-character lowercase digest")
        raw_series = (
            self.raw_prepared_samples_ns,
            self.raw_samples_ns,
            self.raw_end_to_end_samples_ns,
        )
        if any(
            not series or any(sample < 0 for sample in series) for series in raw_series
        ):
            raise ValueError("raw timing series must contain non-negative samples")
        expected_count = len(self.raw_samples_ns)
        if any(len(series) != expected_count for series in raw_series):
            raise ValueError("raw timing series must use the same sample count")
        if any(
            summary.sample_count != expected_count
            for summary in (self.prepared_corpus, self.answer, self.end_to_end)
        ):
            raise ValueError("scenario summaries must use the raw trial sample count")
        if self.answer.total_duration_ns != sum(self.raw_samples_ns):
            raise ValueError("answer summary total must equal sum(raw_samples_ns)")
        if self.prepared_corpus.total_duration_ns != sum(self.raw_prepared_samples_ns):
            raise ValueError("prepared summary total must equal its raw samples")
        if self.end_to_end.total_duration_ns != sum(self.raw_end_to_end_samples_ns):
            raise ValueError("end-to-end summary total must equal its raw samples")
        if self.throughput_cases_per_second < 0:
            raise ValueError("throughput must be non-negative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak memory must be non-negative")
        return self


class SmokeArtifact(_FrozenModel):
    backends: list[Backend]
    dataset_hash: str
    correctness_hash: str
    protocol_hash: str
    workload_hash: str
    warmup_count: int
    trial_count: int
    measurement_iterations: int
    raw_samples_ns: list[int]
    failures: list[FailureRecord] = []
    scenarios: list[SmokeScenarioReport]
    workload: SmokeWorkload
    environment: SmokeEnvironment

    @model_validator(mode="after")
    def _validate_artifact(self) -> SmokeArtifact:
        for field_name in ("correctness_hash", "protocol_hash", "workload_hash"):
            value = getattr(self, field_name)
            if len(value) != 64 or not value.isalnum() or value != value.lower():
                raise ValueError(
                    f"{field_name} must be a 64-character lowercase digest"
                )
        if (
            self.warmup_count < 0
            or self.trial_count < 0
            or self.measurement_iterations <= 0
        ):
            raise ValueError("warmup_count and trial_count must be non-negative")
        if self.trial_count != len(self.raw_samples_ns):
            raise ValueError("trial_count must equal len(raw_samples_ns)")
        if any(sample < 0 for sample in self.raw_samples_ns):
            raise ValueError("raw_samples_ns values must be non-negative")
        if len(self.dataset_hash) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.dataset_hash
        ):
            raise ValueError("dataset_hash must be a 64-character lowercase digest")
        if not self.scenarios:
            raise ValueError("scenarios must not be empty")
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("backends must be non-empty and unique")
        if len({scenario.scenario_id for scenario in self.scenarios}) != len(
            self.scenarios
        ):
            raise ValueError("scenario ids must not contain duplicates")
        if any(
            len(scenario.raw_samples_ns) != self.trial_count
            for scenario in self.scenarios
        ):
            raise ValueError("each scenario must contain trial_count raw samples")
        scenario_backends = {scenario.backend for scenario in self.scenarios}
        if scenario_backends != set(
            self.workload.selected_backend_ids
        ) or scenario_backends != set(self.backends):
            raise ValueError("artifact, scenario, and workload backends must match")
        return self


class SmokeWorkerRequest(_FrozenModel):
    protocol_version: str
    backend: Backend
    warmup_count: int
    cases: list[dict[str, object]]
    scenario: dict[str, object] | None = None

    @model_validator(mode="after")
    def _validate_request(self) -> SmokeWorkerRequest:
        if self.protocol_version != SMOKE_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported smoke worker protocol_version: {self.protocol_version}"
            )
        if self.warmup_count < 0:
            raise ValueError("warmup_count must be non-negative")
        if not self.cases:
            raise ValueError("cases must not be empty")
        return self


class SmokeWorkerSuccessResponse(_FrozenModel):
    ok: Literal[True]
    backend: Backend
    warmup_count: int
    measured_sample_count: int
    failures: list[FailureRecord]
    prepared_total_ns: int
    answer_total_ns: int
    end_to_end_total_ns: int
    raw_sample_ns: int
    correctness_hash: str = "0" * 64
    peak_memory_bytes: int | None = None

    @model_validator(mode="after")
    def _validate_response(self) -> SmokeWorkerSuccessResponse:
        if self.warmup_count < 0 or self.measured_sample_count < 0:
            raise ValueError("counts must be non-negative")
        for field_name in (
            "prepared_total_ns",
            "answer_total_ns",
            "end_to_end_total_ns",
            "raw_sample_ns",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        return self


def summarize_measurements(
    *,
    samples: Sequence[SampleMeasurement],
    warmup_count: int,
    backend: Backend,
    workload: WorkloadSelection,
    environment: EnvironmentMetadata,
) -> BenchmarkReport:
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative")

    ordered_samples = tuple(samples)
    measured_samples = ordered_samples[warmup_count:]
    failures = tuple(
        sample.failure for sample in measured_samples if sample.failure is not None
    )
    prepared_durations = tuple(
        sample.prepared_corpus_duration_ns for sample in measured_samples
    )
    answer_durations = tuple(
        sample.answer_duration_ns
        for sample in measured_samples
        if sample.failure is None or sample.failure.stage != "prepare_corpus"
    )
    total_durations = tuple(sample.total_duration_ns for sample in measured_samples)

    measured_sample_count = len(measured_samples)
    return BenchmarkReport(
        backend=backend,
        workload=workload,
        environment=environment,
        warmup_count=warmup_count,
        measured_sample_count=measured_sample_count,
        failure_count=len(failures),
        prepared_corpus=_duration_summary(prepared_durations),
        answer=_duration_summary(answer_durations),
        end_to_end=_duration_summary(total_durations),
        throughput_cases_per_second=_throughput_cases_per_second(
            sample_count=measured_sample_count,
            total_duration_ns=sum(total_durations),
        ),
        peak_memory_bytes=_max_peak_memory(measured_samples),
        cache_retention=_cache_retention(measured_samples),
        failures=tuple(failure for failure in failures if failure is not None),
        samples=ordered_samples,
    )


def build_environment_metadata(
    *,
    backend: Backend,
    config: BaseModel | Mapping[str, object] | list[object] | tuple[object, ...],
    workload: WorkloadSelection,
) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_version=_package_version(),
        backend=backend,
        config_sha256=sha256_hex(canonical_json_bytes(config)),
        workload_sha256=sha256_hex(
            canonical_json_bytes(workload.model_dump(mode="python"))
        ),
    )


def select_workload(
    cases: Sequence[EvaluationCase],
    *,
    seed: int,
    family_filter: Sequence[str] = (),
) -> WorkloadSelection:
    normalized_family_filter = tuple(sorted(dict.fromkeys(family_filter)))
    eligible_cases = [
        case
        for case in cases
        if case.split in {"train", "dev"}
        and (
            not normalized_family_filter
            or case.document_family_id in normalized_family_filter
        )
    ]
    selected_cases = sorted(
        eligible_cases,
        key=lambda case: (
            _stable_rank(seed, case.document_family_id),
            case.document_family_id,
            _stable_rank(seed, case.document_family_id, case.case_id),
            case.case_id,
        ),
    )
    selected_family_ids = tuple(
        sorted({case.document_family_id for case in selected_cases})
    )
    return WorkloadSelection(
        seed=seed,
        family_filter=normalized_family_filter,
        selected_case_ids=tuple(case.case_id for case in selected_cases),
        selected_document_family_ids=selected_family_ids,
    )


def run_benchmark(
    *,
    cases: Sequence[EvaluationCase],
    backend: Backend,
    config: BaseModel | Mapping[str, object] | list[object] | tuple[object, ...],
    seed: int,
    family_filter: Sequence[str] = (),
    warmup_count: int = 0,
    clock: Clock = time.perf_counter_ns,
    prepare_corpus: PrepareCorpusFn | None = None,
    answer_case: AnswerCaseFn | None = None,
    read_peak_memory_bytes: PeakMemoryReader | None = None,
    read_cache_snapshot: CacheSnapshotReader | None = None,
) -> BenchmarkReport:
    workload = select_workload(cases, seed=seed, family_filter=family_filter)
    selected_cases = _ordered_selected_cases(cases, workload=workload)
    prepare_corpus_impl = prepare_corpus or _default_prepare_corpus
    answer_case_impl = answer_case or _default_answer_case
    peak_memory_reader = read_peak_memory_bytes or _default_peak_memory_reader

    samples: list[SampleMeasurement] = []

    for case in selected_cases:
        sample_started_at = clock()
        cache_before = (
            read_cache_snapshot("before", case, None)
            if read_cache_snapshot is not None
            else None
        )
        try:
            prepared_corpus = prepare_corpus_impl(case)
        except Exception as exc:
            finished_at = clock()
            prepared_duration_ns = _clamp_duration(finished_at - sample_started_at)
            samples.append(
                SampleMeasurement(
                    case_id=case.case_id,
                    document_family_id=case.document_family_id,
                    split=case.split,
                    prepared_corpus_duration_ns=prepared_duration_ns,
                    answer_duration_ns=0,
                    total_duration_ns=prepared_duration_ns,
                    peak_memory_bytes=peak_memory_reader(),
                    cache_before=cache_before,
                    cache_after=(
                        read_cache_snapshot("after", case, None)
                        if read_cache_snapshot is not None
                        else None
                    ),
                    failure=_failure_record(case=case, stage="prepare_corpus", exc=exc),
                )
            )
            continue

        prepared_finished_at = clock()
        answer_started_at = clock()
        try:
            answer_case_impl(case, prepared_corpus)
        except Exception as exc:
            finished_at = clock()
            prepared_duration_ns = _clamp_duration(
                prepared_finished_at - sample_started_at
            )
            answer_duration_ns = _clamp_duration(finished_at - answer_started_at)
            samples.append(
                SampleMeasurement(
                    case_id=case.case_id,
                    document_family_id=case.document_family_id,
                    split=case.split,
                    prepared_corpus_duration_ns=prepared_duration_ns,
                    answer_duration_ns=answer_duration_ns,
                    total_duration_ns=prepared_duration_ns + answer_duration_ns,
                    peak_memory_bytes=peak_memory_reader(),
                    cache_before=cache_before,
                    cache_after=(
                        read_cache_snapshot("after", case, prepared_corpus)
                        if read_cache_snapshot is not None
                        else None
                    ),
                    failure=_failure_record(case=case, stage="answer", exc=exc),
                )
            )
            continue

        finished_at = clock()
        prepared_duration_ns = _clamp_duration(prepared_finished_at - sample_started_at)
        answer_duration_ns = _clamp_duration(finished_at - answer_started_at)
        samples.append(
            SampleMeasurement(
                case_id=case.case_id,
                document_family_id=case.document_family_id,
                split=case.split,
                prepared_corpus_duration_ns=prepared_duration_ns,
                answer_duration_ns=answer_duration_ns,
                total_duration_ns=prepared_duration_ns + answer_duration_ns,
                peak_memory_bytes=peak_memory_reader(),
                cache_before=cache_before,
                cache_after=(
                    read_cache_snapshot("after", case, prepared_corpus)
                    if read_cache_snapshot is not None
                    else None
                ),
            )
        )

    environment = build_environment_metadata(
        backend=backend,
        config=config,
        workload=workload,
    )
    return summarize_measurements(
        samples=tuple(samples),
        warmup_count=warmup_count,
        backend=backend,
        workload=workload,
        environment=environment,
    )


def run_performance_smoke(*, output_path: Path) -> dict[str, object]:
    parent = output_path.parent
    if not parent.exists():
        raise ValueError(f"output parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise ValueError(f"output parent directory must be a directory: {parent}")

    scenarios = _smoke_scenarios()
    cases = tuple(scenario.case for scenario in scenarios)
    workload = SmokeWorkload(
        strata=list(SMOKE_STRATA),
        selected_case_ids=sorted({case.case_id for case in cases}),
        selected_backend_ids=list(
            dict.fromkeys(scenario.backend for scenario in scenarios)
        ),
    )
    scenario_reports: list[SmokeScenarioReport] = []
    failures: list[FailureRecord] = []
    aggregate_samples = [0] * SMOKE_TRIAL_COUNT

    for scenario in scenarios:
        request = SmokeWorkerRequest(
            protocol_version=SMOKE_PROTOCOL_VERSION,
            backend=scenario.backend,
            warmup_count=SMOKE_WARMUP_COUNT,
            cases=[scenario.case.model_dump(mode="json", exclude_computed_fields=True)],
            scenario=scenario.model_dump(mode="json", exclude_computed_fields=True),
        )
        responses = tuple(_run_smoke_trial(request) for _ in range(SMOKE_TRIAL_COUNT))
        correctness_hashes = {response.correctness_hash for response in responses}
        if len(correctness_hashes) != 1:
            raise RuntimeError(
                f"scenario {scenario.scenario_id!r} produced inconsistent correctness hashes"
            )
        prepared_samples = [response.prepared_total_ns for response in responses]
        answer_samples = [response.answer_total_ns for response in responses]
        end_to_end_samples = [response.end_to_end_total_ns for response in responses]
        for index, sample in enumerate(end_to_end_samples):
            aggregate_samples[index] += sample
        for response in responses:
            failures.extend(response.failures)
        peak_values = [
            response.peak_memory_bytes
            for response in responses
            if response.peak_memory_bytes is not None
        ]
        scenario_reports.append(
            SmokeScenarioReport(
                scenario_id=scenario.scenario_id,
                backend=scenario.backend,
                execution_path=scenario.execution_path,
                embeddings=scenario.embeddings,
                candidate_bucket=scenario.candidate_bucket,
                source_length=scenario.source_length,
                answer_shape=scenario.answer_shape,
                correctness_hash=next(iter(correctness_hashes)),
                raw_samples_ns=answer_samples,
                raw_prepared_samples_ns=prepared_samples,
                raw_end_to_end_samples_ns=end_to_end_samples,
                prepared_corpus=_duration_summary(prepared_samples),
                answer=_duration_summary(answer_samples),
                end_to_end=_duration_summary(end_to_end_samples),
                throughput_cases_per_second=_throughput_cases_per_second(
                    sample_count=len(end_to_end_samples),
                    total_duration_ns=sum(end_to_end_samples),
                ),
                peak_memory_bytes=max(peak_values) if peak_values else None,
            )
        )

    dataset_hash = sha256_hex(
        canonical_json_bytes(
            [
                case.model_dump(mode="json", exclude_computed_fields=True)
                for case in sorted(cases, key=lambda item: item.case_id)
            ]
        )
    )
    correctness_hash = sha256_hex(
        canonical_json_bytes(
            [
                {"scenario_id": report.scenario_id, "hash": report.correctness_hash}
                for report in scenario_reports
            ]
        )
    )

    artifact = SmokeArtifact(
        backends=list(workload.selected_backend_ids),
        dataset_hash=dataset_hash,
        correctness_hash=correctness_hash,
        protocol_hash=_smoke_protocol_hash(),
        workload_hash=sha256_hex(
            canonical_json_bytes(workload.model_dump(mode="python"))
        ),
        warmup_count=SMOKE_WARMUP_COUNT,
        trial_count=SMOKE_TRIAL_COUNT,
        measurement_iterations=SMOKE_MEASUREMENT_ITERATIONS,
        raw_samples_ns=list(aggregate_samples),
        failures=list(failures),
        scenarios=scenario_reports,
        workload=workload,
        environment=_build_smoke_environment(),
    )
    _write_atomic_bytes(output_path, canonical_json_bytes(artifact))
    return {
        "backends": list(artifact.backends),
        "command": "performance-smoke",
        "dataset_hash": artifact.dataset_hash,
        "correctness_hash": artifact.correctness_hash,
        "output": str(output_path),
        "protocol_hash": artifact.protocol_hash,
        "raw_samples_ns": list(artifact.raw_samples_ns),
        "workload_hash": artifact.workload_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)
    if len(args) == 3 and args[0] == "compare-smoke":
        return _run_compare_smoke(Path(args[1]), Path(args[2]))
    if args == ("smoke-worker",):
        return _run_smoke_worker()

    print(
        "usage: python -m evaluation.performance smoke-worker | compare-smoke LEFT RIGHT",
        file=sys.stderr,
    )
    return 2


def _run_smoke_worker() -> int:
    try:
        request = _load_smoke_worker_request(sys.stdin.buffer.read())
        if request.scenario is not None:
            scenario = SmokeScenario.model_validate_json(
                canonical_json_bytes(request.scenario)
            )
            expected_case_payload = scenario.case.model_dump(
                mode="json",
                exclude_computed_fields=True,
            )
            if scenario.backend != request.backend:
                raise ValueError("scenario backend must equal request backend")
            if request.cases != [expected_case_payload]:
                raise ValueError("scenario case must equal the sole requested case")
            response = _measure_smoke_scenario(
                scenario,
                warmup_count=request.warmup_count,
            )
            _write_stdout_json(response)
            return 0
        cases = tuple(
            EvaluationCase.model_validate_json(canonical_json_bytes(case))
            for case in request.cases
        )
        report = run_benchmark(
            cases=cases,
            backend=request.backend,
            config={"mode": "smoke", "protocol_version": request.protocol_version},
            seed=7,
            warmup_count=request.warmup_count,
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "failure": {
                "stage": "worker",
                "error_type": type(exc).__name__,
                "message": _truncate_codepoints(
                    str(exc),
                    max_codepoints=MAX_EXCEPTION_MESSAGE_CODEPOINTS,
                ),
            },
        }
        _write_stdout_json(payload)
        return 1

    payload = SmokeWorkerSuccessResponse(
        ok=True,
        backend=report.backend,
        warmup_count=report.warmup_count,
        measured_sample_count=report.measured_sample_count,
        failures=list(report.failures),
        prepared_total_ns=report.prepared_corpus.total_duration_ns,
        answer_total_ns=report.answer.total_duration_ns,
        end_to_end_total_ns=report.end_to_end.total_duration_ns,
        raw_sample_ns=report.answer.total_duration_ns,
    ).model_dump(mode="python")
    _write_stdout_json(payload)
    return 0


class _DeterministicEmbedder:
    """Small offline embedder used only to exercise semantic-index code paths."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float(len(lowered)),
                    float(sum(ch.isalpha() for ch in lowered)),
                    float(sum(ord(ch) for ch in lowered) % 997),
                ]
            )
        return vectors


def _execute_one_shot(
    *,
    case: EvaluationCase,
    backend: Backend,
    embedder: _DeterministicEmbedder | None,
) -> list[object]:
    return list(
        align_citations(
            case.answer,
            _citation_sources(case),
            config=CitationConfig(),
            backend=backend,
            embedder=embedder,
        )
    )


def _execute_prepared(
    *,
    case: EvaluationCase,
    backend: Backend,
    embedder: _DeterministicEmbedder | None,
) -> list[object]:
    corpus = PreparedCitationCorpus.from_sources(
        _citation_sources(case),
        config=CitationConfig(),
        embedder=embedder,
    )
    return list(corpus.align(case.answer, backend=backend))


def _execute_smoke_scenario(scenario: SmokeScenario) -> list[object]:
    embedder = _DeterministicEmbedder() if scenario.embeddings == "on" else None
    if scenario.execution_path == "one-shot":
        return _execute_one_shot(
            case=scenario.case,
            backend=scenario.backend,
            embedder=embedder,
        )
    return _execute_prepared(
        case=scenario.case,
        backend=scenario.backend,
        embedder=embedder,
    )


def _measure_smoke_scenario(
    scenario: SmokeScenario, *, warmup_count: int
) -> SmokeWorkerSuccessResponse:
    for _ in range(warmup_count):
        _execute_smoke_scenario(scenario)

    embedder = _DeterministicEmbedder() if scenario.embeddings == "on" else None
    prepared_ns = 0
    correctness_hashes: set[str] = set()
    if scenario.execution_path == "one-shot":
        started = time.perf_counter_ns()
        for _ in range(SMOKE_MEASUREMENT_ITERATIONS):
            outputs = _execute_one_shot(
                case=scenario.case,
                backend=scenario.backend,
                embedder=embedder,
            )
            correctness_hashes.add(_output_correctness_hash(outputs))
        answer_ns = _average_iteration_duration(
            _clamp_duration(time.perf_counter_ns() - started)
        )
    else:
        prepared_started = time.perf_counter_ns()
        for _ in range(SMOKE_MEASUREMENT_ITERATIONS):
            corpus = PreparedCitationCorpus.from_sources(
                _citation_sources(scenario.case),
                config=CitationConfig(),
                embedder=embedder,
            )
        prepared_ns = _average_iteration_duration(
            _clamp_duration(time.perf_counter_ns() - prepared_started)
        )
        answer_started = time.perf_counter_ns()
        for _ in range(SMOKE_MEASUREMENT_ITERATIONS):
            outputs = list(corpus.align(scenario.case.answer, backend=scenario.backend))
            correctness_hashes.add(_output_correctness_hash(outputs))
        answer_ns = _average_iteration_duration(
            _clamp_duration(time.perf_counter_ns() - answer_started)
        )

    if len(correctness_hashes) != 1:
        raise RuntimeError("repeated smoke iterations changed correctness outputs")
    return SmokeWorkerSuccessResponse(
        ok=True,
        backend=scenario.backend,
        warmup_count=warmup_count,
        measured_sample_count=1,
        failures=[],
        prepared_total_ns=prepared_ns,
        answer_total_ns=answer_ns,
        end_to_end_total_ns=prepared_ns + answer_ns,
        raw_sample_ns=answer_ns,
        correctness_hash=next(iter(correctness_hashes)),
        peak_memory_bytes=_default_peak_memory_reader(),
    )


def _duration_summary(durations: Sequence[int]) -> DurationSummary:
    if not durations:
        return DurationSummary(
            sample_count=0,
            total_duration_ns=0,
            median_duration_ns=0,
            p95_duration_ns=0,
        )

    ordered = sorted(durations)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median: float = float(ordered[mid])
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2

    p95_index = max(0, ceil(0.95 * len(ordered)) - 1)
    return DurationSummary(
        sample_count=len(ordered),
        total_duration_ns=sum(ordered),
        median_duration_ns=median,
        p95_duration_ns=ordered[p95_index],
    )


def _throughput_cases_per_second(*, sample_count: int, total_duration_ns: int) -> float:
    if sample_count == 0 or total_duration_ns == 0:
        return 0.0
    return sample_count * 1_000_000_000 / total_duration_ns


def _max_peak_memory(samples: Sequence[SampleMeasurement]) -> int | None:
    peaks = [
        sample.peak_memory_bytes
        for sample in samples
        if sample.peak_memory_bytes is not None
    ]
    if not peaks:
        return None
    return max(peaks)


def _cache_retention(samples: Sequence[SampleMeasurement]) -> CacheRetention | None:
    before_snapshots = [
        sample.cache_before for sample in samples if sample.cache_before is not None
    ]
    after_snapshots = [
        sample.cache_after for sample in samples if sample.cache_after is not None
    ]
    if not before_snapshots or not after_snapshots:
        return None

    before = before_snapshots[0]
    after = after_snapshots[-1]
    assert before is not None
    assert after is not None
    return CacheRetention(
        before=before,
        after=after,
        retained_entries=min(before.entries, after.entries),
        retained_bytes=min(before.bytes, after.bytes),
        delta_entries=after.entries - before.entries,
        delta_bytes=after.bytes - before.bytes,
    )


def _failure_record(
    *, case: EvaluationCase, stage: FailureStage, exc: BaseException
) -> FailureRecord:
    return FailureRecord(
        case_id=case.case_id,
        document_family_id=case.document_family_id,
        split=case.split,
        stage=stage,
        error_type=type(exc).__name__,
        message=_truncate_codepoints(
            str(exc),
            max_codepoints=MAX_EXCEPTION_MESSAGE_CODEPOINTS,
        ),
    )


def _truncate_codepoints(value: str, *, max_codepoints: int) -> str:
    if len(value) <= max_codepoints:
        return value
    head_length = max_codepoints - len(TRUNCATION_MARKER)
    if head_length <= 0:
        raise ValueError("max_codepoints must be larger than the truncation marker")
    return f"{value[:head_length]}{TRUNCATION_MARKER}"


def _clamp_duration(value: int) -> int:
    return max(value, 0)


def _average_iteration_duration(total_duration_ns: int) -> int:
    return total_duration_ns // SMOKE_MEASUREMENT_ITERATIONS


def _output_correctness_hash(outputs: Sequence[object]) -> str:
    output_payload = [
        output.model_dump(mode="json") if isinstance(output, BaseModel) else output
        for output in outputs
    ]
    return sha256_hex(canonical_json_bytes(output_payload))


def _ordered_selected_cases(
    cases: Sequence[EvaluationCase], *, workload: WorkloadSelection
) -> tuple[EvaluationCase, ...]:
    case_by_id: dict[str, EvaluationCase] = {}
    duplicate_case_ids: list[str] = []

    for case in cases:
        if case.case_id in case_by_id:
            duplicate_case_ids.append(case.case_id)
            continue
        case_by_id[case.case_id] = case

    if duplicate_case_ids:
        duplicates = ", ".join(sorted(set(duplicate_case_ids)))
        raise ValueError(f"duplicate case ids in benchmark input: {duplicates}")

    selected_cases: list[EvaluationCase] = []
    missing_case_ids: list[str] = []
    seen_selected_case_ids: set[str] = set()

    for case_id in workload.selected_case_ids:
        if case_id in seen_selected_case_ids:
            raise ValueError(f"duplicate case ids in workload selection: {case_id}")
        seen_selected_case_ids.add(case_id)

        case = case_by_id.get(case_id)
        if case is None:
            missing_case_ids.append(case_id)
            continue
        selected_cases.append(case)

    if missing_case_ids:
        missing = ", ".join(missing_case_ids)
        raise ValueError(f"missing case ids from benchmark input: {missing}")

    return tuple(selected_cases)


def _stable_rank(seed: int, *parts: str) -> str:
    return sha256_hex(canonical_json_bytes([seed, *parts]))


def _default_prepare_corpus(case: EvaluationCase) -> dict[str, object]:
    total_source_chars = sum(len(source.text) for source in case.sources)
    return {
        "source_ids": tuple(source.source_id for source in case.sources),
        "source_chars": total_source_chars,
    }


def _default_answer_case(
    case: EvaluationCase, prepared_corpus: object
) -> dict[str, object]:
    prepared = prepared_corpus if isinstance(prepared_corpus, Mapping) else {}
    return {
        "case_id": case.case_id,
        "answer_chars": len(case.answer),
        "source_chars": prepared.get("source_chars", 0),
    }


def _default_peak_memory_reader() -> int | None:
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return peak
    return peak * 1024


def _package_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        payload = tomllib.load(handle)
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing the [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml is missing project.version")
    return version


def _build_smoke_environment() -> SmokeEnvironment:
    return SmokeEnvironment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu=platform.processor() or platform.machine() or "unknown",
        git_revision=_git_revision(),
        dependencies=_dependency_versions(),
    )


def _build_smoke_workload(
    *, backend: Backend, cases: Sequence[EvaluationCase]
) -> SmokeWorkload:
    return SmokeWorkload(
        strata=list(SMOKE_STRATA),
        selected_case_ids=list(sorted(case.case_id for case in cases)),
        selected_backend_ids=[backend],
    )


def _dependency_versions() -> dict[str, str]:
    names = {"cite-right", *_project_dependency_names()}
    versions: dict[str, str] = {}
    for name in sorted(names):
        if name == "cite-right":
            versions[name] = _package_version()
            continue
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _git_revision() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if result.returncode == 0 and revision:
        return revision
    return "unknown"


def _load_smoke_artifact(path: Path) -> SmokeArtifact:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError:
        raise
    try:
        artifact = SmokeArtifact.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{path} is not a valid smoke artifact: {exc}") from exc
    canonical_payload = artifact.model_dump(
        mode="python",
        exclude_unset=True,
    )
    if raw_bytes != canonical_json_bytes(canonical_payload):
        raise ValueError(f"{path} must use canonical JSON ordering")
    return artifact


def _load_smoke_worker_request(raw_bytes: bytes) -> SmokeWorkerRequest:
    if not raw_bytes.strip():
        return _default_smoke_worker_request()
    payload = json.loads(raw_bytes)
    request = SmokeWorkerRequest.model_validate(payload)
    if raw_bytes != canonical_json_bytes(request):
        raise ValueError("smoke worker request must use canonical JSON ordering")
    return request


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def _population_variance(values: Sequence[int]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    mean_value = _mean(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def _project_dependency_names() -> tuple[str, ...]:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        payload = tomllib.load(handle)
    project = payload.get("project")
    if not isinstance(project, dict):
        return ()
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        return ()
    names: list[str] = []
    for item in dependencies:
        if not isinstance(item, str):
            continue
        match = _DEPENDENCY_NAME_PATTERN.match(item.strip())
        if match is None:
            continue
        names.append(match.group(0))
    return tuple(sorted(dict.fromkeys(names)))


def _run_compare_smoke(left_path: Path, right_path: Path) -> int:
    try:
        left = _load_smoke_artifact(left_path)
        right = _load_smoke_artifact(right_path)
        _assert_matching_smoke_metadata(left=left, right=right)
    except Exception as exc:
        _write_structured_error(exc)
        return 1

    left_samples = left.raw_samples_ns
    right_samples = right.raw_samples_ns
    left_median = _duration_summary(left_samples).median_duration_ns
    right_median = _duration_summary(right_samples).median_duration_ns
    left_mean = _mean(left_samples)
    right_mean = _mean(right_samples)
    left_variance = _population_variance(left_samples)
    right_variance = _population_variance(right_samples)
    payload = {
        "backends": list(left.backends),
        "correctness_hash": left.correctness_hash,
        "dataset_hash": left.dataset_hash,
        "left": str(left_path),
        "ok": True,
        "protocol_hash": left.protocol_hash,
        "raw_samples_ns": {
            "left": list(left_samples),
            "right": list(right_samples),
        },
        "right": str(right_path),
        "scenario_timing": _compare_scenario_timings(left=left, right=right),
        "timing": {
            "mean_delta_ns": right_mean - left_mean,
            "mean_ratio": _safe_ratio(right_mean, left_mean),
            "median_delta_ns": right_median - left_median,
            "median_ratio": _safe_ratio(float(right_median), float(left_median)),
            "variance_delta_ns": right_variance - left_variance,
            "variance_ratio": _safe_ratio(right_variance, left_variance),
        },
        "workload_hash": left.workload_hash,
    }
    _write_stdout_json(payload)
    return 0


def _run_smoke_trial(request: SmokeWorkerRequest) -> SmokeWorkerSuccessResponse:
    repo_root = Path(__file__).resolve().parents[1]
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "CITE_RIGHT_HOLDOUT_KEY_FILE",
            "CITE_RIGHT_ATTESTATION_KEY_FILE",
            "PYTHONHOME",
            "PYTHONPATH",
        }
    }
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHONSAFEPATH"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CITE_RIGHT_DISABLE_MODEL_DOWNLOADS"] = "1"
    raw_request = canonical_json_bytes(request)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.performance", "smoke-worker"],
            cwd=repo_root,
            env=env,
            input=raw_request,
            check=False,
            capture_output=True,
            timeout=SMOKE_WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"smoke worker exceeded {SMOKE_WORKER_TIMEOUT_SECONDS}s timeout"
        ) from exc
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        stdout_text = result.stdout.decode("utf-8", errors="replace").strip()
        detail = _truncate_codepoints(
            stderr_text or stdout_text or "smoke worker failed",
            max_codepoints=MAX_EXCEPTION_MESSAGE_CODEPOINTS,
        )
        raise RuntimeError(detail)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("smoke worker emitted malformed JSON") from exc
    try:
        response = SmokeWorkerSuccessResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"smoke worker emitted an invalid response: {exc}") from exc
    if result.stdout.rstrip(b"\n") != canonical_json_bytes(response):
        raise ValueError("smoke worker response must use canonical JSON ordering")
    if response.backend != request.backend:
        raise ValueError("smoke worker response backend does not match request")
    if response.warmup_count != request.warmup_count:
        raise ValueError("smoke worker response warmup_count does not match request")
    return response


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _assert_matching_smoke_metadata(
    *, left: SmokeArtifact, right: SmokeArtifact
) -> None:
    for field_name in (
        "backends",
        "dataset_hash",
        "correctness_hash",
        "protocol_hash",
        "workload_hash",
    ):
        if getattr(left, field_name) != getattr(right, field_name):
            raise ValueError(
                f"smoke artifacts differ at {field_name}: "
                f"{getattr(left, field_name)!r} != {getattr(right, field_name)!r}"
            )
    if left.warmup_count != right.warmup_count:
        raise ValueError(
            f"smoke artifacts differ at warmup_count: {left.warmup_count} != {right.warmup_count}"
        )
    if left.trial_count != right.trial_count:
        raise ValueError(
            f"smoke artifacts differ at trial_count: {left.trial_count} != {right.trial_count}"
        )
    if left.measurement_iterations != right.measurement_iterations:
        raise ValueError(
            "smoke artifacts differ at measurement_iterations: "
            f"{left.measurement_iterations} != {right.measurement_iterations}"
        )
    left_scenarios = {
        scenario.scenario_id: (scenario.backend, scenario.correctness_hash)
        for scenario in left.scenarios
    }
    right_scenarios = {
        scenario.scenario_id: (scenario.backend, scenario.correctness_hash)
        for scenario in right.scenarios
    }
    if left_scenarios != right_scenarios:
        raise ValueError(
            "smoke artifacts contain different scenarios or scenario outputs"
        )


def _compare_scenario_timings(
    *, left: SmokeArtifact, right: SmokeArtifact
) -> dict[str, object]:
    right_by_id = {scenario.scenario_id: scenario for scenario in right.scenarios}
    comparison: dict[str, object] = {}
    for left_scenario in left.scenarios:
        right_scenario = right_by_id[left_scenario.scenario_id]
        comparison[left_scenario.scenario_id] = {
            "prepared_corpus": _compare_timing_series(
                left_scenario.raw_prepared_samples_ns,
                right_scenario.raw_prepared_samples_ns,
            ),
            "answer": _compare_timing_series(
                left_scenario.raw_samples_ns,
                right_scenario.raw_samples_ns,
            ),
            "end_to_end": _compare_timing_series(
                left_scenario.raw_end_to_end_samples_ns,
                right_scenario.raw_end_to_end_samples_ns,
            ),
        }
    return comparison


def _compare_timing_series(
    left_samples: Sequence[int], right_samples: Sequence[int]
) -> dict[str, float | None]:
    left_summary = _duration_summary(left_samples)
    right_summary = _duration_summary(right_samples)
    left_mean = _mean(left_samples)
    right_mean = _mean(right_samples)
    left_variance = _population_variance(left_samples)
    right_variance = _population_variance(right_samples)
    return {
        "mean_delta_ns": right_mean - left_mean,
        "mean_ratio": _safe_ratio(right_mean, left_mean),
        "median_delta_ns": (
            right_summary.median_duration_ns - left_summary.median_duration_ns
        ),
        "median_ratio": _safe_ratio(
            right_summary.median_duration_ns,
            left_summary.median_duration_ns,
        ),
        "variance_delta_ns": right_variance - left_variance,
        "variance_ratio": _safe_ratio(right_variance, left_variance),
    }


def _smoke_case(
    *,
    backend: Backend,
    embeddings: Literal["off", "on"],
    execution_path: Literal["one-shot", "prepared"],
    candidate_bucket: Literal["small", "medium", "large"],
    source_length: Literal["short", "long"],
    answer_shape: Literal["single", "multi"],
) -> EvaluationCase:
    source_text = (
        "Paris is in France."
        if source_length == "short"
        else (
            "Paris is in France. The Seine crosses the city. "
            "It is known for museums and dense neighborhoods."
        )
    )
    answer = (
        "Paris is in France."
        if answer_shape == "single"
        else "Paris is in France. The Seine crosses the city."
    )
    source_count = {"small": 1, "medium": 2, "large": 3}[candidate_bucket]
    case_id = (
        f"{backend}:embeddings-{embeddings}:{execution_path}:"
        f"{candidate_bucket}:{source_length}:{answer_shape}"
    )
    sources = tuple(
        {
            "source_id": f"source-{index}",
            "text": source_text,
            "chunk_id": f"chunk-{index}",
            "chunk_char_start": 0,
            "chunk_char_end": len(source_text),
        }
        for index in range(1, source_count + 1)
    )
    return EvaluationCase.model_validate(
        {
            "case_id": case_id,
            "dataset_version": "1.0.0",
            "split": "dev",
            "document_family_id": "smoke-family",
            "transformation_family_id": "summary",
            "provenance": {
                "kind": "authored",
                "title": "Smoke facts",
                "origin": "internal",
                "publisher": "Cite-Right",
                "license": "permissive",
            },
            "sources": sources,
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
                                    "alternatives": tuple(
                                        {
                                            "source_id": f"source-{index}",
                                            "spans": (
                                                {"start": 0, "end": len(source_text)},
                                            ),
                                        }
                                        for index in range(1, source_count + 1)
                                    ),
                                },
                            ),
                        },
                    ),
                },
            ),
        }
    )


def _smoke_cases(*, backend: Backend) -> tuple[EvaluationCase, ...]:
    return tuple(
        sorted(
            (
                _smoke_case(
                    backend=backend,
                    embeddings="off",
                    execution_path="one-shot",
                    candidate_bucket="small",
                    source_length="short",
                    answer_shape="single",
                ),
                _smoke_case(
                    backend=backend,
                    embeddings="on",
                    execution_path="prepared",
                    candidate_bucket="medium",
                    source_length="long",
                    answer_shape="multi",
                ),
                _smoke_case(
                    backend=backend,
                    embeddings="off",
                    execution_path="prepared",
                    candidate_bucket="large",
                    source_length="long",
                    answer_shape="multi",
                ),
            ),
            key=lambda case: case.case_id,
        )
    )


def _rust_backend_supported() -> bool:
    try:
        from cite_right import _core
    except ImportError:
        return False
    return all(
        hasattr(_core, name) for name in ("align_pair_details", "align_batch_details")
    )


def _smoke_scenarios() -> tuple[SmokeScenario, ...]:
    backends: tuple[Backend, ...] = (
        ("python", "rust") if _rust_backend_supported() else ("python",)
    )
    definitions: tuple[
        tuple[
            Literal["one-shot", "prepared"],
            Literal["off", "on"],
            Literal["small", "medium", "large"],
            Literal["short", "long"],
            Literal["single", "multi"],
        ],
        ...,
    ] = (
        ("one-shot", "off", "small", "short", "single"),
        ("one-shot", "on", "medium", "long", "multi"),
        ("one-shot", "off", "large", "long", "multi"),
        ("prepared", "off", "small", "short", "single"),
        ("prepared", "off", "medium", "long", "multi"),
        ("prepared", "on", "large", "long", "multi"),
    )
    scenarios: list[SmokeScenario] = []
    for backend in backends:
        for (
            execution_path,
            embeddings,
            candidate_bucket,
            source_length,
            answer_shape,
        ) in definitions:
            case = _smoke_case(
                backend=backend,
                embeddings=embeddings,
                execution_path=execution_path,
                candidate_bucket=candidate_bucket,
                source_length=source_length,
                answer_shape=answer_shape,
            )
            scenarios.append(
                SmokeScenario(
                    scenario_id=case.case_id,
                    backend=backend,
                    execution_path=execution_path,
                    embeddings=embeddings,
                    candidate_bucket=candidate_bucket,
                    source_length=source_length,
                    answer_shape=answer_shape,
                    case=case,
                )
            )
    return tuple(sorted(scenarios, key=lambda scenario: scenario.scenario_id))


def _citation_sources(
    case: EvaluationCase,
) -> tuple[SourceDocument | SourceChunk, ...]:
    sources: list[SourceDocument | SourceChunk] = []
    for index, source in enumerate(case.sources):
        if source.chunk_char_start is None or source.chunk_char_end is None:
            sources.append(SourceDocument(id=source.source_id, text=source.text))
        else:
            sources.append(
                SourceChunk(
                    source_id=source.source_id,
                    text=source.text,
                    doc_char_start=source.chunk_char_start,
                    doc_char_end=source.chunk_char_end,
                    source_index=index,
                )
            )
    return tuple(sources)


def _smoke_correctness_hash(*, cases: Sequence[EvaluationCase]) -> str:
    payload = tuple(
        {
            "case_id": case.case_id,
            "prepared_corpus": _default_prepare_corpus(case),
            "answer_payload": _default_answer_case(case, _default_prepare_corpus(case)),
        }
        for case in cases
    )
    return sha256_hex(canonical_json_bytes(payload))


def _smoke_protocol_hash() -> str:
    payload = {
        "protocol_version": SMOKE_PROTOCOL_VERSION,
        "raw_sample_definition": (
            "top-level=sum(end_to_end_duration_ns across scenarios); "
            "per-scenario=prepared, answer, and end-to-end nanoseconds"
        ),
        "trial_count": SMOKE_TRIAL_COUNT,
        "warmup_count": SMOKE_WARMUP_COUNT,
        "measurement_iterations": SMOKE_MEASUREMENT_ITERATIONS,
        "worker_entrypoint": "python -m evaluation.performance smoke-worker",
    }
    return sha256_hex(canonical_json_bytes(payload))


def _default_smoke_worker_request() -> SmokeWorkerRequest:
    cases = _smoke_cases(backend="python")
    return SmokeWorkerRequest(
        protocol_version=SMOKE_PROTOCOL_VERSION,
        backend="python",
        warmup_count=SMOKE_WARMUP_COUNT,
        cases=[
            case.model_dump(mode="json", exclude_computed_fields=True) for case in cases
        ],
    )


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    parent = path.parent
    temp_fd, temp_name = tempfile.mkstemp(dir=parent, prefix=f".{path.name}.tmp.")
    try:
        with os.fdopen(temp_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(parent)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _write_structured_error(exc: Exception) -> None:
    payload = {
        "error": {
            "code": "operation_failed",
            "message": str(exc),
            "type": exc.__class__.__name__,
        },
        "ok": False,
    }
    _write_stderr_json(payload)


def _write_stderr_json(
    payload: BaseModel | Mapping[str, object] | list[object] | tuple[object, ...],
) -> None:
    sys.stderr.write(canonical_json_bytes(payload).decode("utf-8"))


def _write_stdout_json(
    payload: BaseModel | Mapping[str, object] | list[object] | tuple[object, ...],
) -> None:
    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
