"""Reproducible performance measurement for evaluation workloads."""

from __future__ import annotations

import platform
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

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
            raise ValueError(
                "selected_document_family_ids must not contain duplicates"
            )
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
                raise ValueError(f"{field_name} must be a 64-character lowercase hex digest")
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
            raise ValueError(
                "end_to_end summary must use the measured sample count"
            )
        if self.failure_count != len(self.failures):
            raise ValueError("failure_count must equal len(failures)")
        if len(self.failures) > len(self.samples):
            raise ValueError("failure count cannot exceed sample count")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)
    if args == ("smoke-worker",):
        return _run_smoke_worker()

    print("usage: python -m evaluation.performance smoke-worker", file=sys.stderr)
    return 2


def _run_smoke_worker() -> int:
    try:
        report = run_benchmark(
            cases=(_smoke_case("case-1"), _smoke_case("case-2")),
            backend="python",
            config={"mode": "smoke"},
            seed=7,
            warmup_count=1,
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
        print(canonical_json_bytes(payload).decode("utf-8"))
        return 1

    payload = {"ok": True, **report.model_dump(mode="python")}
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


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
    peaks = [sample.peak_memory_bytes for sample in samples if sample.peak_memory_bytes is not None]
    if not peaks:
        return None
    return max(peaks)


def _cache_retention(samples: Sequence[SampleMeasurement]) -> CacheRetention | None:
    before_snapshots = [sample.cache_before for sample in samples if sample.cache_before is not None]
    after_snapshots = [sample.cache_after for sample in samples if sample.cache_after is not None]
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


def _default_answer_case(case: EvaluationCase, prepared_corpus: object) -> dict[str, object]:
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


def _smoke_case(case_id: str) -> EvaluationCase:
    source_text = "Paris is in France."
    answer = "Paris is in France."
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
                        },
                    ),
                },
            ),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
