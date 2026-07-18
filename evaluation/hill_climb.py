"""Constrained experiment selection for strict-attribution tuning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median

from pydantic import BaseModel, ConfigDict, field_validator

from cite_right import CitationConfig
from evaluation.baselines import AccuracyReport, accuracy_report
from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.experiments import (
    CandidateMetrics,
    ExperimentRecord,
    ExperimentStore,
    GateDecision,
    ResourceMetrics,
    build_experiment_record,
    canonical_search_space_hash,
    contains_forbidden_holdout_data,
    git_revision,
    load_experiment_store,
    persist_experiment_store,
    resolve_output_path,
)
from evaluation.runner import execute_case
from evaluation.schema import EvaluationCase
from evaluation.tuning_bundle import TuningBundle, load_tuning_bundle

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
SEARCH_SPACE_SCHEMA_VERSION = "evaluation.search-space.v1"


class SyntheticResult(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    train_metrics: CandidateMetrics
    dev_metrics: CandidateMetrics
    resource_metrics: ResourceMetrics


class SearchCandidate(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    candidate_id: str
    code_path_id: str
    config: dict[str, object]
    backend: str = "python"
    embeddings: str = "off"
    parent_candidate_id: str | None = None
    synthetic_result: SyntheticResult | None = None


class SearchSpace(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    schema_version: str
    parent_experiment_id: str | None = None
    baseline: dict[str, object] | None = None
    candidates: list[SearchCandidate]

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != SEARCH_SPACE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SEARCH_SPACE_SCHEMA_VERSION!r}")
        return value

    def synthetic_records(self) -> tuple[ExperimentRecord, ...]:
        baseline_hash, gates = baseline_context(self.baseline)
        records: list[ExperimentRecord] = []
        for candidate in sorted(self.candidates, key=lambda item: item.candidate_id):
            if candidate.synthetic_result is None:
                continue
            gate_decision = evaluate_gate_decision(
                train_metrics=candidate.synthetic_result.train_metrics,
                dev_metrics=candidate.synthetic_result.dev_metrics,
                resource_metrics=candidate.synthetic_result.resource_metrics,
                gates=gates,
                evaluated_dev=True,
            )
            records.append(
                build_experiment_record(
                    candidate_id=candidate.candidate_id,
                    parent_candidate_id=candidate.parent_candidate_id,
                    dataset_hash="0" * 64,
                    baseline_hash=baseline_hash,
                    git_revision="synthetic",
                    code_path_id=candidate.code_path_id,
                    backend=candidate.backend,
                    embeddings=candidate.embeddings,
                    config=candidate.config,
                    train_metrics=candidate.synthetic_result.train_metrics,
                    dev_metrics=candidate.synthetic_result.dev_metrics,
                    resource_metrics=candidate.synthetic_result.resource_metrics,
                    gate_decision=gate_decision,
                )
            )
        return tuple(records)


def load_search_space(path: Path) -> SearchSpace:
    if _forbidden_optimizer_path(path):
        raise ValueError("search space must not reference holdout or release-gate inputs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("search space must be a JSON object")
    if contains_forbidden_holdout_data(payload):
        raise ValueError("search space must not contain holdout fields or paths")
    return SearchSpace.model_validate(payload)


def run_tuning(
    *,
    tuning_bundle: Path,
    search_space_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if _forbidden_optimizer_path(search_space_path):
        raise ValueError("search space must not reference holdout or release-gate inputs")
    if _forbidden_optimizer_path(output_path):
        raise ValueError("release-gate and holdout paths are forbidden optimizer outputs")

    bundle = load_tuning_bundle(tuning_bundle)
    search_space = load_search_space(search_space_path)
    search_space_payload = json.loads(search_space_path.read_text(encoding="utf-8"))
    assert isinstance(search_space_payload, dict)
    baseline_hash, gates = baseline_context(search_space.baseline)
    search_space_hash = canonical_search_space_hash(search_space_payload)

    target = resolve_output_path(output_path)
    existing_records: list[ExperimentRecord] = []
    duplicate_candidate_ids: list[str] = []
    if target.exists():
        store = load_experiment_store(output_path)
        _validate_resume_context(
            store=store,
            dataset_hash=bundle.manifest.dataset_manifest_sha256,
            baseline_hash=baseline_hash,
            search_space_hash=search_space_hash,
        )
        existing_records.extend(store.records)

    records_by_id = {record.candidate_id: record for record in existing_records}
    evaluated = 0
    for candidate in sorted(search_space.candidates, key=lambda item: item.candidate_id):
        if candidate.candidate_id in records_by_id:
            duplicate_candidate_ids.append(candidate.candidate_id)
            continue
        record = _evaluate_candidate(
            candidate=candidate,
            bundle=bundle,
            dataset_hash=bundle.manifest.dataset_manifest_sha256,
            baseline_hash=baseline_hash,
            gates=gates,
        )
        records_by_id[record.candidate_id] = record
        evaluated += 1

    ordered_records = tuple(records_by_id[key] for key in sorted(records_by_id))
    best = select_best_record(ordered_records)
    store = ExperimentStore(
        dataset_hash=bundle.manifest.dataset_manifest_sha256,
        baseline_hash=baseline_hash,
        git_revision=git_revision(),
        search_space_hash=search_space_hash,
        parent_experiment_id=search_space.parent_experiment_id,
        best_candidate_id=None if best is None else best.candidate_id,
        duplicate_candidate_ids=sorted(duplicate_candidate_ids),
        records=list(ordered_records),
    )
    persisted_path = persist_experiment_store(output_path=output_path, store=store)
    return {
        "command": "tune",
        "output": str(persisted_path),
        "best_candidate_id": store.best_candidate_id,
        "evaluated_candidate_count": evaluated,
        "duplicate_candidate_ids": sorted(duplicate_candidate_ids),
        "search_space_hash": search_space_hash,
    }


def select_best_record(records: Sequence[ExperimentRecord]) -> ExperimentRecord | None:
    survivors = [record for record in records if record.gate_decision.gate_pass]
    if not survivors:
        return None
    return max(survivors, key=_ranking_key)


def evaluate_gate_decision(
    *,
    train_metrics: CandidateMetrics,
    dev_metrics: CandidateMetrics | None,
    resource_metrics: ResourceMetrics | None,
    gates: Mapping[str, object],
    evaluated_dev: bool,
) -> GateDecision:
    assessed = dev_metrics if dev_metrics is not None else train_metrics
    offset_tolerance = _coerce_int_default(gates.get("offset_invalid_tolerance"), 0)
    contradiction_tolerance = _coerce_int_default(
        gates.get("contradiction_false_citation_tolerance"), 0
    )
    precision_floor = _coerce_float(
        gates.get("exact_precision_wilson_lower_min"), default=None
    )
    p95_budget = _coerce_int(gates.get("p95_latency_budget_ns"), default=None)
    peak_budget = _coerce_int(gates.get("peak_memory_budget_bytes"), default=None)

    passes_offset = assessed.offset_invalid_count <= offset_tolerance
    passes_precision = precision_floor is None or (
        assessed.exact_precision_lower is not None
        and assessed.exact_precision_lower >= precision_floor
    )
    passes_contradiction = (
        assessed.contradiction_false_citation_count <= contradiction_tolerance
    )
    passes_resource = True
    p95_limit = p95_budget
    if resource_metrics is None:
        if evaluated_dev and (p95_limit is not None or peak_budget is not None):
            passes_resource = False
    elif p95_limit is not None:
        passes_resource = resource_metrics.p95_duration_ns <= p95_limit
    peak_limit = peak_budget
    peak_memory = None if resource_metrics is None else resource_metrics.peak_memory_bytes
    if passes_resource and peak_limit is not None:
        if peak_memory is None:
            passes_resource = not evaluated_dev
        else:
            passes_resource = peak_memory <= peak_limit

    violated: list[str] = []
    if not passes_offset:
        violated.append("offset")
    if not passes_precision:
        violated.append("precision")
    if not passes_contradiction:
        violated.append("contradiction")
    if not passes_resource:
        violated.append("resources")
    return GateDecision(
        evaluated_dev=evaluated_dev,
        passes_offset_gate=passes_offset,
        passes_precision_gate=passes_precision,
        passes_contradiction_gate=passes_contradiction,
        passes_resource_gates=passes_resource,
        gate_pass=not violated,
        violated_gates=violated,
    )


def baseline_context(baseline: Mapping[str, object] | None) -> tuple[str, Mapping[str, object]]:
    if baseline is not None:
        baseline_hash = str(baseline["hash"])
        gates = baseline.get("gates")
        if not isinstance(gates, Mapping):
            raise ValueError("baseline.gates must be a JSON object")
        return baseline_hash, gates
    baseline_path = Path("evaluation/reports/v1/baseline.json")
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline report must be a JSON object")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("baseline report must contain gates")
    return sha256_hex(canonical_json_bytes(payload)), gates


def _evaluate_candidate(
    *,
    candidate: SearchCandidate,
    bundle: TuningBundle,
    dataset_hash: str,
    baseline_hash: str,
    gates: Mapping[str, object],
) -> ExperimentRecord:
    if candidate.synthetic_result is not None:
        gate_decision = evaluate_gate_decision(
            train_metrics=candidate.synthetic_result.train_metrics,
            dev_metrics=candidate.synthetic_result.dev_metrics,
            resource_metrics=candidate.synthetic_result.resource_metrics,
            gates=gates,
            evaluated_dev=True,
        )
        return build_experiment_record(
            candidate_id=candidate.candidate_id,
            parent_candidate_id=candidate.parent_candidate_id,
            dataset_hash=dataset_hash,
            baseline_hash=baseline_hash,
            git_revision=git_revision(),
            code_path_id=candidate.code_path_id,
            backend=candidate.backend,
            embeddings=candidate.embeddings,
            config=candidate.config,
            train_metrics=candidate.synthetic_result.train_metrics,
            dev_metrics=candidate.synthetic_result.dev_metrics,
            resource_metrics=candidate.synthetic_result.resource_metrics,
            gate_decision=gate_decision,
        )

    config = CitationConfig.model_validate(candidate.config)
    train_report, train_durations = _run_accuracy(bundle.train_cases, config=config)
    train_metrics = _candidate_metrics(train_report)
    early_decision = evaluate_gate_decision(
        train_metrics=train_metrics,
        dev_metrics=None,
        resource_metrics=None,
        gates=gates,
        evaluated_dev=False,
    )
    if not early_decision.passes_offset_gate or not early_decision.passes_precision_gate or not early_decision.passes_contradiction_gate:
        return build_experiment_record(
            candidate_id=candidate.candidate_id,
            parent_candidate_id=candidate.parent_candidate_id,
            dataset_hash=dataset_hash,
            baseline_hash=baseline_hash,
            git_revision=git_revision(),
            code_path_id=candidate.code_path_id,
            backend=candidate.backend,
            embeddings=candidate.embeddings,
            config=candidate.config,
            train_metrics=train_metrics,
            dev_metrics=None,
            resource_metrics=None,
            gate_decision=early_decision,
        )

    dev_report, dev_durations = _run_accuracy(bundle.dev_cases, config=config)
    dev_metrics = _candidate_metrics(dev_report)
    resource_metrics = ResourceMetrics(
        median_duration_ns=int(median(dev_durations)) if dev_durations else 0,
        p95_duration_ns=max(dev_durations) if dev_durations else 0,
        peak_memory_bytes=None,
    )
    gate_decision = evaluate_gate_decision(
        train_metrics=train_metrics,
        dev_metrics=dev_metrics,
        resource_metrics=resource_metrics,
        gates=gates,
        evaluated_dev=True,
    )
    return build_experiment_record(
        candidate_id=candidate.candidate_id,
        parent_candidate_id=candidate.parent_candidate_id,
        dataset_hash=dataset_hash,
        baseline_hash=baseline_hash,
        git_revision=git_revision(),
        code_path_id=candidate.code_path_id,
        backend=candidate.backend,
        embeddings=candidate.embeddings,
        config=candidate.config,
        train_metrics=train_metrics,
        dev_metrics=dev_metrics,
        resource_metrics=resource_metrics,
        gate_decision=gate_decision,
    )


def _run_accuracy(
    cases: Sequence[EvaluationCase],
    *,
    config: CitationConfig,
) -> tuple[AccuracyReport, list[int]]:
    runs = [
        execute_case(case=case, backend="python", config=config)
        for case in cases
    ]
    report = accuracy_report(cases=cases, runs=runs)
    return report, [run.duration_ns for run in runs]


def _candidate_metrics(report: AccuracyReport) -> CandidateMetrics:
    metrics = report.metrics
    return CandidateMetrics(
        exact_precision_lower=metrics.exact_precision.lower,
        contradiction_false_citation_count=metrics.contradiction_false_citation_rate.numerator,
        offset_invalid_count=metrics.offset_validity.denominator - metrics.offset_validity.numerator,
        requirement_recall=metrics.requirement_recall.estimate,
        status_macro_f1=metrics.status_macro_f1,
        retrieval_mrr=metrics.retrieval_mrr,
        run_error_count=report.run_error_count,
        output_sha256=report.output_sha256,
    )


def _ranking_key(record: ExperimentRecord) -> tuple[float, float, float, int, str]:
    assert record.dev_metrics is not None
    assert record.resource_metrics is not None
    return (
        record.dev_metrics.requirement_recall or 0.0,
        record.dev_metrics.status_macro_f1 or 0.0,
        record.dev_metrics.retrieval_mrr or 0.0,
        -record.resource_metrics.median_duration_ns,
        record.candidate_id,
    )


def _forbidden_optimizer_path(path: Path) -> bool:
    lowered = str(path).lower()
    return "holdout" in lowered or "release-gate" in lowered


def _validate_resume_context(
    *,
    store: ExperimentStore,
    dataset_hash: str,
    baseline_hash: str,
    search_space_hash: str,
) -> None:
    if store.dataset_hash != dataset_hash:
        raise ValueError("existing experiment store dataset hash does not match")
    if store.baseline_hash != baseline_hash:
        raise ValueError("existing experiment store baseline hash does not match")
    if store.search_space_hash != search_space_hash:
        raise ValueError("existing experiment store search space hash does not match")


def _coerce_int(value: object, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gate values must be numeric")
    return int(value)


def _coerce_int_default(value: object, default: int) -> int:
    resolved = _coerce_int(value, default=default)
    assert resolved is not None
    return resolved


def _coerce_float(value: object, *, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gate values must be numeric")
    return float(value)
