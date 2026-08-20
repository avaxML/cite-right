"""Constrained experiment selection for strict-attribution tuning."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cite_right import CitationConfig
from evaluation.baselines import AccuracyReport, accuracy_report
from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.experiments import (
    CandidateMetrics,
    ExperimentEnvironment,
    ExperimentRecord,
    ExperimentStore,
    GateDecision,
    ResourceMetrics,
    build_experiment_record,
    canonical_search_space_hash,
    contains_forbidden_holdout_data,
    current_code_snapshot_sha256,
    current_environment,
    git_revision,
    load_experiment_store,
    persist_experiment_store,
    resolve_output_path,
    select_best_candidate_id,
)
from evaluation.performance import (
    Backend,
    measure_candidate_smoke,
    smoke_environment_compatibility_hash,
)
from evaluation.runner import canonicalize_config, execute_case
from evaluation.schema import EvaluationCase
from evaluation.tuning_bundle import TuningBundle, load_tuning_bundle

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
SEARCH_SPACE_SCHEMA_VERSION = "evaluation.search-space.v1"
SUPPORTED_CODE_PATH = "config-only"
SUPPORTED_BACKEND: Backend = "python"
SUPPORTED_EMBEDDINGS = "off"


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
    backend: Backend = SUPPORTED_BACKEND
    embeddings: Literal["off", "on"] = SUPPORTED_EMBEDDINGS
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

    @model_validator(mode="after")
    def _validate_candidate_ids(self) -> SearchSpace:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("search-space candidate_id values must be unique")
        return self

    def synthetic_records(self) -> tuple[ExperimentRecord, ...]:
        baseline_hash, gates = baseline_context(self.baseline)
        environment = ExperimentEnvironment(
            python="synthetic",
            platform="synthetic",
            machine="synthetic",
            cpu_count=None,
        )
        code_snapshot_sha256 = "0" * 64
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
                    code_snapshot_sha256=code_snapshot_sha256,
                    code_path_id=candidate.code_path_id,
                    backend=candidate.backend,
                    embeddings=candidate.embeddings,
                    environment=environment,
                    config=CitationConfig.model_validate(candidate.config).model_dump(
                        mode="json"
                    ),
                    train_metrics=candidate.synthetic_result.train_metrics,
                    dev_metrics=candidate.synthetic_result.dev_metrics,
                    resource_metrics=candidate.synthetic_result.resource_metrics,
                    gate_decision=gate_decision,
                )
            )
        return tuple(records)


def load_search_space(path: Path, *, allow_synthetic: bool = False) -> SearchSpace:
    if _forbidden_optimizer_path(path):
        raise ValueError(
            "search space must not reference holdout or release-gate inputs"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("search space must be a JSON object")
    if contains_forbidden_holdout_data(payload):
        raise ValueError("search space must not contain holdout fields or paths")
    search_space = SearchSpace.model_validate(payload)
    if not allow_synthetic and any(
        candidate.synthetic_result is not None for candidate in search_space.candidates
    ):
        raise ValueError("public tune search spaces must not contain synthetic_result")
    return search_space


def run_tuning(
    *,
    tuning_bundle: Path,
    search_space_path: Path,
    output_path: Path,
) -> dict[str, object]:
    git_rev = git_revision()
    code_snapshot_sha = current_code_snapshot_sha256()
    for path in (tuning_bundle, search_space_path, output_path):
        if _forbidden_optimizer_path(path):
            raise ValueError(
                "holdout and release-gate paths are forbidden optimizer inputs"
            )

    search_space = load_search_space(search_space_path, allow_synthetic=False)
    raw_search_space = json.loads(search_space_path.read_text(encoding="utf-8"))
    assert isinstance(raw_search_space, dict)
    bundle = load_tuning_bundle(tuning_bundle)

    if search_space.baseline is not None:
        raise ValueError(
            "public tune search spaces must not override the frozen baseline"
        )
    frozen_baseline = _load_frozen_baseline_report()
    baseline_hash, gates = _baseline_hash_and_gates_from_payload(frozen_baseline)
    baseline_dataset_hash = frozen_baseline.get("dataset_hash")
    if baseline_dataset_hash != bundle.manifest.dataset_manifest_sha256:
        raise ValueError("frozen baseline dataset hash does not match tuning bundle")
    baseline_environment = frozen_baseline.get("environment")
    if isinstance(baseline_environment, Mapping):
        expected_environment = current_environment().model_dump(mode="json")
        comparable = {
            key: baseline_environment.get(key)
            for key in ("python", "platform", "machine", "cpu_count")
        }
        if comparable != expected_environment:
            raise ValueError(
                "frozen baseline environment does not match current environment"
            )
    required_performance_environment_hash = gates.get("performance_environment_hash")
    if required_performance_environment_hash is None:
        raise ValueError("frozen baseline is missing performance_environment_hash")
    if smoke_environment_compatibility_hash() != str(
        required_performance_environment_hash
    ):
        raise ValueError(
            "frozen performance environment does not match current environment"
        )
    search_space_hash = canonical_search_space_hash(raw_search_space)
    environment = current_environment()

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
            git_revision_value=git_rev,
            code_snapshot_sha256=code_snapshot_sha,
            environment=environment,
            parent_experiment_id=search_space.parent_experiment_id,
        )
        existing_records.extend(store.records)
        duplicate_candidate_ids.extend(store.duplicate_candidate_ids)

    records_by_id = {record.candidate_id: record for record in existing_records}
    seen_candidate_identities = {
        _record_identity(record) for record in existing_records
    }
    evaluated = 0
    for candidate in sorted(
        search_space.candidates, key=lambda item: item.candidate_id
    ):
        _validate_real_candidate(candidate)
        candidate_identity = _candidate_identity(candidate)
        if candidate_identity in seen_candidate_identities:
            duplicate_candidate_ids.append(candidate.candidate_id)
            _persist_progress(
                output_path=output_path,
                dataset_hash=bundle.manifest.dataset_manifest_sha256,
                baseline_hash=baseline_hash,
                git_revision_value=git_rev,
                code_snapshot_sha256=code_snapshot_sha,
                search_space_hash=search_space_hash,
                environment=environment,
                parent_experiment_id=search_space.parent_experiment_id,
                duplicate_candidate_ids=sorted(set(duplicate_candidate_ids)),
                records_by_id=records_by_id,
            )
            continue
        record = _evaluate_candidate(
            candidate=candidate,
            bundle=bundle,
            dataset_hash=bundle.manifest.dataset_manifest_sha256,
            baseline_hash=baseline_hash,
            gates=gates,
            git_revision_value=git_rev,
            code_snapshot_sha256=code_snapshot_sha,
            environment=environment,
        )
        records_by_id[record.candidate_id] = record
        seen_candidate_identities.add(_record_identity(record))
        evaluated += 1
        _persist_progress(
            output_path=output_path,
            dataset_hash=bundle.manifest.dataset_manifest_sha256,
            baseline_hash=baseline_hash,
            git_revision_value=git_rev,
            code_snapshot_sha256=code_snapshot_sha,
            search_space_hash=search_space_hash,
            environment=environment,
            parent_experiment_id=search_space.parent_experiment_id,
            duplicate_candidate_ids=sorted(set(duplicate_candidate_ids)),
            records_by_id=records_by_id,
        )
        _assert_code_provenance_unchanged(
            expected_git_revision=git_rev,
            expected_code_snapshot_sha256=code_snapshot_sha,
        )

    ordered_records = [records_by_id[key] for key in sorted(records_by_id)]
    best = select_best_record(ordered_records)
    _assert_code_provenance_unchanged(
        expected_git_revision=git_rev,
        expected_code_snapshot_sha256=code_snapshot_sha,
    )
    store = ExperimentStore(
        dataset_hash=bundle.manifest.dataset_manifest_sha256,
        baseline_hash=baseline_hash,
        git_revision=git_rev,
        code_snapshot_sha256=code_snapshot_sha,
        search_space_hash=search_space_hash,
        environment=environment,
        parent_experiment_id=search_space.parent_experiment_id,
        best_candidate_id=None if best is None else best.candidate_id,
        duplicate_candidate_ids=sorted(set(duplicate_candidate_ids)),
        records=ordered_records,
    )
    persisted_path = persist_experiment_store(output_path=output_path, store=store)
    return {
        "command": "tune",
        "output": str(persisted_path),
        "best_candidate_id": store.best_candidate_id,
        "evaluated_candidate_count": evaluated,
        "duplicate_candidate_ids": sorted(set(duplicate_candidate_ids)),
        "search_space_hash": search_space_hash,
        "git_revision": git_rev,
        "code_snapshot_sha256": code_snapshot_sha,
    }


def select_best_record(records: Sequence[ExperimentRecord]) -> ExperimentRecord | None:
    best_candidate_id = select_best_candidate_id(records)
    if best_candidate_id is None:
        return None
    return next(
        record for record in records if record.candidate_id == best_candidate_id
    )


def evaluate_gate_decision(
    *,
    train_metrics: CandidateMetrics,
    dev_metrics: CandidateMetrics | None,
    resource_metrics: ResourceMetrics | None,
    gates: Mapping[str, object],
    evaluated_dev: bool,
) -> GateDecision:
    assessed = dev_metrics if dev_metrics is not None else train_metrics
    passes_execution = assessed.run_error_count == 0
    offset_tolerance = _coerce_int_default(gates.get("offset_invalid_tolerance"), 0)
    contradiction_tolerance = _coerce_int_default(
        gates.get("contradiction_false_citation_tolerance"), 0
    )
    precision_floor = _coerce_float(
        gates.get("exact_precision_wilson_lower_min"),
        default=None,
    )
    p95_budget = _coerce_int(gates.get("p95_latency_budget_ns"), default=None)
    peak_budget = _coerce_int(gates.get("peak_memory_budget_bytes"), default=None)
    required_protocol_hash = gates.get("performance_protocol_hash")
    required_workload_hash = gates.get("selected_workload_hash")

    passes_offset = assessed.offset_invalid_count <= offset_tolerance
    passes_precision = precision_floor is None or (
        assessed.exact_precision_lower is not None
        and assessed.exact_precision_lower >= precision_floor
    )
    passes_contradiction = (
        assessed.contradiction_false_citation_count <= contradiction_tolerance
    )
    passes_resource = True
    if evaluated_dev and (p95_budget is not None or peak_budget is not None):
        if resource_metrics is None:
            passes_resource = False
        elif required_protocol_hash is None or required_workload_hash is None:
            passes_resource = False
        elif gates.get("performance_environment_hash") is None:
            passes_resource = False
        elif p95_budget is not None and resource_metrics.p95_duration_ns is None:
            passes_resource = False
        elif peak_budget is not None and resource_metrics.peak_memory_bytes is None:
            passes_resource = False
        elif resource_metrics.protocol_hash != str(required_protocol_hash):
            passes_resource = False
        elif resource_metrics.workload_hash != str(required_workload_hash):
            passes_resource = False
        elif resource_metrics.environment_hash != str(
            gates.get("performance_environment_hash")
        ):
            passes_resource = False

    if passes_resource and resource_metrics is not None and p95_budget is not None:
        assert resource_metrics.p95_duration_ns is not None
        passes_resource = resource_metrics.p95_duration_ns <= p95_budget
    if passes_resource and resource_metrics is not None and peak_budget is not None:
        assert resource_metrics.peak_memory_bytes is not None
        passes_resource = resource_metrics.peak_memory_bytes <= peak_budget

    violated: list[str] = []
    if not passes_execution:
        violated.append("execution")
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
        passes_execution_gate=passes_execution,
        passes_offset_gate=passes_offset,
        passes_precision_gate=passes_precision,
        passes_contradiction_gate=passes_contradiction,
        passes_resource_gates=passes_resource,
        gate_pass=not violated,
        violated_gates=violated,
    )


def baseline_context(
    baseline: Mapping[str, object] | None,
) -> tuple[str, Mapping[str, object]]:
    if baseline is not None:
        baseline_hash = str(baseline["hash"])
        gates = baseline.get("gates")
        if not isinstance(gates, Mapping):
            raise ValueError("baseline.gates must be a JSON object")
        return baseline_hash, gates

    return _baseline_hash_and_gates_from_payload(_load_frozen_baseline_report())


def _load_frozen_baseline_report() -> dict[str, object]:
    baseline_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "reports"
        / "v1"
        / "baseline.json"
    )
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline report must be a JSON object")
    return payload


def _baseline_hash_and_gates_from_payload(
    payload: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("baseline report must contain gates")
    for required_key in (
        "performance_protocol_hash",
        "selected_workload_hash",
        "performance_environment_hash",
    ):
        if required_key not in gates:
            raise ValueError(
                f"baseline report is missing required gate metadata: {required_key}"
            )
    return sha256_hex(canonical_json_bytes(payload)), gates


def _evaluate_candidate(
    *,
    candidate: SearchCandidate,
    bundle: TuningBundle,
    dataset_hash: str,
    baseline_hash: str,
    gates: Mapping[str, object],
    git_revision_value: str,
    code_snapshot_sha256: str,
    environment: ExperimentEnvironment,
) -> ExperimentRecord:
    resolved_config = CitationConfig.model_validate(candidate.config)
    resolved_payload = resolved_config.model_dump(mode="json")

    train_report = _run_accuracy(
        bundle.train_cases,
        backend=candidate.backend,
        config=resolved_config,
    )
    train_metrics = _candidate_metrics(train_report)
    early_decision = evaluate_gate_decision(
        train_metrics=train_metrics,
        dev_metrics=None,
        resource_metrics=None,
        gates=gates,
        evaluated_dev=False,
    )
    if not (
        early_decision.passes_execution_gate
        and early_decision.passes_offset_gate
        and early_decision.passes_precision_gate
        and early_decision.passes_contradiction_gate
    ):
        return build_experiment_record(
            candidate_id=candidate.candidate_id,
            parent_candidate_id=candidate.parent_candidate_id,
            dataset_hash=dataset_hash,
            baseline_hash=baseline_hash,
            git_revision=git_revision_value,
            code_snapshot_sha256=code_snapshot_sha256,
            code_path_id=candidate.code_path_id,
            backend=candidate.backend,
            embeddings=candidate.embeddings,
            environment=environment,
            config=resolved_payload,
            train_metrics=train_metrics,
            dev_metrics=None,
            resource_metrics=None,
            gate_decision=early_decision,
        )

    dev_report = _run_accuracy(
        bundle.dev_cases,
        backend=candidate.backend,
        config=resolved_config,
    )
    dev_metrics = _candidate_metrics(dev_report)
    smoke = measure_candidate_smoke(
        backend=candidate.backend,
        embeddings=candidate.embeddings,
        config=resolved_config,
    )
    resource_metrics = ResourceMetrics(
        median_duration_ns=smoke["median_duration_ns"],
        p95_duration_ns=(
            None if smoke["p95_duration_ns"] is None else int(smoke["p95_duration_ns"])
        ),
        peak_memory_bytes=(
            None
            if smoke["peak_memory_bytes"] is None
            else int(smoke["peak_memory_bytes"])
        ),
        config_sha256=(
            None if smoke["config_sha256"] is None else str(smoke["config_sha256"])
        ),
        raw_end_to_end_samples_ns={
            str(scenario_id): [int(sample) for sample in samples]
            for scenario_id, samples in smoke["raw_end_to_end_samples_ns"].items()
        },
        protocol_hash=(
            None if smoke["protocol_hash"] is None else str(smoke["protocol_hash"])
        ),
        workload_hash=(
            None if smoke["workload_hash"] is None else str(smoke["workload_hash"])
        ),
        environment_hash=(
            None
            if smoke["environment_hash"] is None
            else str(smoke["environment_hash"])
        ),
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
        git_revision=git_revision_value,
        code_snapshot_sha256=code_snapshot_sha256,
        code_path_id=candidate.code_path_id,
        backend=candidate.backend,
        embeddings=candidate.embeddings,
        environment=environment,
        config=resolved_payload,
        train_metrics=train_metrics,
        dev_metrics=dev_metrics,
        resource_metrics=resource_metrics,
        gate_decision=gate_decision,
    )


def _run_accuracy(
    cases: Sequence[EvaluationCase],
    *,
    backend: Backend,
    config: CitationConfig,
) -> AccuracyReport:
    runs = [execute_case(case=case, backend=backend, config=config) for case in cases]
    return accuracy_report(cases=cases, runs=runs)


def _candidate_metrics(report: AccuracyReport) -> CandidateMetrics:
    metrics = report.metrics
    return CandidateMetrics(
        exact_precision_lower=metrics.exact_precision.lower,
        contradiction_false_citation_count=metrics.contradiction_false_citation_rate.numerator,
        offset_invalid_count=metrics.offset_validity.denominator
        - metrics.offset_validity.numerator,
        requirement_recall=metrics.requirement_recall.estimate,
        status_macro_f1=metrics.status_macro_f1,
        retrieval_mrr=metrics.retrieval_mrr,
        run_error_count=report.run_error_count,
        output_sha256=report.output_sha256,
    )


def _ranking_key(record: ExperimentRecord) -> tuple[float, float, float, float, str]:
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
    raw = str(path).lower()
    resolved = str(path.resolve(strict=False)).lower()
    normalized = re.sub(r"[^a-z0-9]+", "", raw)
    resolved_normalized = re.sub(r"[^a-z0-9]+", "", resolved)
    return (
        "holdout" in normalized
        or "releasegate" in normalized
        or "aesgcm" in normalized
        or "holdout" in resolved_normalized
        or "releasegate" in resolved_normalized
        or "aesgcm" in resolved_normalized
    )


def _persist_progress(
    *,
    output_path: Path,
    dataset_hash: str,
    baseline_hash: str,
    git_revision_value: str,
    code_snapshot_sha256: str,
    search_space_hash: str,
    environment: ExperimentEnvironment,
    parent_experiment_id: str | None,
    duplicate_candidate_ids: list[str],
    records_by_id: Mapping[str, ExperimentRecord],
) -> None:
    _assert_code_provenance_unchanged(
        expected_git_revision=git_revision_value,
        expected_code_snapshot_sha256=code_snapshot_sha256,
    )
    ordered_records = [records_by_id[key] for key in sorted(records_by_id)]
    best_candidate_id = select_best_candidate_id(ordered_records)
    persist_experiment_store(
        output_path=output_path,
        store=ExperimentStore(
            dataset_hash=dataset_hash,
            baseline_hash=baseline_hash,
            git_revision=git_revision_value,
            code_snapshot_sha256=code_snapshot_sha256,
            search_space_hash=search_space_hash,
            environment=environment,
            parent_experiment_id=parent_experiment_id,
            best_candidate_id=best_candidate_id,
            duplicate_candidate_ids=duplicate_candidate_ids,
            records=ordered_records,
        ),
    )


def _candidate_identity(candidate: SearchCandidate) -> tuple[str, str, str, str]:
    canonical = canonicalize_config(
        CitationConfig.model_validate(candidate.config).model_dump(mode="json")
    )
    return (
        candidate.backend,
        candidate.embeddings,
        candidate.code_path_id,
        canonical.sha256,
    )


def _record_identity(record: ExperimentRecord) -> tuple[str, str, str, str]:
    return (
        record.backend,
        record.embeddings,
        record.code_path_id,
        record.config.sha256,
    )


def _validate_resume_context(
    *,
    store: ExperimentStore,
    dataset_hash: str,
    baseline_hash: str,
    search_space_hash: str,
    git_revision_value: str,
    code_snapshot_sha256: str,
    environment: ExperimentEnvironment,
    parent_experiment_id: str | None,
) -> None:
    if store.dataset_hash != dataset_hash:
        raise ValueError("existing experiment store dataset hash does not match")
    if store.baseline_hash != baseline_hash:
        raise ValueError("existing experiment store baseline hash does not match")
    if store.search_space_hash != search_space_hash:
        raise ValueError("existing experiment store search space hash does not match")
    if store.git_revision != git_revision_value:
        raise ValueError("existing experiment store git revision does not match")
    if store.code_snapshot_sha256 != code_snapshot_sha256:
        raise ValueError("existing experiment store code provenance does not match")
    if store.environment != environment:
        raise ValueError("existing experiment store environment does not match")
    if store.parent_experiment_id != parent_experiment_id:
        raise ValueError("existing experiment store parent experiment does not match")


def _assert_code_provenance_unchanged(
    *,
    expected_git_revision: str,
    expected_code_snapshot_sha256: str,
) -> None:
    if git_revision() != expected_git_revision:
        raise ValueError("working tree git revision changed during tuning")
    if current_code_snapshot_sha256() != expected_code_snapshot_sha256:
        raise ValueError("working tree code snapshot changed during tuning")


def _validate_real_candidate(candidate: SearchCandidate) -> None:
    if candidate.synthetic_result is not None:
        raise ValueError("public tune search spaces must not contain synthetic_result")
    if candidate.backend != SUPPORTED_BACKEND:
        raise ValueError("initial search supports only backend='python'")
    if candidate.embeddings != SUPPORTED_EMBEDDINGS:
        raise ValueError("initial search supports only embeddings='off'")
    if candidate.code_path_id != SUPPORTED_CODE_PATH:
        raise ValueError("initial search supports only code_path_id='config-only'")


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
