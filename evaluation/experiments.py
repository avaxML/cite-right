"""Experiment records and deterministic persistence for tuning runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cite_right import CitationConfig
from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.performance import SMOKE_TRIAL_COUNT, selected_smoke_scenario_ids
from evaluation.runner import CanonicalConfig, canonicalize_config

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
EXPERIMENT_SCHEMA_VERSION = "evaluation.experiments.v1"


class CandidateMetrics(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    exact_precision_lower: float | None
    contradiction_false_citation_count: int
    offset_invalid_count: int
    requirement_recall: float | None
    status_macro_f1: float | None
    retrieval_mrr: float | None
    run_error_count: int = 0
    output_sha256: str

    @field_validator(
        "contradiction_false_citation_count",
        "offset_invalid_count",
        "run_error_count",
    )
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("count fields must be non-negative")
        return value

    @field_validator(
        "exact_precision_lower",
        "requirement_recall",
        "status_macro_f1",
        "retrieval_mrr",
    )
    @classmethod
    def _validate_rate(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("rate fields must stay within [0, 1]")
        return value

    @field_validator("output_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(
                "output_sha256 must be a 64-character lowercase hex digest"
            )
        return value


class ResourceMetrics(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    median_duration_ns: float
    p95_duration_ns: int | None = None
    peak_memory_bytes: int | None = None
    config_sha256: str | None = None
    raw_end_to_end_samples_ns: dict[str, list[int]] = Field(default_factory=dict)
    protocol_hash: str | None = None
    workload_hash: str | None = None
    environment_hash: str | None = None

    @field_validator("median_duration_ns")
    @classmethod
    def _validate_median(cls, value: float) -> float:
        if value < 0:
            raise ValueError("median_duration_ns must be non-negative")
        return value

    @field_validator("p95_duration_ns", "peak_memory_bytes")
    @classmethod
    def _validate_optional_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("resource metrics must be non-negative when present")
        return value

    @field_validator(
        "config_sha256", "protocol_hash", "workload_hash", "environment_hash"
    )
    @classmethod
    def _validate_optional_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError(
                "protocol/workload hashes must be 64-character lowercase hex digests"
            )
        return value

    @field_validator("raw_end_to_end_samples_ns")
    @classmethod
    def _validate_raw_samples(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        if any(not scenario_id for scenario_id in value):
            raise ValueError("resource sample scenario IDs must be non-empty")
        if any(not samples for samples in value.values()):
            raise ValueError("resource sample lists must be non-empty")
        if any(sample < 0 for samples in value.values() for sample in samples):
            raise ValueError("resource samples must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_summaries_match_samples(self) -> ResourceMetrics:
        if not self.raw_end_to_end_samples_ns:
            return self
        medians: list[float] = []
        p95_values: list[int] = []
        for samples in self.raw_end_to_end_samples_ns.values():
            ordered = sorted(samples)
            middle = len(ordered) // 2
            median = (
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            )
            medians.append(median)
            p95_values.append(ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)])
        if self.median_duration_ns != max(medians):
            raise ValueError("median_duration_ns must be derived from raw samples")
        if self.p95_duration_ns != max(p95_values):
            raise ValueError("p95_duration_ns must be derived from raw samples")
        return self


class GateDecision(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    evaluated_dev: bool
    passes_execution_gate: bool
    passes_offset_gate: bool
    passes_precision_gate: bool
    passes_contradiction_gate: bool
    passes_resource_gates: bool
    gate_pass: bool
    violated_gates: list[str] = Field(default_factory=list)

    @field_validator("violated_gates", mode="before")
    @classmethod
    def _coerce_violations(cls, value: object) -> object:
        if isinstance(value, tuple):
            return list(value)
        return value

    @model_validator(mode="after")
    def _validate_state(self) -> GateDecision:
        expected = {
            name
            for name, passed in (
                ("execution", self.passes_execution_gate),
                ("offset", self.passes_offset_gate),
                ("precision", self.passes_precision_gate),
                ("contradiction", self.passes_contradiction_gate),
                ("resources", self.passes_resource_gates),
            )
            if not passed
        }
        actual = set(self.violated_gates)
        if expected != actual:
            raise ValueError(
                "violated_gates must exactly match the failed gate booleans"
            )
        if self.gate_pass != (
            self.passes_execution_gate
            and self.passes_offset_gate
            and self.passes_precision_gate
            and self.passes_contradiction_gate
            and self.passes_resource_gates
        ):
            raise ValueError("gate_pass must reflect the individual gate booleans")
        return self


class ExperimentEnvironment(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    python: str
    platform: str
    machine: str
    cpu_count: int | None = None


class ExperimentRecord(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    candidate_id: str
    parent_candidate_id: str | None = None
    dataset_hash: str
    baseline_hash: str
    git_revision: str
    code_snapshot_sha256: str
    code_path_id: str
    backend: str
    embeddings: str
    environment: ExperimentEnvironment
    config: CanonicalConfig
    train_metrics: CandidateMetrics
    dev_metrics: CandidateMetrics | None = None
    resource_metrics: ResourceMetrics | None = None
    gate_decision: GateDecision

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EXPERIMENT_SCHEMA_VERSION!r}")
        return value

    @field_validator("dataset_hash", "baseline_hash", "code_snapshot_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(
                "hash fields must be lowercase 64-character SHA-256 hex digests"
            )
        return value

    @model_validator(mode="after")
    def _validate_state(self) -> ExperimentRecord:
        if self.gate_decision.evaluated_dev:
            if self.dev_metrics is None or self.resource_metrics is None:
                raise ValueError(
                    "evaluated_dev records require dev_metrics and resource_metrics"
                )
        else:
            if self.dev_metrics is not None or self.resource_metrics is not None:
                raise ValueError(
                    "pre-dev records must not carry dev_metrics or resource_metrics"
                )
        if self.gate_decision.gate_pass and not self.gate_decision.evaluated_dev:
            raise ValueError("gate_pass requires a dev-evaluated record")
        if self.resource_metrics is not None:
            if (
                self.git_revision != "synthetic"
                and self.resource_metrics.config_sha256 is None
            ):
                raise ValueError("non-synthetic resource metrics require config_sha256")
            if (
                self.resource_metrics.config_sha256 is not None
                and self.resource_metrics.config_sha256 != self.config.sha256
            ):
                raise ValueError(
                    "resource_metrics.config_sha256 must match record.config.sha256"
                )
            if (
                self.git_revision != "synthetic"
                and not self.resource_metrics.raw_end_to_end_samples_ns
            ):
                raise ValueError("non-synthetic resource metrics require raw samples")
            if self.git_revision != "synthetic":
                raw_samples = self.resource_metrics.raw_end_to_end_samples_ns
                if set(raw_samples) != _expected_resource_scenario_ids(
                    backend=self.backend,
                    embeddings=self.embeddings,
                ):
                    raise ValueError(
                        "non-synthetic resource metrics require the exact workload scenarios"
                    )
                if any(
                    len(samples) != SMOKE_TRIAL_COUNT
                    for samples in raw_samples.values()
                ):
                    raise ValueError(
                        "non-synthetic resource metrics require the protocol trial count per scenario"
                    )
        if contains_forbidden_holdout_data(self.model_dump(mode="json")):
            raise ValueError("experiment records must not contain holdout data")
        return self


class ExperimentStore(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    dataset_hash: str
    baseline_hash: str
    git_revision: str
    code_snapshot_sha256: str
    search_space_hash: str
    environment: ExperimentEnvironment
    parent_experiment_id: str | None = None
    best_candidate_id: str | None = None
    duplicate_candidate_ids: list[str] = Field(default_factory=list)
    records: list[ExperimentRecord]

    @field_validator("duplicate_candidate_ids", "records", mode="before")
    @classmethod
    def _coerce_lists(cls, value: object) -> object:
        if isinstance(value, tuple):
            return list(value)
        return value

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EXPERIMENT_SCHEMA_VERSION!r}")
        return value

    @field_validator(
        "dataset_hash", "baseline_hash", "code_snapshot_sha256", "search_space_hash"
    )
    @classmethod
    def _validate_store_hashes(cls, value: str) -> str:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(
                "hash fields must be lowercase 64-character SHA-256 hex digests"
            )
        return value

    @model_validator(mode="after")
    def _validate_records(self) -> ExperimentStore:
        candidate_ids = [record.candidate_id for record in self.records]
        if candidate_ids != sorted(candidate_ids):
            raise ValueError("records must be sorted by candidate_id")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("records must not contain duplicate candidate_id values")
        for record in self.records:
            if record.dataset_hash != self.dataset_hash:
                raise ValueError(
                    "record dataset_hash must match the store dataset_hash"
                )
            if record.baseline_hash != self.baseline_hash:
                raise ValueError(
                    "record baseline_hash must match the store baseline_hash"
                )
            if record.git_revision != self.git_revision:
                raise ValueError(
                    "record git_revision must match the store git_revision"
                )
            if record.code_snapshot_sha256 != self.code_snapshot_sha256:
                raise ValueError(
                    "record code_snapshot_sha256 must match the store code_snapshot_sha256"
                )
            if record.environment != self.environment:
                raise ValueError("record environment must match the store environment")
        expected_best = select_best_candidate_id(self.records)
        if self.best_candidate_id != expected_best:
            raise ValueError("best_candidate_id must equal the deterministic winner")
        if contains_forbidden_holdout_data(self.model_dump(mode="json")):
            raise ValueError("experiment store must not contain holdout data")
        return self


def build_experiment_record(
    *,
    candidate_id: str,
    parent_candidate_id: str | None,
    dataset_hash: str,
    baseline_hash: str,
    git_revision: str,
    code_snapshot_sha256: str | None,
    code_path_id: str,
    backend: str,
    embeddings: str,
    environment: ExperimentEnvironment | None,
    config: Mapping[str, object] | CanonicalConfig,
    train_metrics: CandidateMetrics,
    dev_metrics: CandidateMetrics | None,
    resource_metrics: ResourceMetrics | None,
    gate_decision: GateDecision,
) -> ExperimentRecord:
    canonical_config = (
        config
        if isinstance(config, CanonicalConfig)
        else canonicalize_config(
            CitationConfig.model_validate(config).model_dump(mode="json")
        )
    )
    return ExperimentRecord(
        candidate_id=candidate_id,
        parent_candidate_id=parent_candidate_id,
        dataset_hash=dataset_hash,
        baseline_hash=baseline_hash,
        git_revision=git_revision,
        code_snapshot_sha256=code_snapshot_sha256 or current_code_snapshot_sha256(),
        code_path_id=code_path_id,
        backend=backend,
        embeddings=embeddings,
        environment=environment or current_environment(),
        config=canonical_config,
        train_metrics=train_metrics,
        dev_metrics=dev_metrics,
        resource_metrics=resource_metrics,
        gate_decision=gate_decision,
    )


def persist_experiment_store(*, output_path: Path, store: ExperimentStore) -> Path:
    target = resolve_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(store)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def load_experiment_store(path: Path) -> ExperimentStore:
    payload = json.loads(resolve_output_path(path).read_text(encoding="utf-8"))
    return ExperimentStore.model_validate(payload)


def resolve_output_path(output_path: Path) -> Path:
    if output_path.suffix == ".json":
        return output_path
    return output_path / "experiments.json"


def contains_forbidden_holdout_data(value: object) -> bool:
    if isinstance(value, Path):
        return "holdout" in str(value).lower()
    if isinstance(value, str):
        return "holdout" in value.lower()
    if isinstance(value, Mapping):
        return any(
            contains_forbidden_holdout_data(key)
            or contains_forbidden_holdout_data(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_forbidden_holdout_data(item) for item in value)
    return False


def git_revision() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    revision = result.stdout.strip()
    if (
        result.returncode != 0
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError(
            "cannot resolve an exact Git commit for experiment provenance"
        )
    return revision


def canonical_search_space_hash(payload: Mapping[str, object]) -> str:
    return sha256_hex(canonical_json_bytes(payload))


def record_map(records: Sequence[ExperimentRecord]) -> dict[str, ExperimentRecord]:
    return {record.candidate_id: record for record in records}


def select_best_candidate_id(
    records: Sequence[ExperimentRecord],
) -> str | None:
    survivors = [record for record in records if record.gate_decision.gate_pass]
    if not survivors:
        return None
    return max(survivors, key=_ranking_key).candidate_id


def current_environment() -> ExperimentEnvironment:
    return ExperimentEnvironment(
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
    )


def current_code_snapshot_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        path
        for directory in (root / "src", root / "evaluation", root / "rust_core" / "src")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi", ".rs"}
    )
    payload = bytearray()
    for path in files:
        payload.extend(path.relative_to(root).as_posix().encode("utf-8"))
        payload.extend(b"\0")
        payload.extend(path.read_bytes())
        payload.extend(b"\0")
    return sha256_hex(bytes(payload))


def _expected_resource_scenario_ids(*, backend: str, embeddings: str) -> set[str]:
    if backend not in {"python", "rust"} or embeddings not in {"off", "on"}:
        raise ValueError(
            "resource metrics use an unsupported backend or embeddings mode"
        )
    return set(
        selected_smoke_scenario_ids(
            backend=backend,  # type: ignore[arg-type]
            embeddings=embeddings,  # type: ignore[arg-type]
        )
    )


def _ranking_key(record: ExperimentRecord) -> tuple[float, float, float, float, str]:
    if record.dev_metrics is None or record.resource_metrics is None:
        return (-1.0, -1.0, -1.0, 0, record.candidate_id)
    return (
        record.dev_metrics.requirement_recall or 0.0,
        record.dev_metrics.status_macro_f1 or 0.0,
        record.dev_metrics.retrieval_mrr or 0.0,
        -record.resource_metrics.median_duration_ns,
        record.candidate_id,
    )
