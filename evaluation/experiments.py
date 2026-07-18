"""Experiment records and deterministic persistence for tuning runs."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from evaluation.canonical import canonical_json_bytes, sha256_hex
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

    @field_validator("contradiction_false_citation_count", "offset_invalid_count", "run_error_count")
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
            raise ValueError("output_sha256 must be a 64-character lowercase hex digest")
        return value


class ResourceMetrics(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    median_duration_ns: int
    p95_duration_ns: int
    peak_memory_bytes: int | None = None

    @field_validator("median_duration_ns", "p95_duration_ns")
    @classmethod
    def _validate_duration(cls, value: int) -> int:
        if value < 0:
            raise ValueError("duration fields must be non-negative")
        return value

    @field_validator("peak_memory_bytes")
    @classmethod
    def _validate_peak_memory(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("peak_memory_bytes must be non-negative")
        return value


class GateDecision(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    evaluated_dev: bool
    passes_offset_gate: bool
    passes_precision_gate: bool
    passes_contradiction_gate: bool
    passes_resource_gates: bool
    gate_pass: bool
    violated_gates: list[str] = []

    @model_validator(mode="after")
    def _validate_state(self) -> GateDecision:
        expected = not self.violated_gates
        if self.gate_pass != expected:
            raise ValueError("gate_pass must reflect violated_gates")
        return self


class ExperimentRecord(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    candidate_id: str
    parent_candidate_id: str | None = None
    dataset_hash: str
    baseline_hash: str
    git_revision: str
    code_path_id: str
    backend: str
    embeddings: str
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

    @field_validator("dataset_hash", "baseline_hash")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("hash fields must be lowercase 64-character SHA-256 hex digests")
        return value

    @model_validator(mode="after")
    def _reject_holdout_content(self) -> ExperimentRecord:
        if contains_forbidden_holdout_data(self.model_dump(mode="json")):
            raise ValueError("experiment records must not contain holdout data")
        return self


class ExperimentStore(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    dataset_hash: str
    baseline_hash: str
    git_revision: str
    search_space_hash: str
    parent_experiment_id: str | None = None
    best_candidate_id: str | None = None
    duplicate_candidate_ids: list[str] = []
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

    @model_validator(mode="after")
    def _validate_records(self) -> ExperimentStore:
        candidate_ids = [record.candidate_id for record in self.records]
        if candidate_ids != sorted(candidate_ids):
            raise ValueError("records must be sorted by candidate_id")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("records must not contain duplicate candidate_id values")
        for record in self.records:
            if record.dataset_hash != self.dataset_hash:
                raise ValueError("record dataset_hash must match the store dataset_hash")
            if record.baseline_hash != self.baseline_hash:
                raise ValueError("record baseline_hash must match the store baseline_hash")
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
    code_path_id: str,
    backend: str,
    embeddings: str,
    config: Mapping[str, object],
    train_metrics: CandidateMetrics,
    dev_metrics: CandidateMetrics | None,
    resource_metrics: ResourceMetrics | None,
    gate_decision: GateDecision,
) -> ExperimentRecord:
    return ExperimentRecord(
        candidate_id=candidate_id,
        parent_candidate_id=parent_candidate_id,
        dataset_hash=dataset_hash,
        baseline_hash=baseline_hash,
        git_revision=git_revision,
        code_path_id=code_path_id,
        backend=backend,
        embeddings=embeddings,
        config=canonicalize_config(config),
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
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def canonical_search_space_hash(payload: Mapping[str, object]) -> str:
    return sha256_hex(canonical_json_bytes(payload))


def record_map(records: Sequence[ExperimentRecord]) -> dict[str, ExperimentRecord]:
    return {record.candidate_id: record for record in records}
