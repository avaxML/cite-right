"""Honest train/dev accuracy and resource baselines for strict attribution."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from cite_right import CitationConfig
from cite_right.core.results import Citation, SpanCitations
from cite_right.models.base import Embedder
from cite_right.models.sbert_embedder import SentenceTransformerEmbedder
from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.matching import EmittedCitation, match_citations
from evaluation.metrics import (
    CaseMetricRecord,
    MetricReport,
    StatusLabel,
    aggregate_metrics,
)
from evaluation.performance import (
    run_performance_smoke,
    selected_smoke_workload_hash,
    smoke_environment_compatibility_hash,
)
from evaluation.runner import Backend, CaseRun, execute_case
from evaluation.schema import CharSpan, EvaluationCase, EvaluationUnit
from evaluation.tuning_bundle import load_tuning_bundle

SCHEMA_VERSION = "evaluation.baseline.v1"
STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
PINNED_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EmbeddingMode = Literal["off", "on"]


class BaselineConfiguration(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    name: Literal["default", "strict", "permissive"]
    config: dict[str, object]
    selected_for_gates: bool = False


class AccuracyReport(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    case_count: int
    unit_count: int
    run_error_count: int
    metrics: MetricReport
    output_sha256: str


def baseline_configurations() -> tuple[BaselineConfiguration, ...]:
    strict_control = CitationConfig.strict().model_copy(
        update={"require_all_answer_tokens_in_evidence": False}
    )
    return (
        _configuration("default", CitationConfig()),
        _configuration(
            "strict",
            strict_control,
            selected_for_gates=True,
        ),
        _configuration("permissive", CitationConfig.permissive()),
    )


def accuracy_report(
    *, cases: Sequence[EvaluationCase], runs: Sequence[CaseRun]
) -> AccuracyReport:
    if len(cases) != len(runs):
        raise ValueError("accuracy cases and runs must have equal lengths")
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("accuracy cases must have unique case IDs")
    records: list[CaseMetricRecord] = []
    output_payload: list[object] = []
    for run in runs:
        case = case_by_id.get(run.case_id)
        if case is None:
            raise ValueError(f"run references unknown case {run.case_id!r}")
        records.extend(_score_case(case=case, run=run))
        output_payload.append(
            {
                "case_id": run.case_id,
                "backend": run.backend,
                "config_sha256": run.config.sha256,
                "outputs": [output.model_dump(mode="json") for output in run.outputs],
                "output_unit_ids": run.output_unit_ids,
                "error": None
                if run.error is None
                else run.error.model_dump(mode="json"),
            }
        )
    return AccuracyReport(
        case_count=len(cases),
        unit_count=len(records),
        run_error_count=sum(run.error is not None for run in runs),
        metrics=aggregate_metrics(tuple(records)),
        output_sha256=sha256_hex(canonical_json_bytes(output_payload)),
    )


def build_baseline(*, tuning_bundle: Path, output_path: Path) -> dict[str, object]:
    bundle = load_tuning_bundle(tuning_bundle)
    captured_git_revision = _git_revision()
    captured_code_snapshot_sha256 = _code_snapshot_sha256()
    backends: tuple[Backend, ...] = (
        ("python", "rust") if _rust_backend_supported() else ("python",)
    )
    pinned_embedder, embedding_coverage = _load_pinned_embedder()
    embedding_modes: tuple[tuple[EmbeddingMode, Embedder | None], ...] = (
        (("off", None), ("on", pinned_embedder))
        if pinned_embedder is not None
        else (("off", None),)
    )
    matrix: list[dict[str, object]] = []
    execution_inputs: list[
        tuple[
            str,
            Backend,
            CitationConfig,
            tuple[EvaluationCase, ...],
            Embedder | None,
        ]
    ] = []
    for configuration in baseline_configurations():
        config = CitationConfig.model_validate(configuration.config)
        for backend in backends:
            for embedding_mode, embedder in embedding_modes:
                for split, cases in (
                    ("train", bundle.train_cases),
                    ("dev", bundle.dev_cases),
                ):
                    execution_inputs.append(
                        (
                            f"{configuration.name}/{backend}/{embedding_mode}/{split}",
                            backend,
                            config,
                            cases,
                            embedder,
                        )
                    )

    first_reports = _execute_accuracy_inputs(execution_inputs)
    selected_configuration = next(
        configuration
        for configuration in baseline_configurations()
        if configuration.selected_for_gates
    )
    selected_smoke_config = CitationConfig.model_validate(selected_configuration.config)
    with tempfile.TemporaryDirectory(prefix="cite-right-baseline-") as temporary:
        temp_root = Path(temporary)
        performance_trials = []
        for index in (1, 2):
            artifact_path = temp_root / f"performance-{index}.json"
            run_performance_smoke(
                output_path=artifact_path,
                config=selected_smoke_config,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if not isinstance(artifact, dict):
                raise RuntimeError("performance smoke artifact must be a JSON object")
            environment = artifact.get("environment")
            if not isinstance(environment, Mapping):
                raise RuntimeError(
                    "performance smoke artifact must include environment metadata"
                )
            if environment.get("git_revision") != captured_git_revision:
                raise RuntimeError(
                    "performance smoke artifact git_revision does not match captured revision"
                )
            _assert_code_provenance_unchanged(
                expected_git_revision=captured_git_revision,
                expected_code_snapshot_sha256=captured_code_snapshot_sha256,
            )
            performance_trials.append(artifact)
    second_reports = _execute_accuracy_inputs(execution_inputs)
    if _accuracy_hashes(first_reports) != _accuracy_hashes(second_reports):
        raise RuntimeError("performance trials changed correctness outputs")

    by_id = {identifier: report for identifier, report in first_reports}
    for configuration in baseline_configurations():
        for backend in backends:
            for embedding_mode, _embedder in embedding_modes:
                identifier = f"{configuration.name}/{backend}/{embedding_mode}"
                matrix.append(
                    {
                        "id": identifier,
                        "configuration_name": configuration.name,
                        "selected_for_gates": (
                            configuration.selected_for_gates
                            and backend == "python"
                            and embedding_mode == "off"
                        ),
                        "backend": backend,
                        "embeddings": embedding_mode,
                        "embedding_model": (
                            None
                            if embedding_mode == "off"
                            else {
                                "name": PINNED_EMBEDDING_MODEL,
                                "revision": PINNED_EMBEDDING_REVISION,
                            }
                        ),
                        "config": configuration.config,
                        "train": by_id[f"{identifier}/train"].model_dump(mode="json"),
                        "dev": by_id[f"{identifier}/dev"].model_dump(mode="json"),
                    }
                )

    selected = next(item for item in matrix if item["selected_for_gates"])
    selected_config = selected.get("config")
    if not isinstance(selected_config, Mapping):
        raise ValueError("selected baseline config must be a JSON object")
    selected_config_sha256 = sha256_hex(canonical_json_bytes(selected_config))
    protocol_hash, performance_config_sha256 = _validated_performance_metadata(
        performance_trials=performance_trials,
        expected_config_sha256=selected_config_sha256,
    )
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": bundle.manifest.dataset_version,
        "dataset_hash": bundle.manifest.dataset_manifest_sha256,
        "git_revision": captured_git_revision,
        "code_snapshot_sha256": captured_code_snapshot_sha256,
        "worktree_dirty": _worktree_dirty(),
        "environment": _environment(),
        "optional_coverage": {
            "rust_backend": "available" if "rust" in backends else "unavailable",
            "pinned_embedding_model": embedding_coverage,
            "performance_embeddings_on": "deterministic local smoke embedder",
        },
        "matrix": matrix,
        "performance_trials": performance_trials,
        "selected_baseline_id": selected["id"],
        "gates": _freeze_gates(
            selected=selected,
            performance_trials=performance_trials,
            performance_protocol_hash=protocol_hash,
            performance_config_sha256=performance_config_sha256,
        ),
    }
    if report_contains_holdout_data(report):
        raise RuntimeError("baseline report contains forbidden holdout data")
    _assert_code_provenance_unchanged(
        expected_git_revision=captured_git_revision,
        expected_code_snapshot_sha256=captured_code_snapshot_sha256,
    )
    _write_atomic(output_path, canonical_json_bytes(report))
    return report


def compare_baselines(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    metadata_fields = (
        "schema_version",
        "dataset_hash",
        "code_snapshot_sha256",
        "selected_baseline_id",
    )
    if any(left.get(field) != right.get(field) for field in metadata_fields):
        raise ValueError("baseline metadata differs")
    left_correctness = _correctness_signature(left)
    right_correctness = _correctness_signature(right)
    if left_correctness != right_correctness:
        raise ValueError("baseline correctness outputs differ")
    ratios = _performance_median_ratios(left, right)
    margin = _declared_performance_margin(left, right)
    if any(ratio > 1.0 + margin or ratio < 1.0 - margin for ratio in ratios.values()):
        raise ValueError("baseline performance exceeded the declared variance envelope")
    p95_ratios = _scenario_metric_ratios(
        left,
        right,
        extractor=lambda scenario: _duration_metric(scenario, "p95_duration_ns"),
    )
    p95_margin = _declared_gate_margin(
        left,
        right,
        key="all_scenario_p95_noise_margin",
        fallback_key="p95_noise_margin",
    )
    if any(
        ratio > 1.0 + p95_margin or ratio < 1.0 - p95_margin
        for ratio in p95_ratios.values()
    ):
        raise ValueError("baseline p95 latency exceeded the declared variance envelope")
    peak_memory_ratios = _scenario_metric_ratios(
        left,
        right,
        extractor=lambda scenario: _int_metric(scenario, "peak_memory_bytes"),
    )
    peak_memory_margin = _declared_gate_margin(
        left,
        right,
        key="all_scenario_peak_memory_noise_margin",
        fallback_key="peak_memory_noise_margin",
    )
    if any(
        ratio > 1.0 + peak_memory_margin or ratio < 1.0 - peak_memory_margin
        for ratio in peak_memory_ratios.values()
    ):
        raise ValueError("baseline peak memory exceeded the declared variance envelope")
    return {
        "correctness_equal": True,
        "performance_median_ratios": ratios,
        "performance_noise_margin": margin,
        "p95_latency_ratios": p95_ratios,
        "p95_noise_margin": p95_margin,
        "peak_memory_ratios": peak_memory_ratios,
        "peak_memory_noise_margin": peak_memory_margin,
    }


def report_contains_holdout_data(value: object) -> bool:
    if isinstance(value, Path):
        return "holdout" in str(value).lower()
    if isinstance(value, str):
        return "holdout" in value.lower()
    if isinstance(value, Mapping):
        return any(
            report_contains_holdout_data(key) or report_contains_holdout_data(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(report_contains_holdout_data(item) for item in value)
    return False


def _configuration(
    name: Literal["default", "strict", "permissive"],
    config: CitationConfig,
    *,
    selected_for_gates: bool = False,
) -> BaselineConfiguration:
    return BaselineConfiguration(
        name=name,
        config=config.model_dump(mode="json"),
        selected_for_gates=selected_for_gates,
    )


def _execute_accuracy_inputs(
    inputs: Sequence[
        tuple[
            str,
            Backend,
            CitationConfig,
            tuple[EvaluationCase, ...],
            Embedder | None,
        ]
    ],
) -> list[tuple[str, AccuracyReport]]:
    reports: list[tuple[str, AccuracyReport]] = []
    for identifier, backend, config, cases, embedder in inputs:
        runs = tuple(
            execute_case(
                case=case,
                backend=backend,
                config=config,
                embedder=embedder,
            )
            for case in cases
        )
        reports.append((identifier, accuracy_report(cases=cases, runs=runs)))
    return reports


def _score_case(*, case: EvaluationCase, run: CaseRun) -> list[CaseMetricRecord]:
    output_pairs = list(zip(run.outputs, run.output_unit_ids, strict=True))
    return [
        _score_unit(
            case=case,
            unit=unit,
            output_pairs=output_pairs,
            run_failed=run.error is not None,
        )
        for unit in case.evaluation_units
    ]


def _score_unit(
    *,
    case: EvaluationCase,
    unit: EvaluationUnit,
    output_pairs: Sequence[tuple[SpanCitations, tuple[str, ...]]],
    run_failed: bool,
) -> CaseMetricRecord:
    outputs = [output for output, unit_ids in output_pairs if unit.unit_id in unit_ids]
    emissions: list[EmittedCitation] = []
    emitted_offsets_valid = 0
    retrieval_ranks: list[int | None] = []
    for output in outputs:
        for citation in output.citations:  # type: ignore[attr-defined]
            spans = tuple(
                CharSpan(start=span.char_start, end=span.char_end)
                for span in (citation.evidence_spans or [citation])
            )
            emissions.append(EmittedCitation(source_id=citation.source_id, spans=spans))
            emitted_offsets_valid += int(_citation_offsets_valid(case, citation))
    retrieval_support = [
        support for output in outputs for support in output.retrieval_support
    ]
    for claim in unit.claims:
        if not claim.acceptable_retrieval_source_ids:
            continue
        acceptable = set(claim.acceptable_retrieval_source_ids)
        retrieval_ranks.append(
            next(
                (
                    index
                    for index, support in enumerate(retrieval_support, start=1)
                    if support.source_id in acceptable
                ),
                None,
            )
        )

    requirements = tuple(
        requirement
        for claim in unit.claims
        if claim.label == "entailed"
        for requirement in claim.citation_requirements
    )
    exact = match_citations(
        emissions=emissions, requirements=requirements, threshold="exact"
    )
    at_09 = match_citations(
        emissions=emissions, requirements=requirements, threshold="0.9"
    )
    at_05 = match_citations(
        emissions=emissions, requirements=requirements, threshold="0.5"
    )
    exact_ids = {match.requirement_id for match in exact.matches}
    entailed_claims = [claim for claim in unit.claims if claim.label == "entailed"]
    contradicted = [claim for claim in unit.claims if claim.label == "contradicted"]
    observed_status = _observed_status(outputs)
    source_correct = sum(
        any(
            emission.source_id == alternative.source_id
            for requirement in requirements
            for alternative in requirement.alternatives
        )
        for emission in emissions
    )
    multi_requirement_ids = {
        requirement.requirement_id
        for requirement in requirements
        if any(len(alternative.spans) > 1 for alternative in requirement.alternatives)
    }
    multi_emission_count = sum(len(emission.spans) > 1 for emission in emissions)
    multi_tp = sum(
        match.requirement_id in multi_requirement_ids for match in exact.matches
    )
    fully_attributed = sum(
        all(
            requirement.requirement_id in exact_ids
            for requirement in claim.citation_requirements
        )
        for claim in entailed_claims
    )
    contradicted_cited = sum(
        any(
            max(output.answer_span.char_start, claim.answer_span.start)
            < min(output.answer_span.char_end, claim.answer_span.end)
            and bool(output.citations)
            for output in outputs
        )
        for claim in contradicted
    )
    return CaseMetricRecord(
        expected_status=unit.expected_status,
        observed_status=observed_status,
        exact_true_positives=len(exact.matches),
        exact_false_positives=len(exact.unmatched_emission_indices) + len(exact.errors),
        exact_false_negatives=len(exact.unmatched_requirement_ids),
        recall_at_0_9_true_positives=len(at_09.matches),
        recall_at_0_9_false_negatives=len(at_09.unmatched_requirement_ids),
        recall_at_0_5_true_positives=len(at_05.matches),
        recall_at_0_5_false_negatives=len(at_05.unmatched_requirement_ids),
        requirement_count=len(requirements),
        matched_requirement_count=len(exact.matches),
        entailed_claim_count=len(entailed_claims),
        fully_attributed_claim_count=fully_attributed,
        source_selection_attempt_count=len(emissions),
        source_selection_correct_count=source_correct,
        emitted_citation_count=len(emissions),
        valid_offset_count=emitted_offsets_valid,
        multi_span_true_positives=multi_tp,
        multi_span_false_positives=max(multi_emission_count - multi_tp, 0),
        multi_span_false_negatives=len(multi_requirement_ids) - multi_tp,
        contradicted_claim_count=len(contradicted),
        contradicted_claim_citation_count=contradicted_cited,
        retrieval_eligible_claim_count=len(retrieval_ranks),
        retrieval_ranks=tuple(retrieval_ranks),
        evaluator_error=run_failed or bool(exact.errors),
    )


def _observed_status(outputs: Sequence[SpanCitations]) -> StatusLabel:
    statuses = {output.status for output in outputs}
    if "supported" in statuses:
        return "supported"
    if "partial" in statuses:
        return "partial"
    return "unsupported"


def _citation_offsets_valid(case: EvaluationCase, citation: Citation) -> bool:
    source = next(
        (item for item in case.sources if item.source_id == citation.source_id), None
    )
    if source is None:
        return False
    base = source.chunk_char_start or 0
    for span in citation.evidence_spans or [citation]:
        start = span.char_start - base
        end = span.char_end - base
        if start < 0 or end > len(source.text) or start >= end:
            return False
        if source.text[start:end] != span.evidence:
            return False
    return True


def _freeze_gates(
    *,
    selected: Mapping[str, object],
    performance_trials: Sequence[Mapping[str, object]],
    performance_protocol_hash: str,
    performance_config_sha256: str,
) -> dict[str, object]:
    dev = selected["dev"]
    assert isinstance(dev, Mapping)
    metrics = dev["metrics"]
    assert isinstance(metrics, Mapping)
    precision = metrics["exact_precision"]
    assert isinstance(precision, Mapping)
    offsets = metrics["offset_validity"]
    contradiction = metrics["contradiction_false_citation_rate"]
    assert isinstance(offsets, Mapping)
    assert isinstance(contradiction, Mapping)
    selected_scenarios = _selected_performance_scenarios(
        selected=selected,
        performance_trials=performance_trials,
    )
    selected_observations = [
        scenario for scenarios in selected_scenarios.values() for scenario in scenarios
    ]
    p95_margin = _scenario_metric_margin(
        selected_scenarios,
        lambda scenario: _duration_metric(scenario, "p95_duration_ns"),
    )
    peak_margin = _scenario_metric_margin(
        selected_scenarios,
        lambda scenario: _int_metric(scenario, "peak_memory_bytes"),
    )
    p95_values = [
        value
        for value in (
            _duration_metric(scenario, "p95_duration_ns")
            for scenario in selected_observations
        )
        if value is not None
    ]
    peak_values = [
        value
        for value in (
            _int_metric(scenario, "peak_memory_bytes")
            for scenario in selected_observations
        )
        if value is not None
    ]
    lower = precision.get("lower")
    selected_backend = str(selected.get("backend"))
    selected_embeddings = str(selected.get("embeddings"))
    if selected_backend not in {"python", "rust"}:
        raise ValueError("selected baseline backend is invalid")
    if selected_embeddings not in {"off", "on"}:
        raise ValueError("selected baseline embeddings mode is invalid")
    selected_workload_hash = selected_smoke_workload_hash(
        backend=cast(Literal["python", "rust"], selected_backend),
        embeddings=cast(Literal["off", "on"], selected_embeddings),
    )
    return {
        "policy": "strict/python/off selected before the sealed release evaluation",
        "offset_invalid_tolerance": 0,
        "baseline_offset_invalid_count": int(offsets["denominator"])
        - int(offsets["numerator"]),
        "contradiction_false_citation_tolerance": 0,
        "baseline_contradiction_false_citation_count": int(contradiction["numerator"]),
        "exact_precision_wilson_lower_min": None if lower is None else float(lower),
        "p95_latency_budget_ns": None
        if not p95_values
        else int(max(p95_values) * (1.0 + p95_margin)),
        "peak_memory_budget_bytes": None
        if not peak_values
        else int(max(peak_values) * (1.0 + peak_margin)),
        "p95_noise_margin": p95_margin,
        "peak_memory_noise_margin": peak_margin,
        "performance_protocol_hash": performance_protocol_hash,
        "performance_config_sha256": performance_config_sha256,
        "selected_workload_hash": selected_workload_hash,
        "performance_environment_hash": smoke_environment_compatibility_hash(),
        "all_scenario_p95_noise_margin": _scenario_metric_margin(
            _scenario_index(performance_trials),
            lambda scenario: _duration_metric(scenario, "p95_duration_ns"),
        ),
        "all_scenario_peak_memory_noise_margin": _scenario_metric_margin(
            _scenario_index(performance_trials),
            lambda scenario: _int_metric(scenario, "peak_memory_bytes"),
        ),
        "performance_noise_margin": _scenario_metric_margin(
            _scenario_index(performance_trials),
            lambda scenario: _duration_metric(scenario, "median_duration_ns"),
        ),
    }


def _validated_performance_metadata(
    *,
    performance_trials: Sequence[Mapping[str, object]],
    expected_config_sha256: str,
) -> tuple[str, str]:
    protocol_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for trial in performance_trials:
        protocol_hash = trial.get("protocol_hash")
        config_sha256 = trial.get("config_sha256")
        if (
            not isinstance(protocol_hash, str)
            or len(protocol_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in protocol_hash)
        ):
            raise ValueError("performance trials are missing a valid protocol_hash")
        if (
            not isinstance(config_sha256, str)
            or len(config_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in config_sha256)
        ):
            raise ValueError("performance trials are missing a valid config_sha256")
        if config_sha256 != expected_config_sha256:
            raise ValueError(
                "performance trial config hash does not match selected baseline config"
            )
        protocol_hashes.add(protocol_hash)
        config_hashes.add(config_sha256)
    if len(protocol_hashes) != 1:
        raise ValueError("performance trial protocol hashes do not match")
    if len(config_hashes) != 1:
        raise ValueError("performance trial config hashes do not match")
    return next(iter(protocol_hashes)), next(iter(config_hashes))


def _accuracy_hashes(reports: Sequence[tuple[str, AccuracyReport]]) -> dict[str, str]:
    return {identifier: report.output_sha256 for identifier, report in reports}


def _correctness_signature(report: Mapping[str, object]) -> object:
    matrix = report.get("matrix")
    if not isinstance(matrix, list):
        raise ValueError("baseline report matrix is missing")
    return tuple(
        (
            item.get("id"),
            item.get("train", {}).get("output_sha256"),
            item.get("dev", {}).get("output_sha256"),
        )
        for item in matrix
        if isinstance(item, Mapping)
    )


def _performance_median_ratios(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, float]:
    return _scenario_metric_ratios(
        left,
        right,
        extractor=lambda scenario: _duration_metric(scenario, "median_duration_ns"),
    )


def _selected_performance_scenarios(
    *,
    selected: Mapping[str, object],
    performance_trials: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    selected_backend = str(selected.get("backend"))
    selected_embeddings = str(selected.get("embeddings"))
    filtered = {
        scenario_id: scenarios
        for scenario_id, scenarios in _scenario_index(performance_trials).items()
        if all(
            str(scenario.get("backend")) == selected_backend
            and str(scenario.get("embeddings")) == selected_embeddings
            for scenario in scenarios
        )
    }
    if not filtered:
        raise ValueError(
            "baseline performance trials are missing selected baseline scenarios"
        )
    return filtered


def _scenario_index(
    performance_trials: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    collected: dict[str, list[Mapping[str, object]]] = {}
    for trial in performance_trials:
        scenarios = cast(Sequence[object], trial.get("scenarios", []))
        for scenario in scenarios:
            if not isinstance(scenario, Mapping):
                continue
            collected.setdefault(str(scenario.get("scenario_id")), []).append(scenario)
    return collected


def _scenario_metric_margin(
    scenarios_by_id: Mapping[str, Sequence[Mapping[str, object]]],
    extractor: Callable[[Mapping[str, object]], int | None],
) -> float:
    margin = 0.0
    for scenarios in scenarios_by_id.values():
        values = [
            value
            for value in (extractor(scenario) for scenario in scenarios)
            if value is not None
        ]
        if len(values) < 2:
            continue
        baseline = min(values)
        ceiling = max(values)
        if baseline <= 0:
            continue
        margin = max(margin, (ceiling / baseline) - 1.0)
    return margin


def _declared_performance_margin(
    left: Mapping[str, object], right: Mapping[str, object]
) -> float:
    return _declared_gate_margin(left, right, key="performance_noise_margin")


def _declared_gate_margin(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    key: str,
    fallback_key: str | None = None,
) -> float:
    margins: list[float] = []
    for report in (left, right):
        gates = report.get("gates")
        if isinstance(gates, Mapping):
            value = gates.get(key)
            if value is None and fallback_key is not None:
                value = gates.get(fallback_key)
            if isinstance(value, (int, float)):
                margins.append(float(value))
    if not margins:
        return 0.25
    if len(margins) != 2 or any(
        not math.isfinite(margin) or margin < 0.0 for margin in margins
    ):
        raise ValueError(f"baseline {key} values are invalid")
    return max(margins)


def _scenario_metric_ratios(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    extractor: Callable[[Mapping[str, object]], int | None],
) -> dict[str, float]:
    left_values = _scenario_metric_means(left, extractor=extractor)
    right_values = _scenario_metric_means(right, extractor=extractor)
    if left_values.keys() != right_values.keys():
        raise ValueError("baseline performance scenarios differ")
    return {
        key: right_values[key] / left_values[key]
        for key in left_values
        if left_values[key] > 0
    }


def _scenario_metric_means(
    report: Mapping[str, object],
    *,
    extractor: Callable[[Mapping[str, object]], int | None],
) -> dict[str, float]:
    trials = report.get("performance_trials", [])
    collected: dict[str, list[int]] = {}
    if isinstance(trials, list):
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            for scenario in trial.get("scenarios", []):
                if not isinstance(scenario, Mapping):
                    continue
                value = extractor(scenario)
                if value is None:
                    continue
                collected.setdefault(str(scenario.get("scenario_id")), []).append(value)
    return {key: sum(items) / len(items) for key, items in collected.items()}


def _duration_metric(scenario: Mapping[str, object], key: str) -> int | None:
    duration = scenario.get("end_to_end")
    if isinstance(duration, Mapping):
        value = duration.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _int_metric(scenario: Mapping[str, object], key: str) -> int | None:
    value = scenario.get(key)
    if isinstance(value, int):
        return value
    return None


def _load_pinned_embedder() -> tuple[Embedder | None, str]:
    cache_root = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    snapshot = (
        cache_root
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / PINNED_EMBEDDING_REVISION
    )
    model_id = f"{PINNED_EMBEDDING_MODEL}@{PINNED_EMBEDDING_REVISION}"
    if not snapshot.is_dir():
        return None, f"unavailable offline: {model_id} is not cached"
    try:
        return SentenceTransformerEmbedder(str(snapshot)), f"available: {model_id}"
    except (OSError, RuntimeError, ValueError) as exc:
        return (
            None,
            f"unavailable offline: {model_id} could not load ({type(exc).__name__})",
        )


def load_pinned_embedder() -> tuple[Embedder | None, str]:
    """Return the pinned offline embedder used for exact evaluation."""

    return _load_pinned_embedder()


def _rust_backend_supported() -> bool:
    try:
        from cite_right import _core
    except ImportError:
        return False
    return all(
        hasattr(_core, name) for name in ("align_pair_details", "align_batch_details")
    )


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _git_revision() -> str:
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
        raise RuntimeError("cannot resolve an exact Git commit for baseline provenance")
    return revision


def _worktree_dirty() -> bool:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot inspect worktree state for baseline provenance")
    return bool(result.stdout.strip())


def _code_snapshot_sha256() -> str:
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


def _assert_code_provenance_unchanged(
    *,
    expected_git_revision: str,
    expected_code_snapshot_sha256: str,
) -> None:
    if _git_revision() != expected_git_revision:
        raise RuntimeError("Git revision changed during baseline evaluation")
    if _code_snapshot_sha256() != expected_code_snapshot_sha256:
        raise RuntimeError("code snapshot changed during baseline evaluation")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline report must be a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[0] != "compare":
        print(
            "usage: python -m evaluation.baselines compare LEFT RIGHT", file=sys.stderr
        )
        return 2
    result = compare_baselines(_load(Path(args[1])), _load(Path(args[2])))
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
