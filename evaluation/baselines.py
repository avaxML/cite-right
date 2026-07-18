"""Honest train/dev accuracy and resource baselines for strict attribution."""

from __future__ import annotations

import json
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
from evaluation.performance import run_performance_smoke
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
    return (
        _configuration("default", CitationConfig()),
        _configuration("strict", CitationConfig.strict(), selected_for_gates=True),
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
    with tempfile.TemporaryDirectory(prefix="cite-right-baseline-") as temporary:
        temp_root = Path(temporary)
        performance_trials = []
        for index in (1, 2):
            artifact_path = temp_root / f"performance-{index}.json"
            run_performance_smoke(output_path=artifact_path)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if not isinstance(artifact, dict):
                raise RuntimeError("performance smoke artifact must be a JSON object")
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
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": bundle.manifest.dataset_version,
        "dataset_hash": sha256_hex(canonical_json_bytes(bundle.manifest)),
        "git_revision": _git_revision(),
        "code_snapshot_sha256": _code_snapshot_sha256(),
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
            selected=selected, performance_trials=performance_trials
        ),
    }
    if report_contains_holdout_data(report):
        raise RuntimeError("baseline report contains forbidden holdout data")
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
    return {
        "correctness_equal": True,
        "performance_median_ratios": ratios,
        "performance_noise_margin": margin,
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
        "performance_noise_margin": _scenario_metric_margin(
            _scenario_index(performance_trials),
            lambda scenario: _duration_metric(scenario, "median_duration_ns"),
        ),
    }


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
    def values(report: Mapping[str, object]) -> dict[str, float]:
        trials = report.get("performance_trials", [])
        collected: dict[str, list[int]] = {}
        if isinstance(trials, list):
            for trial in trials:
                if not isinstance(trial, Mapping):
                    continue
                for scenario in trial.get("scenarios", []):
                    if not isinstance(scenario, Mapping):
                        continue
                    duration = scenario.get("end_to_end")
                    if isinstance(duration, Mapping) and isinstance(
                        duration.get("median_duration_ns"), (int, float)
                    ):
                        collected.setdefault(
                            str(scenario.get("scenario_id")), []
                        ).append(int(duration["median_duration_ns"]))
        return {key: sum(items) / len(items) for key, items in collected.items()}

    left_values = values(left)
    right_values = values(right)
    if left_values.keys() != right_values.keys():
        raise ValueError("baseline performance scenarios differ")
    return {
        key: right_values[key] / left_values[key]
        for key in left_values
        if left_values[key] > 0
    }


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
    margins: list[float] = []
    for report in (left, right):
        gates = report.get("gates")
        if isinstance(gates, Mapping):
            value = gates.get("performance_noise_margin")
            if isinstance(value, (int, float)):
                margins.append(float(value))
    if not margins:
        return 0.25
    if len(margins) != 2 or any(not 0.0 <= margin < 1.0 for margin in margins):
        raise ValueError("baseline performance noise margins are invalid")
    return max(margins)


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
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _worktree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        check=False,
        text=True,
    )
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
