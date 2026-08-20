from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from cite_right import CitationConfig
from evaluation.experiments import (
    CandidateMetrics,
    ExperimentEnvironment,
    ExperimentStore,
    GateDecision,
    ResourceMetrics,
    build_experiment_record,
    contains_forbidden_holdout_data,
    git_revision,
    load_experiment_store,
    persist_experiment_store,
    resolve_output_path,
)
from evaluation.performance import SMOKE_TRIAL_COUNT, selected_smoke_scenario_ids
from evaluation.runner import canonicalize_config


def test_experiment_store_persists_required_fields_and_deterministic_order(
    tmp_path: Path,
) -> None:
    environment = _environment()
    first = build_experiment_record(
        candidate_id="candidate-a",
        parent_candidate_id=None,
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_snapshot_sha256="c" * 64,
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        environment=environment,
        config={"top_k": 2, "min_final_score": 0.3},
        train_metrics=_metrics("d" * 64, requirement_recall=0.4),
        dev_metrics=_metrics("e" * 64, requirement_recall=0.5),
        resource_metrics=ResourceMetrics(
            median_duration_ns=100,
            p95_duration_ns=120,
            peak_memory_bytes=1024,
            config_sha256=_resolved_config_hash(
                {"top_k": 2, "min_final_score": 0.3}
            ),
            raw_end_to_end_samples_ns=_raw_samples(median=100, p95=120),
            protocol_hash="1" * 64,
            workload_hash="2" * 64,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )
    second = build_experiment_record(
        candidate_id="candidate-b",
        parent_candidate_id="candidate-a",
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_snapshot_sha256="c" * 64,
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        environment=environment,
        config={"min_final_score": 0.35, "top_k": 3},
        train_metrics=_metrics("f" * 64, requirement_recall=0.45),
        dev_metrics=_metrics("0" * 64, requirement_recall=0.55),
        resource_metrics=ResourceMetrics(
            median_duration_ns=95,
            p95_duration_ns=118,
            peak_memory_bytes=1000,
            config_sha256=_resolved_config_hash(
                {"min_final_score": 0.35, "top_k": 3}
            ),
            raw_end_to_end_samples_ns=_raw_samples(median=95, p95=118),
            protocol_hash="1" * 64,
            workload_hash="2" * 64,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )

    store = ExperimentStore(
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_snapshot_sha256="c" * 64,
        search_space_hash="3" * 64,
        environment=environment,
        parent_experiment_id="parent-run",
        best_candidate_id="candidate-b",
        records=[first, second],
    )
    target = persist_experiment_store(output_path=tmp_path / "runs", store=store)

    assert target == tmp_path / "runs" / "experiments.json"
    assert resolve_output_path(tmp_path / "runs") == target
    assert json.loads(target.read_text(encoding="utf-8")) == store.model_dump(mode="json")
    loaded = load_experiment_store(tmp_path / "runs")
    assert loaded == store
    assert isinstance(loaded.records[0].config.payload, tuple)
    assert len(loaded.records[0].config.payload) > 2
    assert ("top_k", 2) in loaded.records[0].config.payload


def test_experiment_records_reject_holdout_fields_and_paths() -> None:
    assert contains_forbidden_holdout_data({"split": "holdout"}) is True
    assert contains_forbidden_holdout_data({"path": Path("release/holdout.aesgcm")}) is True

    with pytest.raises(ValueError, match="holdout"):
        build_experiment_record(
            candidate_id="candidate-holdout",
            parent_candidate_id=None,
            dataset_hash="a" * 64,
            baseline_hash="b" * 64,
            git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
            code_snapshot_sha256="c" * 64,
            code_path_id="holdout-probe",
            backend="python",
            embeddings="off",
            environment=_environment(),
            config={"probe_path": "release/holdout.aesgcm"},
            train_metrics=_metrics("d" * 64, requirement_recall=0.4),
            dev_metrics=_metrics("e" * 64, requirement_recall=0.5),
            resource_metrics=ResourceMetrics(
                median_duration_ns=100,
                p95_duration_ns=120,
                peak_memory_bytes=1024,
                config_sha256=_resolved_config_hash(
                    {"probe_path": "release/holdout.aesgcm"}
                ),
                raw_end_to_end_samples_ns=_raw_samples(median=100, p95=120),
                protocol_hash="1" * 64,
                workload_hash="2" * 64,
            ),
            gate_decision=_gate_decision(gate_pass=True),
        )


def test_gate_decision_requires_internal_boolean_consistency() -> None:
    with pytest.raises(ValueError, match="violated_gates"):
        GateDecision(
            evaluated_dev=True,
            passes_execution_gate=True,
            passes_offset_gate=False,
            passes_precision_gate=True,
            passes_contradiction_gate=True,
            passes_resource_gates=True,
            gate_pass=False,
            violated_gates=[],
        )


def test_real_resource_metrics_require_measured_config_hash() -> None:
    with pytest.raises(ValueError, match="require config_sha256"):
        build_experiment_record(
            candidate_id="candidate-unbound-resource-sample",
            parent_candidate_id=None,
            dataset_hash="a" * 64,
            baseline_hash="b" * 64,
            git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
            code_snapshot_sha256="c" * 64,
            code_path_id="config-only",
            backend="python",
            embeddings="off",
            environment=_environment(),
            config={"top_k": 2},
            train_metrics=_metrics("d" * 64, requirement_recall=0.4),
            dev_metrics=_metrics("e" * 64, requirement_recall=0.5),
            resource_metrics=ResourceMetrics(
                median_duration_ns=100,
                p95_duration_ns=120,
                peak_memory_bytes=1024,
                protocol_hash="1" * 64,
                workload_hash="2" * 64,
            ),
            gate_decision=_gate_decision(gate_pass=True),
        )


def test_real_resource_metrics_require_raw_samples() -> None:
    config = {"top_k": 2}
    with pytest.raises(ValueError, match="require raw samples"):
        build_experiment_record(
            candidate_id="candidate-aggregate-only-resource-sample",
            parent_candidate_id=None,
            dataset_hash="a" * 64,
            baseline_hash="b" * 64,
            git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
            code_snapshot_sha256="c" * 64,
            code_path_id="config-only",
            backend="python",
            embeddings="off",
            environment=_environment(),
            config=config,
            train_metrics=_metrics("d" * 64, requirement_recall=0.4),
            dev_metrics=_metrics("e" * 64, requirement_recall=0.5),
            resource_metrics=ResourceMetrics(
                median_duration_ns=100,
                p95_duration_ns=120,
                peak_memory_bytes=1024,
                config_sha256=_resolved_config_hash(config),
                protocol_hash="1" * 64,
                workload_hash="2" * 64,
            ),
            gate_decision=_gate_decision(gate_pass=True),
        )


def test_resource_summaries_must_be_reconstructable_from_raw_samples() -> None:
    with pytest.raises(ValueError, match="derived from raw samples"):
        ResourceMetrics(
            median_duration_ns=101,
            p95_duration_ns=120,
            peak_memory_bytes=1024,
            raw_end_to_end_samples_ns=_raw_samples(median=100, p95=120),
        )


def test_resource_median_preserves_even_sample_midpoint() -> None:
    metrics = ResourceMetrics(
        median_duration_ns=0.5,
        p95_duration_ns=1,
        peak_memory_bytes=1024,
        raw_end_to_end_samples_ns={"scenario": [0, 1]},
    )

    assert metrics.median_duration_ns == 0.5


@pytest.mark.parametrize("mutation", ["omit-scenario", "drop-sample"])
def test_real_resource_metrics_require_complete_protocol_samples(
    mutation: str,
) -> None:
    raw_samples = _raw_samples(median=100, p95=120)
    if mutation == "omit-scenario":
        raw_samples.pop(next(iter(raw_samples)))
        expected = "exact workload scenarios"
    else:
        raw_samples[next(iter(raw_samples))].pop()
        expected = "protocol trial count"

    with pytest.raises(ValueError, match=expected):
        _build_real_record_with_raw_samples(raw_samples)


def test_real_embeddings_on_resource_metrics_use_canonical_scenarios() -> None:
    raw_samples = _raw_samples(median=100, p95=120, embeddings="on")

    record = _build_real_record_with_raw_samples(raw_samples, embeddings="on")

    assert record is not None


def test_gate_pass_requires_dev_metrics_and_resource_metrics() -> None:
    with pytest.raises(ValueError, match="dev-evaluated"):
        build_experiment_record(
            candidate_id="candidate-a",
            parent_candidate_id=None,
            dataset_hash="a" * 64,
            baseline_hash="b" * 64,
            git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
            code_snapshot_sha256="c" * 64,
            code_path_id="config-only",
            backend="python",
            embeddings="off",
            environment=_environment(),
            config={"top_k": 2},
            train_metrics=_metrics("d" * 64, requirement_recall=0.4),
            dev_metrics=None,
            resource_metrics=None,
            gate_decision=GateDecision(
                evaluated_dev=False,
                passes_execution_gate=True,
                passes_offset_gate=True,
                passes_precision_gate=True,
                passes_contradiction_gate=True,
                passes_resource_gates=True,
                gate_pass=True,
                violated_gates=[],
            ),
        )


def test_git_revision_is_resolved_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    revision = git_revision()

    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)


def test_experiment_store_rejects_missing_or_wrong_best_candidate_id() -> None:
    environment = _environment()
    first = build_experiment_record(
        candidate_id="candidate-a",
        parent_candidate_id=None,
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_snapshot_sha256="c" * 64,
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        environment=environment,
        config={"top_k": 2},
        train_metrics=_metrics("d" * 64, requirement_recall=0.4),
        dev_metrics=_metrics("e" * 64, requirement_recall=0.5),
        resource_metrics=ResourceMetrics(
            median_duration_ns=100,
            p95_duration_ns=120,
            peak_memory_bytes=1024,
            config_sha256=_resolved_config_hash({"top_k": 2}),
            raw_end_to_end_samples_ns=_raw_samples(median=100, p95=120),
            protocol_hash="1" * 64,
            workload_hash="2" * 64,
            environment_hash="3" * 64,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )
    second = build_experiment_record(
        candidate_id="candidate-b",
        parent_candidate_id=None,
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_snapshot_sha256="c" * 64,
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        environment=environment,
        config={"top_k": 3},
        train_metrics=_metrics("f" * 64, requirement_recall=0.6),
        dev_metrics=_metrics("0" * 64, requirement_recall=0.7),
        resource_metrics=ResourceMetrics(
            median_duration_ns=90,
            p95_duration_ns=110,
            peak_memory_bytes=1024,
            config_sha256=_resolved_config_hash({"top_k": 3}),
            raw_end_to_end_samples_ns=_raw_samples(median=90, p95=110),
            protocol_hash="1" * 64,
            workload_hash="2" * 64,
            environment_hash="3" * 64,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )

    common = {
        "dataset_hash": "a" * 64,
        "baseline_hash": "b" * 64,
        "git_revision": "94e7aff4c098390321d5e50f814d00acc99f3428",
        "code_snapshot_sha256": "c" * 64,
        "search_space_hash": "4" * 64,
        "environment": environment,
        "records": [first, second],
    }

    with pytest.raises(ValueError, match="deterministic winner"):
        ExperimentStore(best_candidate_id=None, **common)
    with pytest.raises(ValueError, match="deterministic winner"):
        ExperimentStore(best_candidate_id="candidate-a", **common)


def _metrics(output_sha256: str, *, requirement_recall: float) -> CandidateMetrics:
    return CandidateMetrics(
        exact_precision_lower=0.8,
        contradiction_false_citation_count=0,
        offset_invalid_count=0,
        requirement_recall=requirement_recall,
        status_macro_f1=0.6,
        retrieval_mrr=0.7,
        run_error_count=0,
        output_sha256=output_sha256,
    )


def _resolved_config_hash(config: Mapping[str, object]) -> str:
    resolved = CitationConfig.model_validate(config).model_dump(mode="json")
    return canonicalize_config(resolved).sha256


def _raw_samples(
    *, median: int, p95: int, embeddings: str = "off"
) -> dict[str, list[int]]:
    scenario_ids = selected_smoke_scenario_ids(
        backend="python",
        embeddings=embeddings,  # type: ignore[arg-type]
    )
    return {
        scenario_id: [median] * (SMOKE_TRIAL_COUNT - 3) + [p95] * 3
        for scenario_id in scenario_ids
    }


def _build_real_record_with_raw_samples(
    raw_samples: dict[str, list[int]],
    *,
    embeddings: str = "off",
) -> object:
    config = {"top_k": 2}
    return build_experiment_record(
        candidate_id="candidate-resource-protocol",
        parent_candidate_id=None,
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_snapshot_sha256="c" * 64,
        code_path_id="config-only",
        backend="python",
        embeddings=embeddings,
        environment=_environment(),
        config=config,
        train_metrics=_metrics("d" * 64, requirement_recall=0.4),
        dev_metrics=_metrics("e" * 64, requirement_recall=0.5),
        resource_metrics=ResourceMetrics(
            median_duration_ns=100,
            p95_duration_ns=120,
            peak_memory_bytes=1024,
            config_sha256=_resolved_config_hash(config),
            raw_end_to_end_samples_ns=raw_samples,
            protocol_hash="1" * 64,
            workload_hash="2" * 64,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )


def _gate_decision(*, gate_pass: bool) -> GateDecision:
    violated = [] if gate_pass else ["precision"]
    return GateDecision(
        evaluated_dev=True,
        passes_execution_gate=gate_pass,
        passes_offset_gate=gate_pass,
        passes_precision_gate=gate_pass,
        passes_contradiction_gate=gate_pass,
        passes_resource_gates=gate_pass,
        gate_pass=gate_pass,
        violated_gates=violated,
    )


def _environment() -> ExperimentEnvironment:
    return ExperimentEnvironment(
        python="3.14.2",
        platform="macOS-26.5.2-arm64",
        machine="arm64",
        cpu_count=14,
    )
