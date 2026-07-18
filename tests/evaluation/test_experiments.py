from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.experiments import (
    CandidateMetrics,
    ExperimentStore,
    GateDecision,
    ResourceMetrics,
    build_experiment_record,
    contains_forbidden_holdout_data,
    load_experiment_store,
    persist_experiment_store,
    resolve_output_path,
)


def test_experiment_store_persists_required_fields_and_deterministic_order(
    tmp_path: Path,
) -> None:
    first = build_experiment_record(
        candidate_id="candidate-a",
        parent_candidate_id=None,
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        config={"top_k": 2, "min_final_score": 0.3},
        train_metrics=_metrics("c" * 64, requirement_recall=0.4),
        dev_metrics=_metrics("d" * 64, requirement_recall=0.5),
        resource_metrics=ResourceMetrics(
            median_duration_ns=100,
            p95_duration_ns=120,
            peak_memory_bytes=1024,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )
    second = build_experiment_record(
        candidate_id="candidate-b",
        parent_candidate_id="candidate-a",
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_path_id="prepared-corpus-v1",
        backend="python",
        embeddings="off",
        config={"min_final_score": 0.35, "top_k": 3},
        train_metrics=_metrics("e" * 64, requirement_recall=0.45),
        dev_metrics=_metrics("f" * 64, requirement_recall=0.55),
        resource_metrics=ResourceMetrics(
            median_duration_ns=95,
            p95_duration_ns=118,
            peak_memory_bytes=1000,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )

    store = ExperimentStore(
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        search_space_hash="c" * 64,
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
    assert tuple(record.candidate_id for record in loaded.records) == (
        "candidate-a",
        "candidate-b",
    )


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
            code_path_id="holdout-probe",
            backend="python",
            embeddings="off",
            config={"top_k": 2},
            train_metrics=_metrics("c" * 64, requirement_recall=0.4),
            dev_metrics=_metrics("d" * 64, requirement_recall=0.5),
            resource_metrics=ResourceMetrics(
                median_duration_ns=100,
                p95_duration_ns=120,
                peak_memory_bytes=1024,
            ),
            gate_decision=_gate_decision(gate_pass=True),
        )


def test_experiment_store_requires_context_consistency() -> None:
    record = build_experiment_record(
        candidate_id="candidate-a",
        parent_candidate_id=None,
        dataset_hash="a" * 64,
        baseline_hash="b" * 64,
        git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        config={"top_k": 2},
        train_metrics=_metrics("c" * 64, requirement_recall=0.4),
        dev_metrics=_metrics("d" * 64, requirement_recall=0.5),
        resource_metrics=ResourceMetrics(
            median_duration_ns=100,
            p95_duration_ns=120,
            peak_memory_bytes=1024,
        ),
        gate_decision=_gate_decision(gate_pass=True),
    )

    with pytest.raises(ValueError, match="dataset_hash"):
        ExperimentStore(
            dataset_hash="f" * 64,
            baseline_hash="b" * 64,
            git_revision="94e7aff4c098390321d5e50f814d00acc99f3428",
            search_space_hash="c" * 64,
            records=[record],
        )


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


def _gate_decision(*, gate_pass: bool) -> GateDecision:
    violated = [] if gate_pass else ["precision"]
    return GateDecision(
        evaluated_dev=True,
        passes_offset_gate=gate_pass,
        passes_precision_gate=gate_pass,
        passes_contradiction_gate=gate_pass,
        passes_resource_gates=gate_pass,
        gate_pass=gate_pass,
        violated_gates=violated,
    )
