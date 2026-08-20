from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.experiments import CandidateMetrics, ResourceMetrics
from evaluation.hill_climb import (
    SearchSpace,
    _forbidden_optimizer_path,
    evaluate_gate_decision,
    run_tuning,
    select_best_record,
)
from evaluation.performance import smoke_environment_compatibility_hash
from evaluation.runner import canonicalize_config


def test_select_best_record_uses_lexicographic_gates_and_tiebreakers() -> None:
    search_space = SearchSpace.model_validate(_synthetic_search_space())

    best = select_best_record(search_space.synthetic_records())

    assert best is not None
    assert best.candidate_id == "candidate-recall-win"


def test_search_space_rejects_duplicate_candidate_ids() -> None:
    payload = _synthetic_search_space()
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidates[1]["candidate_id"] = candidates[0]["candidate_id"]

    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        SearchSpace.model_validate(payload)


def test_forbidden_optimizer_path_checks_symlink_target(tmp_path: Path) -> None:
    forbidden = tmp_path / "release-gate"
    forbidden.mkdir()
    alias = tmp_path / "public"
    alias.symlink_to(forbidden, target_is_directory=True)

    assert _forbidden_optimizer_path(alias / "result.json") is True


def test_select_best_record_breaks_ties_by_status_then_retrieval_then_latency() -> None:
    status_space = SearchSpace.model_validate(_tiebreak_status_search_space())
    retrieval_space = SearchSpace.model_validate(_tiebreak_retrieval_search_space())
    latency_space = SearchSpace.model_validate(_tiebreak_latency_search_space())

    status_best = select_best_record(status_space.synthetic_records())
    retrieval_best = select_best_record(retrieval_space.synthetic_records())
    latency_best = select_best_record(latency_space.synthetic_records())

    assert status_best is not None
    assert retrieval_best is not None
    assert latency_best is not None
    assert status_best.candidate_id == "candidate-status-win"
    assert retrieval_best.candidate_id == "candidate-retrieval-win"
    assert latency_best.candidate_id == "candidate-latency-win"


def test_run_tuning_executes_real_fixture_search_and_resumes(tmp_path: Path) -> None:
    output_path = tmp_path / "resume.json"
    fixture_path = Path("tests/evaluation/fixtures/three-candidates.json")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_frozen_baseline_report()), encoding="utf-8")

    from evaluation import hill_climb as module

    original_loader = module._load_frozen_baseline_report
    module._load_frozen_baseline_report = lambda: json.loads(
        baseline_path.read_text(encoding="utf-8")
    )
    try:
        first = run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=fixture_path,
            output_path=output_path,
        )
        second = run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=fixture_path,
            output_path=output_path,
        )
    finally:
        module._load_frozen_baseline_report = original_loader

    assert first["best_candidate_id"] is None
    assert first["evaluated_candidate_count"] == 3
    assert second["evaluated_candidate_count"] == 0
    assert second["duplicate_candidate_ids"] == [
        "strict-baseline",
        "strict-guard-on",
        "strict-top-k-3",
    ]

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["best_candidate_id"] is None
    assert [record["candidate_id"] for record in payload["records"]] == [
        "strict-baseline",
        "strict-guard-on",
        "strict-top-k-3",
    ]
    assert (
        payload["records"][0]["config"]["sha256"]
        == canonicalize_config(_strict_control_payload()).sha256
    )


def test_run_tuning_rejects_code_drift_before_persisting_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import hill_climb as module

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_frozen_baseline_report()), encoding="utf-8")
    original_evaluate = module._evaluate_candidate
    state = {"snapshot": "c" * 64}

    monkeypatch.setattr(
        module,
        "_load_frozen_baseline_report",
        lambda: json.loads(baseline_path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        module, "current_code_snapshot_sha256", lambda: state["snapshot"]
    )

    def drifting_evaluate(*args: object, **kwargs: object):
        record = original_evaluate(*args, **kwargs)
        state["snapshot"] = "d" * 64
        return record

    monkeypatch.setattr(module, "_evaluate_candidate", drifting_evaluate)
    output_path = tmp_path / "experiments.json"

    with pytest.raises(ValueError, match="code snapshot changed"):
        run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=Path("tests/evaluation/fixtures/three-candidates.json"),
            output_path=output_path,
        )

    assert not output_path.exists()


def test_run_tuning_rejects_holdout_release_and_synthetic_inputs(
    tmp_path: Path,
) -> None:
    synthetic_path = tmp_path / "synthetic.json"
    synthetic_path.write_text(json.dumps(_synthetic_search_space()), encoding="utf-8")
    holdout_path = tmp_path / "holdout-tuning"
    holdout_path.mkdir()

    with pytest.raises(ValueError, match="synthetic_result"):
        run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=synthetic_path,
            output_path=tmp_path / "out.json",
        )

    inline_baseline = _synthetic_search_space()
    inline_baseline["candidates"] = [
        {
            "candidate_id": "strict-baseline",
            "code_path_id": "config-only",
            "backend": "python",
            "embeddings": "off",
            "config": {"top_k": 2},
        }
    ]
    baseline_override_path = tmp_path / "baseline-override.json"
    baseline_override_path.write_text(json.dumps(inline_baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen baseline"):
        run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=baseline_override_path,
            output_path=tmp_path / "out.json",
        )

    with pytest.raises(ValueError, match="holdout"):
        run_tuning(
            tuning_bundle=holdout_path,
            search_space_path=Path("tests/evaluation/fixtures/three-candidates.json"),
            output_path=tmp_path / "out.json",
        )

    for suffix in (
        "release_gate.json",
        "release gate.json",
        "release.gate.json",
        "releasegate.json",
        "output.aesgcm",
    ):
        with pytest.raises(ValueError, match="release-gate|holdout"):
            run_tuning(
                tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
                search_space_path=Path(
                    "tests/evaluation/fixtures/three-candidates.json"
                ),
                output_path=tmp_path / suffix,
            )


def test_run_tuning_rejects_resume_with_changed_search_space(tmp_path: Path) -> None:
    output_path = tmp_path / "resume.json"
    baseline_path = Path("tests/evaluation/fixtures/three-candidates.json")
    changed_path = tmp_path / "changed.json"
    frozen_baseline_path = tmp_path / "baseline.json"
    frozen_baseline_path.write_text(
        json.dumps(_frozen_baseline_report()), encoding="utf-8"
    )
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["candidates"][2]["config"]["top_k"] = 9
    changed_path.write_text(json.dumps(payload), encoding="utf-8")

    from evaluation import hill_climb as module

    original_loader = module._load_frozen_baseline_report
    module._load_frozen_baseline_report = lambda: json.loads(
        frozen_baseline_path.read_text(encoding="utf-8")
    )
    try:
        run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=baseline_path,
            output_path=output_path,
        )

        with pytest.raises(ValueError, match="search space"):
            run_tuning(
                tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
                search_space_path=changed_path,
                output_path=output_path,
            )
    finally:
        module._load_frozen_baseline_report = original_loader


def test_candidate_identity_suppresses_aliases_with_equivalent_resolved_configs() -> (
    None
):
    from evaluation import hill_climb as module

    full_config = _strict_control_payload()
    partial_config = {
        "top_k": 2,
        "min_final_score": 0.3,
        "min_answer_coverage": 0.4,
        "supported_answer_coverage": 0.7,
        "max_citations_per_source": 1,
        "max_retrieval_support": 2,
        "require_all_answer_tokens_in_evidence": False,
    }
    full = module.SearchCandidate(
        candidate_id="full",
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        config=full_config,
    )
    alias = module.SearchCandidate(
        candidate_id="alias",
        code_path_id="config-only",
        backend="python",
        embeddings="off",
        config=partial_config,
    )

    assert module._candidate_identity(full) == module._candidate_identity(alias)


def test_run_tuning_suppresses_canonical_aliases_across_resume(tmp_path: Path) -> None:
    from evaluation import hill_climb as module

    full_config = _strict_control_payload()
    partial_config = {
        "top_k": 2,
        "min_final_score": 0.3,
        "min_answer_coverage": 0.4,
        "supported_answer_coverage": 0.7,
        "max_citations_per_source": 1,
        "max_retrieval_support": 2,
        "require_all_answer_tokens_in_evidence": False,
    }
    search_path = tmp_path / "aliases.json"
    search_path.write_text(
        json.dumps(
            {
                "schema_version": "evaluation.search-space.v1",
                "parent_experiment_id": "alias-test",
                "candidates": [
                    {
                        "candidate_id": "alias-full",
                        "code_path_id": "config-only",
                        "backend": "python",
                        "embeddings": "off",
                        "config": full_config,
                    },
                    {
                        "candidate_id": "alias-partial",
                        "code_path_id": "config-only",
                        "backend": "python",
                        "embeddings": "off",
                        "config": partial_config,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "experiments.json"
    original_loader = module._load_frozen_baseline_report
    module._load_frozen_baseline_report = _frozen_baseline_report
    try:
        first = run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=search_path,
            output_path=output_path,
        )
        second = run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=search_path,
            output_path=output_path,
        )
    finally:
        module._load_frozen_baseline_report = original_loader

    assert first["evaluated_candidate_count"] == 1
    assert first["duplicate_candidate_ids"] == ["alias-partial"]
    assert second["evaluated_candidate_count"] == 0
    assert second["duplicate_candidate_ids"] == ["alias-full", "alias-partial"]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [record["candidate_id"] for record in payload["records"]] == ["alias-full"]


def test_checked_in_search_space_is_bounded_one_coordinate_neighborhood() -> None:
    search_space = json.loads(
        Path("evaluation/search_spaces/v1.json").read_text(encoding="utf-8")
    )
    candidates = search_space["candidates"]
    assert 7 <= len(candidates) <= 12

    resolved = {
        candidate["candidate_id"]: _flatten_json(
            canonicalize_config(candidate["config"]).payload
        )
        for candidate in candidates
    }
    control = resolved["strict-baseline"]
    baseline = json.loads(
        Path("evaluation/reports/v1/baseline.json").read_text(encoding="utf-8")
    )
    assert (
        canonicalize_config(candidates[0]["config"]).sha256
        == baseline["gates"]["performance_config_sha256"]
    )

    changed_paths: set[str] = set()
    identities: set[str] = set()
    for candidate in candidates:
        identity = canonicalize_config(candidate["config"]).sha256
        assert identity not in identities
        identities.add(identity)
        if candidate["candidate_id"] == "strict-baseline":
            continue
        candidate_values = resolved[candidate["candidate_id"]]
        differences = {
            key
            for key in control.keys() | candidate_values.keys()
            if control.get(key) != candidate_values.get(key)
        }
        assert len(differences) == 1, candidate["candidate_id"]
        changed_paths.update(differences)

    assert {
        "require_all_answer_tokens_in_evidence",
        "top_k",
        "min_final_score",
        "max_candidates_total",
        "weights.lexical",
        "window_size_sentences",
        "min_alignment_score",
    } <= changed_paths


def test_run_tuning_rejects_frozen_baseline_dataset_hash_mismatch(
    tmp_path: Path,
) -> None:
    fixture_path = Path("tests/evaluation/fixtures/three-candidates.json")
    frozen_baseline_path = tmp_path / "baseline.json"
    payload = _frozen_baseline_report()
    payload["dataset_hash"] = "f" * 64
    frozen_baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    from evaluation import hill_climb as module

    original_loader = module._load_frozen_baseline_report
    module._load_frozen_baseline_report = lambda: json.loads(
        frozen_baseline_path.read_text(encoding="utf-8")
    )
    try:
        with pytest.raises(ValueError, match="dataset hash"):
            run_tuning(
                tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
                search_space_path=fixture_path,
                output_path=tmp_path / "out.json",
            )
    finally:
        module._load_frozen_baseline_report = original_loader


def test_run_tuning_persists_completed_work_before_interruption(tmp_path: Path) -> None:
    output_path = tmp_path / "resume.json"
    fixture_path = Path("tests/evaluation/fixtures/three-candidates.json")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_frozen_baseline_report()), encoding="utf-8")

    from evaluation import hill_climb as module

    original_loader = module._load_frozen_baseline_report
    original_evaluate = module._evaluate_candidate
    call_count = {"count": 0}

    def flaky_evaluate(*args: object, **kwargs: object):
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise RuntimeError("boom")
        return original_evaluate(*args, **kwargs)

    module._load_frozen_baseline_report = lambda: json.loads(
        baseline_path.read_text(encoding="utf-8")
    )
    module._evaluate_candidate = flaky_evaluate  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="boom"):
            run_tuning(
                tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
                search_space_path=fixture_path,
                output_path=output_path,
            )
    finally:
        module._load_frozen_baseline_report = original_loader
        module._evaluate_candidate = original_evaluate  # type: ignore[assignment]

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [record["candidate_id"] for record in payload["records"]] == [
        "strict-baseline"
    ]


def test_resource_gates_fail_closed_when_frozen_hashes_are_missing() -> None:
    decision = evaluate_gate_decision(
        train_metrics=_metrics_model("a" * 64, 0.8),
        dev_metrics=_metrics_model("b" * 64, 0.8),
        resource_metrics=ResourceMetrics(
            median_duration_ns=100,
            p95_duration_ns=110,
            peak_memory_bytes=1000,
            protocol_hash="a" * 64,
            workload_hash="b" * 64,
        ),
        gates={
            "offset_invalid_tolerance": 0,
            "contradiction_false_citation_tolerance": 0,
            "exact_precision_wilson_lower_min": 0.0,
            "p95_latency_budget_ns": 200,
            "peak_memory_budget_bytes": 1200,
        },
        evaluated_dev=True,
    )
    assert decision.passes_resource_gates is False
    assert "resources" in decision.violated_gates


def _synthetic_search_space() -> dict[str, object]:
    return {
        "schema_version": "evaluation.search-space.v1",
        "parent_experiment_id": "task16-synthetic",
        "baseline": {
            "hash": "b" * 64,
            "gates": {
                "offset_invalid_tolerance": 0,
                "contradiction_false_citation_tolerance": 0,
                "exact_precision_wilson_lower_min": 0.75,
                "p95_latency_budget_ns": 150,
                "peak_memory_budget_bytes": 1200,
                "performance_protocol_hash": "1" * 64,
                "selected_workload_hash": "2" * 64,
                "performance_environment_hash": smoke_environment_compatibility_hash(),
            },
        },
        "candidates": [
            {
                "candidate_id": "candidate-offset-loss",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 2, "min_final_score": 0.25},
                "synthetic_result": {
                    "train_metrics": _metrics("c" * 64, 0.9, 1, 0.9, 0.8),
                    "dev_metrics": _metrics("a" * 64, 0.95, 1, 0.9, 0.9),
                    "resource_metrics": {
                        "median_duration_ns": 90,
                        "p95_duration_ns": 100,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
            {
                "candidate_id": "candidate-contradiction-loss",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 2, "min_final_score": 0.26},
                "synthetic_result": {
                    "train_metrics": _metrics(
                        "b" * 64,
                        0.9,
                        0,
                        0.9,
                        0.8,
                        contradiction_false_citation_count=1,
                    ),
                    "dev_metrics": _metrics(
                        "1" * 64,
                        0.95,
                        0,
                        0.9,
                        0.9,
                        contradiction_false_citation_count=1,
                    ),
                    "resource_metrics": {
                        "median_duration_ns": 85,
                        "p95_duration_ns": 95,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
            {
                "candidate_id": "candidate-resource-loss",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 2, "min_final_score": 0.27},
                "synthetic_result": {
                    "train_metrics": _metrics("2" * 64, 0.92, 0, 0.9, 0.8),
                    "dev_metrics": _metrics("3" * 64, 0.97, 0, 0.9, 0.9),
                    "resource_metrics": {
                        "median_duration_ns": 90,
                        "p95_duration_ns": 151,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
            {
                "candidate_id": "candidate-recall-lose",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 2, "min_final_score": 0.3},
                "synthetic_result": {
                    "train_metrics": _metrics("e" * 64, 0.75, 0, 0.75, 0.85),
                    "dev_metrics": _metrics("d" * 64, 0.8, 0, 0.8, 0.9),
                    "resource_metrics": {
                        "median_duration_ns": 80,
                        "p95_duration_ns": 120,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
            {
                "candidate_id": "candidate-recall-win",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 3, "min_final_score": 0.28},
                "synthetic_result": {
                    "train_metrics": _metrics("9" * 64, 0.8, 0, 0.7, 0.7),
                    "dev_metrics": _metrics("f" * 64, 0.85, 0, 0.7, 0.7),
                    "resource_metrics": {
                        "median_duration_ns": 95,
                        "p95_duration_ns": 130,
                        "peak_memory_bytes": 1100,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
        ],
    }


def _metrics(
    output_sha256: str,
    requirement_recall: float,
    offset_invalid_count: int,
    status_macro_f1: float,
    retrieval_mrr: float,
    *,
    contradiction_false_citation_count: int = 0,
    run_error_count: int = 0,
) -> dict[str, object]:
    return {
        "exact_precision_lower": 0.8,
        "contradiction_false_citation_count": contradiction_false_citation_count,
        "offset_invalid_count": offset_invalid_count,
        "requirement_recall": requirement_recall,
        "status_macro_f1": status_macro_f1,
        "retrieval_mrr": retrieval_mrr,
        "run_error_count": run_error_count,
        "output_sha256": output_sha256,
    }


def _flatten_json(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            flattened: dict[str, object] = {}
            for key, item in value:
                path = f"{prefix}.{key}" if prefix else key
                flattened.update(_flatten_json(item, path))
            return flattened
        return {prefix: value}
    return {prefix: value}


def _metrics_model(output_sha256: str, requirement_recall: float) -> CandidateMetrics:
    return CandidateMetrics.model_validate(
        _metrics(
            output_sha256,
            requirement_recall,
            0,
            0.7,
            0.7,
        )
    )


def _frozen_baseline_report() -> dict[str, object]:
    return {
        "dataset_hash": "0c99b9086371850e91dc5972b9f101e4a566ed14b11281e4f3741c6b0743defd",
        "gates": {
            "offset_invalid_tolerance": 0,
            "contradiction_false_citation_tolerance": 0,
            "exact_precision_wilson_lower_min": 0.0,
            "p95_latency_budget_ns": 10_000_000,
            "peak_memory_budget_bytes": 100_000_000,
            "performance_protocol_hash": "a" * 64,
            "selected_workload_hash": "b" * 64,
            "performance_environment_hash": smoke_environment_compatibility_hash(),
            "performance_config_sha256": canonicalize_config(
                _strict_control_payload()
            ).sha256,
        },
    }


def _strict_control_payload() -> dict[str, object]:
    from cite_right import CitationConfig

    return (
        CitationConfig.strict()
        .model_copy(update={"require_all_answer_tokens_in_evidence": False})
        .model_dump(mode="json")
    )


def _tiebreak_status_search_space() -> dict[str, object]:
    return {
        "schema_version": "evaluation.search-space.v1",
        "baseline": _synthetic_search_space()["baseline"],
        "candidates": [
            {
                "candidate_id": "candidate-status-lose",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 2},
                "synthetic_result": {
                    "train_metrics": _metrics("4" * 64, 0.8, 0, 0.8, 0.6),
                    "dev_metrics": _metrics("5" * 64, 0.8, 0, 0.8, 0.7),
                    "resource_metrics": {
                        "median_duration_ns": 100,
                        "p95_duration_ns": 120,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
            {
                "candidate_id": "candidate-status-win",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 3},
                "synthetic_result": {
                    "train_metrics": _metrics("6" * 64, 0.8, 0, 0.9, 0.6),
                    "dev_metrics": _metrics("7" * 64, 0.8, 0, 0.9, 0.7),
                    "resource_metrics": {
                        "median_duration_ns": 105,
                        "p95_duration_ns": 125,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
        ],
    }


def _tiebreak_retrieval_search_space() -> dict[str, object]:
    return {
        "schema_version": "evaluation.search-space.v1",
        "baseline": _synthetic_search_space()["baseline"],
        "candidates": [
            {
                "candidate_id": "candidate-retrieval-lose",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 2},
                "synthetic_result": {
                    "train_metrics": _metrics("8" * 64, 0.8, 0, 0.9, 0.7),
                    "dev_metrics": _metrics("9" * 64, 0.8, 0, 0.9, 0.7),
                    "resource_metrics": {
                        "median_duration_ns": 100,
                        "p95_duration_ns": 120,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
            {
                "candidate_id": "candidate-retrieval-win",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 3},
                "synthetic_result": {
                    "train_metrics": _metrics("a" * 64, 0.8, 0, 0.9, 0.8),
                    "dev_metrics": _metrics("b" * 64, 0.8, 0, 0.9, 0.8),
                    "resource_metrics": {
                        "median_duration_ns": 105,
                        "p95_duration_ns": 125,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
        ],
    }


def _tiebreak_latency_search_space() -> dict[str, object]:
    return {
        "schema_version": "evaluation.search-space.v1",
        "baseline": _synthetic_search_space()["baseline"],
        "candidates": [
            {
                "candidate_id": "candidate-latency-lose",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 2},
                "synthetic_result": {
                    "train_metrics": _metrics("c" * 64, 0.8, 0, 0.9, 0.8),
                    "dev_metrics": _metrics("d" * 64, 0.8, 0, 0.9, 0.8),
                    "resource_metrics": {
                        "median_duration_ns": 110,
                        "p95_duration_ns": 130,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
            {
                "candidate_id": "candidate-latency-win",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"top_k": 3},
                "synthetic_result": {
                    "train_metrics": _metrics("e" * 64, 0.8, 0, 0.9, 0.8),
                    "dev_metrics": _metrics("f" * 64, 0.8, 0, 0.9, 0.8),
                    "resource_metrics": {
                        "median_duration_ns": 90,
                        "p95_duration_ns": 120,
                        "peak_memory_bytes": 1000,
                        "protocol_hash": "1" * 64,
                        "workload_hash": "2" * 64,
                        "environment_hash": smoke_environment_compatibility_hash(),
                    },
                },
            },
        ],
    }
