from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from cite_right import CitationConfig
from evaluation.baselines import (
    accuracy_report,
    baseline_configurations,
    build_baseline,
    compare_baselines,
    report_contains_holdout_data,
)
from evaluation.canonical import canonical_json_bytes
from evaluation.runner import CaseRun, canonicalize_config
from evaluation.schema import EvaluationCase
from evaluation.tuning_bundle import TuningBundle, TuningBundleManifest


def test_baseline_configurations_are_complete_and_policy_ordered() -> None:
    configurations = baseline_configurations()

    assert tuple(item.name for item in configurations) == (
        "default",
        "strict",
        "permissive",
    )
    assert configurations[1].selected_for_gates is True
    assert sum(item.selected_for_gates for item in configurations) == 1
    for item in configurations:
        assert item.config == CitationConfig.model_validate(item.config).model_dump(
            mode="json"
        )


def test_accuracy_report_scores_exact_evidence_and_raw_counts() -> None:
    case = _case()
    run = CaseRun.model_validate(
        {
            "case_id": case.case_id,
            "backend": "python",
            "config": canonicalize_config(CitationConfig()),
            "outputs": (
                {
                    "answer_span": {
                        "text": case.answer,
                        "char_start": 0,
                        "char_end": len(case.answer),
                    },
                    "citations": [
                        {
                            "score": 1.0,
                            "source_id": "source-1",
                            "source_index": 0,
                            "candidate_index": 0,
                            "char_start": 0,
                            "char_end": len(case.answer),
                            "evidence": case.answer,
                            "evidence_spans": [
                                {
                                    "char_start": 0,
                                    "char_end": len(case.answer),
                                    "evidence": case.answer,
                                }
                            ],
                        }
                    ],
                    "status": "supported",
                },
            ),
            "output_unit_ids": (("unit-1",),),
            "duration_ns": 1,
        }
    )

    report = accuracy_report(cases=(case,), runs=(run,))

    assert report.metrics.exact_precision.numerator == 1
    assert report.metrics.exact_precision.denominator == 1
    assert report.metrics.exact_recall.numerator == 1
    assert report.metrics.offset_validity.numerator == 1
    assert report.output_sha256


def test_reports_reject_holdout_names_and_comparison_checks_correctness() -> None:
    assert report_contains_holdout_data({"split": "holdout"}) is True
    assert report_contains_holdout_data({"path": Path("bundle/holdout.aesgcm")}) is True
    assert report_contains_holdout_data({"train": 1, "dev": 2}) is False

    left = {
        "schema_version": "evaluation.baseline.v1",
        "dataset_hash": "a" * 64,
        "code_snapshot_sha256": "b" * 64,
        "selected_baseline_id": "strict/python/off",
        "matrix": [{"id": "strict/python/off", "train": {"output_sha256": "a"}}],
        "performance_trials": [
            {
                "scenarios": [
                    {
                        "scenario_id": "s1",
                        "end_to_end": {
                            "median_duration_ns": 100,
                            "p95_duration_ns": 120,
                        },
                        "peak_memory_bytes": 1000,
                    }
                ]
            }
        ],
        "gates": {
            "performance_noise_margin": 0.25,
            "p95_noise_margin": 0.15,
            "peak_memory_noise_margin": 0.10,
        },
    }
    assert compare_baselines(left, left)["correctness_equal"] is True
    right = {
        **left,
        "matrix": [{"id": "strict/python/off", "train": {"output_sha256": "b"}}],
    }
    with pytest.raises(ValueError, match="correctness"):
        compare_baselines(left, right)
    with pytest.raises(ValueError, match="metadata"):
        compare_baselines(left, {**left, "dataset_hash": "c" * 64})
    assert compare_baselines(
        left,
        {
            **left,
            "gates": {
                "performance_noise_margin": 0.35,
                "p95_noise_margin": 0.20,
                "peak_memory_noise_margin": 0.12,
            },
        },
    )["performance_noise_margin"] == pytest.approx(0.35)
    with pytest.raises(ValueError, match="p95 latency"):
        compare_baselines(
            left,
            {
                **left,
                "performance_trials": [
                    {
                        "scenarios": [
                            {
                                "scenario_id": "s1",
                                "end_to_end": {
                                    "median_duration_ns": 100,
                                    "p95_duration_ns": 150,
                                },
                                "peak_memory_bytes": 1000,
                            }
                        ]
                    }
                ],
            },
        )
    with pytest.raises(ValueError, match="peak memory"):
        compare_baselines(
            left,
            {
                **left,
                "performance_trials": [
                    {
                        "scenarios": [
                            {
                                "scenario_id": "s1",
                                "end_to_end": {
                                    "median_duration_ns": 100,
                                    "p95_duration_ns": 120,
                                },
                                "peak_memory_bytes": 1110,
                            }
                        ]
                    }
                ],
            },
        )


def test_build_baseline_freezes_selected_resource_gates_from_selected_policy_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    bundle = TuningBundle(
        root_dir=tmp_path / "bundle",
        manifest=TuningBundleManifest.model_validate(
            {
                "bundle_version": "1.0.0",
                "dataset_version": "1.0.0",
                "schema_version": "1.0.0",
                "train_case_count": 1,
                "dev_case_count": 1,
                "train_claim_count": 1,
                "dev_claim_count": 1,
                "train_source_count": 1,
                "dev_source_count": 1,
                "train_sha256": "1" * 64,
                "dev_sha256": "2" * 64,
                "dataset_manifest_sha256": "3" * 64,
                "provenance_sha256": "4" * 64,
                "source_catalog_sha256": "5" * 64,
                "dev_review_ledger_sha256": "6" * 64,
            }
        ),
        train_cases=(case,),
        dev_cases=(case,),
    )

    monkeypatch.setattr("evaluation.baselines.load_tuning_bundle", lambda _: bundle)
    monkeypatch.setattr("evaluation.baselines._rust_backend_supported", lambda: True)
    monkeypatch.setattr("evaluation.baselines._git_revision", lambda: "deadbeef")
    monkeypatch.setattr("evaluation.baselines._code_snapshot_sha256", lambda: "c" * 64)
    monkeypatch.setattr("evaluation.baselines._worktree_dirty", lambda: False)
    monkeypatch.setattr(
        "evaluation.baselines._load_pinned_embedder",
        lambda: (
            _DummyEmbedder(),
            "available: sentence-transformers/all-MiniLM-L6-v2@test",
        ),
    )
    monkeypatch.setattr(
        "evaluation.baselines._environment",
        lambda: {
            "python": "3.14.2",
            "platform": "test",
            "machine": "arm64",
            "cpu_count": 8,
        },
    )
    monkeypatch.setattr(
        "evaluation.baselines._execute_accuracy_inputs", _fake_accuracy_reports
    )

    artifacts = {
        "performance-1.json": _performance_artifact(
            python_off_median=100,
            python_off_p95=120,
            python_off_peak=1_000,
            python_on_median=200,
            python_on_p95=240,
            python_on_peak=2_000,
            rust_off_median=300,
            rust_off_p95=360,
            rust_off_peak=3_000,
        ),
        "performance-2.json": _performance_artifact(
            python_off_median=110,
            python_off_p95=125,
            python_off_peak=1_050,
            python_on_median=260,
            python_on_p95=300,
            python_on_peak=2_300,
            rust_off_median=315,
            rust_off_p95=372,
            rust_off_peak=3_150,
        ),
    }

    def fake_performance_smoke(*, output_path: Path) -> dict[str, object]:
        artifact = artifacts[output_path.name]
        output_path.write_text(json.dumps(artifact), encoding="utf-8")
        return artifact

    monkeypatch.setattr(
        "evaluation.baselines.run_performance_smoke", fake_performance_smoke
    )

    output_path = tmp_path / "baseline.json"
    report = build_baseline(tuning_bundle=tmp_path / "unused", output_path=output_path)
    report_map = report if isinstance(report, Mapping) else {}

    optional_coverage = report_map["optional_coverage"]
    gates = report_map["gates"]
    matrix = report_map["matrix"]

    assert report_map["selected_baseline_id"] == "strict/python/off"
    assert report_map["dataset_hash"] == "3" * 64
    assert isinstance(optional_coverage, Mapping)
    assert optional_coverage["rust_backend"] == "available"
    assert (
        optional_coverage["pinned_embedding_model"]
        == "available: sentence-transformers/all-MiniLM-L6-v2@test"
    )
    assert report_contains_holdout_data(report_map) is False
    assert isinstance(matrix, list)
    assert len(matrix) == 12
    assert any(
        item["id"] == "strict/python/on"
        and item["embedding_model"]["name"] == "sentence-transformers/all-MiniLM-L6-v2"
        for item in matrix
        if isinstance(item, Mapping)
    )
    assert isinstance(gates, Mapping)
    assert gates["performance_noise_margin"] == pytest.approx(0.3)
    assert gates["p95_noise_margin"] == pytest.approx((125 / 120) - 1.0)
    assert gates["peak_memory_noise_margin"] == pytest.approx(0.05)
    assert gates["p95_latency_budget_ns"] == 130
    assert gates["peak_memory_budget_bytes"] == 1102
    assert gates["p95_latency_budget_ns"] < 200
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_compare_baselines_accepts_large_non_negative_declared_noise_margins() -> None:
    left = {
        "schema_version": "evaluation.baseline.v1",
        "dataset_hash": "a" * 64,
        "code_snapshot_sha256": "b" * 64,
        "selected_baseline_id": "strict/python/off",
        "matrix": [{"id": "strict/python/off", "train": {"output_sha256": "a"}}],
        "performance_trials": [
            {
                "scenarios": [
                    {
                        "scenario_id": "s1",
                        "end_to_end": {
                            "median_duration_ns": 100,
                            "p95_duration_ns": 100,
                        },
                        "peak_memory_bytes": 1000,
                    }
                ]
            }
        ],
        "gates": {
            "performance_noise_margin": 1.5,
            "p95_noise_margin": 2.0,
            "peak_memory_noise_margin": 1.1,
        },
    }
    result = compare_baselines(left, left)

    assert result["performance_noise_margin"] == pytest.approx(1.5)
    assert result["p95_noise_margin"] == pytest.approx(2.0)
    assert result["peak_memory_noise_margin"] == pytest.approx(1.1)


def _case() -> EvaluationCase:
    text = "Paris is in France."
    return EvaluationCase.model_validate(
        {
            "case_id": "baseline-case",
            "dataset_version": "1.0.0",
            "split": "dev",
            "document_family_id": "family",
            "transformation_family_id": "exact",
            "provenance": {"kind": "authored"},
            "sources": ({"source_id": "source-1", "text": text},),
            "answer": text,
            "evaluation_units": (
                {
                    "unit_id": "unit-1",
                    "answer_span": {"start": 0, "end": len(text)},
                    "text": text,
                    "claims": (
                        {
                            "claim_id": "claim-1",
                            "answer_span": {"start": 0, "end": len(text)},
                            "text": text,
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "requirement-1",
                                    "alternatives": (
                                        {
                                            "source_id": "source-1",
                                            "spans": ({"start": 0, "end": len(text)},),
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


def test_fixture_is_canonical_json_serializable() -> None:
    assert canonical_json_bytes(_case()).startswith(b"{")


def _fake_accuracy_reports(inputs: object) -> list[tuple[str, object]]:
    reports = []
    for identifier, backend, config, cases, _embedder in inputs:  # type: ignore[misc]
        runs = tuple(
            _supported_run(case=item, backend=backend, config=config) for item in cases
        )
        reports.append((identifier, accuracy_report(cases=cases, runs=runs)))
    return reports


def _supported_run(
    *, case: EvaluationCase, backend: str, config: CitationConfig
) -> CaseRun:
    return CaseRun.model_validate(
        {
            "case_id": case.case_id,
            "backend": backend,
            "config": canonicalize_config(config),
            "outputs": (
                {
                    "answer_span": {
                        "text": case.answer,
                        "char_start": 0,
                        "char_end": len(case.answer),
                    },
                    "citations": [
                        {
                            "score": 1.0,
                            "source_id": "source-1",
                            "source_index": 0,
                            "candidate_index": 0,
                            "char_start": 0,
                            "char_end": len(case.answer),
                            "evidence": case.answer,
                            "evidence_spans": [
                                {
                                    "char_start": 0,
                                    "char_end": len(case.answer),
                                    "evidence": case.answer,
                                }
                            ],
                        }
                    ],
                    "status": "supported",
                },
            ),
            "output_unit_ids": (("unit-1",),),
            "duration_ns": 1,
        }
    )


def _performance_artifact(
    *,
    python_off_median: int,
    python_off_p95: int,
    python_off_peak: int,
    python_on_median: int,
    python_on_p95: int,
    python_on_peak: int,
    rust_off_median: int,
    rust_off_p95: int,
    rust_off_peak: int,
) -> dict[str, object]:
    return {
        "scenarios": [
            {
                "scenario_id": "python:embeddings-off:prepared:small:short:single",
                "backend": "python",
                "embeddings": "off",
                "end_to_end": {
                    "median_duration_ns": python_off_median,
                    "p95_duration_ns": python_off_p95,
                },
                "peak_memory_bytes": python_off_peak,
            },
            {
                "scenario_id": "python:embeddings-on:prepared:small:short:single",
                "backend": "python",
                "embeddings": "on",
                "end_to_end": {
                    "median_duration_ns": python_on_median,
                    "p95_duration_ns": python_on_p95,
                },
                "peak_memory_bytes": python_on_peak,
            },
            {
                "scenario_id": "rust:embeddings-off:prepared:small:short:single",
                "backend": "rust",
                "embeddings": "off",
                "end_to_end": {
                    "median_duration_ns": rust_off_median,
                    "p95_duration_ns": rust_off_p95,
                },
                "peak_memory_bytes": rust_off_peak,
            },
        ]
    }


class _DummyEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]
