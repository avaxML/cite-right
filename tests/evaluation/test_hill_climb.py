from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.hill_climb import load_search_space, run_tuning, select_best_record


def test_select_best_record_uses_lexicographic_gates_and_tiebreakers(
    tmp_path: Path,
) -> None:
    search_space = load_search_space(
        Path("tests/evaluation/fixtures/three-candidates.json")
    )
    result = run_tuning(
        tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
        search_space_path=Path("tests/evaluation/fixtures/three-candidates.json"),
        output_path=tmp_path / "tune.json",
    )

    assert search_space.candidates[0].candidate_id == "candidate-offset-loss"
    assert result["best_candidate_id"] == "candidate-recall-win"
    assert result["evaluated_candidate_count"] == 3
    payload = json.loads((tmp_path / "tune.json").read_text(encoding="utf-8"))
    assert payload["best_candidate_id"] == "candidate-recall-win"
    assert [record["candidate_id"] for record in payload["records"]] == [
        "candidate-offset-loss",
        "candidate-recall-lose",
        "candidate-recall-win",
    ]
    best = select_best_record(tuple(search_space.synthetic_records()))
    assert best is not None
    assert best.candidate_id == "candidate-recall-win"


def test_run_tuning_resumes_and_suppresses_duplicate_candidates(tmp_path: Path) -> None:
    output_path = tmp_path / "resume.json"
    first = run_tuning(
        tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
        search_space_path=Path("tests/evaluation/fixtures/three-candidates.json"),
        output_path=output_path,
    )
    second = run_tuning(
        tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
        search_space_path=Path("tests/evaluation/fixtures/three-candidates.json"),
        output_path=output_path,
    )

    assert first["evaluated_candidate_count"] == 3
    assert second["evaluated_candidate_count"] == 0
    assert second["duplicate_candidate_ids"] == [
        "candidate-offset-loss",
        "candidate-recall-lose",
        "candidate-recall-win",
    ]


def test_run_tuning_rejects_holdout_and_release_gate_inputs(tmp_path: Path) -> None:
    holdout_space = tmp_path / "holdout-space.json"
    holdout_space.write_text("{}", encoding="utf-8")
    release_gate_path = tmp_path / "release-holdout.json"
    release_gate_path.write_text("{}", encoding="utf-8")

    try:
        run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=Path("tests/evaluation/fixtures/three-candidates.json"),
            output_path=release_gate_path,
        )
    except ValueError as exc:
        assert "release-gate" in str(exc) or "holdout" in str(exc)
    else:
        raise AssertionError("expected release-gate output rejection")

    try:
        run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=holdout_space,
            output_path=tmp_path / "ok.json",
        )
    except ValueError as exc:
        assert "search space" in str(exc) or "holdout" in str(exc)
    else:
        raise AssertionError("expected holdout search-space rejection")


def test_run_tuning_rejects_holdout_content_inside_search_space(tmp_path: Path) -> None:
    search_space = {
        "schema_version": "evaluation.search-space.v1",
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "code_path_id": "config-only",
                "backend": "python",
                "embeddings": "off",
                "config": {"probe_path": "release/holdout.aesgcm"},
            }
        ],
    }
    search_space_path = tmp_path / "safe-name.json"
    search_space_path.write_text(json.dumps(search_space), encoding="utf-8")

    with pytest.raises(ValueError, match="holdout"):
        run_tuning(
            tuning_bundle=Path("tests/evaluation/fixtures/tuning"),
            search_space_path=search_space_path,
            output_path=tmp_path / "out.json",
        )


def test_run_tuning_rejects_resume_with_changed_search_space(tmp_path: Path) -> None:
    output_path = tmp_path / "resume.json"
    baseline_path = Path("tests/evaluation/fixtures/three-candidates.json")
    changed_path = tmp_path / "changed.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["candidates"][2]["config"] = {"min_final_score": 0.31, "top_k": 9}
    changed_path.write_text(json.dumps(payload), encoding="utf-8")

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
