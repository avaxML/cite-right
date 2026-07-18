from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from evaluation.canonical import canonical_json_bytes
from evaluation.cli import main
from evaluation.manifest import build_private_manifest
from evaluation.schema import EvaluationCase
from evaluation.tuning_bundle import worker_launch_spec
from tests.evaluation.test_tuning_bundle import _write_dataset_fixture


def test_cli_help_lists_exact_six_foundational_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [
        "build",
        "validate",
        "seal",
        "verify-public-manifest",
        "build-tuning-bundle",
        "promote",
    ]
    command_block = result.stdout.split("{", 1)[1].split("}", 1)[0].split(",")
    assert command_block == commands
    for command in commands:
        assert command in result.stdout


def test_cli_unknown_and_missing_arguments_exit_two() -> None:
    assert main(["unknown"]) == 2
    assert main(["build"]) == 2
    assert main(["validate"]) == 2


def test_cli_operational_failures_emit_structured_json_on_stderr() -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        exit_code = main(["validate", "--bundle", "/definitely/missing"])
    assert exit_code == 1
    payload = json.loads(stderr.getvalue())
    assert payload["ok"] is False
    assert payload["error"]["type"] in {"ValueError", "FileNotFoundError"}
    assert "traceback" not in payload["error"]["message"].lower()


def test_build_command_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        _freeze_cli_date("2026-07-17")
        assert main(["build", "--output", str(left), "--seed", "20260717"]) == 0
        _freeze_cli_date("2026-07-18")
        assert main(["build", "--output", str(right), "--seed", "20260717"]) == 0
    assert stderr.getvalue() == ""
    for relative_path in (
        "train.json",
        "dev.json",
        "holdout.json",
        "manifest.json",
        "dev_reviews.json",
        "provenance.json",
        "sources/real.json",
    ):
        assert (left / relative_path).read_bytes() == (right / relative_path).read_bytes()
    assert main(["validate", "--bundle", str(left)]) == 0


def test_build_command_emits_success_json_and_excludes_computed_fields_from_case_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bundle"
    stdout = io.StringIO()
    stderr = io.StringIO()

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        _freeze_cli_date("2026-07-18")
        exit_code = main(["build", "--output", str(output_dir), "--seed", "20260717"])

    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["command"] == "build"

    for case_file in ("train.json", "dev.json", "holdout.json"):
        case_payload = json.loads((output_dir / case_file).read_text(encoding="utf-8"))
        assert all("expected_status" not in unit for item in case_payload for unit in item["evaluation_units"])


def test_seal_and_verify_public_manifest_commands_succeed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)
    ciphertext_path = tmp_path / "sealed-holdout.aesgcm"
    public_manifest_path = tmp_path / "sealed-holdout.public.json"

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = main(
            [
                "seal",
                "--plaintext",
                str(dataset_dir / "holdout.json"),
                "--output",
                str(ciphertext_path),
                "--public-manifest",
                str(public_manifest_path),
                "--public-key",
                str(dataset_dir / "holdout_public_key.pem"),
            ]
        )
    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["ciphertext_path"] == str(ciphertext_path)
    assert payload["public_manifest_path"] == str(public_manifest_path)

    verify_dir = tmp_path / "verify-bundle"
    verify_dir.mkdir()
    (verify_dir / "holdout.aesgcm").write_bytes(ciphertext_path.read_bytes())
    (verify_dir / "holdout.public.json").write_bytes(public_manifest_path.read_bytes())
    (verify_dir / "holdout_public_key.pem").write_bytes(
        (dataset_dir / "holdout_public_key.pem").read_bytes()
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = main(["verify-public-manifest", "--bundle", str(verify_dir)])
    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["command"] == "verify-public-manifest"


def test_build_tuning_bundle_and_worker_subprocess_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)
    bundle_dir = tmp_path / "tuning"

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = main(
            [
                "build-tuning-bundle",
                "--dataset",
                str(dataset_dir),
                "--output",
                str(bundle_dir),
            ]
        )
    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True

    env = dict(os.environ)
    env["CITE_RIGHT_HOLDOUT_KEY_FILE"] = "/secret/holdout.key"
    env["CITE_RIGHT_ATTESTATION_KEY_FILE"] = "/secret/attestation.pem"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.worker"],
        cwd=bundle_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "sensitive holdout environment variables" in result.stderr

    spec = worker_launch_spec(bundle_dir, base_env=env)
    assert spec.command == (sys.executable, "-m", "evaluation.worker")
    assert spec.cwd == bundle_dir
    assert spec.env["PYTHONPATH"] == str(Path(__file__).resolve().parents[2])
    assert "CITE_RIGHT_HOLDOUT_KEY_FILE" not in spec.env
    assert "CITE_RIGHT_ATTESTATION_KEY_FILE" not in spec.env

    result = subprocess.run(
        spec.command,
        cwd=spec.cwd,
        env=dict(spec.env),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["train_case_count"] == 1
    assert payload["dev_case_count"] == 1

    result = subprocess.run(
        [*spec.command, str(dataset_dir / "train.json")],
        cwd=spec.cwd,
        env=dict(spec.env),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert (
        "accepts no positional or option arguments" in result.stderr
        or "unrecognized arguments" in result.stderr
        or "usage:" in result.stderr
    )


def test_promote_rejects_plaintext_holdout_and_rolls_back_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)
    dataset_dir = tmp_path / "dataset-live"
    dataset_dir.mkdir()
    (dataset_dir / "sentinel.txt").write_text("original", encoding="utf-8")

    with contextlib.redirect_stderr(io.StringIO()):
        exit_code = main(["promote", "--staging", str(staging_dir), "--dataset", str(dataset_dir)])
    assert exit_code == 1

    plaintext_removed = staging_dir / "holdout.json"
    plaintext_removed.unlink()

    import evaluation.cli as cli

    original_replace = cli.os.replace
    replace_calls: list[tuple[str, str]] = []

    def flaky_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        replace_calls.append((str(src), str(dst)))
        if str(dst) == str(dataset_dir):
            raise OSError("simulated replace failure")
        original_replace(src, dst)

    monkeypatch.setattr(cli.os, "replace", flaky_replace)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        exit_code = main(["promote", "--staging", str(staging_dir), "--dataset", str(dataset_dir)])
    assert exit_code == 1
    assert (dataset_dir / "sentinel.txt").read_text(encoding="utf-8") == "original"
    assert not any(path.name.startswith(".dataset-live.tmp.") for path in tmp_path.iterdir() if path.is_dir())


def test_promote_success_replaces_dataset_with_allowlisted_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_dir = _write_promotion_staging_fixture(tmp_path / "staging", monkeypatch)
    dataset_dir = tmp_path / "dataset-live"
    dataset_dir.mkdir()
    (dataset_dir / "sentinel.txt").write_text("original", encoding="utf-8")

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = main(["promote", "--staging", str(staging_dir), "--dataset", str(dataset_dir)])

    assert exit_code == 0
    assert stderr.getvalue() == ""
    payload = json.loads(stdout.getvalue())
    assert payload["ok"] is True
    assert payload["command"] == "promote"
    assert not (dataset_dir / "sentinel.txt").exists()
    assert sorted(
        str(path.relative_to(dataset_dir))
        for path in dataset_dir.rglob("*")
        if path.is_file()
    ) == [
        "dev.json",
        "dev_reviews.json",
        "holdout.aesgcm",
        "holdout.public.json",
        "holdout_public_key.pem",
        "manifest.json",
        "provenance.json",
        "sources/real.json",
        "train.json",
    ]


def test_promote_rejects_unknown_empty_directories_and_manifest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_dir = _write_promotion_staging_fixture(tmp_path / "staging", monkeypatch)
    dataset_dir = tmp_path / "dataset-live"

    (staging_dir / "unknown-empty-dir").mkdir()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        exit_code = main(["promote", "--staging", str(staging_dir), "--dataset", str(dataset_dir)])
    assert exit_code == 1
    payload = json.loads(stderr.getvalue())
    assert payload["ok"] is False
    assert "unknown director" in payload["error"]["message"]

    (staging_dir / "unknown-empty-dir").rmdir()
    manifest_payload = json.loads((staging_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload["split_case_counts"]["train"] = 999
    (staging_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest_payload))
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        exit_code = main(["promote", "--staging", str(staging_dir), "--dataset", str(dataset_dir)])
    assert exit_code == 1
    payload = json.loads(stderr.getvalue())
    assert payload["ok"] is False
    assert "manifest" in payload["error"]["message"]


def test_promote_preserves_backup_when_rollback_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_dir = _write_promotion_staging_fixture(tmp_path / "staging", monkeypatch)
    dataset_dir = tmp_path / "dataset-live"
    dataset_dir.mkdir()
    (dataset_dir / "sentinel.txt").write_text("original", encoding="utf-8")

    import evaluation.cli as cli

    original_replace = cli.os.replace

    def broken_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        if dst_path == dataset_dir and src_path.name.startswith(".dataset-live.tmp."):
            raise OSError("simulated publish failure")
        if dst_path == dataset_dir and src_path.name.startswith(".dataset-live.backup."):
            raise OSError("simulated rollback failure")
        original_replace(src, dst)

    monkeypatch.setattr(cli.os, "replace", broken_replace)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        exit_code = main(["promote", "--staging", str(staging_dir), "--dataset", str(dataset_dir)])
    assert exit_code == 1
    payload = json.loads(stderr.getvalue())
    assert payload["ok"] is False
    assert "rollback was incomplete" in payload["error"]["message"]
    backup_dirs = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.startswith(".dataset-live.backup.")
    ]
    assert len(backup_dirs) == 1
    assert (backup_dirs[0] / "sentinel.txt").read_text(encoding="utf-8") == "original"


def _freeze_cli_date(iso_date: str) -> None:
    import evaluation.cli as cli

    frozen_day = date.fromisoformat(iso_date)

    class _FrozenDate:
        @classmethod
        def today(cls) -> date:
            return frozen_day

    cli.date = _FrozenDate


def _write_promotion_staging_fixture(base_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dataset_dir, _ = _write_dataset_fixture(base_dir, monkeypatch)
    train_cases = tuple(
        EvaluationCase.model_validate_json(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for item in json.loads((dataset_dir / "train.json").read_text(encoding="utf-8"))
    )
    dev_cases = tuple(
        EvaluationCase.model_validate_json(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for item in json.loads((dataset_dir / "dev.json").read_text(encoding="utf-8"))
    )
    non_holdout_manifest = build_private_manifest(
        train_cases + dev_cases,
        generated_at="2026-07-18",
    )
    (dataset_dir / "manifest.json").write_bytes(canonical_json_bytes(non_holdout_manifest))
    (dataset_dir / "holdout.json").unlink()
    (dataset_dir / "holdout_reviews.json").unlink()
    assert train_cases and dev_cases
    return dataset_dir
