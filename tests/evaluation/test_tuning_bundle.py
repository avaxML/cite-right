from __future__ import annotations

import dataclasses
import json
import os
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.manifest import build_private_manifest
from evaluation.review import ReviewLedger, make_review_record
from evaluation.schema import EvaluationCase
from evaluation.sealing import seal_holdout
from evaluation.tuning_bundle import (
    build_tuning_bundle,
    load_tuning_bundle,
    worker_launch_spec,
)


def test_build_tuning_bundle_redacts_holdout_tokens_review_metadata_and_dataset_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, holdout_case = _write_dataset_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "tuning"

    manifest = build_tuning_bundle(dataset_dir, output_dir)
    bundle = load_tuning_bundle(output_dir)

    assert manifest == bundle.manifest
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert {path.name for path in output_dir.iterdir()} == {
        "dev.json",
        "manifest.json",
        "train.json",
    }
    assert [case.case_id for case in bundle.train_cases] == ["case-train"]
    assert [case.case_id for case in bundle.dev_cases] == ["case-dev"]
    assert all(case.split == "train" for case in bundle.train_cases)
    assert all(case.split == "dev" for case in bundle.dev_cases)
    assert all(case.review is None for case in (*bundle.train_cases, *bundle.dev_cases))

    serialized = _bundle_secret_scan(output_dir)
    forbidden_tokens = (
        str(dataset_dir),
        "holdout.aesgcm",
        "holdout.public.json",
        "holdout_public_key.pem",
        "holdout_reviews.json",
        "Sensitive holdout note.",
        "Approved dev claim.",
        "Approved case.",
        "case-holdout",
        holdout_case.answer,
        "ciphertext_b64",
    )
    for token in forbidden_tokens:
        assert token not in serialized


def test_build_tuning_bundle_rejects_existing_output_unknown_dataset_entries_and_holdout_alias_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)

    existing_output = tmp_path / "tuning"
    existing_output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_tuning_bundle(dataset_dir, existing_output)

    rogue_dataset_dir, _ = _write_dataset_fixture(tmp_path / "rogue", monkeypatch)
    (rogue_dataset_dir / "rogue.secret").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown dataset artifact"):
        build_tuning_bundle(rogue_dataset_dir, tmp_path / "rogue-bundle")

    restricted_dir = dataset_dir / "holdout"
    restricted_dir.mkdir()
    with pytest.raises(ValueError, match="holdout or review directories"):
        build_tuning_bundle(dataset_dir, restricted_dir / "bundle")

    alias_path = tmp_path / "holdout-alias"
    alias_path.symlink_to(restricted_dir, target_is_directory=True)
    with pytest.raises(
        ValueError, match="must not be a symlink|holdout or review directories"
    ):
        build_tuning_bundle(dataset_dir, alias_path / "bundle")


def test_build_tuning_bundle_rejects_symlink_nonregular_noncanonical_duplicate_and_wrong_split_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symlink_dataset, _ = _write_dataset_fixture(tmp_path / "symlink", monkeypatch)
    canonical_train = tmp_path / "canonical-train.json"
    canonical_train.write_bytes((symlink_dataset / "train.json").read_bytes())
    (symlink_dataset / "train.json").unlink()
    (symlink_dataset / "train.json").symlink_to(canonical_train)
    with pytest.raises(ValueError, match="must not be a symlink"):
        build_tuning_bundle(symlink_dataset, tmp_path / "symlink-bundle")

    nonregular_dataset, _ = _write_dataset_fixture(tmp_path / "nonregular", monkeypatch)
    (nonregular_dataset / "manifest.json").unlink()
    (nonregular_dataset / "manifest.json").mkdir()
    with pytest.raises(ValueError, match="must be a regular file"):
        build_tuning_bundle(nonregular_dataset, tmp_path / "nonregular-bundle")

    noncanonical_dataset, _ = _write_dataset_fixture(
        tmp_path / "noncanonical", monkeypatch
    )
    dev_payload = json.loads(
        (noncanonical_dataset / "dev.json").read_text(encoding="utf-8")
    )
    (noncanonical_dataset / "dev.json").write_text(
        json.dumps(dev_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        build_tuning_bundle(noncanonical_dataset, tmp_path / "noncanonical-bundle")

    duplicate_dataset, _ = _write_dataset_fixture(tmp_path / "duplicate", monkeypatch)
    duplicate_dev = json.loads(
        (duplicate_dataset / "dev.json").read_text(encoding="utf-8")
    )
    duplicate_dev[0]["case_id"] = "case-train"
    (duplicate_dataset / "dev.json").write_bytes(canonical_json_bytes(duplicate_dev))
    with pytest.raises(ValueError, match="duplicate case id"):
        build_tuning_bundle(duplicate_dataset, tmp_path / "duplicate-bundle")

    wrong_split_dataset, _ = _write_dataset_fixture(
        tmp_path / "wrong-split", monkeypatch
    )
    wrong_train = json.loads(
        (wrong_split_dataset / "train.json").read_text(encoding="utf-8")
    )
    wrong_train[0]["split"] = "holdout"
    (wrong_split_dataset / "train.json").write_bytes(canonical_json_bytes(wrong_train))
    with pytest.raises(ValueError, match="must contain only train cases"):
        build_tuning_bundle(wrong_split_dataset, tmp_path / "wrong-split-bundle")


def test_load_tuning_bundle_rejects_hash_tamper_unknown_entries_symlinks_duplicates_wrong_split_and_manifest_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)

    tampered_bundle = tmp_path / "tampered"
    build_tuning_bundle(dataset_dir, tampered_bundle)
    train_payload = json.loads(
        (tampered_bundle / "train.json").read_text(encoding="utf-8")
    )
    train_payload[0]["answer"] += " tampered"
    (tampered_bundle / "train.json").write_bytes(canonical_json_bytes(train_payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_tuning_bundle(tampered_bundle)

    unexpected_bundle = tmp_path / "unexpected"
    build_tuning_bundle(dataset_dir, unexpected_bundle)
    (unexpected_bundle / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        load_tuning_bundle(unexpected_bundle)

    symlink_bundle = tmp_path / "symlink-bundle"
    build_tuning_bundle(dataset_dir, symlink_bundle)
    manifest_copy = tmp_path / "manifest-copy.json"
    manifest_copy.write_bytes((symlink_bundle / "manifest.json").read_bytes())
    (symlink_bundle / "manifest.json").unlink()
    (symlink_bundle / "manifest.json").symlink_to(manifest_copy)
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_tuning_bundle(symlink_bundle)

    duplicate_bundle = tmp_path / "duplicate-bundle"
    build_tuning_bundle(dataset_dir, duplicate_bundle)
    duplicate_dev = json.loads(
        (duplicate_bundle / "dev.json").read_text(encoding="utf-8")
    )
    duplicate_dev[0]["case_id"] = "case-train"
    duplicate_dev_bytes = canonical_json_bytes(duplicate_dev)
    (duplicate_bundle / "dev.json").write_bytes(duplicate_dev_bytes)
    duplicate_manifest = json.loads(
        (duplicate_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    duplicate_manifest["dev_sha256"] = sha256_hex(duplicate_dev_bytes)
    (duplicate_bundle / "manifest.json").write_bytes(
        canonical_json_bytes(duplicate_manifest)
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_tuning_bundle(duplicate_bundle)

    wrong_split_bundle = tmp_path / "wrong-split-bundle"
    build_tuning_bundle(dataset_dir, wrong_split_bundle)
    wrong_dev = json.loads(
        (wrong_split_bundle / "dev.json").read_text(encoding="utf-8")
    )
    wrong_dev[0]["split"] = "holdout"
    wrong_dev_bytes = canonical_json_bytes(wrong_dev)
    (wrong_split_bundle / "dev.json").write_bytes(wrong_dev_bytes)
    wrong_split_manifest = json.loads(
        (wrong_split_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    wrong_split_manifest["dev_sha256"] = sha256_hex(wrong_dev_bytes)
    (wrong_split_bundle / "manifest.json").write_bytes(
        canonical_json_bytes(wrong_split_manifest)
    )
    with pytest.raises(ValueError, match="must contain only dev cases"):
        load_tuning_bundle(wrong_split_bundle)

    review_bundle = tmp_path / "review-bundle"
    build_tuning_bundle(dataset_dir, review_bundle)
    review_train = json.loads(
        (review_bundle / "train.json").read_text(encoding="utf-8")
    )
    review_train[0]["review"] = {
        "state": "approved",
        "reviewer": "leaky-reviewer",
        "reviewed_at": "2026-07-18",
        "notes": "Leaked notes.",
    }
    review_train_bytes = canonical_json_bytes(review_train)
    (review_bundle / "train.json").write_bytes(review_train_bytes)
    review_manifest = json.loads(
        (review_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    review_manifest["train_sha256"] = sha256_hex(review_train_bytes)
    (review_bundle / "manifest.json").write_bytes(canonical_json_bytes(review_manifest))
    with pytest.raises(ValueError, match="must not include review metadata"):
        load_tuning_bundle(review_bundle)

    leaky_bundle = tmp_path / "leaky-bundle"
    build_tuning_bundle(dataset_dir, leaky_bundle)
    leaky_manifest = json.loads(
        (leaky_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    leaky_manifest["dataset_root"] = str(dataset_dir)
    (leaky_bundle / "manifest.json").write_bytes(canonical_json_bytes(leaky_manifest))
    with pytest.raises(ValueError, match="manifest.json is invalid"):
        load_tuning_bundle(leaky_bundle)


def test_build_tuning_bundle_cleans_up_temp_directory_after_atomic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "tuning"

    import evaluation.tuning_bundle as tuning_bundle

    original_replace = tuning_bundle.os.replace

    def fail_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst) == output_dir:
            raise OSError("simulated atomic replace failure")
        original_replace(src, dst)

    monkeypatch.setattr(tuning_bundle.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        build_tuning_bundle(dataset_dir, output_dir)

    assert not output_dir.exists()
    assert not any(path.name.startswith(".tuning.tmp.") for path in tmp_path.iterdir())


def test_load_tuning_bundle_returns_immutable_train_dev_only_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "tuning"
    build_tuning_bundle(dataset_dir, output_dir)

    bundle = load_tuning_bundle(output_dir)

    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.__setattr__("root_dir", tmp_path)


def test_worker_launch_spec_scrubs_sensitive_environment_variables(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "train.json").write_text("[]", encoding="utf-8")
    (bundle_dir / "dev.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        worker_launch_spec(
            bundle_dir,
            base_env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "/tmp/untrusted",
                "PYTHONSAFEPATH": "0",
                "CITE_RIGHT_HOLDOUT_KEY_FILE": "/secret/holdout.key",
                "CITE_RIGHT_ATTESTATION_KEY_FILE": "/secret/attestation.pem",
            },
        )


def test_worker_launch_spec_launches_repo_worker_from_bundle_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)
    bundle_dir = tmp_path / "tuning"
    build_tuning_bundle(dataset_dir, bundle_dir)

    spec = worker_launch_spec(
        bundle_dir,
        base_env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "/tmp/untrusted",
            "PYTHONSAFEPATH": "0",
            "CITE_RIGHT_HOLDOUT_KEY_FILE": "/secret/holdout.key",
            "CITE_RIGHT_ATTESTATION_KEY_FILE": "/secret/attestation.pem",
        },
    )

    assert spec.command[1:] == ("-m", "evaluation.worker")
    assert spec.command[0] == sys.executable
    assert spec.cwd == bundle_dir
    assert spec.env["PYTHONPATH"] == str(Path(__file__).resolve().parents[2])
    assert spec.env["PYTHONSAFEPATH"] == "1"
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


def test_worker_launch_spec_safe_import_path_blocks_bundle_module_hijack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir, _ = _write_dataset_fixture(tmp_path, monkeypatch)
    bundle_dir = tmp_path / "tuning"
    build_tuning_bundle(dataset_dir, bundle_dir)

    spec = worker_launch_spec(
        bundle_dir,
        base_env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "/tmp/untrusted",
            "PYTHONSAFEPATH": "0",
        },
    )

    marker_path = tmp_path / "malicious-imported.txt"
    malicious_package = bundle_dir / "evaluation"
    malicious_package.mkdir()
    (malicious_package / "__init__.py").write_text("", encoding="utf-8")
    (malicious_package / "worker.py").write_text(
        (
            "from pathlib import Path\n"
            f"Path({str(marker_path)!r}).write_text('executed', encoding='utf-8')\n"
            "raise SystemExit(99)\n"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        spec.command,
        cwd=spec.cwd,
        env=dict(spec.env),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert not marker_path.exists()
    assert "unexpected files in tuning bundle: evaluation" in result.stderr


def _write_dataset_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, EvaluationCase]:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "sources").mkdir(parents=True)
    train_case = _build_case(case_id="case-train", split="train")
    dev_case = _build_case(case_id="case-dev", split="dev")
    holdout_case = _build_case(case_id="case-holdout", split="holdout")
    (dataset_dir / "train.json").write_bytes(
        _canonical_json_bytes([_case_payload(train_case)])
    )
    (dataset_dir / "dev.json").write_bytes(
        _canonical_json_bytes([_case_payload(dev_case)])
    )
    (dataset_dir / "holdout.json").write_bytes(
        _canonical_json_bytes([_case_payload(holdout_case)])
    )
    (dataset_dir / "manifest.json").write_bytes(
        canonical_json_bytes(
            build_private_manifest(
                (train_case, dev_case, holdout_case),
                generated_at="2026-07-18",
            )
        )
    )
    (dataset_dir / "provenance.json").write_bytes(_canonical_json_bytes([]))
    (dataset_dir / "sources" / "authored.json").write_bytes(_canonical_json_bytes([]))
    (dataset_dir / "sources" / "real.json").write_bytes(_canonical_json_bytes([]))
    dev_ledger = ReviewLedger(
        dataset_version="1.0.0",
        schema_version="1.0.0",
        entries=(
            make_review_record(
                dev_case,
                dev_case.evaluation_units[0].claims[0],
                reviewer="dev-reviewer",
                reviewed_at="2026-07-18",
                decision="approve",
                notes="Approved dev claim.",
            ),
        ),
    )
    (dataset_dir / "dev_reviews.json").write_bytes(canonical_json_bytes(dev_ledger))
    public_key_path = dataset_dir / "holdout_public_key.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)
    holdout_ledger = ReviewLedger(
        dataset_version="1.0.0",
        schema_version="1.0.0",
        entries=(
            make_review_record(
                holdout_case,
                holdout_case.evaluation_units[0].claims[0],
                reviewer="holdout-reviewer",
                reviewed_at="2026-07-18",
                decision="approve",
                notes="Sensitive holdout note.",
            ),
        ),
    )
    (dataset_dir / "holdout_reviews.json").write_bytes(
        canonical_json_bytes(holdout_ledger)
    )
    seal_holdout(
        (holdout_case,),
        ledger=holdout_ledger,
        ciphertext_path=dataset_dir / "holdout.aesgcm",
        public_manifest_path=dataset_dir / "holdout.public.json",
        public_key_path=public_key_path,
        generated_at="2026-07-18T12:00:00Z",
    )
    return dataset_dir, holdout_case


def _build_case(*, case_id: str, split: str) -> EvaluationCase:
    answer = f"{case_id} evidence"
    return EvaluationCase.model_validate(
        {
            "case_id": case_id,
            "dataset_version": "1.0.0",
            "split": split,
            "document_family_id": f"family-{case_id}",
            "transformation_family_id": "transform-fixture",
            "provenance": {
                "kind": "public_domain",
                "title": f"Title for {case_id}",
                "origin": f"https://example.test/{case_id}",
                "publisher": "Example Publisher",
                "license": "Public domain",
                "retrieval_date": date(2026, 7, 18),
                "snapshot_hash": f"snapshot-{case_id}",
            },
            "sources": (
                {
                    "source_id": "source-1",
                    "text": answer,
                },
            ),
            "answer": answer,
            "evaluation_units": (
                {
                    "unit_id": "unit-1",
                    "answer_span": {"start": 0, "end": len(answer)},
                    "text": answer,
                    "claims": (
                        {
                            "claim_id": "claim-1",
                            "answer_span": {"start": 0, "end": len(answer)},
                            "text": answer,
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "req-1",
                                    "alternatives": (
                                        {
                                            "source_id": "source-1",
                                            "spans": (
                                                {"start": 0, "end": len(answer)},
                                            ),
                                        },
                                    ),
                                },
                            ),
                            "acceptable_retrieval_source_ids": ("source-1",),
                        },
                    ),
                },
            ),
            "difficulty_tags": ("fixture", split),
            "review": {
                "state": "approved",
                "reviewer": "case-reviewer",
                "reviewed_at": date(2026, 7, 18),
                "notes": "Approved case.",
            },
        }
    )


def _bundle_secret_scan(bundle_dir: Path) -> str:
    chunks = ["\n".join(sorted(path.name for path in bundle_dir.iterdir()))]
    chunks.extend(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(bundle_dir.iterdir())
        if path.is_file()
    )
    return "\n".join(chunks)


def _case_payload(case: EvaluationCase) -> dict[str, object]:
    return case.model_dump(mode="json", exclude_computed_fields=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _install_signing_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    public_key_path: Path,
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    holdout_key_path = tmp_path / "holdout.key"
    holdout_key_path.write_bytes(bytes(range(32)))
    os.chmod(holdout_key_path, stat.S_IRUSR | stat.S_IWUSR)

    private_key = Ed25519PrivateKey.generate()
    private_key_path = tmp_path / "attestation-private.pem"
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    os.chmod(private_key_path, stat.S_IRUSR | stat.S_IWUSR)

    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("CITE_RIGHT_HOLDOUT_KEY_FILE", str(holdout_key_path))
    monkeypatch.setenv("CITE_RIGHT_ATTESTATION_KEY_FILE", str(private_key_path))
