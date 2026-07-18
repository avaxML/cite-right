"""Operational CLI for evaluation dataset artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Sequence

from evaluation.baselines import build_baseline
from evaluation.builders.authored_sources import AUTHORED_FACT_TEMPLATES
from evaluation.builders.cases import generate_all_authored_cases
from evaluation.builders.real_sources import (
    PROVENANCE_PATH,
    REAL_SOURCES_PATH,
    generate_real_cases,
)
from evaluation.canonical import canonical_json_bytes
from evaluation.manifest import (
    DatasetManifest,
    build_private_manifest,
    verify_private_manifest_expectations,
)
from evaluation.performance import run_performance_smoke
from evaluation.review import ReviewLedger, assert_review_complete, load_review_ledger
from evaluation.schema import EvaluationCase
from evaluation.sealing import (
    DEFAULT_PUBLIC_KEY_PATH,
    seal_holdout,
    verify_public_manifest,
)
from evaluation.splitting import apply_split_assignments, assign_splits
from evaluation.tuning_bundle import build_tuning_bundle, load_tuning_bundle
from evaluation.validation import DatasetBundle, validate_dataset

_PROMOTION_BASELINE_FILES = frozenset(
    {
        "dev.json",
        "dev_reviews.json",
        "manifest.json",
        "provenance.json",
        "sources/authored.json",
        "sources/real.json",
        "train.json",
    }
)
_PROMOTION_OPTIONAL_FILES = frozenset(
    {"holdout.aesgcm", "holdout.public.json", "holdout_public_key.pem"}
)
_PROMOTION_DISALLOWED_FILES = frozenset({"holdout.json"})


class _CliUsageError(RuntimeError):
    pass


class _AtomicReplaceRollbackError(RuntimeError):
    def __init__(
        self,
        *,
        original_error: BaseException,
        rollback_error: BaseException,
        recoverable_backup_path: Path,
    ) -> None:
        self.original_error = original_error
        self.rollback_error = rollback_error
        self.recoverable_backup_path = recoverable_backup_path
        super().__init__(
            "directory publication failed and rollback was incomplete; "
            f"publication_error={original_error}; rollback_error={rollback_error}; "
            f"recoverable_backup={recoverable_backup_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="python -m evaluation.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--seed", required=True, type=int)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--bundle", required=True)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--plaintext", required=True)
    seal_parser.add_argument("--output", required=True)
    seal_parser.add_argument("--public-manifest", required=True, dest="public_manifest")
    seal_parser.add_argument("--public-key", default=None, dest="public_key")

    verify_parser = subparsers.add_parser("verify-public-manifest")
    verify_parser.add_argument("--bundle", required=True)

    tuning_parser = subparsers.add_parser("build-tuning-bundle")
    tuning_parser.add_argument("--dataset", required=True)
    tuning_parser.add_argument("--output", required=True)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--staging", required=True)
    promote_parser.add_argument("--dataset", required=True)

    performance_parser = subparsers.add_parser("performance-smoke")
    performance_parser.add_argument("--output", required=True)

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--tuning-bundle", required=True, dest="tuning_bundle")
    baseline_parser.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        payload = _dispatch(args)
    except _CliUsageError as exc:
        _write_usage_error(parser, str(exc))
        return 2
    except Exception as exc:
        _write_error_json(exc)
        return 1
    _write_success_json(payload)
    return 0


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    command = args.command
    if command == "build":
        return _command_build(output_dir=Path(args.output), seed=args.seed)
    if command == "validate":
        return _command_validate(bundle_dir=Path(args.bundle))
    if command == "seal":
        return _command_seal(
            plaintext_path=Path(args.plaintext),
            ciphertext_path=Path(args.output),
            public_manifest_path=Path(args.public_manifest),
            public_key_path=Path(args.public_key) if args.public_key else None,
        )
    if command == "verify-public-manifest":
        return _command_verify_public_manifest(bundle_dir=Path(args.bundle))
    if command == "build-tuning-bundle":
        return _command_build_tuning_bundle(
            dataset_dir=Path(args.dataset), output_dir=Path(args.output)
        )
    if command == "promote":
        return _command_promote(
            staging_dir=Path(args.staging), dataset_dir=Path(args.dataset)
        )
    if command == "performance-smoke":
        return _command_performance_smoke(output_path=Path(args.output))
    if command == "baseline":
        return build_baseline(
            tuning_bundle=Path(args.tuning_bundle), output_path=Path(args.output)
        )
    raise _CliUsageError(f"unknown command {command!r}")


def _command_build(*, output_dir: Path, seed: int) -> dict[str, object]:
    authored_cases = generate_all_authored_cases(seed)
    real_cases = generate_real_cases()
    combined_cases = authored_cases + real_cases
    assignment_report = assign_splits(combined_cases, seed=seed)
    assigned_cases = apply_split_assignments(
        combined_cases, assignment_report.assignment_by_case_id
    )
    generated_at = None
    manifest = build_private_manifest(assigned_cases, generated_at=generated_at)
    validation_report = validate_dataset(
        DatasetBundle(
            case_records=assigned_cases,
            expected_private_manifest=manifest,
            actual_manifest_generated_at=generated_at,
        )
    )
    validation_report.assert_valid()

    train_cases = _canonical_case_order(
        case for case in assigned_cases if case.split == "train"
    )
    dev_cases = _canonical_case_order(
        case for case in assigned_cases if case.split == "dev"
    )
    holdout_cases = _canonical_case_order(
        case for case in assigned_cases if case.split == "holdout"
    )
    dev_ledger = ReviewLedger(
        dataset_version=manifest.dataset_version, schema_version=manifest.schema_version
    )

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    parent = _require_parent_directory(output_dir)
    temp_dir = Path(tempfile.mkdtemp(dir=parent, prefix=f".{output_dir.name}.tmp."))
    try:
        (temp_dir / "sources").mkdir()
        _write_json(
            temp_dir / "train.json", [_case_payload(case) for case in train_cases]
        )
        _write_json(temp_dir / "dev.json", [_case_payload(case) for case in dev_cases])
        _write_json(
            temp_dir / "holdout.json", [_case_payload(case) for case in holdout_cases]
        )
        _write_json(temp_dir / "manifest.json", manifest)
        _write_json(temp_dir / "dev_reviews.json", dev_ledger)
        _write_json(
            temp_dir / "sources" / "authored.json",
            [
                template.model_dump(mode="json")
                for template in sorted(
                    AUTHORED_FACT_TEMPLATES,
                    key=lambda item: item.family_id,
                )
            ],
        )
        _write_json(
            temp_dir / "sources" / "real.json",
            json.loads(_read_regular_file(REAL_SOURCES_PATH)),
        )
        _write_json(
            temp_dir / "provenance.json",
            json.loads(_read_regular_file(PROVENANCE_PATH)),
        )
        _fsync_tree(temp_dir)
        os.replace(temp_dir, output_dir)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "command": "build",
        "bundle": str(output_dir),
        "dataset_version": manifest.dataset_version,
        "dev_case_count": len(dev_cases),
        "holdout_case_count": len(holdout_cases),
        "seed": seed,
        "train_case_count": len(train_cases),
    }


def _command_validate(*, bundle_dir: Path) -> dict[str, object]:
    bundle_root = _require_directory(bundle_dir, label="bundle directory")
    if _looks_like_tuning_bundle(bundle_root):
        bundle = load_tuning_bundle(bundle_root)
        return {
            "bundle": str(bundle_root),
            "command": "validate",
            "dev_case_count": len(bundle.dev_cases),
            "error_count": 0,
            "finding_count": 0,
            "is_valid": True,
            "train_case_count": len(bundle.train_cases),
            "warning_count": 0,
        }

    cases = _load_present_case_files(bundle_root)
    if not cases:
        raise ValueError("bundle directory does not contain any case files")

    expected_manifest = _load_private_manifest_if_present(bundle_root, cases=cases)
    validation_report = validate_dataset(
        DatasetBundle(
            case_records=cases,
            expected_private_manifest=expected_manifest,
            actual_manifest_generated_at=expected_manifest.generated_at
            if expected_manifest
            else None,
        )
    )
    validation_report.assert_valid()

    if (bundle_root / "dev_reviews.json").exists():
        load_review_ledger(bundle_root / "dev_reviews.json")

    return {
        "bundle": str(bundle_root),
        "command": "validate",
        "error_count": sum(
            1 for finding in validation_report.findings if finding.severity == "error"
        ),
        "finding_count": len(validation_report.findings),
        "is_valid": validation_report.is_valid,
        "warning_count": sum(
            1 for finding in validation_report.findings if finding.severity == "warning"
        ),
    }


def _command_seal(
    *,
    plaintext_path: Path,
    ciphertext_path: Path,
    public_manifest_path: Path,
    public_key_path: Path | None,
) -> dict[str, object]:
    cases = _load_cases_file(plaintext_path)
    ledger_path = _resolve_holdout_review_ledger_path(plaintext_path)
    ledger = load_review_ledger(ledger_path)
    resolved_public_key = public_key_path or _resolve_public_key_path(
        public_manifest_path.parent
    )
    manifest = seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=public_manifest_path,
        public_key_path=resolved_public_key,
        generated_at=date.today().isoformat(),
    )
    return {
        "command": "seal",
        "ciphertext_path": str(ciphertext_path),
        "dataset_version": manifest.dataset_version,
        "public_manifest_path": str(public_manifest_path),
    }


def _command_verify_public_manifest(*, bundle_dir: Path) -> dict[str, object]:
    bundle_root = _require_directory(bundle_dir, label="bundle directory")
    manifest = verify_public_manifest(
        bundle_root / "holdout.public.json",
        ciphertext_path=bundle_root / "holdout.aesgcm",
        public_key_path=bundle_root / "holdout_public_key.pem",
    )
    return {
        "bundle": str(bundle_root),
        "command": "verify-public-manifest",
        "dataset_version": manifest.dataset_version,
        "holdout_case_count": manifest.holdout_case_count,
    }


def _command_build_tuning_bundle(
    *, dataset_dir: Path, output_dir: Path
) -> dict[str, object]:
    manifest = build_tuning_bundle(dataset_dir, output_dir)
    return {
        "command": "build-tuning-bundle",
        "dataset_version": manifest.dataset_version,
        "dev_case_count": manifest.dev_case_count,
        "output": str(output_dir),
        "train_case_count": manifest.train_case_count,
    }


def _command_promote(*, staging_dir: Path, dataset_dir: Path) -> dict[str, object]:
    staging_root = _require_directory(staging_dir, label="staging directory")
    staged_files = _validate_promotion_staging(staging_root)

    if dataset_dir.exists() and dataset_dir.is_symlink():
        raise ValueError(f"dataset directory must not be a symlink: {dataset_dir}")
    dataset_parent = _require_parent_directory(dataset_dir)
    temp_dir = Path(
        tempfile.mkdtemp(dir=dataset_parent, prefix=f".{dataset_dir.name}.tmp.")
    )
    try:
        (temp_dir / "sources").mkdir()
        for relative_path in staged_files:
            destination = temp_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_regular_file(staging_root / relative_path, destination)
        _validate_promotion_staging(temp_dir)
        _fsync_tree(temp_dir)
        _atomic_replace_directory(temp_dir, dataset_dir)
        _fsync_directory(dataset_parent)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "command": "promote",
        "dataset": str(dataset_dir),
        "staging": str(staging_dir),
    }


def _command_performance_smoke(*, output_path: Path) -> dict[str, object]:
    return run_performance_smoke(output_path=output_path)


def _validate_promotion_staging(staging_root: Path) -> tuple[str, ...]:
    present_directories = {
        str(path.relative_to(staging_root))
        for path in sorted(staging_root.rglob("*"))
        if path.is_dir()
    }
    unknown_directories = sorted(
        directory for directory in present_directories if directory not in {"sources"}
    )
    if unknown_directories:
        raise ValueError(
            f"staging directory contains unknown directories: {', '.join(unknown_directories)}"
        )

    present_files = {
        str(path.relative_to(staging_root))
        for path in sorted(staging_root.rglob("*"))
        if path.is_file() or path.is_symlink()
    }
    disallowed = sorted(present_files & _PROMOTION_DISALLOWED_FILES)
    if disallowed:
        raise ValueError(
            f"staging directory contains disallowed plaintext holdout artifacts: {', '.join(disallowed)}"
        )

    unknown = sorted(
        present_files - _PROMOTION_BASELINE_FILES - _PROMOTION_OPTIONAL_FILES
    )
    if unknown:
        raise ValueError(
            f"staging directory contains unknown artifacts: {', '.join(unknown)}"
        )

    missing_baseline = sorted(_PROMOTION_BASELINE_FILES - present_files)
    if missing_baseline:
        raise ValueError(
            f"staging directory is missing required artifacts: {', '.join(missing_baseline)}"
        )

    holdout_present = present_files & _PROMOTION_OPTIONAL_FILES
    if holdout_present and holdout_present != _PROMOTION_OPTIONAL_FILES:
        missing_optional = sorted(_PROMOTION_OPTIONAL_FILES - holdout_present)
        raise ValueError(
            f"staging directory is missing required holdout artifacts: {', '.join(missing_optional)}"
        )

    for relative_path in sorted(present_files):
        _read_regular_file(staging_root / relative_path)

    train_cases = _load_cases_file(staging_root / "train.json")
    dev_cases = _load_cases_file(staging_root / "dev.json")
    if any(case.split != "train" for case in train_cases):
        raise ValueError("train.json may contain only train cases")
    if any(case.split != "dev" for case in dev_cases):
        raise ValueError("dev.json may contain only dev cases")

    ledger = load_review_ledger(staging_root / "dev_reviews.json")
    assert_review_complete(dev_cases, ledger, split="dev")
    manifest = _load_dataset_manifest(
        staging_root / "manifest.json",
        cases=train_cases + dev_cases,
    )
    validation_report = validate_dataset(
        DatasetBundle(
            case_records=train_cases + dev_cases,
            expected_private_manifest=manifest,
            actual_manifest_generated_at=manifest.generated_at,
        )
    )
    validation_report.assert_valid()

    if holdout_present:
        public_manifest = verify_public_manifest(
            staging_root / "holdout.public.json",
            ciphertext_path=staging_root / "holdout.aesgcm",
            public_key_path=staging_root / "holdout_public_key.pem",
        )
        if public_manifest.dataset_version != manifest.dataset_version:
            raise ValueError(
                "public holdout manifest dataset_version does not match manifest.json"
            )
        if public_manifest.schema_version != manifest.schema_version:
            raise ValueError(
                "public holdout manifest schema_version does not match manifest.json"
            )

    return tuple(sorted(present_files))


def _load_private_manifest_if_present(
    bundle_root: Path,
    *,
    cases: tuple[EvaluationCase, ...],
) -> DatasetManifest | None:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.exists():
        return None
    return _load_dataset_manifest(manifest_path, cases=cases)


def _load_present_case_files(bundle_root: Path) -> tuple[EvaluationCase, ...]:
    cases: list[EvaluationCase] = []
    for name in ("train.json", "dev.json", "holdout.json"):
        path = bundle_root / name
        if not path.exists():
            continue
        cases.extend(_load_cases_file(path))
    return _canonical_case_order(cases)


def _resolve_holdout_review_ledger_path(plaintext_path: Path) -> Path:
    if plaintext_path.name.endswith(".json"):
        return plaintext_path.with_name(f"{plaintext_path.stem}_reviews.json")
    return plaintext_path.with_name(f"{plaintext_path.name}_reviews.json")


def _resolve_public_key_path(bundle_root: Path) -> Path:
    sibling = bundle_root / "holdout_public_key.pem"
    if sibling.exists():
        return sibling
    return DEFAULT_PUBLIC_KEY_PATH


def _load_cases_file(path: Path) -> tuple[EvaluationCase, ...]:
    raw_bytes = _read_regular_file(path)
    payload = json.loads(raw_bytes)
    if raw_bytes != canonical_json_bytes(payload):
        raise ValueError(f"{path} must use canonical JSON ordering")
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a canonical JSON array")
    cases = tuple(
        EvaluationCase.model_validate_json(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for item in payload
    )
    if cases != _canonical_case_order(cases):
        raise ValueError(f"{path} case records must use canonical case order")
    return cases


def _canonical_case_order(
    cases: Iterable[EvaluationCase],
) -> tuple[EvaluationCase, ...]:
    return tuple(
        sorted(
            cases,
            key=lambda case: (
                case.case_id,
                canonical_json_bytes(case.model_dump(mode="json")),
            ),
        )
    )


def _load_dataset_manifest(
    manifest_path: Path,
    *,
    cases: tuple[EvaluationCase, ...],
) -> DatasetManifest:
    raw_bytes = _read_regular_file(manifest_path)
    payload = json.loads(raw_bytes)
    manifest = DatasetManifest.model_validate(payload)
    if raw_bytes != canonical_json_bytes(manifest):
        raise ValueError(f"{manifest_path} must use canonical JSON ordering")
    expected_manifest = build_private_manifest(
        cases, generated_at=manifest.generated_at
    )
    mismatches = verify_private_manifest_expectations(
        actual=manifest, expected=expected_manifest
    )
    if mismatches:
        mismatch = mismatches[0]
        raise ValueError(
            f"manifest.json mismatch at {mismatch.path}: {mismatch.message}"
        )
    return manifest


def _looks_like_tuning_bundle(bundle_root: Path) -> bool:
    return {path.name for path in bundle_root.iterdir()} == {
        "dev.json",
        "manifest.json",
        "train.json",
    }


def _case_payload(case: EvaluationCase) -> dict[str, object]:
    return case.model_dump(mode="json", exclude_computed_fields=True)


def _copy_regular_file(source: Path, destination: Path) -> None:
    destination.write_bytes(_read_regular_file(source))


def _write_json(
    path: Path,
    payload: DatasetManifest
    | ReviewLedger
    | EvaluationCase
    | Mapping[str, object]
    | list[object]
    | tuple[object, ...],
) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _write_success_json(payload: dict[str, object]) -> None:
    if "ok" not in payload:
        payload = {"ok": True, **payload}
    print(canonical_json_bytes(payload).decode("utf-8"))


def _write_error_json(exc: Exception) -> None:
    payload = {
        "error": {
            "code": "operation_failed",
            "message": str(exc),
            "type": exc.__class__.__name__,
        },
        "ok": False,
    }
    print(canonical_json_bytes(payload).decode("utf-8"), file=sys.stderr)


def _write_usage_error(parser: argparse.ArgumentParser, message: str) -> None:
    print(parser.format_usage().rstrip(), file=sys.stderr)
    print(f"{parser.prog}: error: {message}", file=sys.stderr)


def _require_directory(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{path} must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _require_parent_directory(path: Path) -> Path:
    parent = path.parent
    if not parent.exists():
        raise ValueError(f"output parent directory does not exist: {parent}")
    return _require_directory(parent, label="output parent directory")


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(file_descriptor)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{path} must not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{path} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(file_descriptor)


def _atomic_replace_directory(source_dir: Path, destination_dir: Path) -> None:
    if not destination_dir.exists():
        os.replace(source_dir, destination_dir)
        return
    backup_dir = destination_dir.with_name(
        f".{destination_dir.name}.backup.{uuid.uuid4().hex}"
    )
    os.replace(destination_dir, backup_dir)
    try:
        os.replace(source_dir, destination_dir)
    except Exception as exc:
        try:
            os.replace(backup_dir, destination_dir)
        except Exception as rollback_exc:
            raise _AtomicReplaceRollbackError(
                original_error=exc,
                rollback_error=rollback_exc,
                recoverable_backup_path=backup_dir,
            ) from rollback_exc
        raise
    shutil.rmtree(backup_dir)


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(root)
    for path in root.rglob("*"):
        if path.is_dir():
            _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - exercised via tests
        raise _CliUsageError(message)


if __name__ == "__main__":
    raise SystemExit(main())
