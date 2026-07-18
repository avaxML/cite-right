"""Train/dev-only tuning bundle creation and validation."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.manifest import SCHEMA_VERSION
from evaluation.review import ReviewLedger, assert_review_complete
from evaluation.schema import EvaluationCase

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
TUNING_BUNDLE_VERSION = "1.0.0"
_REQUIRED_FILES = frozenset({"dev.json", "manifest.json", "train.json"})
_ALLOWED_DATASET_DIRECTORIES = frozenset({".", "holdout", "review", "reviews", "sources"})
_ALLOWED_DATASET_FILES = frozenset(
    {
        "dev.json",
        "dev_reviews.json",
        "holdout.aesgcm",
        "holdout.json",
        "holdout.public.json",
        "holdout_public_key.pem",
        "holdout_reviews.json",
        "manifest.json",
        "provenance.json",
        "sources/real.json",
        "train.json",
    }
)


class TuningBundleManifest(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    bundle_version: str
    dataset_version: str
    schema_version: str
    train_case_count: int
    dev_case_count: int
    train_claim_count: int
    dev_claim_count: int
    train_source_count: int
    dev_source_count: int
    train_sha256: str
    dev_sha256: str
    dataset_manifest_sha256: str
    provenance_sha256: str
    source_catalog_sha256: str
    dev_review_ledger_sha256: str

    @field_validator(
        "train_case_count",
        "dev_case_count",
        "train_claim_count",
        "dev_claim_count",
        "train_source_count",
        "dev_source_count",
    )
    @classmethod
    def _validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("count fields must be non-negative")
        return value

    @field_validator(
        "train_sha256",
        "dev_sha256",
        "dataset_manifest_sha256",
        "provenance_sha256",
        "source_catalog_sha256",
        "dev_review_ledger_sha256",
    )
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("hash fields must be lowercase 64-character SHA-256 hex digests")
        return value

    @field_validator("bundle_version")
    @classmethod
    def _validate_bundle_version(cls, value: str) -> str:
        if value != TUNING_BUNDLE_VERSION:
            raise ValueError(f"bundle_version must be {TUNING_BUNDLE_VERSION!r}")
        return value

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        return value


@dataclass(frozen=True, slots=True)
class TuningBundle:
    root_dir: Path
    manifest: TuningBundleManifest
    train_cases: tuple[EvaluationCase, ...]
    dev_cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class WorkerLaunchSpec:
    command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]


def build_tuning_bundle(
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> TuningBundleManifest:
    dataset_root = _require_directory(Path(dataset_dir), label="dataset directory")
    output_path = Path(output_dir)
    _reject_unsafe_output_location(dataset_root, output_path)
    _validate_dataset_tree(dataset_root)

    train_input_bytes = _read_canonical_file(dataset_root / "train.json")
    dev_input_bytes = _read_canonical_file(dataset_root / "dev.json")
    dataset_manifest_bytes = _read_canonical_file(dataset_root / "manifest.json")
    provenance_bytes = _read_canonical_file(dataset_root / "provenance.json")
    source_catalog_bytes = _read_canonical_file(dataset_root / "sources" / "real.json")
    dev_review_bytes = _read_canonical_file(dataset_root / "dev_reviews.json")
    dev_ledger = _load_review_ledger_bytes(
        dev_review_bytes,
        artifact_name="dev_reviews.json",
    )

    train_cases = _load_cases_from_bytes(
        train_input_bytes,
        artifact_name="train.json",
        expected_split="train",
        reject_review_metadata=False,
    )
    dev_cases = _load_cases_from_bytes(
        dev_input_bytes,
        artifact_name="dev.json",
        expected_split="dev",
        reject_review_metadata=False,
    )
    all_cases = train_cases + dev_cases
    _validate_unique_case_ids(all_cases)
    redacted_train_cases = _redact_cases(train_cases)
    redacted_dev_cases = _redact_cases(dev_cases)
    assert_review_complete(dev_cases, dev_ledger, split="dev")

    dataset_version = _shared_dataset_version(all_cases)
    if dev_ledger.dataset_version != dataset_version:
        raise ValueError("dev review incomplete: dev review ledger dataset_version mismatch")

    train_bytes = _canonical_cases_bytes(redacted_train_cases)
    dev_bytes = _canonical_cases_bytes(redacted_dev_cases)
    manifest = TuningBundleManifest(
        bundle_version=TUNING_BUNDLE_VERSION,
        dataset_version=dataset_version,
        schema_version=SCHEMA_VERSION,
        train_case_count=len(redacted_train_cases),
        dev_case_count=len(redacted_dev_cases),
        train_claim_count=_claim_count(redacted_train_cases),
        dev_claim_count=_claim_count(redacted_dev_cases),
        train_source_count=_source_count(redacted_train_cases),
        dev_source_count=_source_count(redacted_dev_cases),
        train_sha256=sha256_hex(train_bytes),
        dev_sha256=sha256_hex(dev_bytes),
        dataset_manifest_sha256=sha256_hex(dataset_manifest_bytes),
        provenance_sha256=sha256_hex(provenance_bytes),
        source_catalog_sha256=sha256_hex(source_catalog_bytes),
        dev_review_ledger_sha256=sha256_hex(dev_review_bytes),
    )

    if output_path.exists():
        raise FileExistsError(f"tuning bundle output already exists: {output_path}")

    parent = _require_parent_directory(output_path)
    temp_dir = Path(tempfile.mkdtemp(dir=parent, prefix=f".{output_path.name}.tmp."))
    published = False
    try:
        os.chmod(temp_dir, 0o700)
        _write_bundle_file(temp_dir / "train.json", train_bytes)
        _write_bundle_file(temp_dir / "dev.json", dev_bytes)
        _write_bundle_file(temp_dir / "manifest.json", canonical_json_bytes(manifest))
        _fsync_tree(temp_dir)
        os.replace(temp_dir, output_path)
        published = True
        _fsync_directory(parent)
    except Exception:
        if published and output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def load_tuning_bundle(bundle_dir: str | Path) -> TuningBundle:
    bundle_root = _require_directory(Path(bundle_dir), label="tuning bundle")
    entries = _list_directory_entries(bundle_root)
    names = {path.name for path in entries}
    unexpected = sorted(path.name for path in entries if path.name not in _REQUIRED_FILES)
    if unexpected:
        raise ValueError(f"unexpected files in tuning bundle: {', '.join(unexpected)}")
    if names != _REQUIRED_FILES:
        missing = ", ".join(sorted(_REQUIRED_FILES - names))
        raise ValueError(f"tuning bundle is missing required artifacts: {missing}")

    manifest_bytes = _read_canonical_file(bundle_root / "manifest.json")
    try:
        manifest = TuningBundleManifest.model_validate_json(manifest_bytes)
    except ValidationError as exc:
        raise ValueError("tuning bundle manifest.json is invalid") from exc

    train_bytes = _read_canonical_file(bundle_root / "train.json")
    dev_bytes = _read_canonical_file(bundle_root / "dev.json")
    train_cases = _load_cases_from_bytes(
        train_bytes,
        artifact_name="train.json",
        expected_split="train",
        reject_review_metadata=True,
    )
    dev_cases = _load_cases_from_bytes(
        dev_bytes,
        artifact_name="dev.json",
        expected_split="dev",
        reject_review_metadata=True,
    )
    all_cases = train_cases + dev_cases
    _validate_unique_case_ids(all_cases)

    actual_train_sha = sha256_hex(train_bytes)
    actual_dev_sha = sha256_hex(dev_bytes)
    if manifest.train_sha256 != actual_train_sha:
        raise ValueError("train.json hash mismatch")
    if manifest.dev_sha256 != actual_dev_sha:
        raise ValueError("dev.json hash mismatch")

    bundle_dataset_version = _shared_dataset_version(all_cases)
    if manifest.dataset_version != bundle_dataset_version:
        raise ValueError("tuning bundle dataset_version does not match its case files")
    if len(train_cases) != manifest.train_case_count:
        raise ValueError("tuning bundle train_case_count does not match train.json")
    if len(dev_cases) != manifest.dev_case_count:
        raise ValueError("tuning bundle dev_case_count does not match dev.json")
    if _claim_count(train_cases) != manifest.train_claim_count:
        raise ValueError("tuning bundle train_claim_count does not match train.json")
    if _claim_count(dev_cases) != manifest.dev_claim_count:
        raise ValueError("tuning bundle dev_claim_count does not match dev.json")
    if _source_count(train_cases) != manifest.train_source_count:
        raise ValueError("tuning bundle train_source_count does not match train.json")
    if _source_count(dev_cases) != manifest.dev_source_count:
        raise ValueError("tuning bundle dev_source_count does not match dev.json")

    return TuningBundle(
        root_dir=bundle_root,
        manifest=manifest,
        train_cases=train_cases,
        dev_cases=dev_cases,
    )


def worker_launch_spec(
    bundle_dir: str | Path,
    *,
    base_env: Mapping[str, str] | None = None,
) -> WorkerLaunchSpec:
    env = dict(os.environ if base_env is None else base_env)
    env.pop("CITE_RIGHT_ATTESTATION_KEY_FILE", None)
    env.pop("CITE_RIGHT_HOLDOUT_KEY_FILE", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    return WorkerLaunchSpec(
        command=(sys.executable, "-m", "evaluation.worker"),
        cwd=Path(bundle_dir),
        env=env,
    )


def _shared_dataset_version(cases: tuple[EvaluationCase, ...]) -> str:
    dataset_versions = {case.dataset_version for case in cases}
    if len(dataset_versions) != 1:
        raise ValueError("tuning bundle cases must share one dataset_version")
    return next(iter(dataset_versions))


def _load_cases_from_bytes(
    payload: bytes,
    *,
    artifact_name: str,
    expected_split: str,
    reject_review_metadata: bool,
) -> tuple[EvaluationCase, ...]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_name} is not valid JSON") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{artifact_name} must be a canonical JSON array")
    if not raw:
        raise ValueError(f"{artifact_name} must not be empty")

    cases = tuple(
        EvaluationCase.model_validate_json(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for item in raw
    )
    if any(case.split != expected_split for case in cases):
        raise ValueError(f"{artifact_name} must contain only {expected_split} cases")
    if reject_review_metadata and any(case.review is not None for case in cases):
        raise ValueError(f"{artifact_name} must not include review metadata")
    return cases


def _load_review_ledger_bytes(payload: bytes, *, artifact_name: str) -> ReviewLedger:
    try:
        return ReviewLedger.model_validate_json(payload)
    except ValidationError as exc:
        raise ValueError(f"{artifact_name} is invalid") from exc


def _redact_cases(cases: tuple[EvaluationCase, ...]) -> tuple[EvaluationCase, ...]:
    return tuple(case.model_copy(update={"review": None}) for case in cases)


def _validate_unique_case_ids(cases: tuple[EvaluationCase, ...]) -> None:
    counts = Counter(case.case_id for case in cases)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate case id {duplicates[0]!r}")


def _canonical_cases_bytes(cases: tuple[EvaluationCase, ...]) -> bytes:
    return canonical_json_bytes(
        [case.model_dump(mode="json", exclude_computed_fields=True) for case in cases]
    )


def _claim_count(cases: tuple[EvaluationCase, ...]) -> int:
    return sum(len(unit.claims) for case in cases for unit in case.evaluation_units)


def _source_count(cases: tuple[EvaluationCase, ...]) -> int:
    return sum(len(case.sources) for case in cases)


def _validate_dataset_tree(dataset_root: Path) -> None:
    for path in _iter_tree(dataset_root):
        relative = path.relative_to(dataset_root)
        relative_name = "." if relative == Path(".") else relative.as_posix()
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{path} must not be a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            if relative_name in _ALLOWED_DATASET_FILES:
                raise ValueError(f"{path} must be a regular file")
            if relative_name not in _ALLOWED_DATASET_DIRECTORIES:
                raise ValueError(f"unknown dataset artifact: {relative_name}")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{path} must be a regular file")
        if relative_name not in _ALLOWED_DATASET_FILES:
            raise ValueError(f"unknown dataset artifact: {relative_name}")


def _iter_tree(root: Path) -> tuple[Path, ...]:
    paths = [root]
    for path in sorted(root.rglob("*")):
        paths.append(path)
    return tuple(paths)


def _reject_unsafe_output_location(dataset_root: Path, output_path: Path) -> None:
    _assert_existing_ancestors_are_not_symlinks(output_path)
    resolved_output = output_path.resolve(strict=False)
    forbidden_roots = tuple(
        (dataset_root / name).resolve(strict=False) for name in ("holdout", "review", "reviews")
    )
    for forbidden_root in forbidden_roots:
        if resolved_output == forbidden_root or _is_relative_to(resolved_output, forbidden_root):
            raise ValueError(
                "tuning bundle output must not be nested inside dataset holdout or review directories"
            )


def _require_directory(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{path} must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _require_parent_directory(path: Path) -> Path:
    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError(f"output parent directory does not exist: {parent}")
    return _require_directory(parent, label="output parent directory")


def _write_bundle_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    os.chmod(path, 0o600)


def _read_canonical_file(path: Path) -> bytes:
    raw_bytes = _read_regular_file(path)
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if raw_bytes != canonical_json_bytes(payload):
        raise ValueError(f"{path} must use canonical JSON ordering")
    return raw_bytes


def _read_regular_file(path: Path) -> bytes:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{path} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{path} must be a regular file")
    flags = os.O_RDONLY
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow:
        flags |= no_follow
    file_descriptor = os.open(path, flags)
    try:
        reopened = os.fstat(file_descriptor)
        if not stat.S_ISREG(reopened.st_mode):
            raise ValueError(f"{path} must be a regular file")
        if (metadata.st_dev, metadata.st_ino) != (reopened.st_dev, reopened.st_ino):
            raise ValueError(f"{path} changed while being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 8192)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(file_descriptor)
    return b"".join(chunks)


def _list_directory_entries(root: Path) -> tuple[Path, ...]:
    entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    for path in entries:
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{path} must not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"unexpected files in tuning bundle: {path.name}")
    return entries


def _assert_existing_ancestors_are_not_symlinks(path: Path) -> None:
    current = path.resolve(strict=False)
    while True:
        if current.exists():
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{current} must not be a symlink")
        if current == current.parent:
            return
        current = current.parent


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _fsync_tree(root: Path) -> None:
    for path in root.iterdir():
        metadata = os.lstat(path)
        if stat.S_ISREG(metadata.st_mode):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


__all__ = [
    "TUNING_BUNDLE_VERSION",
    "TuningBundle",
    "TuningBundleManifest",
    "WorkerLaunchSpec",
    "build_tuning_bundle",
    "load_tuning_bundle",
    "worker_launch_spec",
]
