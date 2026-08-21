"""Seal holdout evaluation labels while preserving public verification."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.manifest import (
    SCHEMA_VERSION,
    PublicHoldoutManifest,
    build_private_manifest,
    build_public_holdout_manifest,
)
from evaluation.review import ReviewLedger, assert_review_complete, review_completion
from evaluation.schema import EvaluationCase

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
SEALING_FORMAT_VERSION = "1.0.0"
SEALING_ALGORITHM = "AES-256-GCM"
AES256_KEY_BYTES = 32
AESGCM_NONCE_BYTES = 12
DEFAULT_PUBLIC_KEY_PATH = Path("evaluation/data/v1/holdout_public_key.pem")


class ArtifactPublicationRollbackError(RuntimeError):
    def __init__(
        self,
        *,
        original_error: BaseException,
        rollback_error: BaseException,
        recoverable_backup_paths: tuple[str, ...],
    ) -> None:
        self.original_error = original_error
        self.rollback_error = rollback_error
        self.recoverable_backup_paths = recoverable_backup_paths
        backup_display = ", ".join(recoverable_backup_paths) or "none"
        super().__init__(
            "holdout artifact publication failed and rollback was incomplete; "
            f"publication_error={original_error}; rollback_error={rollback_error}; "
            f"recoverable_backups={backup_display}"
        )


class SealedHoldoutEnvelope(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    format_version: str
    algorithm: str
    dataset_version: str
    schema_version: str
    nonce_b64: str
    ciphertext_b64: str

    @field_validator("format_version")
    @classmethod
    def _validate_format_version(cls, value: str) -> str:
        if value != SEALING_FORMAT_VERSION:
            raise ValueError(f"format_version must be {SEALING_FORMAT_VERSION!r}")
        return value

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, value: str) -> str:
        if value != SEALING_ALGORITHM:
            raise ValueError(f"algorithm must be {SEALING_ALGORITHM!r}")
        return value

    @field_validator("nonce_b64")
    @classmethod
    def _validate_nonce_b64(cls, value: str) -> str:
        decoded = _decode_canonical_base64(value, field_name="nonce_b64")
        if len(decoded) != AESGCM_NONCE_BYTES:
            raise ValueError(
                f"nonce_b64 must encode exactly {AESGCM_NONCE_BYTES} bytes"
            )
        return value

    @field_validator("ciphertext_b64")
    @classmethod
    def _validate_ciphertext_b64(cls, value: str) -> str:
        decoded = _decode_canonical_base64(value, field_name="ciphertext_b64")
        if not decoded:
            raise ValueError("ciphertext_b64 must not be empty")
        return value


def seal_holdout(
    cases: Iterable[EvaluationCase],
    *,
    ledger: ReviewLedger,
    ciphertext_path: str | Path,
    public_manifest_path: str | Path,
    public_key_path: str | Path = DEFAULT_PUBLIC_KEY_PATH,
    generated_at: str | None = None,
) -> PublicHoldoutManifest:
    holdout_cases = _validated_holdout_cases(cases)
    if ledger.dataset_version != holdout_cases[0].dataset_version:
        raise ValueError("ledger dataset_version does not match the provided cases")

    assert_review_complete(holdout_cases, ledger, split="holdout")
    completion = review_completion(holdout_cases, ledger, splits=("holdout",))
    if completion.approved_claims != completion.total_claims:
        raise ValueError(
            "holdout review incomplete: all holdout claims must have current approve decisions"
        )
    if completion.reviewed_claims != completion.total_claims:
        raise ValueError(
            "holdout review incomplete: all holdout claims must be reviewed before sealing"
        )

    private_manifest = build_private_manifest(holdout_cases, generated_at=generated_at)
    holdout_key = load_holdout_key_from_env()
    attestation_private_key = load_attestation_private_key_from_env()
    committed_public_key = load_public_attestation_key(public_key_path)
    derived_public_key = attestation_private_key.public_key()
    if _public_key_fingerprint(committed_public_key) != _public_key_fingerprint(
        derived_public_key
    ):
        raise ValueError("public key does not match the attestation private key")

    try:
        plaintext = canonical_json_bytes(
            {
                "cases": [
                    case.model_dump(mode="json", exclude_computed_fields=True)
                    for case in holdout_cases
                ]
            }
        )
        nonce = os.urandom(AESGCM_NONCE_BYTES)
        aad = _aad_bytes(
            dataset_version=private_manifest.dataset_version,
            schema_version=private_manifest.schema_version,
        )
        ciphertext = AESGCM(holdout_key).encrypt(nonce, plaintext, aad)
    finally:
        _zero_bytes(holdout_key)

    envelope = SealedHoldoutEnvelope(
        format_version=SEALING_FORMAT_VERSION,
        algorithm=SEALING_ALGORITHM,
        dataset_version=private_manifest.dataset_version,
        schema_version=private_manifest.schema_version,
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
    )
    ciphertext_bytes = canonical_json_bytes(envelope)
    ciphertext_sha256 = sha256_hex(ciphertext_bytes)
    public_key_fingerprint = _public_key_fingerprint(committed_public_key)
    unsigned_manifest = build_public_holdout_manifest(
        private_manifest,
        ciphertext_sha256=ciphertext_sha256,
        total_claim_count=completion.total_claims,
        reviewed_claim_count=completion.approved_claims,
        public_key_fingerprint=public_key_fingerprint,
        signature=None,
    )
    signature = base64.b64encode(
        attestation_private_key.sign(_public_attestation_bytes(unsigned_manifest))
    ).decode("ascii")
    signed_manifest = unsigned_manifest.model_copy(update={"signature": signature})
    public_manifest_bytes = canonical_json_bytes(signed_manifest)

    _write_artifacts_atomically(
        ciphertext_path=Path(ciphertext_path),
        ciphertext_bytes=ciphertext_bytes,
        public_manifest_path=Path(public_manifest_path),
        public_manifest_bytes=public_manifest_bytes,
    )
    return signed_manifest


def unseal_holdout(ciphertext_path: str | Path) -> tuple[EvaluationCase, ...]:
    envelope_path = Path(ciphertext_path)
    raw_bytes = envelope_path.read_bytes()
    envelope = _load_canonical_envelope(envelope_path, raw_bytes)
    holdout_key = load_holdout_key_from_env()
    try:
        aad = _aad_bytes(
            dataset_version=envelope.dataset_version,
            schema_version=envelope.schema_version,
        )
        try:
            plaintext = AESGCM(holdout_key).decrypt(
                _decode_canonical_base64(envelope.nonce_b64, field_name="nonce_b64"),
                _decode_canonical_base64(
                    envelope.ciphertext_b64, field_name="ciphertext_b64"
                ),
                aad,
            )
        except Exception as exc:
            raise ValueError(
                "failed authentication while unsealing holdout dataset"
            ) from exc
    finally:
        _zero_bytes(holdout_key)

    try:
        payload = json.loads(plaintext)
    except json.JSONDecodeError as exc:
        raise ValueError("sealed holdout payload is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("sealed holdout payload must be a JSON object")
    cases_payload = payload.get("cases")
    if not isinstance(cases_payload, list):
        raise ValueError("sealed holdout payload must contain a cases array")
    cases = tuple(
        EvaluationCase.model_validate_json(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for item in cases_payload
    )
    return _validated_holdout_cases(cases)


def verify_public_manifest(
    manifest_path: str | Path,
    *,
    ciphertext_path: str | Path,
    public_key_path: str | Path = DEFAULT_PUBLIC_KEY_PATH,
) -> PublicHoldoutManifest:
    manifest_file = Path(manifest_path)
    raw_bytes = manifest_file.read_bytes()
    manifest = _load_canonical_public_manifest(manifest_file, raw_bytes)
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"public manifest schema_version must be {SCHEMA_VERSION!r}")
    if manifest.reviewed_claim_count != manifest.total_claim_count:
        raise ValueError(
            "public manifest reviewed_claim_count must equal total_claim_count"
        )

    ciphertext_bytes = Path(ciphertext_path).read_bytes()
    actual_ciphertext_sha256 = sha256_hex(ciphertext_bytes)
    if manifest.ciphertext_sha256 != actual_ciphertext_sha256:
        raise ValueError(
            "public manifest ciphertext_sha256 does not match the sealed ciphertext"
        )

    public_key = load_public_attestation_key(public_key_path)
    actual_fingerprint = _public_key_fingerprint(public_key)
    if manifest.public_key_fingerprint != actual_fingerprint:
        raise ValueError(
            "public manifest public_key_fingerprint does not match the provided public key"
        )
    if manifest.signature is None:
        raise ValueError("public manifest signature is required")

    signature = _decode_canonical_base64(manifest.signature, field_name="signature")
    try:
        public_key.verify(signature, _public_attestation_bytes(manifest))
    except InvalidSignature as exc:
        raise ValueError("public manifest signature verification failed") from exc

    envelope = _load_canonical_envelope(Path(ciphertext_path), ciphertext_bytes)
    if envelope.dataset_version != manifest.dataset_version:
        raise ValueError(
            "sealed holdout envelope dataset_version does not match the signed public manifest"
        )
    if envelope.schema_version != manifest.schema_version:
        raise ValueError(
            "sealed holdout envelope schema_version does not match the signed public manifest"
        )
    return manifest


def load_holdout_key_from_env(
    env_var: str = "CITE_RIGHT_HOLDOUT_KEY_FILE",
) -> bytearray:
    key_bytes = _read_protected_file_bytes(env_var)
    if len(key_bytes) != AES256_KEY_BYTES:
        raise ValueError("holdout key must be exactly 32 bytes")
    return key_bytes


def load_attestation_private_key_from_env(
    env_var: str = "CITE_RIGHT_ATTESTATION_KEY_FILE",
) -> Ed25519PrivateKey:
    key_bytes = _read_protected_file_bytes(env_var)
    try:
        private_key = serialization.load_pem_private_key(
            bytes(key_bytes), password=None
        )
    except ValueError as exc:
        raise ValueError("attestation private key must be a valid Ed25519 PEM") from exc
    finally:
        _zero_bytes(key_bytes)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("attestation private key must be a valid Ed25519 PEM")
    return private_key


def load_public_attestation_key(path: str | Path) -> Ed25519PublicKey:
    raw_bytes = _read_file_bytes(Path(path), require_private_permissions=False)
    try:
        public_key = serialization.load_pem_public_key(bytes(raw_bytes))
    except ValueError as exc:
        raise ValueError("public attestation key must be a valid Ed25519 PEM") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("public attestation key must be a valid Ed25519 PEM")
    return public_key


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.sealing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cases", required=True)
    seal_parser.add_argument("--ledger", required=True)
    seal_parser.add_argument("--ciphertext-output", required=True)
    seal_parser.add_argument("--public-manifest-output", required=True)
    seal_parser.add_argument("--public-key", default=str(DEFAULT_PUBLIC_KEY_PATH))
    seal_parser.add_argument("--generated-at")

    verify_parser = subparsers.add_parser("verify-public-manifest")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--ciphertext", required=True)
    verify_parser.add_argument("--public-key", default=str(DEFAULT_PUBLIC_KEY_PATH))

    args = parser.parse_args(argv)
    if args.command == "seal":
        cases = _load_cases_file(Path(args.cases))
        ledger = _load_review_ledger(Path(args.ledger))
        seal_holdout(
            cases,
            ledger=ledger,
            ciphertext_path=args.ciphertext_output,
            public_manifest_path=args.public_manifest_output,
            public_key_path=args.public_key,
            generated_at=args.generated_at,
        )
        return 0
    if args.command == "verify-public-manifest":
        verify_public_manifest(
            args.manifest,
            ciphertext_path=args.ciphertext,
            public_key_path=args.public_key,
        )
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


def _validated_holdout_cases(
    cases: Iterable[EvaluationCase],
) -> tuple[EvaluationCase, ...]:
    ordered_cases = tuple(cases)
    if not ordered_cases:
        raise ValueError("cases must not be empty")
    if any(case.split != "holdout" for case in ordered_cases):
        raise ValueError("holdout sealing requires every case to use the holdout split")
    build_private_manifest(ordered_cases)
    return ordered_cases


def _load_cases_file(path: Path) -> tuple[EvaluationCase, ...]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, list):
        raise ValueError("cases file must be a canonical JSON array")
    return tuple(
        EvaluationCase.model_validate_json(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for item in payload
    )


def _load_review_ledger(path: Path) -> ReviewLedger:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"review ledger {path} is not valid JSON") from exc
    try:
        ledger = ReviewLedger.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"review ledger {path} is invalid") from exc
    if raw_bytes != canonical_json_bytes(ledger):
        raise ValueError(f"review ledger {path} must use canonical JSON ordering")
    return ledger


def _aad_bytes(*, dataset_version: str, schema_version: str) -> bytes:
    return canonical_json_bytes(
        {
            "dataset_version": dataset_version,
            "format_version": SEALING_FORMAT_VERSION,
            "schema_version": schema_version,
        }
    )


def _public_attestation_bytes(manifest: PublicHoldoutManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    payload.pop("signature", None)
    return canonical_json_bytes(payload)


def _public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    return sha256_hex(
        public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def _read_protected_file_bytes(env_var: str) -> bytearray:
    path_value = os.environ.get(env_var)
    if path_value is None or not path_value.strip():
        raise ValueError(f"{env_var} must be set to a key file path")
    return _read_file_bytes(Path(path_value), require_private_permissions=True)


def _read_file_bytes(path: Path, *, require_private_permissions: bool) -> bytearray:
    try:
        before_stat = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if stat.S_ISLNK(before_stat.st_mode):
        raise ValueError(f"{path} must not be a symlink")
    if not stat.S_ISREG(before_stat.st_mode):
        raise ValueError(f"{path} must be a regular file")
    if require_private_permissions and os.name == "posix":
        if before_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(f"{path} must not be group/world-readable")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    fd = os.open(path, flags)
    try:
        after_stat = os.fstat(fd)
        if not stat.S_ISREG(after_stat.st_mode):
            raise ValueError(f"{path} must be a regular file")
        if (before_stat.st_dev, before_stat.st_ino) != (
            after_stat.st_dev,
            after_stat.st_ino,
        ):
            raise ValueError(f"{path} changed while being opened")
        if require_private_permissions and os.name == "posix":
            if after_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise PermissionError(f"{path} must not be group/world-readable")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    return bytearray(b"".join(chunks))


def _write_artifacts_atomically(
    *,
    ciphertext_path: Path,
    ciphertext_bytes: bytes,
    public_manifest_path: Path,
    public_manifest_bytes: bytes,
) -> None:
    temp_paths: list[str] = []
    backup_paths: list[Path] = []
    rollback_paths: list[tuple[Path, Path | None]] = []
    cleanup_backups = False
    parents = {ciphertext_path.parent, public_manifest_path.parent}
    for parent in parents:
        if not parent.exists():
            raise FileNotFoundError(
                f"artifact parent directory does not exist: {parent}"
            )
    try:
        ciphertext_temp = _write_temp_file(
            ciphertext_path.parent,
            ciphertext_path.name,
            ciphertext_bytes,
        )
        temp_paths.append(ciphertext_temp)
        manifest_temp = _write_temp_file(
            public_manifest_path.parent,
            public_manifest_path.name,
            public_manifest_bytes,
        )
        temp_paths.append(manifest_temp)

        ciphertext_backup = _backup_existing_path(ciphertext_path)
        if ciphertext_backup is not None:
            backup_paths.append(ciphertext_backup)
        rollback_paths.append((ciphertext_path, ciphertext_backup))
        os.replace(ciphertext_temp, ciphertext_path)
        _fsync_directory(ciphertext_path.parent)

        manifest_backup = _backup_existing_path(public_manifest_path)
        if manifest_backup is not None:
            backup_paths.append(manifest_backup)
        rollback_paths.append((public_manifest_path, manifest_backup))
        os.replace(manifest_temp, public_manifest_path)
        _fsync_directory(public_manifest_path.parent)
        cleanup_backups = True
    except Exception as original_exc:
        rollback_parent_order: list[Path] = []
        rollback_parent_seen: set[Path] = set()
        try:
            for target_path, backup_path in reversed(rollback_paths):
                rollback_parent = target_path.parent
                if backup_path is None:
                    try:
                        target_path.unlink()
                    except FileNotFoundError:
                        continue
                    if rollback_parent not in rollback_parent_seen:
                        rollback_parent_seen.add(rollback_parent)
                        rollback_parent_order.append(rollback_parent)
                    continue
                try:
                    if backup_path.exists():
                        os.replace(backup_path, target_path)
                        if rollback_parent not in rollback_parent_seen:
                            rollback_parent_seen.add(rollback_parent)
                            rollback_parent_order.append(rollback_parent)
                except FileNotFoundError:
                    continue
            for rollback_parent in rollback_parent_order:
                _fsync_directory(rollback_parent)
            cleanup_backups = True
        except Exception as rollback_exc:
            recoverable_backup_paths = tuple(
                str(path) for path in backup_paths if path.exists()
            )
            raise ArtifactPublicationRollbackError(
                original_error=original_exc,
                rollback_error=rollback_exc,
                recoverable_backup_paths=recoverable_backup_paths,
            ) from rollback_exc
        raise
    finally:
        for temp_path in temp_paths:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
        if cleanup_backups:
            for backup_path in backup_paths:
                try:
                    backup_path.unlink()
                except FileNotFoundError:
                    pass


def _backup_existing_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_fd, backup_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.bak."
    )
    os.close(backup_fd)
    os.replace(path, backup_name)
    return Path(backup_name)


def _write_temp_file(parent: Path, name: str, payload: bytes) -> str:
    temp_fd, temp_name = tempfile.mkstemp(dir=parent, prefix=f".{name}.tmp.")
    with os.fdopen(temp_fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return temp_name


def _load_canonical_envelope(path: Path, raw_bytes: bytes) -> SealedHoldoutEnvelope:
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"sealed holdout envelope {path} is not valid JSON") from exc
    try:
        envelope = SealedHoldoutEnvelope.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"sealed holdout envelope {path} is invalid: {exc}") from exc
    if raw_bytes != canonical_json_bytes(envelope):
        raise ValueError(
            f"sealed holdout envelope {path} must use canonical JSON ordering"
        )
    return envelope


def _load_canonical_public_manifest(
    path: Path, raw_bytes: bytes
) -> PublicHoldoutManifest:
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"public holdout manifest {path} is not valid JSON") from exc
    try:
        manifest = PublicHoldoutManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"public holdout manifest {path} is invalid: {exc}") from exc
    if raw_bytes != canonical_json_bytes(manifest):
        raise ValueError(
            f"public holdout manifest {path} must use canonical JSON ordering"
        )
    return manifest


def _decode_canonical_base64(value: str, *, field_name: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"{field_name} must be canonical base64") from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{field_name} must be canonical base64")
    return decoded


def _zero_bytes(value: bytearray) -> None:
    value[:] = b"\x00" * len(value)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


if __name__ == "__main__":
    raise SystemExit(main())
