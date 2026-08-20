from __future__ import annotations

import base64
import json
import os
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Literal

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.review import ReviewLedger, make_review_record
from evaluation.schema import EvaluationCase


@pytest.mark.parametrize("nonce_count", [2])
def test_seal_holdout_round_trips_and_uses_fresh_nonces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nonce_count: int,
) -> None:
    from evaluation.sealing import seal_holdout, unseal_holdout, verify_public_manifest

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    envelopes: list[bytes] = []
    manifests: list[bytes] = []
    for index in range(nonce_count):
        ciphertext_path = tmp_path / f"holdout-{index}.sealed.json"
        manifest_path = tmp_path / f"holdout-{index}.public.json"
        sealed_manifest = seal_holdout(
            cases,
            ledger=ledger,
            ciphertext_path=ciphertext_path,
            public_manifest_path=manifest_path,
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

        verified_manifest = verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )
        recovered_cases = unseal_holdout(ciphertext_path)

        assert tuple(case.model_dump(mode="json") for case in recovered_cases) == tuple(
            case.model_dump(mode="json") for case in cases
        )
        assert verified_manifest == sealed_manifest
        assert sealed_manifest.total_claim_count == 2
        assert sealed_manifest.reviewed_claim_count == 2
        envelopes.append(ciphertext_path.read_bytes())
        manifests.append(manifest_path.read_bytes())

    assert len(set(envelopes)) == nonce_count
    assert len(set(manifests)) == nonce_count


def test_unseal_holdout_rejects_ciphertext_tampering_wrong_key_and_aad_version_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.sealing import seal_holdout, unseal_holdout

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    ciphertext_path = tmp_path / "holdout.sealed.json"
    manifest_path = tmp_path / "holdout.public.json"
    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )

    tampered_payload = json.loads(ciphertext_path.read_text(encoding="utf-8"))
    tampered_payload["ciphertext_b64"] = (
        tampered_payload["ciphertext_b64"][:-4] + "AAAA"
    )
    ciphertext_path.write_bytes(canonical_json_bytes(tampered_payload))
    with pytest.raises(ValueError, match="failed authentication"):
        unseal_holdout(ciphertext_path)

    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )
    wrong_key_path = tmp_path / "wrong-holdout.key"
    wrong_key_path.write_bytes(bytes(reversed(range(32))))
    os.chmod(wrong_key_path, stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv("CITE_RIGHT_HOLDOUT_KEY_FILE", str(wrong_key_path))
    with pytest.raises(ValueError, match="failed authentication"):
        unseal_holdout(ciphertext_path)

    monkeypatch.setenv("CITE_RIGHT_HOLDOUT_KEY_FILE", str(tmp_path / "holdout.key"))
    aad_tampered_payload = json.loads(ciphertext_path.read_text(encoding="utf-8"))
    aad_tampered_payload["dataset_version"] = "9.9.9"
    ciphertext_path.write_bytes(canonical_json_bytes(aad_tampered_payload))
    with pytest.raises(ValueError, match="failed authentication"):
        unseal_holdout(ciphertext_path)


def test_verify_public_manifest_rejects_noncanonical_hash_signature_count_schema_and_fingerprint_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.sealing import seal_holdout, verify_public_manifest

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    ciphertext_path = tmp_path / "holdout.sealed.json"
    manifest_path = tmp_path / "holdout.public.json"
    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    (
        ("dataset_version", "9.9.9", "dataset_version"),
        ("schema_version", "9.9.9", "schema_version"),
    ),
)
def test_verify_public_manifest_rejects_envelope_manifest_version_mismatches_even_when_hash_and_signature_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    field_value: str,
    message: str,
) -> None:
    from evaluation.sealing import seal_holdout, verify_public_manifest

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    ciphertext_path = tmp_path / "holdout.sealed.json"
    manifest_path = tmp_path / "holdout.public.json"
    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )

    envelope_payload = json.loads(ciphertext_path.read_text(encoding="utf-8"))
    envelope_payload[field_name] = field_value
    ciphertext_path.write_bytes(canonical_json_bytes(envelope_payload))

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["ciphertext_sha256"] = sha256_hex(ciphertext_path.read_bytes())
    _resign_manifest_payload(tmp_path / "attestation-private.pem", manifest_payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest_payload))

    with pytest.raises(ValueError, match=message):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )


def test_verify_public_manifest_rejects_malformed_and_noncanonical_envelopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.sealing import seal_holdout, verify_public_manifest

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    ciphertext_path = tmp_path / "holdout.sealed.json"
    manifest_path = tmp_path / "holdout.public.json"
    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )

    ciphertext_path.write_text("[]", encoding="utf-8")
    malformed_manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    malformed_manifest_payload["ciphertext_sha256"] = sha256_hex(
        ciphertext_path.read_bytes()
    )
    _resign_manifest_payload(
        tmp_path / "attestation-private.pem", malformed_manifest_payload
    )
    manifest_path.write_bytes(canonical_json_bytes(malformed_manifest_payload))

    with pytest.raises(ValueError, match="sealed holdout envelope"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )

    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )
    envelope_payload = json.loads(ciphertext_path.read_text(encoding="utf-8"))
    ciphertext_path.write_text(
        json.dumps(envelope_payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    noncanonical_manifest_payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    noncanonical_manifest_payload["ciphertext_sha256"] = sha256_hex(
        ciphertext_path.read_bytes()
    )
    _resign_manifest_payload(
        tmp_path / "attestation-private.pem", noncanonical_manifest_payload
    )
    manifest_path.write_bytes(canonical_json_bytes(noncanonical_manifest_payload))

    with pytest.raises(ValueError, match="canonical JSON"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )

    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )
    tampered_ciphertext = json.loads(ciphertext_path.read_text(encoding="utf-8"))
    tampered_ciphertext["nonce_b64"] = tampered_ciphertext["nonce_b64"][:-4] + "AAAA"
    ciphertext_path.write_bytes(canonical_json_bytes(tampered_ciphertext))
    with pytest.raises(ValueError, match="ciphertext_sha256"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )

    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["reviewed_claim_count"] = payload["reviewed_claim_count"] - 1
    manifest_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="reviewed_claim_count"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )

    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "9.9.9"
    manifest_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="schema_version"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )

    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )
    alternate_public_key = tmp_path / "alternate-public.pem"
    alternate_public_key.write_bytes(
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    with pytest.raises(ValueError, match="public_key_fingerprint"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=alternate_public_key,
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    decoded_signature = bytearray(base64.b64decode(payload["signature"], validate=True))
    decoded_signature[0] ^= 0x01
    payload["signature"] = base64.b64encode(bytes(decoded_signature)).decode("ascii")
    manifest_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="signature"):
        verify_public_manifest(
            manifest_path,
            ciphertext_path=ciphertext_path,
            public_key_path=public_key_path,
        )


def test_key_loading_rejects_bad_lengths_formats_permissions_and_unsafe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.sealing import (
        load_attestation_private_key_from_env,
        load_holdout_key_from_env,
        load_public_attestation_key,
    )

    public_key_path = Path("evaluation/data/v1/holdout_public_key.pem")
    assert load_public_attestation_key(public_key_path) is not None

    holdout_key_path = tmp_path / "holdout.key"
    holdout_key_path.write_bytes(b"short")
    os.chmod(holdout_key_path, stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv("CITE_RIGHT_HOLDOUT_KEY_FILE", str(holdout_key_path))
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        load_holdout_key_from_env()

    holdout_key_path.write_bytes(bytes(range(32)))
    if os.name == "posix":
        os.chmod(holdout_key_path, 0o644)
        with pytest.raises(PermissionError, match="group/world-readable"):
            load_holdout_key_from_env()
        os.chmod(holdout_key_path, 0o600)

    holdout_directory = tmp_path / "not-a-file"
    holdout_directory.mkdir()
    monkeypatch.setenv("CITE_RIGHT_HOLDOUT_KEY_FILE", str(holdout_directory))
    with pytest.raises(ValueError, match="regular file"):
        load_holdout_key_from_env()

    holdout_symlink = tmp_path / "holdout-symlink.key"
    holdout_symlink.symlink_to(holdout_key_path)
    monkeypatch.setenv("CITE_RIGHT_HOLDOUT_KEY_FILE", str(holdout_symlink))
    with pytest.raises(ValueError, match="symlink"):
        load_holdout_key_from_env()

    private_key_path = tmp_path / "attestation-private.pem"
    private_key_path.write_text("not a pem", encoding="utf-8")
    os.chmod(private_key_path, stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv("CITE_RIGHT_ATTESTATION_KEY_FILE", str(private_key_path))
    with pytest.raises(ValueError, match="Ed25519"):
        load_attestation_private_key_from_env()

    private_key = Ed25519PrivateKey.generate()
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    if os.name == "posix":
        os.chmod(private_key_path, 0o644)
        with pytest.raises(PermissionError, match="group/world-readable"):
            load_attestation_private_key_from_env()


def test_seal_holdout_rejects_incomplete_stale_correct_reject_non_holdout_and_empty_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.sealing import seal_holdout

    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)
    holdout_cases = _build_holdout_cases()

    with pytest.raises(ValueError, match="cases must not be empty"):
        seal_holdout(
            (),
            ledger=ReviewLedger(
                dataset_version="1.0.0", schema_version="1.0.0", entries=()
            ),
            ciphertext_path=tmp_path / "empty.sealed.json",
            public_manifest_path=tmp_path / "empty.public.json",
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

    with pytest.raises(ValueError, match="holdout review incomplete"):
        seal_holdout(
            holdout_cases,
            ledger=ReviewLedger(
                dataset_version="1.0.0", schema_version="1.0.0", entries=()
            ),
            ciphertext_path=tmp_path / "missing.sealed.json",
            public_manifest_path=tmp_path / "missing.public.json",
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

    stale_cases = (
        holdout_cases[0].model_copy(
            update={
                "sources": (
                    holdout_cases[0]
                    .sources[0]
                    .model_copy(
                        update={"text": holdout_cases[0].sources[0].text + " changed"}
                    ),
                )
            }
        ),
        holdout_cases[1],
    )
    with pytest.raises(ValueError, match="holdout review incomplete"):
        seal_holdout(
            stale_cases,
            ledger=_approved_review_ledger(holdout_cases),
            ciphertext_path=tmp_path / "stale.sealed.json",
            public_manifest_path=tmp_path / "stale.public.json",
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

    with pytest.raises(ValueError, match="holdout review incomplete"):
        seal_holdout(
            holdout_cases,
            ledger=_decision_ledger(holdout_cases, "correct"),
            ciphertext_path=tmp_path / "correct.sealed.json",
            public_manifest_path=tmp_path / "correct.public.json",
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

    with pytest.raises(ValueError, match="holdout review incomplete"):
        seal_holdout(
            holdout_cases,
            ledger=_decision_ledger(holdout_cases, "reject"),
            ciphertext_path=tmp_path / "reject.sealed.json",
            public_manifest_path=tmp_path / "reject.public.json",
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

    mixed_cases = holdout_cases + (
        _build_case(case_id="case-train", family_id="family-train", split="train"),
    )
    with pytest.raises(ValueError, match="holdout split"):
        seal_holdout(
            mixed_cases,
            ledger=_approved_review_ledger(holdout_cases),
            ciphertext_path=tmp_path / "mixed.sealed.json",
            public_manifest_path=tmp_path / "mixed.public.json",
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )


def test_seal_holdout_redacts_public_manifest_and_avoids_partial_publication_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.sealing import seal_holdout

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    ciphertext_path = tmp_path / "holdout.sealed.json"
    manifest_path = tmp_path / "holdout.public.json"
    seal_holdout(
        cases,
        ledger=ledger,
        ciphertext_path=ciphertext_path,
        public_manifest_path=manifest_path,
        public_key_path=public_key_path,
        generated_at="2026-07-18",
    )

    envelope_payload = json.loads(ciphertext_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert set(envelope_payload) == {
        "algorithm",
        "ciphertext_b64",
        "dataset_version",
        "format_version",
        "nonce_b64",
        "schema_version",
    }
    assert set(manifest_payload) == {
        "ciphertext_sha256",
        "dataset_version",
        "distributions",
        "generated_at",
        "holdout_case_count",
        "public_key_fingerprint",
        "reviewed_claim_count",
        "schema_version",
        "signature",
        "total_claim_count",
    }
    manifest_text = json.dumps(manifest_payload, sort_keys=True)
    for forbidden in (
        cases[0].case_id,
        cases[0].answer,
        cases[0].sources[0].text,
        "reviewer-1",
        "Approved for sealing.",
    ):
        assert forbidden not in manifest_text

    mismatched_public_key_path = tmp_path / "mismatched-public.pem"
    mismatched_public_key_path.write_bytes(
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    failing_ciphertext_path = tmp_path / "failing.sealed.json"
    failing_manifest_path = tmp_path / "failing.public.json"
    with pytest.raises(ValueError, match="public key"):
        seal_holdout(
            cases,
            ledger=ledger,
            ciphertext_path=failing_ciphertext_path,
            public_manifest_path=failing_manifest_path,
            public_key_path=mismatched_public_key_path,
            generated_at="2026-07-18",
        )
    assert not failing_ciphertext_path.exists()
    assert not failing_manifest_path.exists()


@pytest.mark.parametrize("shared_parent", [True, False])
def test_seal_holdout_rolls_back_if_public_manifest_replace_fails_and_fsyncs_rollback_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_parent: bool,
) -> None:
    import evaluation.sealing as sealing

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    if shared_parent:
        ciphertext_dir = tmp_path
        manifest_dir = tmp_path
    else:
        ciphertext_dir = tmp_path / "ciphertext"
        manifest_dir = tmp_path / "manifest"
        ciphertext_dir.mkdir()
        manifest_dir.mkdir()

    ciphertext_path = ciphertext_dir / "holdout.sealed.json"
    manifest_path = manifest_dir / "holdout.public.json"
    ciphertext_path.write_bytes(b"ciphertext-old")
    manifest_path.write_bytes(b"manifest-old")

    original_replace = os.replace
    call_count = {"value": 0}
    fsync_calls: list[Path] = []
    original_fsync_directory = sealing._fsync_directory

    def failing_replace(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> None:
        call_count["value"] += 1
        if call_count["value"] == 4:
            raise RuntimeError("simulated manifest replace failure")
        original_replace(src, dst)

    def tracking_fsync_directory(path: Path) -> None:
        fsync_calls.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(os, "replace", failing_replace)
    monkeypatch.setattr(sealing, "_fsync_directory", tracking_fsync_directory)

    with pytest.raises(RuntimeError, match="manifest replace failure"):
        sealing.seal_holdout(
            cases,
            ledger=ledger,
            ciphertext_path=ciphertext_path,
            public_manifest_path=manifest_path,
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

    assert ciphertext_path.read_bytes() == b"ciphertext-old"
    assert manifest_path.read_bytes() == b"manifest-old"
    assert not any(path.name.startswith(".holdout") for path in tmp_path.rglob("*"))

    if shared_parent:
        assert fsync_calls == [tmp_path, tmp_path]
    else:
        assert fsync_calls[0] == ciphertext_dir
        assert fsync_calls[1:] == [manifest_dir, ciphertext_dir]


@pytest.mark.parametrize("shared_parent", [True, False])
def test_seal_holdout_preserves_backups_if_rollback_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_parent: bool,
) -> None:
    import evaluation.sealing as sealing

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)

    if shared_parent:
        ciphertext_dir = tmp_path
        manifest_dir = tmp_path
    else:
        ciphertext_dir = tmp_path / "ciphertext"
        manifest_dir = tmp_path / "manifest"
        ciphertext_dir.mkdir()
        manifest_dir.mkdir()

    ciphertext_path = ciphertext_dir / "holdout.sealed.json"
    manifest_path = manifest_dir / "holdout.public.json"
    ciphertext_path.write_bytes(b"ciphertext-old")
    manifest_path.write_bytes(b"manifest-old")

    original_replace = os.replace
    call_count = {"value": 0}

    def failing_replace(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> None:
        call_count["value"] += 1
        if call_count["value"] == 4:
            raise RuntimeError("simulated manifest replace failure")
        if call_count["value"] == 5:
            raise RuntimeError("simulated rollback restore failure")
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(sealing.ArtifactPublicationRollbackError) as exc_info:
        sealing.seal_holdout(
            cases,
            ledger=ledger,
            ciphertext_path=ciphertext_path,
            public_manifest_path=manifest_path,
            public_key_path=public_key_path,
            generated_at="2026-07-18",
        )

    assert "simulated manifest replace failure" in str(exc_info.value)
    assert "simulated rollback restore failure" in str(exc_info.value)
    assert ".bak." in str(exc_info.value)
    assert exc_info.value.recoverable_backup_paths

    backup_paths = tuple(Path(path) for path in exc_info.value.recoverable_backup_paths)
    assert all(path.exists() for path in backup_paths)
    assert ciphertext_path.read_bytes() != b"ciphertext-old"
    assert not manifest_path.exists()
    for backup_path in backup_paths:
        backup_bytes = backup_path.read_bytes()
        assert backup_bytes in {b"ciphertext-old", b"manifest-old"}
    assert not any(".tmp." in path.name for path in tmp_path.rglob("*"))


def test_sealing_cli_uses_env_keys_and_verifies_public_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.sealing import load_public_attestation_key

    cases = _build_holdout_cases()
    ledger = _approved_review_ledger(cases)
    public_key_path = tmp_path / "attestation-public.pem"
    _install_signing_keys(tmp_path, monkeypatch, public_key_path=public_key_path)
    assert load_public_attestation_key(public_key_path) is not None

    cases_path = tmp_path / "holdout-cases.json"
    ledger_path = tmp_path / "holdout-ledger.json"
    ciphertext_path = tmp_path / "cli.sealed.json"
    manifest_path = tmp_path / "cli.public.json"
    cases_path.write_bytes(
        json.dumps(
            [
                case.model_dump(mode="json", exclude_computed_fields=True)
                for case in cases
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    ledger_path.write_bytes(canonical_json_bytes(ledger))

    env = os.environ.copy()
    env["CITE_RIGHT_HOLDOUT_KEY_FILE"] = str(tmp_path / "holdout.key")
    env["CITE_RIGHT_ATTESTATION_KEY_FILE"] = str(tmp_path / "attestation-private.pem")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.sealing",
            "seal",
            "--cases",
            str(cases_path),
            "--ledger",
            str(ledger_path),
            "--ciphertext-output",
            str(ciphertext_path),
            "--public-manifest-output",
            str(manifest_path),
            "--public-key",
            str(public_key_path),
            "--generated-at",
            "2026-07-18",
        ],
        cwd=Path.cwd(),
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.sealing",
            "verify-public-manifest",
            "--manifest",
            str(manifest_path),
            "--ciphertext",
            str(ciphertext_path),
            "--public-key",
            str(public_key_path),
        ],
        cwd=Path.cwd(),
        env=env,
        check=True,
    )


def test_repo_commits_only_the_public_holdout_pem() -> None:
    from evaluation.sealing import load_public_attestation_key

    data_dir = Path("evaluation/data/v1")
    pem_files = tuple(sorted(path.name for path in data_dir.glob("*.pem")))

    assert pem_files == ("holdout_public_key.pem",)
    assert load_public_attestation_key(data_dir / "holdout_public_key.pem") is not None


def _build_holdout_cases() -> tuple[EvaluationCase, ...]:
    return (
        _build_case(
            case_id="case-holdout-a", family_id="family-holdout-a", split="holdout"
        ),
        _build_case(
            case_id="case-holdout-b", family_id="family-holdout-b", split="holdout"
        ),
    )


def _build_case(
    *,
    case_id: str,
    family_id: str,
    split: str,
) -> EvaluationCase:
    answer = f"{case_id} supporting evidence"
    return EvaluationCase.model_validate(
        {
            "case_id": case_id,
            "dataset_version": "1.0.0",
            "split": split,
            "document_family_id": family_id,
            "transformation_family_id": "transform-evidence",
            "provenance": {
                "kind": "public_domain",
                "title": f"Title for {case_id}",
                "origin": f"https://example.test/{case_id}",
                "publisher": "Example Publisher",
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
                "notes": "Case review approved.",
            },
        }
    )


def _approved_review_ledger(cases: tuple[EvaluationCase, ...]) -> ReviewLedger:
    records = [
        make_review_record(
            case,
            case.evaluation_units[0].claims[0],
            reviewer=f"reviewer-{index}",
            reviewed_at="2026-07-18",
            decision="approve",
            notes="Approved for sealing.",
        )
        for index, case in enumerate(cases, start=1)
    ]
    return ReviewLedger(
        dataset_version="1.0.0",
        schema_version="1.0.0",
        entries=tuple(records),
    )


def _decision_ledger(
    cases: tuple[EvaluationCase, ...],
    decision: Literal["approve", "correct", "reject"],
) -> ReviewLedger:
    records = [
        make_review_record(
            case,
            case.evaluation_units[0].claims[0],
            reviewer=f"reviewer-{index}",
            reviewed_at="2026-07-18",
            decision=decision,
            notes=f"{decision} claim.",
            correction_summary="Needs correction." if decision == "correct" else None,
        )
        for index, case in enumerate(cases, start=1)
    ]
    return ReviewLedger(
        dataset_version="1.0.0",
        schema_version="1.0.0",
        entries=tuple(records),
    )


def _install_signing_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    public_key_path: Path,
) -> None:
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


def _resign_manifest_payload(
    private_key_path: Path, manifest_payload: dict[str, object]
) -> None:
    private_key = serialization.load_pem_private_key(
        private_key_path.read_bytes(),
        password=None,
    )
    assert isinstance(private_key, Ed25519PrivateKey)
    unsigned_payload = {
        key: value for key, value in manifest_payload.items() if key != "signature"
    }
    manifest_payload["signature"] = base64.b64encode(
        private_key.sign(canonical_json_bytes(unsigned_payload))
    ).decode("ascii")
