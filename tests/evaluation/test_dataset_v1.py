from __future__ import annotations

import json
from pathlib import Path

from evaluation.builders.authored_sources import AUTHORED_FACT_TEMPLATES
from evaluation.builders.cases import generate_all_authored_cases
from evaluation.builders.real_sources import generate_real_cases
from evaluation.canonical import canonical_json_bytes
from evaluation.leakage import detect_leakage
from evaluation.manifest import (
    DatasetManifest,
    PublicHoldoutManifest,
    build_private_manifest,
)
from evaluation.review import assert_review_complete, load_review_ledger
from evaluation.schema import EvaluationCase
from evaluation.sealing import verify_public_manifest
from evaluation.splitting import apply_split_assignments, assign_splits
from evaluation.tuning_bundle import load_tuning_bundle

DATASET_ROOT = Path(__file__).resolve().parents[2] / "evaluation" / "data" / "v1"
SEED = 20260717


def test_dataset_v1_contains_only_the_required_public_and_tuning_artifacts() -> None:
    required = {
        "dev.json",
        "dev_reviews.json",
        "holdout.aesgcm",
        "holdout.public.json",
        "holdout_public_key.pem",
        "manifest.json",
        "provenance.json",
        "sources/authored.json",
        "sources/real.json",
        "train.json",
        "tuning/dev.json",
        "tuning/manifest.json",
        "tuning/train.json",
    }
    present = {
        str(path.relative_to(DATASET_ROOT))
        for path in DATASET_ROOT.rglob("*")
        if path.is_file()
    }

    assert required.issubset(present)
    assert "holdout.json" not in present
    assert "holdout_reviews.json" not in present
    assert not any(
        "private" in path.casefold() or path.endswith(".key") for path in present
    )


def test_dataset_v1_has_target_size_grouped_split_balance_and_coverage() -> None:
    train = _load_cases(DATASET_ROOT / "train.json")
    dev = _load_cases(DATASET_ROOT / "dev.json")
    public_holdout = PublicHoldoutManifest.model_validate_json(
        (DATASET_ROOT / "holdout.public.json").read_bytes()
    )
    counts = {
        "train": len(train),
        "dev": len(dev),
        "holdout": public_holdout.holdout_case_count,
    }
    total = sum(counts.values())

    assert 725 <= total <= 775
    for split, target in {"train": 0.60, "dev": 0.20, "holdout": 0.20}.items():
        assert abs(counts[split] / total - target) <= 0.05

    cases = train + dev
    domains = {case.document_family_id.split("-", 1)[0] for case in cases}
    assert {
        "science",
        "finance",
        "policy",
        "technology",
        "health",
        "history",
    } <= domains
    assert len({case.document_family_id for case in cases}) >= 40
    assert {case.provenance.kind for case in cases} >= {"authored", "public_domain"}
    assert len({case.transformation_family_id for case in cases}) >= 12


def test_dataset_v1_dev_reviews_and_public_holdout_attestation_are_complete() -> None:
    dev = _load_cases(DATASET_ROOT / "dev.json")
    ledger = load_review_ledger(DATASET_ROOT / "dev_reviews.json")

    assert_review_complete(dev, ledger, split="dev")
    public_manifest = verify_public_manifest(
        DATASET_ROOT / "holdout.public.json",
        ciphertext_path=DATASET_ROOT / "holdout.aesgcm",
        public_key_path=DATASET_ROOT / "holdout_public_key.pem",
    )
    assert public_manifest.total_claim_count > 0
    assert public_manifest.reviewed_claim_count == public_manifest.total_claim_count


def test_dataset_v1_regenerates_canonical_splits_without_cross_split_leakage() -> None:
    regenerated = _regenerated_cases()
    train = tuple(
        sorted(
            (case for case in regenerated if case.split == "train"),
            key=lambda case: case.case_id,
        )
    )
    dev = tuple(
        sorted(
            (case for case in regenerated if case.split == "dev"),
            key=lambda case: case.case_id,
        )
    )
    expected_manifest = build_private_manifest(train + dev, generated_at=None)
    actual_manifest = DatasetManifest.model_validate_json(
        (DATASET_ROOT / "manifest.json").read_bytes()
    )

    assert (
        canonical_json_bytes([_case_payload(case) for case in train])
        == (DATASET_ROOT / "train.json").read_bytes()
    )
    assert (
        canonical_json_bytes([_case_payload(case) for case in dev])
        == (DATASET_ROOT / "dev.json").read_bytes()
    )
    assert actual_manifest == expected_manifest
    assert (
        canonical_json_bytes(expected_manifest)
        == (DATASET_ROOT / "manifest.json").read_bytes()
    )
    leakage = detect_leakage(regenerated)
    assert leakage.error_count == 0


def test_dataset_v1_authored_snapshot_and_tuning_bundle_are_reproducible() -> None:
    authored_payload: tuple[object, ...] = tuple(
        template.model_dump(mode="json")
        for template in sorted(AUTHORED_FACT_TEMPLATES, key=lambda item: item.family_id)
    )
    assert (
        canonical_json_bytes(authored_payload)
        == (DATASET_ROOT / "sources" / "authored.json").read_bytes()
    )

    tuning = load_tuning_bundle(DATASET_ROOT / "tuning")
    assert all(
        case.split == "train" and case.review is None for case in tuning.train_cases
    )
    assert all(case.split == "dev" and case.review is None for case in tuning.dev_cases)
    assert {case.case_id for case in tuning.train_cases}.isdisjoint(
        case.case_id for case in tuning.dev_cases
    )


def _regenerated_cases() -> tuple[EvaluationCase, ...]:
    candidates = generate_all_authored_cases(SEED) + generate_real_cases()
    report = assign_splits(candidates, seed=SEED)
    return apply_split_assignments(candidates, report.assignment_by_case_id)


def _load_cases(path: Path) -> tuple[EvaluationCase, ...]:
    payload = json.loads(path.read_bytes())
    assert isinstance(payload, list)
    return tuple(
        EvaluationCase.model_validate_json(canonical_json_bytes(item))
        for item in payload
    )


def _case_payload(case: EvaluationCase) -> dict[str, object]:
    return case.model_dump(mode="json", exclude_computed_fields=True)
