"""Deterministic manifest builders for evaluation datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase, ExpectedStatus, ProvenanceKind, Split

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
SCHEMA_VERSION = "1.0.0"
SPLIT_NAMES: tuple[Split, ...] = ("train", "dev", "holdout")
REVIEW_STATES: tuple[Literal["missing", "pending", "approved", "rejected"], ...] = (
    "missing",
    "pending",
    "approved",
    "rejected",
)
EXPECTED_STATUSES: tuple[ExpectedStatus, ...] = ("supported", "partial", "unsupported")
PROVENANCE_KINDS: tuple[ProvenanceKind, ...] = (
    "authored",
    "public_domain",
    "permissive_license",
)


class DatasetManifest(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    schema_version: str
    generated_at: str | None = None
    overall_sha256: str
    split_sha256: Mapping[Split, str]
    total_case_count: int
    split_case_counts: Mapping[Split, int]
    distributions: Mapping[str, Mapping[str, Mapping[str, int]]]
    review_state_counts: Mapping[str, Mapping[str, int]]


class PublicHoldoutManifest(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    schema_version: str
    generated_at: str | None = None
    holdout_case_count: int
    distributions: Mapping[str, Mapping[str, int]]
    ciphertext_sha256: str
    public_key_fingerprint: str | None = None
    signature: str | None = None


class ManifestMismatch(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    path: str
    message: str


def build_private_manifest(
    cases: Iterable[EvaluationCase],
    *,
    generated_at: str | None = None,
) -> DatasetManifest:
    ordered_cases = _validated_cases(cases)
    dataset_version = _validated_dataset_version(ordered_cases)
    overall_payload = {
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "cases": tuple(case.model_dump(mode="json") for case in ordered_cases),
    }
    split_sha256 = {
        split_name: sha256_hex(
            canonical_json_bytes(
                {
                    "split": split_name,
                    "cases": tuple(
                        case.model_dump(mode="json")
                        for case in ordered_cases
                        if case.split == split_name
                    ),
                }
            )
        )
        for split_name in SPLIT_NAMES
    }
    split_case_counts = {
        split_name: sum(1 for case in ordered_cases if case.split == split_name)
        for split_name in SPLIT_NAMES
    }
    distributions = {"overall": _distribution_for(ordered_cases)}
    distributions.update(
        {
            split_name: _distribution_for(
                tuple(case for case in ordered_cases if case.split == split_name)
            )
            for split_name in SPLIT_NAMES
        }
    )
    review_state_counts = {"overall": _review_state_counts(ordered_cases)}
    review_state_counts.update(
        {
            split_name: _review_state_counts(
                tuple(case for case in ordered_cases if case.split == split_name)
            )
            for split_name in SPLIT_NAMES
        }
    )
    return DatasetManifest(
        dataset_version=dataset_version,
        schema_version=SCHEMA_VERSION,
        generated_at=generated_at,
        overall_sha256=sha256_hex(canonical_json_bytes(overall_payload)),
        split_sha256=cast(Mapping[Split, str], split_sha256),
        total_case_count=len(ordered_cases),
        split_case_counts=cast(Mapping[Split, int], split_case_counts),
        distributions=distributions,
        review_state_counts=review_state_counts,
    )


def build_public_holdout_manifest(
    private_manifest: DatasetManifest,
    *,
    ciphertext_sha256: str,
    public_key_fingerprint: str | None = None,
    signature: str | None = None,
) -> PublicHoldoutManifest:
    holdout_distributions = private_manifest.distributions["holdout"]
    return PublicHoldoutManifest(
        dataset_version=private_manifest.dataset_version,
        schema_version=private_manifest.schema_version,
        generated_at=private_manifest.generated_at,
        holdout_case_count=private_manifest.split_case_counts["holdout"],
        distributions={
            "expected_status": dict(holdout_distributions["expected_status"]),
            "domain": dict(holdout_distributions["domain"]),
            "transformation_family": dict(holdout_distributions["transformation_family"]),
            "difficulty_family": dict(holdout_distributions["difficulty_family"]),
            "provenance_kind": dict(holdout_distributions["provenance_kind"]),
        },
        ciphertext_sha256=ciphertext_sha256,
        public_key_fingerprint=public_key_fingerprint,
        signature=signature,
    )


def verify_private_manifest_expectations(
    actual: DatasetManifest,
    expected: DatasetManifest,
) -> tuple[ManifestMismatch, ...]:
    mismatches: list[ManifestMismatch] = []
    _compare_manifest_value(
        mismatches,
        path="manifest",
        actual=actual.model_dump(mode="json"),
        expected=expected.model_dump(mode="json"),
    )
    return tuple(sorted(mismatches, key=lambda mismatch: (mismatch.path, mismatch.message)))


def _validated_cases(cases: Iterable[EvaluationCase]) -> tuple[EvaluationCase, ...]:
    ordered_cases = tuple(cases)
    if not ordered_cases:
        raise ValueError("cases must not be empty")
    seen_case_ids: set[str] = set()
    for case in ordered_cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case id {case.case_id!r}")
        seen_case_ids.add(case.case_id)
    return tuple(
        sorted(
            ordered_cases,
            key=lambda case: (
                case.split,
                case.case_id,
                canonical_json_bytes(case.model_dump(mode="json")),
            ),
        )
    )


def _validated_dataset_version(cases: tuple[EvaluationCase, ...]) -> str:
    dataset_versions = {case.dataset_version for case in cases}
    if len(dataset_versions) != 1:
        raise ValueError("cases must all share one dataset_version")
    return next(iter(dataset_versions))


def _distribution_for(cases: tuple[EvaluationCase, ...]) -> dict[str, dict[str, int]]:
    expected_status = Counter(_case_expected_status(case) for case in cases)
    domains = Counter(case.difficulty_tags[0] for case in cases if case.difficulty_tags)
    difficulty_families = Counter(
        case.difficulty_tags[1] for case in cases if len(case.difficulty_tags) > 1
    )
    transformation_families = Counter(case.transformation_family_id for case in cases)
    provenance_kinds = Counter(case.provenance.kind for case in cases)
    return {
        "expected_status": {
            status: expected_status.get(status, 0) for status in EXPECTED_STATUSES
        },
        "domain": dict(sorted(domains.items())),
        "transformation_family": dict(sorted(transformation_families.items())),
        "difficulty_family": dict(sorted(difficulty_families.items())),
        "provenance_kind": {
            kind: provenance_kinds.get(kind, 0) for kind in PROVENANCE_KINDS
        },
    }


def _review_state_counts(cases: tuple[EvaluationCase, ...]) -> dict[str, int]:
    counts = Counter(
        "missing" if case.review is None else case.review.state for case in cases
    )
    return {state: counts.get(state, 0) for state in REVIEW_STATES}


def _case_expected_status(case: EvaluationCase) -> ExpectedStatus:
    unit_statuses = {unit.expected_status for unit in case.evaluation_units}
    if unit_statuses == {"supported"}:
        return "supported"
    if unit_statuses == {"unsupported"}:
        return "unsupported"
    return "partial"


def _compare_manifest_value(
    mismatches: list[ManifestMismatch],
    *,
    path: str,
    actual: object,
    expected: object,
) -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        actual_keys = set(actual)
        expected_keys = set(expected)
        for key in sorted(actual_keys | expected_keys):
            child_path = f"{path}.{key}"
            if key not in expected:
                mismatches.append(
                    ManifestMismatch(
                        path=child_path,
                        message=f"{child_path} unexpected key in actual mapping",
                    )
                )
                continue
            if key not in actual:
                mismatches.append(
                    ManifestMismatch(
                        path=child_path,
                        message=f"{child_path} expected key missing from actual mapping",
                    )
                )
                continue
            _compare_manifest_value(
                mismatches,
                path=child_path,
                actual=actual[key],
                expected=expected[key],
            )
        return
    if isinstance(actual, list | tuple) and isinstance(expected, list | tuple):
        max_length = max(len(actual), len(expected))
        for index in range(max_length):
            child_path = f"{path}.{index}"
            if index >= len(expected):
                mismatches.append(
                    ManifestMismatch(
                        path=child_path,
                        message=f"{child_path} unexpected index in actual sequence",
                    )
                )
                continue
            if index >= len(actual):
                mismatches.append(
                    ManifestMismatch(
                        path=child_path,
                        message=f"{child_path} expected index missing from actual sequence",
                    )
                )
                continue
            _compare_manifest_value(
                mismatches,
                path=child_path,
                actual=actual[index],
                expected=expected[index],
            )
        return
    if actual == expected:
        return
    mismatches.append(
        ManifestMismatch(
            path=path,
            message=f"{path} expected {expected!r} but found {actual!r}",
        )
    )


__all__ = [
    "DatasetManifest",
    "ManifestMismatch",
    "PublicHoldoutManifest",
    "SCHEMA_VERSION",
    "build_private_manifest",
    "build_public_holdout_manifest",
    "verify_private_manifest_expectations",
]
