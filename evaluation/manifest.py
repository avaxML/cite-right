"""Deterministic manifest builders for evaluation datasets."""

from __future__ import annotations

import base64
import binascii
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import Generic, Literal, Self, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase, ExpectedStatus, ProvenanceKind, Split

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
SCHEMA_VERSION = "1.0.0"
SPLIT_NAMES: tuple[Split, ...] = ("train", "dev", "holdout")
PUBLIC_DOMAIN_BUCKETS: tuple[str, ...] = (
    "science",
    "finance",
    "policy",
    "technology",
    "health",
    "history",
    "environment",
    "other",
)
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
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_ValueT = TypeVar("_ValueT")


class FrozenMapping(Mapping[str, _ValueT], Generic[_ValueT]):
    """Concrete immutable mapping preserved by Pydantic validation."""

    __slots__ = ("_items", "_lookup")
    _items: tuple[tuple[str, _ValueT], ...]
    _lookup: dict[str, _ValueT]

    def __init__(self, items: Mapping[str, object]) -> None:
        tuple_items = tuple((key, cast(_ValueT, value)) for key, value in items.items())
        self._items = tuple_items
        self._lookup = dict(tuple_items)
        if len(self._lookup) != len(self._items):
            raise ValueError("frozen mappings must not contain duplicate keys")

    def __getitem__(self, key: str) -> _ValueT:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        del source_type, handler
        return core_schema.no_info_plain_validator_function(cls._validate_input)

    @classmethod
    def _validate_input(cls, value: object) -> Self:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("frozen mappings must be provided as mappings")
        for key in value:
            if not isinstance(key, str):
                raise TypeError("frozen mappings must use only string keys")
        return cls(cast(Mapping[str, object], value))

    def to_dict(self) -> dict[str, object]:
        return {
            key: _materialize_json_like(value)
            for key, value in self._items
        }


class DatasetManifest(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    schema_version: str
    generated_at: str | None = None
    overall_sha256: str
    split_sha256: FrozenMapping[str]
    total_case_count: int
    split_case_counts: FrozenMapping[int]
    distributions: FrozenMapping[FrozenMapping[FrozenMapping[int]]]
    review_state_counts: FrozenMapping[FrozenMapping[int]]

    @field_validator("split_sha256", mode="before")
    @classmethod
    def _freeze_split_sha256(cls, value: object) -> FrozenMapping[str]:
        return _freeze_string_mapping(value)

    @field_validator("split_case_counts", mode="before")
    @classmethod
    def _freeze_split_case_counts(cls, value: object) -> FrozenMapping[int]:
        return _freeze_int_mapping(value)

    @field_validator("distributions", mode="before")
    @classmethod
    def _freeze_distributions(
        cls,
        value: object,
    ) -> FrozenMapping[FrozenMapping[FrozenMapping[int]]]:
        return _freeze_three_level_int_mapping(value)

    @field_validator("review_state_counts", mode="before")
    @classmethod
    def _freeze_review_state_counts(
        cls,
        value: object,
    ) -> FrozenMapping[FrozenMapping[int]]:
        return _freeze_two_level_int_mapping(value)

    @field_serializer("split_sha256", "split_case_counts", "distributions", "review_state_counts")
    def _serialize_frozen_mapping(self, value: object) -> dict[str, object]:
        materialized = _materialize_json_like(value)
        if not isinstance(materialized, dict):
            raise TypeError("manifest mapping fields must serialize to dictionaries")
        return materialized


class PublicHoldoutManifest(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    schema_version: str
    generated_at: str | None = None
    holdout_case_count: int
    distributions: FrozenMapping[FrozenMapping[int]]
    ciphertext_sha256: str
    public_key_fingerprint: str | None = None
    signature: str | None = None

    @field_validator("distributions", mode="before")
    @classmethod
    def _freeze_public_distributions(
        cls,
        value: object,
    ) -> FrozenMapping[FrozenMapping[int]]:
        return _freeze_two_level_int_mapping(value)

    @field_validator("ciphertext_sha256")
    @classmethod
    def _validate_ciphertext_sha256(cls, value: str) -> str:
        return _validate_hex_64(
            value,
            field_name="ciphertext_sha256",
        )

    @field_validator("public_key_fingerprint")
    @classmethod
    def _validate_public_key_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_hex_64(
            value,
            field_name="public_key_fingerprint",
        )

    @field_validator("signature")
    @classmethod
    def _validate_signature(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            decoded = base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError(
                "signature must be canonical base64 for exactly 64 bytes"
            ) from exc
        if len(decoded) != 64:
            raise ValueError("signature must be canonical base64 for exactly 64 bytes")
        if base64.b64encode(decoded).decode("ascii") != value:
            raise ValueError("signature must be canonical base64 for exactly 64 bytes")
        return value

    @field_serializer("distributions")
    def _serialize_distributions(self, value: object) -> dict[str, object]:
        materialized = _materialize_json_like(value)
        if not isinstance(materialized, dict):
            raise TypeError("public manifest distributions must serialize to dictionaries")
        return materialized

    @model_validator(mode="after")
    def _validate_distribution_allowlist(self) -> PublicHoldoutManifest:
        if tuple(sorted(self.distributions)) != tuple(sorted(("domain", "expected_status", "provenance_kind"))):
            raise ValueError(
                "public holdout distributions must expose only domain, expected_status, and provenance_kind"
            )
        domain_keys = tuple(sorted(self.distributions["domain"]))
        if domain_keys != tuple(sorted(PUBLIC_DOMAIN_BUCKETS)):
            raise ValueError("public holdout domain distributions must use the public allowlist buckets")
        if tuple(sorted(self.distributions["expected_status"])) != tuple(sorted(EXPECTED_STATUSES)):
            raise ValueError("public holdout expected_status distributions must use the fixed buckets")
        if tuple(sorted(self.distributions["provenance_kind"])) != tuple(sorted(PROVENANCE_KINDS)):
            raise ValueError("public holdout provenance_kind distributions must use the fixed buckets")
        return self


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
        split_sha256=cast(FrozenMapping[str], _freeze_string_mapping(split_sha256)),
        total_case_count=len(ordered_cases),
        split_case_counts=cast(FrozenMapping[int], _freeze_int_mapping(split_case_counts)),
        distributions=cast(
            FrozenMapping[FrozenMapping[FrozenMapping[int]]],
            _freeze_three_level_int_mapping(distributions),
        ),
        review_state_counts=cast(
            FrozenMapping[FrozenMapping[int]],
            _freeze_two_level_int_mapping(review_state_counts),
        ),
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
        distributions=cast(
            FrozenMapping[FrozenMapping[int]],
            _freeze_two_level_int_mapping(
                {
                    "expected_status": {
                        status: _int_value(
                            holdout_distributions["expected_status"].get(status, 0)
                        )
                        for status in EXPECTED_STATUSES
                    },
                    "domain": _public_domain_distribution(holdout_distributions["domain"]),
                    "provenance_kind": {
                        kind: _int_value(
                            holdout_distributions["provenance_kind"].get(kind, 0)
                        )
                        for kind in PROVENANCE_KINDS
                    },
                }
            ),
        ),
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
        path="/manifest",
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
        for key in sorted(actual_keys | expected_keys, key=lambda item: str(item)):
            child_path = _json_pointer_child(path, key)
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
            child_path = _json_pointer_child(path, index)
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
    if type(actual) is type(expected) and actual == expected:
        return
    if type(actual) is not type(expected):
        mismatches.append(
            ManifestMismatch(
                path=path,
                message=(
                    f"{path} expected {expected!r} ({type(expected).__name__}) "
                    f"but found {actual!r} ({type(actual).__name__})"
                ),
            )
        )
        return
    mismatches.append(
        ManifestMismatch(
            path=path,
            message=f"{path} expected {expected!r} but found {actual!r}",
        )
    )


def _freeze_string_mapping(value: object) -> FrozenMapping[str]:
    raw_mapping = _require_mapping(value)
    frozen: dict[str, str] = {}
    for key, item in raw_mapping.items():
        if not isinstance(item, str):
            raise TypeError("manifest string mappings must contain only string values")
        frozen[key] = item
    return FrozenMapping(frozen)


def _freeze_int_mapping(value: object) -> FrozenMapping[int]:
    raw_mapping = _require_mapping(value)
    frozen: dict[str, int] = {}
    for key, item in raw_mapping.items():
        frozen[key] = _int_value(item)
    return FrozenMapping(frozen)


def _freeze_two_level_int_mapping(value: object) -> FrozenMapping[FrozenMapping[int]]:
    raw_mapping = _require_mapping(value)
    return FrozenMapping(
        {key: _freeze_int_mapping(item) for key, item in raw_mapping.items()}
    )


def _freeze_three_level_int_mapping(
    value: object,
) -> FrozenMapping[FrozenMapping[FrozenMapping[int]]]:
    raw_mapping = _require_mapping(value)
    return FrozenMapping(
        {key: _freeze_two_level_int_mapping(item) for key, item in raw_mapping.items()}
    )


def _require_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, FrozenMapping):
        return cast(Mapping[str, object], value)
    if not isinstance(value, Mapping):
        raise ValueError("manifest mappings must be provided as mappings")
    frozen_input: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("manifest mappings must use only string keys")
        frozen_input[key] = item
    return frozen_input


def _int_value(value: object) -> int:
    if type(value) is not int:
        raise TypeError("manifest integer mappings must contain only int values")
    return cast(int, value)


def _materialize_json_like(value: object) -> object:
    if isinstance(value, FrozenMapping):
        return value.to_dict()
    if isinstance(value, Mapping):
        materialized: dict[str, object] = {}
        for key, item in value.items():
            materialized[str(key)] = _materialize_json_like(item)
        return materialized
    if isinstance(value, tuple):
        return [_materialize_json_like(item) for item in value]
    if isinstance(value, list):
        return [_materialize_json_like(item) for item in value]
    return value


def _validate_hex_64(value: str, *, field_name: str) -> str:
    if _HEX_64_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")
    return value


def _public_domain_distribution(domain_counts: Mapping[str, int]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in PUBLIC_DOMAIN_BUCKETS}
    for key, raw_value in domain_counts.items():
        bucket = key if key in counts and key != "other" else "other"
        counts[bucket] += _int_value(raw_value)
    return counts


def _json_pointer_child(path: str, token: object) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


__all__ = [
    "DatasetManifest",
    "FrozenMapping",
    "ManifestMismatch",
    "PUBLIC_DOMAIN_BUCKETS",
    "PublicHoldoutManifest",
    "SCHEMA_VERSION",
    "build_private_manifest",
    "build_public_holdout_manifest",
    "verify_private_manifest_expectations",
]
