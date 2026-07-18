"""Local real-world public-domain source snapshots for evaluation cases."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

from evaluation import DATASET_VERSION
from evaluation.canonical import authoritative_case_id, canonical_json_bytes, sha256_hex
from evaluation.schema import (
    CharSpan,
    CitationRequirement,
    CitationTarget,
    ClaimAnnotation,
    EvaluationCase,
    EvaluationUnit,
    Provenance,
    Source,
)

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
RETRIEVAL_DATE = date(2026, 7, 17)
STATUTORY_URL = (
    "https://uscode.house.gov/view.xhtml?edition=2023&num=0&req="
    "granuleid%3AUSC-2023-title17-section105"
)
REAL_TRANSFORMATION_ORDER = (
    "real-positive",
    "real-contradicted",
    "real-partial",
    "real-distractor",
)
REAL_DOMAIN = Literal[
    "science",
    "environment",
    "health",
    "history",
    "finance",
    "technology",
    "policy",
]
REAL_CHALLENGE_KIND = Literal["contradicted", "partial", "distractor"]
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "v1"
REAL_SOURCES_PATH = DATA_ROOT / "sources" / "real.json"
PROVENANCE_PATH = DATA_ROOT / "provenance.json"
_DNS_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_PUBLISHER_ORIGIN_HOSTS = MappingProxyType(
    {
        "NASA": "www.nasa.gov",
        "U.S. Environmental Protection Agency": "www.epa.gov",
        "U.S. Geological Survey": "www.usgs.gov",
        "U.S. Department of Energy": "www.energy.gov",
        "NOAA Pacific Marine Environmental Laboratory": "www.pmel.noaa.gov",
        "Centers for Disease Control and Prevention": "www.cdc.gov",
        "National Archives": "www.archives.gov",
        "Federal Deposit Insurance Corporation": "www.fdic.gov",
        "National Institute of Standards and Technology": "www.nist.gov",
        "USAGov": "www.usa.gov",
    }
)


class RealSourceProvenance(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    family_id: str
    domain: REAL_DOMAIN
    source_text: str
    origin_url: str
    page_title: str
    publisher: str
    license_basis: str
    policy_url: str
    statutory_url: str
    retrieval_date: date
    snapshot_hash: str
    third_party_credit: bool

    @field_validator(
        "family_id",
        "source_text",
        "page_title",
        "publisher",
        "license_basis",
        "snapshot_hash",
    )
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        field_name = str(info.field_name)
        if not value.strip():
            raise ValueError(f"{field_name} must be non-empty")
        if field_name == "source_text":
            return _require_non_empty(value, "real source text must be non-empty")
        return value

    @field_validator("origin_url")
    @classmethod
    def _validate_origin_url(cls, value: str) -> str:
        return _validate_official_https_url(
            value,
            message="origin_url must use https:// and point to an official source",
        )

    @field_validator("policy_url")
    @classmethod
    def _validate_policy_url(cls, value: str) -> str:
        return _validate_official_https_url(
            value,
            message="policy_url must use https:// and point to an official source",
        )

    @field_validator("statutory_url")
    @classmethod
    def _validate_statutory_url(cls, value: str) -> str:
        return _validate_official_https_url(
            value,
            message="statutory_url must use https:// and point to an official source",
        )

    @field_validator("retrieval_date", mode="before")
    @classmethod
    def _parse_retrieval_date(cls, value: object) -> object:
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _validate_record(self) -> RealSourceProvenance:
        if not self.family_id.startswith(f"{self.domain}-"):
            raise ValueError("family_id must be prefixed by domain")
        if self.third_party_credit:
            raise ValueError("third_party_credit must be false")
        if self.retrieval_date != RETRIEVAL_DATE:
            raise ValueError("retrieval_date must equal 2026-07-17")
        if self.snapshot_hash != sha256_hex(self.source_text.encode("utf-8")):
            raise ValueError("snapshot_hash must equal sha256 of source_text")
        if self.statutory_url != STATUTORY_URL:
            raise ValueError("statutory_url must equal the 17 U.S.C. 105 basis URL")
        expected_origin_host = _PUBLISHER_ORIGIN_HOSTS.get(self.publisher)
        if expected_origin_host is None:
            raise ValueError("origin_url hostname must match the declared publisher")
        if _canonical_origin_hostname(self.origin_url) != expected_origin_host:
            raise ValueError("origin_url hostname must match the declared publisher")
        return self


class RealSourceChallenge(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    kind: REAL_CHALLENGE_KIND
    answer: str
    distractor_family_id: str | None = None
    unsupported_suffix: str | None = None

    @field_validator("answer")
    @classmethod
    def _validate_answer(cls, value: str) -> str:
        return _require_non_empty(value, "challenge answer must be non-empty")

    @field_validator("distractor_family_id", "unsupported_suffix")
    @classmethod
    def _validate_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("optional challenge fields must be non-empty when provided")
        return value

    @model_validator(mode="after")
    def _validate_kind_specific_fields(self) -> RealSourceChallenge:
        if self.kind == "contradicted":
            if self.distractor_family_id is not None or self.unsupported_suffix is not None:
                raise ValueError(
                    "contradicted challenges must not define distractor_family_id or unsupported_suffix"
                )
            return self
        if self.kind == "partial":
            if self.unsupported_suffix is None:
                raise ValueError("partial challenges must define unsupported_suffix")
            if self.distractor_family_id is not None:
                raise ValueError("partial challenges must not define distractor_family_id")
            return self
        if self.distractor_family_id is None:
            raise ValueError("distractor challenges must define distractor_family_id")
        if self.unsupported_suffix is not None:
            raise ValueError("distractor challenges must not define unsupported_suffix")
        return self


class RealSourceFamily(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    family_id: str
    domain: REAL_DOMAIN
    source_text: str
    supported_answer: str
    snapshot_hash: str
    provenance: RealSourceProvenance
    challenge: RealSourceChallenge

    @field_validator("family_id", "snapshot_hash")
    @classmethod
    def _validate_non_empty_identifiers(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty(value, f"{info.field_name} must be non-empty")

    @field_validator("source_text")
    @classmethod
    def _validate_source_text(cls, value: str) -> str:
        return _require_non_empty(value, "real source text must be non-empty")

    @field_validator("supported_answer")
    @classmethod
    def _validate_supported_answer(cls, value: str) -> str:
        return _require_non_empty(value, "supported_answer must be non-empty")

    @model_validator(mode="after")
    def _validate_family(self) -> RealSourceFamily:
        if not self.family_id.startswith(f"{self.domain}-"):
            raise ValueError("family_id must be prefixed by domain")
        if self.supported_answer != self.source_text:
            raise ValueError("supported_answer must equal source_text")
        if len(self.source_text.split()) > 20:
            raise ValueError("source_text must be 20 words or fewer")
        if self.snapshot_hash != sha256_hex(self.source_text.encode("utf-8")):
            raise ValueError("snapshot_hash must equal sha256 of source_text")
        if self.family_id != self.provenance.family_id:
            raise ValueError("family_id must match provenance.family_id")
        if self.domain != self.provenance.domain:
            raise ValueError("domain must match provenance.domain")
        if self.source_text != self.provenance.source_text:
            raise ValueError("source_text must match provenance.source_text")
        if self.snapshot_hash != self.provenance.snapshot_hash:
            raise ValueError("snapshot_hash must match provenance.snapshot_hash")
        if self.challenge.kind == "partial":
            assert self.challenge.unsupported_suffix is not None
            allowed_answers = (
                self.source_text + self.challenge.unsupported_suffix,
                self.source_text + "." + self.challenge.unsupported_suffix,
            )
            if self.challenge.answer not in allowed_answers:
                raise ValueError(
                    "partial challenge answer must equal source_text plus unsupported_suffix"
                )
        elif self.challenge.kind == "distractor":
            if self.challenge.answer != self.supported_answer:
                raise ValueError("distractor challenge answer must equal supported_answer")
        return self


def load_real_source_provenance() -> tuple[RealSourceProvenance, ...]:
    payload = _load_json_array(
        PROVENANCE_PATH,
        artifact_name="provenance.json",
    )
    records = tuple(RealSourceProvenance.model_validate(item) for item in payload)
    _validate_unique_ids(records, context="provenance records")
    _validate_sorted_ids(records, context="provenance records")
    return records


def load_real_source_families() -> tuple[RealSourceFamily, ...]:
    payload = _load_json_array(
        REAL_SOURCES_PATH,
        artifact_name="real.json",
    )
    provenance_by_id = {
        record.family_id: record
        for record in load_real_source_provenance()
    }
    families: list[RealSourceFamily] = []
    seen_family_ids: set[str] = set()
    for raw_family in payload:
        if not isinstance(raw_family, dict):
            raise ValueError("real.json entries must be objects")
        family_id = raw_family.get("family_id")
        if not isinstance(family_id, str) or not family_id.strip():
            raise ValueError("family_id must be non-empty")
        if family_id in seen_family_ids:
            raise ValueError(f"duplicate family_id {family_id!r}")
        provenance = provenance_by_id.get(family_id)
        if provenance is None:
            raise ValueError(f"missing provenance record for family_id {family_id!r}")
        families.append(
            RealSourceFamily.model_validate(
                {
                    **raw_family,
                    "provenance": provenance.model_dump(mode="python"),
                }
            )
        )
        seen_family_ids.add(family_id)
    missing_family_ids = tuple(
        family_id
        for family_id in provenance_by_id
        if family_id not in seen_family_ids
    )
    if missing_family_ids:
        raise ValueError(
            "provenance.json contains family_ids that are missing from real.json"
        )
    ordered = tuple(sorted(families, key=lambda family: family.family_id))
    if tuple(family.family_id for family in ordered) != tuple(
        family.family_id for family in families
    ):
        raise ValueError("real source families must be sorted by family_id")
    return ordered


def generate_real_cases() -> tuple[EvaluationCase, ...]:
    families = load_real_source_families()
    families_by_id = {family.family_id: family for family in families}
    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for family in families:
        for case in (
            _build_positive_case(family),
            _build_challenge_case(family, families_by_id),
        ):
            if case.case_id in seen_case_ids:
                raise ValueError(f"duplicate case id {case.case_id!r}")
            seen_case_ids.add(case.case_id)
            cases.append(case)
    ordered = tuple(
        sorted(
            cases,
            key=lambda case: (
                case.document_family_id,
                REAL_TRANSFORMATION_ORDER.index(case.transformation_family_id),
                case.case_id,
            ),
        )
    )
    return ordered


def _build_positive_case(family: RealSourceFamily) -> EvaluationCase:
    source = Source(source_id="source-primary", text=family.source_text)
    unit = EvaluationUnit(
        unit_id="unit-answer",
        answer_span=CharSpan(start=0, end=len(family.supported_answer)),
        text=family.supported_answer,
        claims=(
            ClaimAnnotation(
                claim_id="claim-supported",
                answer_span=CharSpan(start=0, end=len(family.supported_answer)),
                text=family.supported_answer,
                label="entailed",
                citation_requirements=_whole_source_requirements("source-primary", source.text),
                acceptable_retrieval_source_ids=("source-primary",),
            ),
        ),
    )
    return _finalize_case(
        document_family_id=family.family_id,
        transformation_family_id="real-positive",
        provenance=family.provenance,
        sources=(source,),
        answer=family.supported_answer,
        evaluation_units=(unit,),
        difficulty_tags=(family.domain, "real", "positive"),
    )


def _build_challenge_case(
    family: RealSourceFamily,
    families_by_id: dict[str, RealSourceFamily],
) -> EvaluationCase:
    source = Source(source_id="source-primary", text=family.source_text)
    challenge = family.challenge
    if challenge.kind == "contradicted":
        unit = EvaluationUnit(
            unit_id="unit-answer",
            answer_span=CharSpan(start=0, end=len(challenge.answer)),
            text=challenge.answer,
            claims=(
                ClaimAnnotation(
                    claim_id="claim-contradicted",
                    answer_span=CharSpan(start=0, end=len(challenge.answer)),
                    text=challenge.answer,
                    label="contradicted",
                ),
            ),
        )
        return _finalize_case(
            document_family_id=family.family_id,
            transformation_family_id="real-contradicted",
            provenance=family.provenance,
            sources=(source,),
            answer=challenge.answer,
            evaluation_units=(unit,),
            difficulty_tags=(family.domain, "real", "contradicted"),
        )

    if challenge.kind == "partial":
        assert challenge.unsupported_suffix is not None
        supported_end = len(family.source_text)
        unsupported_text = challenge.answer[supported_end:].lstrip(" \t\n\r.,;:!?-")
        unsupported_start = len(challenge.answer) - len(unsupported_text)
        unit = EvaluationUnit(
            unit_id="unit-answer",
            answer_span=CharSpan(start=0, end=len(challenge.answer)),
            text=challenge.answer,
            claims=(
                ClaimAnnotation(
                    claim_id="claim-supported",
                    answer_span=CharSpan(start=0, end=supported_end),
                    text=family.source_text,
                    label="entailed",
                    citation_requirements=_whole_source_requirements("source-primary", source.text),
                    acceptable_retrieval_source_ids=("source-primary",),
                ),
                ClaimAnnotation(
                    claim_id="claim-unsupported",
                    answer_span=CharSpan(
                        start=unsupported_start,
                        end=len(challenge.answer),
                    ),
                    text=unsupported_text,
                    label="not_in_sources",
                ),
            ),
        )
        return _finalize_case(
            document_family_id=family.family_id,
            transformation_family_id="real-partial",
            provenance=family.provenance,
            sources=(source,),
            answer=challenge.answer,
            evaluation_units=(unit,),
            difficulty_tags=(family.domain, "real", "partial"),
        )

    assert challenge.kind == "distractor"
    assert challenge.distractor_family_id is not None
    distractor_family = families_by_id.get(challenge.distractor_family_id)
    if distractor_family is None:
        raise ValueError(
            f"distractor_family_id {challenge.distractor_family_id!r} must reference a known family"
        )
    if distractor_family.family_id == family.family_id:
        raise ValueError("distractor_family_id must reference a different family")
    distractor_source = Source(
        source_id="source-distractor",
        text=distractor_family.source_text,
    )
    unit = EvaluationUnit(
        unit_id="unit-answer",
        answer_span=CharSpan(start=0, end=len(challenge.answer)),
        text=challenge.answer,
        claims=(
            ClaimAnnotation(
                claim_id="claim-supported",
                answer_span=CharSpan(start=0, end=len(challenge.answer)),
                text=challenge.answer,
                label="entailed",
                citation_requirements=_whole_source_requirements("source-primary", source.text),
                acceptable_retrieval_source_ids=("source-primary",),
            ),
        ),
    )
    return _finalize_case(
        document_family_id=family.family_id,
        transformation_family_id="real-distractor",
        provenance=family.provenance,
        sources=(source, distractor_source),
        answer=challenge.answer,
        evaluation_units=(unit,),
        difficulty_tags=(family.domain, "real", "distractor"),
    )


def _finalize_case(
    *,
    document_family_id: str,
    transformation_family_id: str,
    provenance: RealSourceProvenance,
    sources: tuple[Source, ...],
    answer: str,
    evaluation_units: tuple[EvaluationUnit, ...],
    difficulty_tags: tuple[str, ...],
) -> EvaluationCase:
    temporary_case = EvaluationCase(
        case_id="case-pending",
        dataset_version=DATASET_VERSION,
        split="train",
        document_family_id=document_family_id,
        transformation_family_id=transformation_family_id,
        provenance=Provenance(
            kind="public_domain",
            title=provenance.page_title,
            origin=provenance.origin_url,
            publisher=provenance.publisher,
            license=provenance.license_basis,
            retrieval_date=provenance.retrieval_date,
            snapshot_hash=provenance.snapshot_hash,
        ),
        sources=sources,
        answer=answer,
        evaluation_units=evaluation_units,
        difficulty_tags=difficulty_tags,
        generation=None,
        review=None,
    )
    case = temporary_case.model_copy(
        update={"case_id": authoritative_case_id(temporary_case)}
    )
    return EvaluationCase.model_validate(case.model_dump(mode="python", round_trip=True))


def _whole_source_requirements(
    source_id: str,
    source_text: str,
) -> tuple[CitationRequirement, ...]:
    return (
        CitationRequirement(
            requirement_id="req-primary",
            alternatives=(
                CitationTarget(
                    source_id=source_id,
                    spans=(CharSpan(start=0, end=len(source_text)),),
                ),
            ),
        ),
    )


def _canonical_case_digest(cases: tuple[EvaluationCase, ...]) -> str:
    payload = tuple(
        case.model_dump(mode="json")
        for case in sorted(cases, key=lambda case: case.case_id)
    )
    return sha256_hex(canonical_json_bytes({"cases": payload}))


def _load_json_array(path: Path, *, artifact_name: str) -> list[object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact {artifact_name}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{artifact_name} must contain a JSON array")
    return payload


def _validate_unique_ids(
    records: tuple[RealSourceProvenance, ...],
    *,
    context: str,
) -> None:
    seen: set[str] = set()
    for record in records:
        if record.family_id in seen:
            raise ValueError(f"duplicate family_id {record.family_id!r} in {context}")
        seen.add(record.family_id)


def _validate_sorted_ids(
    records: tuple[RealSourceProvenance, ...],
    *,
    context: str,
) -> None:
    family_ids = tuple(record.family_id for record in records)
    if family_ids != tuple(sorted(family_ids)):
        raise ValueError(f"{context} must be sorted by family_id")


def _validate_official_https_url(value: str, *, message: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError(message) from exc

    if parsed.scheme != "https" or parsed.fragment:
        raise ValueError(message)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(message)
    try:
        if parsed.port is not None:
            raise ValueError(message)
    except ValueError as exc:
        raise ValueError(message) from exc

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(message)
    if hostname.startswith(".") or hostname.endswith("."):
        raise ValueError(message)
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(message) from exc

    labels = hostname.split(".")
    if len(labels) < 2 or labels[-1] != "gov":
        raise ValueError(message)
    if any(not label for label in labels):
        raise ValueError(message)
    if any(_DNS_LABEL_PATTERN.fullmatch(label) is None for label in labels):
        raise ValueError(message)
    return value


def _canonical_origin_hostname(value: str) -> str:
    return _validated_hostname(value)


def _validated_hostname(value: str) -> str:
    parsed = urlsplit(value)
    assert parsed.hostname is not None
    return parsed.hostname


def _require_non_empty(value: str, message: str) -> str:
    if not value.strip():
        raise ValueError(message)
    return value


REAL_CASES_CANONICAL_DIGEST = (
    "7d5bc92d9020a35ff730c720bae70b632d87d74d1d21b788e0f024a03b851637"
)
ALL_CASES_CANONICAL_DIGEST = (
    "6d1d9725c4dddddc426f03d009e7fb6d680fdb31e737ed05257d057e479796b6"
)


__all__ = [
    "ALL_CASES_CANONICAL_DIGEST",
    "PROVENANCE_PATH",
    "REAL_CASES_CANONICAL_DIGEST",
    "REAL_SOURCES_PATH",
    "REAL_TRANSFORMATION_ORDER",
    "RETRIEVAL_DATE",
    "RealSourceChallenge",
    "RealSourceFamily",
    "RealSourceProvenance",
    "generate_real_cases",
    "load_real_source_families",
    "load_real_source_provenance",
]
