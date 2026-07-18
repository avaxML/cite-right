"""Dataset validation for evaluation corpora and manifest expectations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from evaluation.canonical import canonical_json_bytes
from evaluation.leakage import LeakageFinding, detect_leakage
from evaluation.manifest import (
    DatasetManifest,
    build_private_manifest,
    verify_private_manifest_expectations,
)
from evaluation.schema import EvaluationCase

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
RecordInput = EvaluationCase | Mapping[str, object]


class ValidationFinding(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    severity: Literal["error", "warning"]
    code: str
    case_id: str | None
    path: str
    message: str


class ValidationReport(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    findings: tuple[ValidationFinding, ...]
    total_case_records: int
    valid_case_records: int
    invalid_case_records: int

    @property
    def is_valid(self) -> bool:
        return all(finding.severity != "error" for finding in self.findings)

    def assert_valid(self) -> None:
        if self.is_valid:
            return
        raise ValueError(
            "dataset validation failed with "
            f"{len(self.findings)} findings "
            f"({sum(1 for finding in self.findings if finding.severity == 'error')} errors, "
            f"{sum(1 for finding in self.findings if finding.severity == 'warning')} warnings)"
        )


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    case_records: tuple[RecordInput, ...]
    expected_private_manifest: DatasetManifest | None
    actual_manifest_generated_at: str | None
    require_reviews: bool

    def __init__(
        self,
        *,
        case_records: Iterable[RecordInput],
        expected_private_manifest: DatasetManifest | None = None,
        actual_manifest_generated_at: str | None = None,
        require_reviews: bool = False,
    ) -> None:
        frozen_records = tuple(_freeze_record(record) for record in case_records)
        if not frozen_records:
            raise ValueError("case_records must not be empty")
        object.__setattr__(self, "case_records", frozen_records)
        object.__setattr__(self, "expected_private_manifest", expected_private_manifest)
        object.__setattr__(
            self, "actual_manifest_generated_at", actual_manifest_generated_at
        )
        object.__setattr__(self, "require_reviews", require_reviews)


def validate_dataset(bundle: DatasetBundle) -> ValidationReport:
    findings: list[ValidationFinding] = []
    error_record_indexes: set[int] = set()
    validated_cases: list[tuple[int, EvaluationCase]] = []
    raw_case_id_indexes: defaultdict[str, list[int]] = defaultdict(list)

    for index, record in enumerate(bundle.case_records):
        raw_case_id = _extract_case_id(record)
        if raw_case_id is not None:
            raw_case_id_indexes[raw_case_id].append(index)
        raw_mapping = _thaw_record(record)
        try:
            validated_case = EvaluationCase.model_validate(raw_mapping)
        except ValidationError as exc:
            error_record_indexes.add(index)
            findings.extend(
                _findings_from_validation_error(
                    exc,
                    raw_mapping=raw_mapping,
                    raw_case_id=raw_case_id,
                )
            )
            continue
        validated_cases.append((index, validated_case))

    has_duplicate_case_ids = False
    for case_id, indexes in sorted(raw_case_id_indexes.items()):
        if len(indexes) < 2:
            continue
        has_duplicate_case_ids = True
        error_record_indexes.update(indexes)
        findings.append(
            ValidationFinding(
                severity="error",
                code="duplicate_case_id",
                case_id=case_id,
                path="case_id",
                message=(
                    f"duplicate case id {case_id!r} appears in records "
                    f"{', '.join(str(index) for index in indexes)}"
                ),
            )
        )

    ordered_valid_cases = tuple(case for _, case in validated_cases)
    case_index_by_id = {case.case_id: index for index, case in validated_cases}
    dataset_versions = tuple(sorted({case.dataset_version for case in ordered_valid_cases}))
    if len(dataset_versions) > 1:
        error_record_indexes.update(index for index, _ in validated_cases)
        findings.append(
            ValidationFinding(
                severity="error",
                code="mixed_dataset_version",
                case_id=None,
                path="/dataset_version",
                message=(
                    "dataset records must all share one dataset_version; found "
                    + ", ".join(repr(version) for version in dataset_versions)
                ),
            )
        )
    for index, case in validated_cases:
        provenance_missing = _missing_provenance_fields(case)
        if provenance_missing:
            error_record_indexes.add(index)
            findings.append(
                ValidationFinding(
                    severity="error",
                    code="provenance_incomplete",
                    case_id=case.case_id,
                    path="provenance",
                message=(
                    f"{case.provenance.kind} provenance requires "
                    f"{', '.join(provenance_missing)}"
                ),
            )
            )
        if bundle.require_reviews and case.split in {"dev", "holdout"}:
            if case.review is None or case.review.state != "approved":
                error_record_indexes.add(index)
                findings.append(
                    ValidationFinding(
                        severity="error",
                        code="review_required",
                        case_id=case.case_id,
                        path="review",
                        message=(
                            f"{case.split} cases require an approved review record "
                            "with audit metadata"
                        ),
                    )
                )

    ordering_finding = _case_order_finding(ordered_valid_cases)
    if ordering_finding is not None:
        findings.append(ordering_finding)

    if ordered_valid_cases and not has_duplicate_case_ids:
        leakage_report = detect_leakage(ordered_valid_cases)
        for leakage_finding in leakage_report.findings:
            findings.append(_convert_leakage_finding(leakage_finding))
            if leakage_finding.severity == "error":
                for case_id in leakage_finding.case_ids:
                    error_record_indexes.add(case_index_by_id[case_id])
    elif validated_cases:
        findings.append(
            ValidationFinding(
                severity="warning",
                code="leakage_analysis_partial",
                case_id=None,
                path="case_records",
                message="leakage analysis was partial because duplicate case ids were present",
            )
        )

    if len(validated_cases) != len(bundle.case_records):
        findings.append(
            ValidationFinding(
                severity="warning",
                code="leakage_analysis_partial",
                case_id=None,
                path="case_records",
                message=(
                    "leakage analysis was partial because schema-invalid records were "
                    "excluded from the analyzed subset"
                ),
            )
        )

    if bundle.expected_private_manifest is not None:
        manifest_blockers = _manifest_precondition_reasons(
            total_record_count=len(bundle.case_records),
            validated_record_count=len(validated_cases),
            has_duplicate_case_ids=has_duplicate_case_ids,
            has_mixed_dataset_versions=len(dataset_versions) > 1,
        )
        if manifest_blockers:
            findings.append(
                ValidationFinding(
                    severity="error",
                    code="manifest_unverifiable",
                    case_id=None,
                    path="/manifest",
                    message=(
                        "private manifest expectations could not be verified because "
                        + ", ".join(manifest_blockers)
                    ),
                )
            )
        elif ordered_valid_cases:
            try:
                actual_manifest = build_private_manifest(
                    ordered_valid_cases,
                    generated_at=bundle.actual_manifest_generated_at,
                )
            except ValueError as exc:
                findings.append(
                    ValidationFinding(
                        severity="error",
                        code="manifest_unverifiable",
                        case_id=None,
                        path="/manifest",
                        message=(
                            "private manifest expectations could not be verified because "
                            + str(exc)
                        ),
                    )
                )
            else:
                for mismatch in verify_private_manifest_expectations(
                    actual_manifest,
                    bundle.expected_private_manifest,
                ):
                    findings.append(
                        ValidationFinding(
                            severity="error",
                            code="manifest_mismatch",
                            case_id=None,
                            path=mismatch.path,
                            message=mismatch.message,
                        )
                    )

    ordered_findings = tuple(sorted(findings, key=_finding_sort_key))
    invalid_case_records = len(error_record_indexes)
    return ValidationReport(
        findings=ordered_findings,
        total_case_records=len(bundle.case_records),
        valid_case_records=len(bundle.case_records) - invalid_case_records,
        invalid_case_records=invalid_case_records,
    )


def _freeze_record(record: RecordInput) -> RecordInput:
    if isinstance(record, EvaluationCase):
        return record.model_copy(deep=True)
    if not isinstance(record, Mapping):
        raise TypeError("case records must be EvaluationCase or Mapping[str, object]")
    return cast(Mapping[str, object], _deep_freeze_jsonish(record, active_ids=set()))


def _deep_freeze_jsonish(value: object, *, active_ids: set[int]) -> object:
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in active_ids:
            raise ValueError("case records must not contain reference cycles")
        active_ids.add(value_id)
        try:
            frozen: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("case record mappings must use only string keys")
                frozen[key] = _deep_freeze_jsonish(item, active_ids=active_ids)
            return MappingProxyType(frozen)
        finally:
            active_ids.remove(value_id)
    if isinstance(value, list | tuple):
        value_id = id(value)
        if value_id in active_ids:
            raise ValueError("case records must not contain reference cycles")
        active_ids.add(value_id)
        try:
            return tuple(
                _deep_freeze_jsonish(item, active_ids=active_ids) for item in value
            )
        finally:
            active_ids.remove(value_id)
    return value


def _thaw_record(record: RecordInput) -> dict[str, object]:
    if isinstance(record, EvaluationCase):
        return cast(dict[str, object], record.model_dump(mode="python", round_trip=True))
    return cast(dict[str, object], _deep_thaw_jsonish(record))


def _deep_thaw_jsonish(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw_jsonish(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_deep_thaw_jsonish(item) for item in value)
    return value


def _extract_case_id(record: RecordInput) -> str | None:
    if isinstance(record, EvaluationCase):
        return record.case_id
    raw_case_id = record.get("case_id")
    return raw_case_id if isinstance(raw_case_id, str) else None


def _findings_from_validation_error(
    exc: ValidationError,
    *,
    raw_mapping: Mapping[str, object],
    raw_case_id: str | None,
) -> tuple[ValidationFinding, ...]:
    extracted: list[ValidationFinding] = []
    for error in exc.errors():
        message = _strip_pydantic_prefix(error["msg"])
        path = _error_path(error["loc"], raw_mapping, message)
        extracted.append(
            ValidationFinding(
                severity="error",
                code="schema_validation_error",
                case_id=raw_case_id,
                path=path,
                message=message,
            )
        )
    return tuple(extracted)


def _strip_pydantic_prefix(message: str) -> str:
    prefix = "Value error, "
    if message.startswith(prefix):
        return message[len(prefix) :]
    return message


def _error_path(
    location: tuple[object, ...],
    raw_mapping: Mapping[str, object],
    message: str,
) -> str:
    if location:
        return ".".join(str(part) for part in location)
    inferred_path = _infer_path_from_message(raw_mapping, message)
    return inferred_path if inferred_path is not None else "case"


def _infer_path_from_message(
    raw_mapping: Mapping[str, object],
    message: str,
) -> str | None:
    answer = raw_mapping.get("answer")
    evaluation_units = raw_mapping.get("evaluation_units")
    sources = raw_mapping.get("sources")
    if not isinstance(answer, str) or not isinstance(evaluation_units, tuple):
        return None
    source_map = _source_text_map(sources)
    if message == "evaluation unit text must equal the referenced answer slice":
        for unit_index, unit in enumerate(evaluation_units):
            if not isinstance(unit, Mapping):
                continue
            unit_path = f"evaluation_units.{unit_index}"
            text = unit.get("text")
            span = unit.get("answer_span")
            if isinstance(text, str) and isinstance(span, Mapping):
                start = span.get("start")
                end = span.get("end")
                if isinstance(start, int) and isinstance(end, int):
                    if answer[start:end] != text:
                        return f"{unit_path}.text"
        return "evaluation_units"
    if message == "claim text must equal the referenced answer slice":
        for claim_index, claim in _claim_paths(evaluation_units):
            span = claim.get("answer_span")
            text = claim.get("text")
            if isinstance(text, str) and isinstance(span, Mapping):
                start = span.get("start")
                end = span.get("end")
                if isinstance(start, int) and isinstance(end, int):
                    if answer[start:end] != text:
                        return f"evaluation_units.{claim_index[0]}.claims.{claim_index[1]}.text"
        return "evaluation_units"
    if message == "citation target spans must stay within the referenced source text":
        for indexes, target in _target_paths(evaluation_units):
            source_id = target.get("source_id")
            spans = target.get("spans")
            source_text = source_map.get(source_id) if isinstance(source_id, str) else None
            if source_text is None or not isinstance(spans, tuple):
                continue
            for span_index, span in enumerate(spans):
                if not isinstance(span, Mapping):
                    continue
                start = span.get("start")
                end = span.get("end")
                if isinstance(start, int) and start < 0:
                    return (
                        "evaluation_units."
                        f"{indexes[0]}.claims.{indexes[1]}.citation_requirements."
                        f"{indexes[2]}.alternatives.{indexes[3]}.spans.{span_index}.start"
                    )
                if isinstance(end, int) and end > len(source_text):
                    return (
                        "evaluation_units."
                        f"{indexes[0]}.claims.{indexes[1]}.citation_requirements."
                        f"{indexes[2]}.alternatives.{indexes[3]}.spans.{span_index}.end"
                    )
        return "sources"
    if message == "evaluation units must be ordered and non-overlapping":
        previous_end = -1
        for unit_index, unit in enumerate(evaluation_units):
            if not isinstance(unit, Mapping):
                continue
            span = unit.get("answer_span")
            if not isinstance(span, Mapping):
                continue
            start = span.get("start")
            end = span.get("end")
            if isinstance(start, int) and isinstance(end, int):
                if start < previous_end:
                    return f"evaluation_units.{unit_index}.answer_span.start"
                previous_end = end
        return "evaluation_units"
    duplicate_paths = {
        "source ids must be unique within a case": _duplicate_path(
            raw_mapping.get("sources"), "source_id", "sources"
        ),
        "evaluation unit ids must be unique within a case": _duplicate_path(
            evaluation_units, "unit_id", "evaluation_units"
        ),
        "claim ids must be unique within an evaluation unit": _duplicate_nested_path(
            evaluation_units, "claims", "claim_id", "evaluation_units"
        ),
        "citation requirement ids must be unique within a claim": _duplicate_requirement_path(
            evaluation_units
        ),
    }
    return duplicate_paths.get(message)


def _source_text_map(raw_sources: object) -> dict[str, str]:
    if not isinstance(raw_sources, tuple):
        return {}
    source_map: dict[str, str] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            continue
        source_id = source.get("source_id")
        text = source.get("text")
        if isinstance(source_id, str) and isinstance(text, str):
            source_map[source_id] = text
    return source_map


def _claim_paths(
    evaluation_units: tuple[object, ...],
) -> list[tuple[tuple[int, int], Mapping[str, object]]]:
    claims: list[tuple[tuple[int, int], Mapping[str, object]]] = []
    for unit_index, unit in enumerate(evaluation_units):
        if not isinstance(unit, Mapping):
            continue
        raw_claims = unit.get("claims")
        if not isinstance(raw_claims, tuple):
            continue
        for claim_index, claim in enumerate(raw_claims):
            if isinstance(claim, Mapping):
                claims.append(((unit_index, claim_index), claim))
    return claims


def _target_paths(
    evaluation_units: tuple[object, ...],
) -> list[tuple[tuple[int, int, int, int], Mapping[str, object]]]:
    targets: list[tuple[tuple[int, int, int, int], Mapping[str, object]]] = []
    for unit_index, unit in enumerate(evaluation_units):
        if not isinstance(unit, Mapping):
            continue
        raw_claims = unit.get("claims")
        if not isinstance(raw_claims, tuple):
            continue
        for claim_index, claim in enumerate(raw_claims):
            if not isinstance(claim, Mapping):
                continue
            requirements = claim.get("citation_requirements")
            if not isinstance(requirements, tuple):
                continue
            for requirement_index, requirement in enumerate(requirements):
                if not isinstance(requirement, Mapping):
                    continue
                alternatives = requirement.get("alternatives")
                if not isinstance(alternatives, tuple):
                    continue
                for alternative_index, target in enumerate(alternatives):
                    if isinstance(target, Mapping):
                        targets.append(
                            (
                                (unit_index, claim_index, requirement_index, alternative_index),
                                target,
                            )
                        )
    return targets


def _duplicate_path(
    raw_items: object,
    field_name: str,
    path_prefix: str,
) -> str | None:
    if not isinstance(raw_items, tuple):
        return None
    seen: set[str] = set()
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        value = item.get(field_name)
        if not isinstance(value, str):
            continue
        if value in seen:
            return f"{path_prefix}.{index}.{field_name}"
        seen.add(value)
    return None


def _duplicate_nested_path(
    evaluation_units: tuple[object, ...],
    nested_field: str,
    field_name: str,
    path_prefix: str,
) -> str | None:
    for unit_index, unit in enumerate(evaluation_units):
        if not isinstance(unit, Mapping):
            continue
        nested_items = unit.get(nested_field)
        if not isinstance(nested_items, tuple):
            continue
        seen: set[str] = set()
        for item_index, item in enumerate(nested_items):
            if not isinstance(item, Mapping):
                continue
            value = item.get(field_name)
            if not isinstance(value, str):
                continue
            if value in seen:
                return f"{path_prefix}.{unit_index}.{nested_field}.{item_index}.{field_name}"
            seen.add(value)
    return None


def _duplicate_requirement_path(evaluation_units: tuple[object, ...]) -> str | None:
    for unit_index, unit in enumerate(evaluation_units):
        if not isinstance(unit, Mapping):
            continue
        claims = unit.get("claims")
        if not isinstance(claims, tuple):
            continue
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                continue
            requirements = claim.get("citation_requirements")
            if not isinstance(requirements, tuple):
                continue
            seen: set[str] = set()
            for requirement_index, requirement in enumerate(requirements):
                if not isinstance(requirement, Mapping):
                    continue
                requirement_id = requirement.get("requirement_id")
                if not isinstance(requirement_id, str):
                    continue
                if requirement_id in seen:
                    return (
                        "evaluation_units."
                        f"{unit_index}.claims.{claim_index}.citation_requirements."
                        f"{requirement_index}.requirement_id"
                    )
                seen.add(requirement_id)
    return None


def _missing_provenance_fields(case: EvaluationCase) -> tuple[str, ...]:
    if case.provenance.kind == "authored":
        return ()
    required_fields = (
        ("title", case.provenance.title),
        ("origin", case.provenance.origin),
        ("publisher", case.provenance.publisher),
        ("license", case.provenance.license),
        ("retrieval_date", case.provenance.retrieval_date),
        ("snapshot_hash", case.provenance.snapshot_hash),
    )
    return tuple(field for field, value in required_fields if value is None)


def _case_order_finding(cases: tuple[EvaluationCase, ...]) -> ValidationFinding | None:
    if len(cases) < 2:
        return None
    canonical_order = tuple(
        sorted(
            cases,
            key=lambda case: (case.case_id, canonical_json_bytes(case.model_dump(mode="json"))),
        )
    )
    if cases == canonical_order:
        return None
    for observed, expected in zip(cases, canonical_order, strict=False):
        if observed != expected:
            return ValidationFinding(
                severity="warning",
                code="case_order_not_canonical",
                case_id=observed.case_id,
                path="/case_records",
                message=(
                    "case ordering is not canonical; expected "
                    f"{expected.case_id!r} before {observed.case_id!r}"
                ),
            )
    return None


def _convert_leakage_finding(finding: LeakageFinding) -> ValidationFinding:
    similarity_suffix = (
        f"; similarity={finding.similarity}"
        if finding.similarity is not None
        else ""
    )
    return ValidationFinding(
        severity=cast(Literal["error", "warning"], finding.severity),
        code=f"leakage_{finding.code}",
        case_id=finding.case_ids[0] if len(finding.case_ids) == 1 else None,
        path="/case_records",
        message=(
            f"cases={', '.join(finding.case_ids)}; "
            f"splits={', '.join(finding.split_names)}; "
            f"evidence={finding.evidence}{similarity_suffix}"
        ),
    )


def _manifest_precondition_reasons(
    *,
    total_record_count: int,
    validated_record_count: int,
    has_duplicate_case_ids: bool,
    has_mixed_dataset_versions: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if has_duplicate_case_ids:
        reasons.append("duplicate case ids were present")
    if has_mixed_dataset_versions:
        reasons.append("mixed dataset versions were present")
    if validated_record_count != total_record_count:
        reasons.append("schema-invalid records were present")
    return tuple(reasons)


def _finding_sort_key(finding: ValidationFinding) -> tuple[object, ...]:
    severity_rank = 0 if finding.severity == "error" else 1
    return (
        severity_rank,
        finding.code,
        "" if finding.case_id is None else finding.case_id,
        finding.path,
        finding.message,
    )


__all__ = [
    "DatasetBundle",
    "ValidationFinding",
    "ValidationReport",
    "validate_dataset",
]
