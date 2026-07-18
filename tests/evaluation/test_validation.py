from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import date
from typing import Any, cast

import pytest

from evaluation.builders.cases import generate_all_authored_cases
from evaluation.builders.real_sources import generate_real_cases
from evaluation.canonical import authoritative_case_id, canonical_json_bytes, sha256_hex
from evaluation.manifest import (
    build_private_manifest,
    build_public_holdout_manifest,
    verify_private_manifest_expectations,
)
from evaluation.schema import (
    CharSpan,
    CitationRequirement,
    CitationTarget,
    ClaimAnnotation,
    EvaluationCase,
    EvaluationUnit,
    Provenance,
    ProvenanceKind,
    ReviewRecord,
    Source,
    Split,
)
from evaluation.splitting import apply_split_assignments, assign_splits
from evaluation.validation import DatasetBundle, validate_dataset


def test_validate_dataset_reports_raw_schema_failures_and_preserves_denominator() -> None:
    valid_case = _build_case(
        family_id="family-valid",
        transformation_id="positive",
        split="train",
    )
    invalid_answer_slice = _make_valid_case_mapping(case_id="case-bad-answer")
    invalid_answer_slice["evaluation_units"][0]["text"] = "incorrect slice"

    invalid_target_slice = _make_valid_case_mapping(case_id="case-bad-target")
    invalid_target_slice["evaluation_units"][0]["claims"][0]["citation_requirements"][0][
        "alternatives"
    ] = (
        {
            "source_id": "source-1",
            "spans": ({"start": 0, "end": 999},),
        },
    )

    report = validate_dataset(
        DatasetBundle(case_records=(valid_case, invalid_answer_slice, invalid_target_slice))
    )

    assert report.total_case_records == 3
    assert report.valid_case_records == 1
    assert report.invalid_case_records == 2
    assert report.is_valid is False
    assert _finding_codes(report) == (
        "schema_validation_error",
        "schema_validation_error",
        "leakage_analysis_partial",
    )
    assert report.findings[0].case_id == "case-bad-answer"
    assert report.findings[0].path.endswith("evaluation_units.0.text")
    assert "referenced answer slice" in report.findings[0].message
    assert report.findings[1].case_id == "case-bad-target"
    assert report.findings[1].path.endswith(
        "evaluation_units.0.claims.0.citation_requirements.0.alternatives.0.spans.0.end"
    )
    assert "referenced source text" in report.findings[1].message


def test_validate_dataset_reports_claim_contract_and_unit_overlap_errors() -> None:
    missing_requirement = _make_valid_case_mapping(case_id="case-missing-requirement")
    missing_requirement["evaluation_units"][0]["claims"][0]["citation_requirements"] = ()

    forbidden_negative_requirement = _make_valid_case_mapping(case_id="case-forbidden-negative")
    forbidden_negative_requirement["evaluation_units"][0]["claims"][0]["label"] = (
        "contradicted"
    )

    overlapping_units = _make_valid_case_mapping(case_id="case-overlap")
    overlapping_units["evaluation_units"] = (
        overlapping_units["evaluation_units"][0],
        {
            "unit_id": "unit-2",
            "answer_span": {"start": 3, "end": 9},
            "text": overlapping_units["answer"][3:9],
            "claims": (
                {
                    "claim_id": "claim-2",
                    "answer_span": {"start": 3, "end": 9},
                    "text": overlapping_units["answer"][3:9],
                    "label": "not_in_sources",
                },
            ),
        },
    )

    report = validate_dataset(
        DatasetBundle(
            case_records=(
                missing_requirement,
                forbidden_negative_requirement,
                overlapping_units,
            )
        )
    )

    assert report.invalid_case_records == 3
    assert report.valid_case_records == 0
    assert report.total_case_records == 3
    assert _finding_codes(report) == (
        "schema_validation_error",
        "schema_validation_error",
        "schema_validation_error",
        "leakage_analysis_partial",
    )
    messages = {finding.message for finding in report.findings}
    assert "entailed claims must define at least one citation requirement" in messages
    assert "negative claims must not define citation requirements" in messages
    assert "evaluation units must be ordered and non-overlapping" in messages


def test_validate_dataset_reports_duplicates_at_all_supported_scopes() -> None:
    duplicate_case_a = _build_case(
        family_id="family-duplicate-case-a",
        transformation_id="dup-case",
        split="train",
    )
    duplicate_case_b = duplicate_case_a.model_copy(
        update={
            "document_family_id": "family-duplicate-case-b",
        }
    )

    duplicate_source = _make_valid_case_mapping(case_id="case-duplicate-source")
    duplicate_source["sources"] = (
        duplicate_source["sources"][0],
        {
            **duplicate_source["sources"][0],
            "text": "Duplicated source id with different payload.",
        },
    )

    duplicate_unit = _make_valid_case_mapping(case_id="case-duplicate-unit")
    duplicate_unit["evaluation_units"] = (
        duplicate_unit["evaluation_units"][0],
        deepcopy(duplicate_unit["evaluation_units"][0]),
    )

    duplicate_claim = _make_valid_case_mapping(case_id="case-duplicate-claim")
    duplicate_claim["evaluation_units"][0]["claims"] = (
        duplicate_claim["evaluation_units"][0]["claims"][0],
        deepcopy(duplicate_claim["evaluation_units"][0]["claims"][0]),
    )

    duplicate_requirement = _make_valid_case_mapping(case_id="case-duplicate-requirement")
    duplicate_requirement["evaluation_units"][0]["claims"][0]["citation_requirements"] = (
        duplicate_requirement["evaluation_units"][0]["claims"][0]["citation_requirements"][0],
        deepcopy(
            duplicate_requirement["evaluation_units"][0]["claims"][0]["citation_requirements"][
                0
            ]
        ),
    )

    report = validate_dataset(
        DatasetBundle(
            case_records=(
                duplicate_case_a,
                duplicate_case_b,
                duplicate_source,
                duplicate_unit,
                duplicate_claim,
                duplicate_requirement,
            )
        )
    )

    assert report.total_case_records == 6
    assert report.valid_case_records == 0
    assert report.invalid_case_records == 6
    assert Counter(finding.code for finding in report.findings)["schema_validation_error"] == 4
    assert Counter(finding.code for finding in report.findings)["duplicate_case_id"] == 1
    assert any("source ids must be unique within a case" in finding.message for finding in report.findings)
    assert any(
        "evaluation unit ids must be unique within a case" in finding.message
        for finding in report.findings
    )
    assert any(
        "claim ids must be unique within an evaluation unit" in finding.message
        for finding in report.findings
    )
    assert any(
        "citation requirement ids must be unique within a claim" in finding.message
        for finding in report.findings
    )


def test_validate_dataset_requires_complete_real_and_permissive_provenance() -> None:
    real_missing_snapshot = _build_case(
        family_id="family-real",
        transformation_id="provenance",
        split="dev",
        provenance_kind="public_domain",
    ).model_copy(
        update={
            "provenance": Provenance(
                kind="public_domain",
                title="Real title",
                origin="https://example.com/article",
                publisher="Example News",
                license="CC-BY-4.0",
                retrieval_date=date(2026, 7, 17),
                snapshot_hash=None,
            )
        }
    )
    permissive_missing_origin = _build_case(
        family_id="family-permissive",
        transformation_id="provenance",
        split="holdout",
        provenance_kind="permissive_license",
    ).model_copy(
        update={
            "provenance": Provenance(
                kind="permissive_license",
                title="Permissive title",
                origin=None,
                publisher="Permissive Publisher",
                license="Apache-2.0",
                retrieval_date=date(2026, 7, 17),
                snapshot_hash="snapshot-permissive",
            )
        }
    )

    report = validate_dataset(
        DatasetBundle(case_records=(real_missing_snapshot, permissive_missing_origin))
    )

    assert report.total_case_records == 2
    assert report.valid_case_records == 0
    assert report.invalid_case_records == 2
    assert Counter(finding.code for finding in report.findings)["provenance_incomplete"] == 2
    assert {finding.case_id for finding in report.findings if finding.code == "provenance_incomplete"} == {
        real_missing_snapshot.case_id,
        permissive_missing_origin.case_id,
    }


def test_validate_dataset_requires_reviews_for_dev_and_holdout_only() -> None:
    approved_review = ReviewRecord(
        state="approved",
        reviewer="qa-reviewer",
        reviewed_at=date(2026, 7, 17),
        notes="Approved.",
    )
    train_case = _build_case(
        family_id="family-train",
        transformation_id="review",
        split="train",
        review=None,
    )
    dev_missing = _build_case(
        family_id="family-dev-missing",
        transformation_id="review",
        split="dev",
        review=None,
    )
    holdout_pending = _build_case(
        family_id="family-holdout-pending",
        transformation_id="review",
        split="holdout",
        review=ReviewRecord(state="pending"),
    )
    dev_approved = _build_case(
        family_id="family-dev-approved",
        transformation_id="review",
        split="dev",
        review=approved_review,
    )

    report = validate_dataset(
        DatasetBundle(
            case_records=(train_case, dev_missing, holdout_pending, dev_approved),
            require_reviews=True,
        )
    )

    review_gap_findings = tuple(
        finding for finding in report.findings if finding.code == "review_required"
    )
    assert len(review_gap_findings) == 2
    assert {finding.case_id for finding in review_gap_findings} == {
        dev_missing.case_id,
        holdout_pending.case_id,
    }
    assert train_case.case_id not in {finding.case_id for finding in review_gap_findings}
    assert dev_approved.case_id not in {finding.case_id for finding in review_gap_findings}


def test_validate_dataset_converts_leakage_findings_and_retains_shingle_warnings() -> None:
    duplicate_train = _build_case(
        family_id="family-exact-a",
        transformation_id="exact-a",
        split="train",
        answer="Exact answer duplicate.",
        source_texts=("Exact answer duplicate.",),
    )
    duplicate_dev = _build_case(
        family_id="family-exact-b",
        transformation_id="exact-b",
        split="dev",
        answer="Exact answer duplicate.",
        source_texts=("Exact answer duplicate.",),
    )
    shingle_train = _build_case(
        family_id="family-shingle-a",
        transformation_id="shingle-a",
        split="train",
        answer=(
            "mars mission status remains green after engine check orbit update landing plan"
        ),
        source_texts=(
            "mars mission status remains green after engine check orbit update landing plan",
        ),
    )
    shingle_holdout = _build_case(
        family_id="family-shingle-b",
        transformation_id="shingle-b",
        split="holdout",
        answer=(
            "mars mission status remains green after engine repair orbit update landing drill"
        ),
        source_texts=(
            "mars mission status remains green after engine repair orbit update landing drill",
        ),
    )

    report = validate_dataset(
        DatasetBundle(
            case_records=(
                shingle_holdout,
                duplicate_dev,
                shingle_train,
                duplicate_train,
            )
        )
    )

    assert "leakage_exact_duplicate_cross_split" in _finding_codes(report)
    assert "leakage_shingle_overlap_warning" in _finding_codes(report)
    exact_finding = next(
        finding
        for finding in report.findings
        if finding.code == "leakage_exact_duplicate_cross_split"
    )
    warning_finding = next(
        finding
        for finding in report.findings
        if finding.code == "leakage_shingle_overlap_warning"
    )
    assert exact_finding.severity == "error"
    assert warning_finding.severity == "warning"
    assert duplicate_train.case_id in exact_finding.message
    assert shingle_train.case_id in warning_finding.message


def test_validate_dataset_reports_non_canonical_ordering() -> None:
    alpha = _build_case(family_id="family-alpha", transformation_id="ordered", split="train")
    beta = _build_case(family_id="family-beta", transformation_id="ordered", split="train")
    gamma = _build_case(family_id="family-gamma", transformation_id="ordered", split="train")
    ordered = tuple(sorted((alpha, beta, gamma), key=lambda case: case.case_id))
    reversed_cases = tuple(reversed(ordered))

    report = validate_dataset(DatasetBundle(case_records=reversed_cases))

    ordering_findings = tuple(
        finding for finding in report.findings if finding.code == "case_order_not_canonical"
    )
    assert len(ordering_findings) == 1
    assert report.valid_case_records == 3
    assert report.invalid_case_records == 0
    assert report.total_case_records == 3
    assert ordered[0].case_id in ordering_findings[0].message


def test_assert_valid_raises_summary_exception_after_collecting_all_findings() -> None:
    invalid_mapping = _make_valid_case_mapping(case_id="case-summary-error")
    invalid_mapping["evaluation_units"][0]["claims"][0]["label"] = "contradicted"

    report = validate_dataset(DatasetBundle(case_records=(invalid_mapping,)))

    with pytest.raises(ValueError, match="dataset validation failed with 2 finding"):
        report.assert_valid()


def test_build_private_manifest_is_deterministic_and_order_invariant() -> None:
    approved_review = ReviewRecord(
        state="approved",
        reviewer="manifest-reviewer",
        reviewed_at=date(2026, 7, 17),
        notes="Approved.",
    )
    train_case = _build_case(
        family_id="family-train",
        transformation_id="man-train",
        split="train",
        difficulty_tags=("science", "family-a"),
        review=None,
    )
    dev_case = _build_case(
        family_id="family-dev",
        transformation_id="man-dev",
        split="dev",
        difficulty_tags=("science", "family-b"),
        provenance_kind="public_domain",
        review=approved_review,
    )
    holdout_case = _build_case(
        family_id="family-holdout",
        transformation_id="man-holdout",
        split="holdout",
        difficulty_tags=("history", "family-c"),
        provenance_kind="permissive_license",
        review=approved_review,
    )

    cases = (dev_case, holdout_case, train_case)
    manifest = build_private_manifest(cases, generated_at="2026-07-17")
    reversed_manifest = build_private_manifest(tuple(reversed(cases)), generated_at="2026-07-17")

    ordered_cases = tuple(
        sorted(cases, key=lambda case: (case.split, case.case_id, canonical_json_bytes(case)))
    )
    overall_payload = {
        "dataset_version": "1.0.0",
        "schema_version": "1.0.0",
        "cases": tuple(case.model_dump(mode="json") for case in ordered_cases),
    }
    train_payload = {
        "split": "train",
        "cases": tuple(
            case.model_dump(mode="json") for case in ordered_cases if case.split == "train"
        ),
    }

    assert manifest == reversed_manifest
    assert manifest.total_case_count == 3
    assert manifest.split_case_counts == {"train": 1, "dev": 1, "holdout": 1}
    assert manifest.overall_sha256 == sha256_hex(canonical_json_bytes(overall_payload))
    assert manifest.split_sha256["train"] == sha256_hex(canonical_json_bytes(train_payload))
    assert manifest.distributions["overall"]["expected_status"] == {
        "supported": 3,
        "partial": 0,
        "unsupported": 0,
    }
    assert manifest.distributions["overall"]["domain"] == {"history": 1, "science": 2}
    assert manifest.distributions["overall"]["provenance_kind"] == {
        "authored": 1,
        "public_domain": 1,
        "permissive_license": 1,
    }
    assert manifest.review_state_counts == {
        "overall": {"missing": 1, "pending": 0, "approved": 2, "rejected": 0},
        "train": {"missing": 1, "pending": 0, "approved": 0, "rejected": 0},
        "dev": {"missing": 0, "pending": 0, "approved": 1, "rejected": 0},
        "holdout": {"missing": 0, "pending": 0, "approved": 1, "rejected": 0},
    }


def test_build_public_holdout_manifest_redacts_case_level_data() -> None:
    holdout_case = _build_case(
        family_id="family-holdout-public",
        transformation_id="man-holdout",
        split="holdout",
        review=ReviewRecord(
            state="approved",
            reviewer="public-reviewer",
            reviewed_at=date(2026, 7, 17),
            notes="Approved for publication.",
        ),
    )
    private_manifest = build_private_manifest((holdout_case,), generated_at="2026-07-17")

    public_manifest = build_public_holdout_manifest(
        private_manifest,
        ciphertext_sha256="ciphertext-hash-placeholder",
        public_key_fingerprint="fingerprint-001",
        signature="signature-placeholder",
    )
    payload = public_manifest.model_dump(mode="json")
    payload_json = json.dumps(payload, sort_keys=True)

    assert payload["dataset_version"] == "1.0.0"
    assert payload["holdout_case_count"] == 1
    assert payload["ciphertext_sha256"] == "ciphertext-hash-placeholder"
    assert payload["public_key_fingerprint"] == "fingerprint-001"
    assert payload["signature"] == "signature-placeholder"
    assert "overall_sha256" not in payload
    assert "split_sha256" not in payload
    assert "review_state_counts" not in payload
    for forbidden in (
        holdout_case.case_id,
        holdout_case.answer,
        holdout_case.sources[0].text,
        holdout_case.document_family_id,
        "reviewer",
        "notes",
        "file_path",
    ):
        assert forbidden not in payload_json


def test_validate_dataset_reports_expected_manifest_mismatch() -> None:
    train_case = _build_case(
        family_id="family-manifest-train",
        transformation_id="manifest",
        split="train",
    )
    expected = build_private_manifest((train_case,), generated_at="2026-07-17")
    mismatched = expected.model_copy(
        update={"overall_sha256": "0" * 64, "total_case_count": 999}
    )

    report = validate_dataset(
        DatasetBundle(case_records=(train_case,), expected_private_manifest=mismatched)
    )

    assert verify_private_manifest_expectations(
        build_private_manifest((train_case,), generated_at="2026-07-17"),
        mismatched,
    )
    assert "manifest_mismatch" in _finding_codes(report)


def test_current_corpus_manifest_is_deterministic_and_validation_only_flags_reviews() -> None:
    corpus = generate_all_authored_cases(seed=31) + generate_real_cases()
    assignment_report = assign_splits(corpus, seed=20260717)
    assigned = apply_split_assignments(corpus, assignment_report.assignment_by_case_id)
    reversed_assigned = tuple(reversed(assigned))

    manifest = build_private_manifest(assigned, generated_at="2026-07-17")
    reversed_manifest = build_private_manifest(reversed_assigned, generated_at="2026-07-17")
    validation_report = validate_dataset(
        DatasetBundle(
            case_records=assigned,
            expected_private_manifest=manifest,
            require_reviews=True,
        )
    )

    assert len(assigned) == 750
    assert manifest == reversed_manifest
    assert manifest.total_case_count == 750
    assert manifest.overall_sha256 == reversed_manifest.overall_sha256
    assert validation_report.total_case_records == 750
    assert validation_report.invalid_case_records == sum(
        1 for case in assigned if case.split in {"dev", "holdout"}
    )
    assert validation_report.valid_case_records == (
        validation_report.total_case_records - validation_report.invalid_case_records
    )
    non_review_errors = [
        finding
        for finding in validation_report.findings
        if finding.severity == "error" and finding.code != "review_required"
    ]
    assert non_review_errors == []
    review_gap_findings = [
        finding for finding in validation_report.findings if finding.code == "review_required"
    ]
    assert len(review_gap_findings) == validation_report.invalid_case_records
    shingle_warnings = [
        finding
        for finding in validation_report.findings
        if finding.code == "leakage_shingle_overlap_warning"
    ]
    assert len(shingle_warnings) in {0, 109}


def _build_case(
    *,
    family_id: str,
    transformation_id: str,
    split: str = "train",
    answer: str | None = None,
    source_texts: tuple[str, ...] | None = None,
    provenance_kind: str = "authored",
    difficulty_tags: tuple[str, ...] | None = None,
    review: ReviewRecord | None = None,
) -> EvaluationCase:
    answer_text = answer or f"{family_id} answer for {transformation_id}."
    source_values = source_texts or (answer_text,)
    sources = tuple(
        Source(source_id=f"source-{index}", text=text)
        for index, text in enumerate(source_values, start=1)
    )
    answer_span = CharSpan(start=0, end=len(answer_text))
    primary_source = sources[0]
    unit = EvaluationUnit(
        unit_id="unit-answer",
        answer_span=answer_span,
        text=answer_text,
        claims=(
            ClaimAnnotation(
                claim_id="claim-answer",
                answer_span=answer_span,
                text=answer_text,
                label="entailed",
                citation_requirements=(
                    CitationRequirement(
                        requirement_id="req-primary",
                        alternatives=(
                            CitationTarget(
                                source_id=primary_source.source_id,
                                spans=(CharSpan(start=0, end=len(primary_source.text)),),
                            ),
                        ),
                    ),
                ),
                acceptable_retrieval_source_ids=(primary_source.source_id,),
            ),
        ),
    )
    provenance = Provenance(
        kind=cast(ProvenanceKind, provenance_kind),
        title=f"{family_id} title",
        origin="https://example.com/source",
        publisher="Cite-Right",
        license="CC-BY-4.0",
        retrieval_date=date(2026, 7, 17),
        snapshot_hash=f"snapshot-{family_id}",
    )
    base_case = EvaluationCase(
        case_id="case-pending",
        dataset_version="1.0.0",
        split=cast(Split, split),
        document_family_id=family_id,
        transformation_family_id=transformation_id,
        provenance=provenance,
        sources=sources,
        answer=answer_text,
        evaluation_units=(unit,),
        difficulty_tags=difficulty_tags or ("science", transformation_id),
        generation=None,
        review=review,
    )
    return base_case.model_copy(update={"case_id": authoritative_case_id(base_case)})


def _make_valid_case_mapping(*, case_id: str) -> dict[str, Any]:
    answer = "Paris is in France."
    source_text = "Paris is in France."
    return {
        "case_id": case_id,
        "dataset_version": "1.0.0",
        "split": "dev",
        "document_family_id": f"family-{case_id}",
        "transformation_family_id": "mapping",
        "provenance": {
            "kind": "authored",
            "title": "Worksheet",
            "origin": "internal",
            "publisher": "Cite-Right",
            "license": "proprietary-draft",
            "retrieval_date": date(2026, 7, 17),
            "snapshot_hash": f"snapshot-{case_id}",
        },
        "sources": (
            {
                "source_id": "source-1",
                "text": source_text,
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
                                            {"start": 0, "end": len(source_text)},
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
        "difficulty_tags": ("geography", "mapping"),
        "generation": None,
        "review": None,
    }


def _finding_codes(report: Any) -> tuple[str, ...]:
    return tuple(finding.code for finding in report.findings)
