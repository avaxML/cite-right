from __future__ import annotations

from datetime import date
from typing import cast

import pytest

from evaluation.builders.cases import generate_all_authored_cases
from evaluation.builders.real_sources import generate_real_cases
from evaluation.canonical import authoritative_case_id
from evaluation.schema import (
    CharSpan,
    CitationRequirement,
    CitationTarget,
    ClaimAnnotation,
    EvaluationCase,
    EvaluationUnit,
    Provenance,
    ProvenanceKind,
    Source,
    Split,
)


def test_detect_leakage_reports_cross_split_duplicates_and_lineage_only():
    from evaluation.leakage import detect_leakage
    from evaluation.splitting import CaseLineage

    lineage_train = _build_case(
        family_id="family-lineage-a",
        transformation_id="positive-a",
        source_texts=("Lineage source one.",),
        answer="Lineage answer one.",
        split="train",
    )
    lineage_dev = _build_case(
        family_id="family-lineage-b",
        transformation_id="positive-b",
        source_texts=("Lineage source two.",),
        answer="Lineage answer two.",
        split="dev",
    )
    exact_train = _build_case(
        family_id="family-exact-a",
        transformation_id="exact-a",
        source_texts=("Exact source duplicate.",),
        answer="Exact answer duplicate.",
        split="train",
    )
    exact_holdout = _build_case(
        family_id="family-exact-b",
        transformation_id="exact-b",
        source_texts=("Exact source duplicate.",),
        answer="Exact answer duplicate.",
        split="holdout",
    )
    unicode_train = _build_case(
        family_id="family-unicode-a",
        transformation_id="unicode-a",
        source_texts=("Cafe\u0301 closes at eleven.",),
        answer="Cafe\u0301 closes at eleven.",
        split="train",
    )
    unicode_dev = _build_case(
        family_id="family-unicode-b",
        transformation_id="unicode-b",
        source_texts=("Café closes at eleven.",),
        answer="Café closes at eleven.",
        split="dev",
    )
    normalized_train = _build_case(
        family_id="family-normalized-a",
        transformation_id="normalized-a",
        source_texts=("Alpha, beta: gamma.",),
        answer="Alpha, beta: gamma.",
        split="train",
    )
    normalized_dev = _build_case(
        family_id="family-normalized-b",
        transformation_id="normalized-b",
        source_texts=("alpha beta gamma",),
        answer="alpha beta gamma",
        split="dev",
    )
    shingle_error_train = _build_case(
        family_id="family-shingle-error-a",
        transformation_id="shingle-error-a",
        source_texts=(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
        ),
        answer="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
        split="train",
    )
    shingle_error_dev = _build_case(
        family_id="family-shingle-error-b",
        transformation_id="shingle-error-b",
        source_texts=("alpha beta gamma delta epsilon zeta eta theta iota kappa mu",),
        answer="alpha beta gamma delta epsilon zeta eta theta iota kappa mu",
        split="dev",
    )
    shingle_warning_train = _build_case(
        family_id="family-shingle-warning-a",
        transformation_id="shingle-warning-a",
        source_texts=(
            "mars mission status remains green after engine check orbit update landing plan",
        ),
        answer="mars mission status remains green after engine check orbit update landing plan",
        split="train",
    )
    shingle_warning_holdout = _build_case(
        family_id="family-shingle-warning-b",
        transformation_id="shingle-warning-b",
        source_texts=(
            "mars mission status remains green after engine repair orbit update landing drill",
        ),
        answer="mars mission status remains green after engine repair orbit update landing drill",
        split="holdout",
    )
    same_split_duplicate_a = _build_case(
        family_id="family-same-split-a",
        transformation_id="same-split-a",
        source_texts=("Same split duplicate text.",),
        answer="Same split duplicate text.",
        split="train",
    )
    same_split_duplicate_b = _build_case(
        family_id="family-same-split-b",
        transformation_id="same-split-b",
        source_texts=("Same split duplicate text.",),
        answer="Same split duplicate text.",
        split="train",
    )

    cases = (
        same_split_duplicate_b,
        shingle_warning_holdout,
        exact_holdout,
        lineage_dev,
        normalized_train,
        shingle_error_dev,
        unicode_dev,
        normalized_dev,
        unicode_train,
        shingle_error_train,
        exact_train,
        same_split_duplicate_a,
        lineage_train,
        shingle_warning_train,
    )
    lineage = (
        CaseLineage(
            case_id=lineage_train.case_id, template_lineage_ids=("template-x",)
        ),
        CaseLineage(case_id=lineage_dev.case_id, template_lineage_ids=("template-x",)),
    )

    report = detect_leakage(cases, explicit_lineage=lineage)
    codes = [finding.code for finding in report.findings]

    assert codes == [
        "lineage_cross_split",
        "exact_duplicate_cross_split",
        "unicode_normalized_duplicate_cross_split",
        "normalized_duplicate_cross_split",
        "shingle_overlap_error",
        "shingle_overlap_warning",
    ]
    assert report.error_count == 5
    assert report.warning_count == 1
    assert tuple(finding.severity for finding in report.findings[:5]) == (
        "error",
        "error",
        "error",
        "error",
        "error",
    )
    assert report.findings[-1].severity == "warning"
    assert report.findings[-1].similarity is not None


def test_detect_leakage_is_permutation_invariant_and_rejects_duplicate_ids():
    from evaluation.leakage import detect_leakage

    first = _build_case(
        family_id="family-permute-a",
        transformation_id="permute-a",
        source_texts=("Shared source duplication.",),
        answer="Shared answer duplication.",
        split="train",
    )
    second = _build_case(
        family_id="family-permute-b",
        transformation_id="permute-b",
        source_texts=("Shared source duplication.",),
        answer="Shared answer duplication.",
        split="dev",
    )

    forward = detect_leakage((first, second))
    reverse = detect_leakage((second, first))

    assert forward == reverse

    duplicate = first.model_copy(update={"document_family_id": "family-permute-c"})
    with pytest.raises(ValueError, match="duplicate case id"):
        detect_leakage((first, duplicate))


def test_grouped_splitting_prevents_fatal_leakage_on_current_corpus():
    from evaluation.leakage import detect_leakage
    from evaluation.splitting import apply_split_assignments, assign_splits

    corpus = generate_all_authored_cases(seed=31) + generate_real_cases()
    report = assign_splits(corpus, seed=20260717)
    assigned = apply_split_assignments(corpus, report.assignment_by_case_id)

    leakage = detect_leakage(assigned)
    reversed_leakage = detect_leakage(tuple(reversed(assigned)))

    assert leakage == reversed_leakage
    assert leakage.error_count == 0
    assert all(finding.severity == "warning" for finding in leakage.findings)


def _build_case(
    *,
    family_id: str,
    transformation_id: str,
    source_texts: tuple[str, ...],
    answer: str,
    split: str,
    provenance_kind: str = "authored",
) -> EvaluationCase:
    sources = tuple(
        Source(source_id=f"source-{index}", text=text)
        for index, text in enumerate(source_texts, start=1)
    )
    answer_span = CharSpan(start=0, end=len(answer))
    primary_source = sources[0]
    unit = EvaluationUnit(
        unit_id="unit-answer",
        answer_span=answer_span,
        text=answer,
        claims=(
            ClaimAnnotation(
                claim_id="claim-answer",
                answer_span=answer_span,
                text=answer,
                label="entailed",
                citation_requirements=(
                    CitationRequirement(
                        requirement_id="req-primary",
                        alternatives=(
                            CitationTarget(
                                source_id=primary_source.source_id,
                                spans=(
                                    CharSpan(start=0, end=len(primary_source.text)),
                                ),
                            ),
                        ),
                    ),
                ),
                acceptable_retrieval_source_ids=(primary_source.source_id,),
            ),
        ),
    )
    base_case = EvaluationCase(
        case_id="case-pending",
        dataset_version="1.0.0",
        split=cast(Split, split),
        document_family_id=family_id,
        transformation_family_id=transformation_id,
        provenance=Provenance(
            kind=cast(ProvenanceKind, provenance_kind),
            title=f"{family_id} title",
            origin="internal-evaluation",
            publisher="Cite-Right",
            license="CC-BY-4.0",
            retrieval_date=date(2026, 7, 17),
        ),
        sources=sources,
        answer=answer,
        evaluation_units=(unit,),
        difficulty_tags=("science", transformation_id),
        generation=None,
        review=None,
    )
    return base_case.model_copy(update={"case_id": authoritative_case_id(base_case)})
