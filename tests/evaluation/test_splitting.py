from __future__ import annotations

from collections import Counter
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


def test_build_lineage_components_scopes_transformation_ids_and_keeps_transitive_links():
    from evaluation.splitting import CaseLineage, build_lineage_components

    alpha = _build_case(
        family_id="family-alpha",
        transformation_id="negation",
        source_texts=("Alpha source packet.",),
        answer="Alpha answer.",
    )
    beta = _build_case(
        family_id="family-beta",
        transformation_id="template-share",
        source_texts=("Bridge source packet.",),
        answer="Bridge answer.",
    )
    gamma = _build_case(
        family_id="family-gamma",
        transformation_id="template-share",
        source_texts=("Gamma source packet.",),
        answer="Gamma answer.",
    )
    delta = _build_case(
        family_id="family-delta",
        transformation_id="negation",
        source_texts=("Delta source packet.",),
        answer="Delta answer.",
    )

    components = build_lineage_components(
        (delta, gamma, beta, alpha),
        explicit_lineage=(
            CaseLineage(
                case_id=alpha.case_id,
                template_lineage_ids=("template-cluster",),
            ),
            CaseLineage(
                case_id=beta.case_id,
                template_lineage_ids=("template-cluster",),
                transformation_lineage_ids=("transform-cluster",),
            ),
            CaseLineage(
                case_id=gamma.case_id,
                transformation_lineage_ids=("transform-cluster",),
            ),
        ),
    )

    component_sets = {frozenset(component.case_ids) for component in components}

    assert frozenset({alpha.case_id, beta.case_id, gamma.case_id}) in component_sets
    assert frozenset({delta.case_id}) in component_sets
    assert len(components) == 2


def test_assign_splits_rejects_invalid_inputs_and_apply_preserves_authoritative_ids():
    from evaluation.splitting import (
        CaseLineage,
        apply_split_assignments,
        assign_splits,
    )

    case = _build_case(
        family_id="family-invalid",
        transformation_id="multi_source",
        source_texts=("A valid primary source.",),
        answer="A valid answer.",
    )
    duplicate = case.model_copy(update={"document_family_id": "family-duplicate"})

    with pytest.raises(ValueError, match="duplicate case id"):
        assign_splits((case, duplicate), seed=20260717)

    with pytest.raises(ValueError, match="seed must be an integer"):
        assign_splits((case,), seed=cast(int, True))

    with pytest.raises(ValueError, match="ratios must define exactly three positive values"):
        assign_splits((case,), ratios=cast(tuple[float, float, float], (0.6, 0.4)))

    with pytest.raises(ValueError, match="ratios must sum to 1.0"):
        assign_splits((case,), ratios=(0.6, 0.2, 0.1))

    with pytest.raises(ValueError, match="duplicate lineage metadata for case id"):
        assign_splits(
            (case,),
            explicit_lineage=(
                CaseLineage(case_id=case.case_id, template_lineage_ids=("template-a",)),
                CaseLineage(case_id=case.case_id, transformation_lineage_ids=("xf-a",)),
            ),
        )

    with pytest.raises(ValueError, match="unknown case id in explicit lineage metadata"):
        assign_splits(
            (case,),
            explicit_lineage=(CaseLineage(case_id="case-missing", template_lineage_ids=("x",)),),
        )

    report = assign_splits((case,), seed=20260717)
    applied = apply_split_assignments((case,), report.assignment_by_case_id)

    assert applied[0].case_id == case.case_id
    assert authoritative_case_id(applied[0]) == case.case_id


def test_assign_splits_current_corpus_is_deterministic_component_safe_and_balanced():
    from evaluation.splitting import apply_split_assignments, assign_splits

    corpus = generate_all_authored_cases(seed=31) + generate_real_cases()
    reversed_corpus = tuple(reversed(corpus))

    report = assign_splits(corpus, seed=20260717)
    reversed_report = assign_splits(reversed_corpus, seed=20260717)

    assert len(corpus) == 750
    assert len(report.assignment_by_case_id) == 750
    assert report.assignment_by_case_id == reversed_report.assignment_by_case_id
    assert report.assignment_hash == reversed_report.assignment_hash

    for component in report.components:
        assigned_splits = {
            report.assignment_by_case_id[case_id] for case_id in component.case_ids
        }
        assert len(assigned_splits) == 1

    total_cases = len(corpus)
    counts = Counter(report.assignment_by_case_id.values())
    targets: dict[Split, float] = {"train": 0.6, "dev": 0.2, "holdout": 0.2}
    for split_name, target_ratio in targets.items():
        observed_ratio = counts[split_name] / total_cases
        assert abs(observed_ratio - target_ratio) <= 0.05
        assert abs(report.deviation_by_split[split_name]) <= 0.05

    applied = apply_split_assignments(corpus, report.assignment_by_case_id)
    applied_ids = tuple(case.case_id for case in applied)

    assert applied_ids == tuple(case.case_id for case in corpus)
    assert all(authoritative_case_id(case) == case.case_id for case in applied)

    provenance_by_split = {
        split_name: {case.provenance.kind for case in applied if case.split == split_name}
        for split_name in ("train", "dev", "holdout")
    }
    assert all("authored" in kinds for kinds in provenance_by_split.values())
    assert all("public_domain" in kinds for kinds in provenance_by_split.values())


def _build_case(
    *,
    family_id: str,
    transformation_id: str,
    source_texts: tuple[str, ...],
    answer: str,
    split: str = "train",
    provenance_kind: str = "authored",
    difficulty_tags: tuple[str, ...] | None = None,
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
                                spans=(CharSpan(start=0, end=len(primary_source.text)),),
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
        difficulty_tags=difficulty_tags or ("science", transformation_id),
        generation=None,
        review=None,
    )
    return base_case.model_copy(update={"case_id": authoritative_case_id(base_case)})
