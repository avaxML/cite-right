"""Deterministic orchestration for authored evaluation case generation."""

from __future__ import annotations

from evaluation.builders.authored_sources import AUTHORED_FACT_TEMPLATES, FactTemplate
from evaluation.builders.transformations import TRANSFORMATIONS, Transformation
from evaluation.canonical import authoritative_case_id
from evaluation.schema import EvaluationCase


def generate_cases_for_template(
    template: FactTemplate,
    seed: int,
    transformations: tuple[Transformation, ...] = TRANSFORMATIONS,
) -> tuple[EvaluationCase, ...]:
    """Expand one template using the provided stable transformation ordering."""

    ordered_transformations = _validated_transformations(transformations)
    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()

    for transformation in ordered_transformations:
        generated_cases = tuple(transformation.generate(template, seed))
        if len(generated_cases) != 1:
            raise ValueError(
                "each transformation must generate exactly one case per template"
            )
        case = generated_cases[0]
        _validate_generated_case(
            case=case,
            document_family_id=template.family_id,
            transformation_family_id=transformation.name,
        )
        _remember_unique_case_id(
            case_id=case.case_id,
            seen_case_ids=seen_case_ids,
            context=f"template {template.family_id!r}",
        )
        cases.append(case)

    return tuple(cases)


def generate_all_authored_cases(
    seed: int,
    templates: tuple[FactTemplate, ...] = AUTHORED_FACT_TEMPLATES,
    transformations: tuple[Transformation, ...] = TRANSFORMATIONS,
) -> tuple[EvaluationCase, ...]:
    """Expand the full authored catalog independent of caller template ordering."""

    ordered_transformations = _validated_transformations(transformations)
    transformation_rank = {
        transformation.name: index
        for index, transformation in enumerate(ordered_transformations)
    }
    ordered_templates = _sorted_unique_templates(templates)

    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for template in ordered_templates:
        for case in generate_cases_for_template(
            template=template,
            seed=seed,
            transformations=ordered_transformations,
        ):
            _remember_unique_case_id(
                case_id=case.case_id,
                seen_case_ids=seen_case_ids,
                context="full authored catalog",
            )
            cases.append(case)

    return tuple(
        sorted(
            cases,
            key=lambda case: (
                case.document_family_id,
                transformation_rank[case.transformation_family_id],
                case.case_id,
            ),
        )
    )


def _validated_transformations(
    transformations: tuple[Transformation, ...],
) -> tuple[Transformation, ...]:
    seen_names: set[str] = set()
    for transformation in transformations:
        if transformation.name in seen_names:
            raise ValueError(
                f"duplicate transformation name {transformation.name!r} is not allowed"
            )
        seen_names.add(transformation.name)
    return transformations


def _sorted_unique_templates(
    templates: tuple[FactTemplate, ...],
) -> tuple[FactTemplate, ...]:
    family_ids = tuple(template.family_id for template in templates)
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("template family ids must be unique")
    return tuple(sorted(templates, key=lambda template: template.family_id))


def _validate_generated_case(
    *,
    case: EvaluationCase,
    document_family_id: str,
    transformation_family_id: str,
) -> None:
    if case.document_family_id != document_family_id:
        raise ValueError(
            "generated case document family does not match the input template"
        )
    if case.transformation_family_id != transformation_family_id:
        raise ValueError(
            "generated case transformation family does not match the input transformation"
        )
    expected_case_id = authoritative_case_id(case)
    if case.case_id != expected_case_id:
        raise ValueError("generated case id must match the authoritative case id")


def _remember_unique_case_id(
    *,
    case_id: str,
    seen_case_ids: set[str],
    context: str,
) -> None:
    if case_id in seen_case_ids:
        raise ValueError(f"duplicate case id {case_id!r} in {context}")
    seen_case_ids.add(case_id)


__all__ = [
    "TRANSFORMATIONS",
    "Transformation",
    "generate_all_authored_cases",
    "generate_cases_for_template",
]
