"""Deterministic transformation engine for authored evaluation cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from evaluation import DATASET_VERSION
from evaluation.builders.authored_sources import Evidence, Fact, FactTemplate
from evaluation.canonical import authoritative_case_id
from evaluation.schema import (
    CharSpan,
    CitationRequirement,
    CitationTarget,
    ClaimAnnotation,
    EvaluationCase,
    EvaluationUnit,
    GenerationRecipe,
    Provenance,
    Source,
    SupportLabel,
)

_NEGATIVE_TRANSFORMATIONS = frozenset(
    {"negation", "number", "unit", "date", "entity", "relation"}
)
_POSITIVE_TRANSFORMATIONS = frozenset(
    {
        "unicode",
        "duplicate_distractor",
        "multi_span",
        "multi_source",
        "unsupported_clause",
    }
)


class Transformation(Protocol):
    @property
    def name(self) -> str:
        ...

    def generate(self, template: FactTemplate, seed: int) -> tuple[EvaluationCase, ...]:
        ...


@dataclass(frozen=True)
class _ConfiguredTransformation:
    name: str

    def generate(self, template: FactTemplate, seed: int) -> tuple[EvaluationCase, ...]:
        fact = _fact_for_transformation(template, self.name)
        return (
            _build_case(
                template=template,
                fact=fact,
                transformation_name=self.name,
                seed=seed,
            ),
        )


TRANSFORMATIONS: tuple[Transformation, ...] = (
    _ConfiguredTransformation("negation"),
    _ConfiguredTransformation("number"),
    _ConfiguredTransformation("unit"),
    _ConfiguredTransformation("date"),
    _ConfiguredTransformation("entity"),
    _ConfiguredTransformation("relation"),
    _ConfiguredTransformation("modality"),
    _ConfiguredTransformation("unsupported_clause"),
    _ConfiguredTransformation("unicode"),
    _ConfiguredTransformation("duplicate_distractor"),
    _ConfiguredTransformation("multi_span"),
    _ConfiguredTransformation("multi_source"),
)


def _build_case(
    *,
    template: FactTemplate,
    fact: Fact,
    transformation_name: str,
    seed: int,
) -> EvaluationCase:
    answer = _answer_for_transformation(fact, transformation_name)
    unit = _build_evaluation_unit(
        answer=answer,
        fact=fact,
        transformation_name=transformation_name,
    )
    temporary_case = EvaluationCase(
        case_id="case-pending",
        dataset_version=DATASET_VERSION,
        split="train",
        document_family_id=template.family_id,
        transformation_family_id=transformation_name,
        provenance=Provenance(
            kind="authored",
            title=template.provenance_title,
            origin=template.provenance_origin,
            publisher=template.provenance_publisher,
            license=template.provenance_license,
            retrieval_date=template.provenance_retrieval_date,
        ),
        sources=_sources_for_transformation(template, fact, transformation_name),
        answer=answer,
        evaluation_units=(unit,),
        difficulty_tags=(template.domain, transformation_name),
        generation=GenerationRecipe(
            recipe_id=f"recipe-{template.family_id}-{transformation_name}-seed-{seed}",
            generator_name="evaluation.builders.transformations",
            prompt_version="authored-v1",
            seed=seed,
            notes=f"family={template.family_id}; transformation={transformation_name}",
        ),
        review=None,
    )
    resolved_case = temporary_case.model_copy(
        update={"case_id": authoritative_case_id(temporary_case)}
    )
    return EvaluationCase.model_validate(
        resolved_case.model_dump(mode="python", round_trip=True)
    )


def _build_evaluation_unit(
    *,
    answer: str,
    fact: Fact,
    transformation_name: str,
) -> EvaluationUnit:
    if transformation_name == "unsupported_clause":
        faithful_claim = _claim_text(fact, transformation_name, use_variant=False)
        suffix = _variant_string(fact, transformation_name, "unsupported_suffix")
        suffix_text = suffix.lstrip()
        suffix_start = len(faithful_claim) + (len(suffix) - len(suffix_text))
        claims = (
            ClaimAnnotation(
                claim_id="claim-supported",
                answer_span=CharSpan(start=0, end=len(faithful_claim)),
                text=faithful_claim,
                label="entailed",
                citation_requirements=_single_source_requirements(
                    requirement_id="req-primary",
                    source_id="source-primary",
                    evidence=fact.evidence,
                ),
                acceptable_retrieval_source_ids=("source-primary",),
            ),
            ClaimAnnotation(
                claim_id="claim-unsupported-clause",
                answer_span=CharSpan(start=suffix_start, end=len(answer)),
                text=suffix_text,
                label="not_in_sources",
            ),
        )
        return EvaluationUnit(
            unit_id="unit-answer",
            answer_span=CharSpan(start=0, end=len(answer)),
            text=answer,
            claims=claims,
        )

    claim = ClaimAnnotation(
        claim_id="claim-answer",
        answer_span=CharSpan(start=0, end=len(answer)),
        text=answer,
        label=_label_for_transformation(transformation_name),
        citation_requirements=_citation_requirements_for_transformation(
            fact=fact,
            transformation_name=transformation_name,
        ),
        acceptable_retrieval_source_ids=_retrieval_source_ids_for_transformation(
            transformation_name
        ),
        requires_non_contiguous_evidence=transformation_name == "multi_span",
    )
    return EvaluationUnit(
        unit_id="unit-answer",
        answer_span=CharSpan(start=0, end=len(answer)),
        text=answer,
        claims=(claim,),
    )


def _label_for_transformation(transformation_name: str) -> SupportLabel:
    if transformation_name in _NEGATIVE_TRANSFORMATIONS:
        return "contradicted"
    if transformation_name == "modality":
        return "not_in_sources"
    if transformation_name in _POSITIVE_TRANSFORMATIONS:
        return "entailed"
    raise ValueError(f"unsupported transformation family: {transformation_name}")


def _citation_requirements_for_transformation(
    *,
    fact: Fact,
    transformation_name: str,
) -> tuple[CitationRequirement, ...]:
    if transformation_name not in _POSITIVE_TRANSFORMATIONS:
        return ()
    if transformation_name == "multi_source":
        secondary_text = _variant_string(fact, transformation_name, "secondary_source_text")
        secondary_evidence = tuple(
            evidence for evidence in fact.evidence if evidence.text in secondary_text
        )
        if len(secondary_evidence) != 1:
            raise ValueError(
                "multi_source transformations require exactly one evidence fragment in the secondary source"
            )
        shared_evidence = secondary_evidence[0]
        primary_evidence = tuple(
            evidence
            for evidence in fact.evidence
            if evidence.slot_id != shared_evidence.slot_id
        )
        if not primary_evidence:
            raise ValueError(
                "multi_source transformations require primary-only evidence spans"
            )
        return (
            CitationRequirement(
                requirement_id="req-primary",
                alternatives=(
                    CitationTarget(
                        source_id="source-primary",
                        spans=tuple(evidence.span for evidence in primary_evidence),
                    ),
                ),
            ),
            CitationRequirement(
                requirement_id="req-secondary",
                alternatives=(
                    CitationTarget(
                        source_id="source-secondary",
                        spans=(_find_unique_span(secondary_text, shared_evidence.text),),
                    ),
                ),
            ),
        )
    if transformation_name == "multi_span":
        config = _variant_config(fact, transformation_name)
        slot_ids = tuple(_string_tuple(config, "evidence_slot_ids"))
        evidence_by_slot = {evidence.slot_id: evidence for evidence in fact.evidence}
        evidence = tuple(evidence_by_slot[slot_id] for slot_id in slot_ids)
    else:
        evidence = fact.evidence
    return _single_source_requirements(
        requirement_id="req-primary",
        source_id="source-primary",
        evidence=evidence,
    )


def _single_source_requirements(
    *,
    requirement_id: str,
    source_id: str,
    evidence: tuple[Evidence, ...],
) -> tuple[CitationRequirement, ...]:
    if not evidence:
        raise ValueError("entailed transformations require at least one evidence span")
    return (
        CitationRequirement(
            requirement_id=requirement_id,
            alternatives=(
                CitationTarget(
                    source_id=source_id,
                    spans=tuple(item.span for item in evidence),
                ),
            ),
        ),
    )


def _retrieval_source_ids_for_transformation(
    transformation_name: str,
) -> tuple[str, ...]:
    if transformation_name == "multi_source":
        return ("source-primary", "source-secondary")
    if transformation_name in _POSITIVE_TRANSFORMATIONS:
        return ("source-primary",)
    return ()


def _sources_for_transformation(
    template: FactTemplate,
    fact: Fact,
    transformation_name: str,
) -> tuple[Source, ...]:
    sources = [Source(source_id="source-primary", text=template.source_text)]
    if transformation_name == "duplicate_distractor":
        sources.append(
            Source(
                source_id="source-distractor",
                text=_variant_string(fact, transformation_name, "distractor_source_text"),
            )
        )
    elif transformation_name == "multi_source":
        sources.append(
            Source(
                source_id="source-secondary",
                text=_variant_string(fact, transformation_name, "secondary_source_text"),
            )
        )
    return tuple(sources)


def _answer_for_transformation(fact: Fact, transformation_name: str) -> str:
    if transformation_name == "unsupported_clause":
        return _claim_text(fact, transformation_name, use_variant=False) + _variant_string(
            fact,
            transformation_name,
            "unsupported_suffix",
        )
    return _claim_text(fact, transformation_name, use_variant=True)


def _claim_text(fact: Fact, transformation_name: str, *, use_variant: bool) -> str:
    config = _variant_config(fact, transformation_name)
    slots = dict(fact.slots)
    slot_updates = _string_mapping(config.get("slots"))
    slots.update(slot_updates)
    claim_template = fact.claim_template
    if use_variant and "claim_template" in config:
        claim_template = _require_string(config["claim_template"], "claim_template")
    return claim_template.format(**slots)


def _fact_for_transformation(
    template: FactTemplate,
    transformation_name: str,
) -> Fact:
    matches = tuple(
        fact
        for fact in template.facts
        if transformation_name in fact.adversarial_variants
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one fact for transformation {transformation_name!r}"
        )
    return matches[0]


def _variant_config(fact: Fact, transformation_name: str) -> Mapping[str, object]:
    config = fact.adversarial_variants[transformation_name]
    if not isinstance(config, Mapping):
        raise ValueError("transformation variants must be mappings")
    return cast(Mapping[str, object], config)


def _variant_string(fact: Fact, transformation_name: str, key: str) -> str:
    config = _variant_config(fact, transformation_name)
    if key not in config:
        raise ValueError(f"missing required variant key {key!r}")
    return _require_string(config[key], key)


def _string_mapping(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("slot overrides must be mappings")
    converted: dict[str, str] = {}
    for key, item in value.items():
        converted[_require_string(key, "slot override key")] = _require_string(
            item,
            "slot override value",
        )
    return converted


def _string_tuple(config: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw = config.get(key)
    if not isinstance(raw, tuple):
        raise ValueError(f"{key} must be a tuple of strings")
    values = tuple(_require_string(item, key) for item in raw)
    if not values:
        raise ValueError(f"{key} must be non-empty")
    return values


def _find_unique_span(source_text: str, fragment: str) -> CharSpan:
    if source_text.count(fragment) != 1:
        raise ValueError("citation evidence fragments must appear exactly once")
    start = source_text.index(fragment)
    return CharSpan(start=start, end=start + len(fragment))


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


__all__ = ["TRANSFORMATIONS", "Transformation"]
