from __future__ import annotations

import ast
import json
import socket
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from importlib import import_module
from pathlib import Path

import pytest

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import CharSpan, EvaluationCase

ALL_TRANSFORMATION_NAMES = (
    "negation",
    "number",
    "unit",
    "date",
    "entity",
    "relation",
    "modality",
    "unsupported_clause",
    "unicode",
    "duplicate_distractor",
    "multi_span",
    "multi_source",
)
BALANCED_DOMAINS = (
    "finance",
    "health",
    "history",
    "policy",
    "science",
    "technology",
)


@dataclass(frozen=True)
class ExpectedFamilyBehavior:
    claim_label: str
    expected_answer: str
    requires_citations: bool
    expected_status: str
    expected_source_count: int
    expected_claim_count: int


EXPECTED_BEHAVIOR_BY_TRANSFORMATION = {
    "negation": ExpectedFamilyBehavior(
        claim_label="contradicted",
        expected_answer="Mercury does not complete one orbit every 88 days.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "number": ExpectedFamilyBehavior(
        claim_label="contradicted",
        expected_answer="Mercury completes one orbit every 89 days.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "unit": ExpectedFamilyBehavior(
        claim_label="contradicted",
        expected_answer="Mercury completes one orbit every 88 weeks.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "date": ExpectedFamilyBehavior(
        claim_label="contradicted",
        expected_answer="The mission launched in 1978.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "entity": ExpectedFamilyBehavior(
        claim_label="not_in_sources",
        expected_answer="Venus completes one orbit every 88 days.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "relation": ExpectedFamilyBehavior(
        claim_label="not_in_sources",
        expected_answer="Mercury documents one orbit every 88 days.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "modality": ExpectedFamilyBehavior(
        claim_label="not_in_sources",
        expected_answer="The report states the probe will remain active through 2030.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "unsupported_clause": ExpectedFamilyBehavior(
        claim_label="not_in_sources",
        expected_answer="The report states the probe should remain active through 2030. It is powered by a thorium battery.",
        requires_citations=False,
        expected_status="partial",
        expected_source_count=1,
        expected_claim_count=2,
    ),
    "unicode": ExpectedFamilyBehavior(
        claim_label="entailed",
        expected_answer="Engineers call the guidance mode café-safe in internal notes.",
        requires_citations=True,
        expected_status="supported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "duplicate_distractor": ExpectedFamilyBehavior(
        claim_label="entailed",
        expected_answer="The mission launched in 1977.",
        requires_citations=True,
        expected_status="supported",
        expected_source_count=2,
        expected_claim_count=1,
    ),
    "multi_span": ExpectedFamilyBehavior(
        claim_label="entailed",
        expected_answer="Mercury completes one orbit every 88 days.",
        requires_citations=True,
        expected_status="supported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "multi_source": ExpectedFamilyBehavior(
        claim_label="entailed",
        expected_answer="Mercury completes one orbit every 88 days. The mission launched in 1977.",
        requires_citations=True,
        expected_status="supported",
        expected_source_count=2,
        expected_claim_count=2,
    ),
}


def test_authored_catalog_is_balanced_and_structurally_valid() -> None:
    authored_sources = _authored_sources_module()
    templates = authored_sources.AUTHORED_FACT_TEMPLATES
    fact_type = authored_sources.Fact
    template_type = authored_sources.FactTemplate

    assert len(templates) == 60
    assert len({template.family_id for template in templates}) == len(templates)
    assert tuple(template.family_id for template in templates) == tuple(
        sorted(template.family_id for template in templates)
    )

    domain_counts = Counter(template.domain for template in templates)
    assert tuple(sorted(domain_counts)) == BALANCED_DOMAINS
    for domain in BALANCED_DOMAINS:
        assert domain_counts[domain] == 10

    for template in templates:
        assert isinstance(template, template_type)
        assert template.family_id.startswith(f"{template.domain}-")
        assert template.source_text
        assert template.facts
        assert template.provenance_title
        assert template.provenance_origin
        assert template.provenance_publisher
        assert template.provenance_license
        assert template.provenance_retrieval_date == date(2026, 7, 17)
        assert template.source_text.count("  ") == 0
        assert tuple(fact.fact_id for fact in template.facts) == tuple(
            sorted(fact.fact_id for fact in template.facts)
        )
        assert len({fact.fact_id for fact in template.facts}) == len(template.facts)
        advertised_transformations = {
            transformation_name
            for fact in template.facts
            for transformation_name in fact.adversarial_variants
        }
        assert advertised_transformations == set(ALL_TRANSFORMATION_NAMES)
        for fact in template.facts:
            assert isinstance(fact, fact_type)
            assert fact.claim_template.format(**fact.slots)
            assert set(fact.answer_slots).issubset(fact.slots)
            assert fact.evidence
            for evidence in fact.evidence:
                assert template.source_text[evidence.span.start : evidence.span.end] == evidence.text


def test_transformations_cover_all_required_families_by_stable_name() -> None:
    transformations = _transformations_module()
    names = tuple(transformation.name for transformation in transformations.TRANSFORMATIONS)

    assert names == ALL_TRANSFORMATION_NAMES
    assert len(set(names)) == len(ALL_TRANSFORMATION_NAMES)


@pytest.mark.parametrize("transformation_name", ALL_TRANSFORMATION_NAMES)
def test_transformations_are_deterministic_for_fixed_seed(
    transformation_name: str,
) -> None:
    template = _fixture_template()
    first = _generate_case(
        template=template,
        transformation_name=transformation_name,
        seed=23,
    )
    second = _generate_case(
        template=template,
        transformation_name=transformation_name,
        seed=23,
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


@pytest.mark.parametrize("transformation_name", ALL_TRANSFORMATION_NAMES)
def test_transformation_family_semantics_and_lineage(
    transformation_name: str,
) -> None:
    template = _fixture_template()
    case = _generate_case(
        template=template,
        transformation_name=transformation_name,
        seed=23,
    )
    expected = EXPECTED_BEHAVIOR_BY_TRANSFORMATION[transformation_name]

    assert case.document_family_id == template.family_id
    assert case.transformation_family_id == transformation_name
    assert case.provenance.kind == "authored"
    assert case.generation is not None
    assert case.generation.seed == 23
    assert case.generation.generator_name == "evaluation.builders.transformations"
    assert case.generation.prompt_version == "authored-v1"
    assert case.generation.notes is not None
    assert f"family={template.family_id}" in case.generation.notes
    assert f"transformation={transformation_name}" in case.generation.notes

    assert case.case_id
    assert case.case_id != ""
    assert case.answer == expected.expected_answer
    assert case.answer == _expected_answer_for_transformation(template, transformation_name)

    assert tuple(source.source_id for source in case.sources[:1]) == ("source-primary",)
    assert case.dataset_version == "1.0.0"
    assert case.review is None
    assert case.split == "train"
    assert len(case.evaluation_units) == 1
    assert case.evaluation_units[0].expected_status == expected.expected_status
    assert len(case.sources) == expected.expected_source_count

    claims = case.evaluation_units[0].claims
    if transformation_name == "unsupported_clause":
        assert len(claims) == 2
        assert claims[0].label == "entailed"
        assert claims[1].label == "not_in_sources"
    elif transformation_name == "multi_source":
        assert len(claims) == 2
        assert all(claim.label == "entailed" for claim in claims)
    else:
        assert len(claims) == 1
        assert claims[0].label == expected.claim_label
    assert len(claims) == expected.expected_claim_count

    for claim in claims:
        if claim.label == "entailed":
            assert claim.citation_requirements
            assert claim.acceptable_retrieval_source_ids
        else:
            assert claim.citation_requirements == ()

    _assert_positive_source_targets_slice_exact_text(
        case,
        template=template,
        transformation_name=transformation_name,
    )
    _assert_answer_targets_are_authored(
        case,
        template=template,
        transformation_name=transformation_name,
    )

    if transformation_name == "duplicate_distractor":
        assert case.sources[1].source_id == "source-distractor"
        assert _positive_claim(case).citation_requirements[0].alternatives[0].source_id == "source-primary"
    elif transformation_name == "multi_span":
        positive_claim = _positive_claim(case)
        target = positive_claim.citation_requirements[0].alternatives[0]
        assert len(target.spans) == 2
        assert positive_claim.requires_non_contiguous_evidence is True
        gap_text = case.sources[0].text[target.spans[0].end : target.spans[1].start]
        assert any(character.isalpha() for character in gap_text)
    elif transformation_name == "multi_source":
        entailed_claims = [claim for claim in claims if claim.label == "entailed"]
        assert len(entailed_claims) == 2
        assert tuple(claim.text for claim in entailed_claims) == (
            "Mercury completes one orbit every 88 days.",
            "The mission launched in 1977.",
        )
        assert case.sources[0].text == "Mercury completes one orbit every 88 days."
        assert case.sources[1].text == "Mission logs confirm the launch year was 1977."
        assert tuple(
            requirement.alternatives[0].source_id
            for claim in entailed_claims
            for requirement in claim.citation_requirements
        ) == ("source-primary", "source-secondary")


def test_authored_entity_and_relation_variants_are_not_in_sources_across_catalog() -> None:
    authored_sources = _authored_sources_module()

    for template in authored_sources.AUTHORED_FACT_TEMPLATES:
        for transformation_name in ("entity", "relation"):
            case = _generate_case(
                template=template,
                transformation_name=transformation_name,
                seed=23,
            )
            claim = case.evaluation_units[0].claims[0]

            assert claim.label == "not_in_sources"
            assert claim.citation_requirements == ()
            assert claim.text == case.answer
            assert all(claim.text not in source.text for source in case.sources)


def test_authored_entailed_targets_are_minimal_propositions_not_bare_slots() -> None:
    authored_sources = _authored_sources_module()

    for template in authored_sources.AUTHORED_FACT_TEMPLATES:
        for transformation_name in (
            "unicode",
            "duplicate_distractor",
            "multi_span",
            "multi_source",
            "unsupported_clause",
        ):
            case = _generate_case(
                template=template,
                transformation_name=transformation_name,
                seed=23,
            )
            target_texts = _entailed_target_texts(case)
            fact = _fact_for_transformation(template, transformation_name)

            assert target_texts
            assert all(text != evidence.text for text in target_texts for evidence in fact.evidence)
            assert all(len(text.split()) >= 3 for text in target_texts)
            assert all(any(character.isalpha() for character in text) for text in target_texts)


@pytest.mark.parametrize("transformation_name", ALL_TRANSFORMATION_NAMES)
def test_second_template_transformations_derive_from_template_not_fixture_constants(
    transformation_name: str,
) -> None:
    template = _second_catalog_template()
    case = _generate_case(
        template=template,
        transformation_name=transformation_name,
        seed=19,
    )

    _assert_case_derives_from_template(
        case,
        template=template,
        transformation_name=transformation_name,
    )
    assert case.answer != EXPECTED_BEHAVIOR_BY_TRANSFORMATION[transformation_name].expected_answer


@pytest.mark.parametrize(
    "transformation_name",
    ("unicode", "duplicate_distractor", "multi_span", "multi_source", "unsupported_clause"),
)
def test_exact_target_spans_match_authored_evidence_for_fixture_template(
    transformation_name: str,
) -> None:
    template = _fixture_template()
    case = _generate_case(
        template=template,
        transformation_name=transformation_name,
        seed=29,
    )

    _assert_positive_source_targets_slice_exact_text(
        case,
        template=template,
        transformation_name=transformation_name,
    )
    _assert_answer_targets_are_authored(
        case,
        template=template,
        transformation_name=transformation_name,
    )


def test_catalog_multi_span_cases_require_separated_proposition_evidence() -> None:
    authored_sources = _authored_sources_module()

    for template in authored_sources.AUTHORED_FACT_TEMPLATES:
        case = _generate_case(
            template=template,
            transformation_name="multi_span",
            seed=29,
        )
        expected_first, expected_second = _string_tuple(
            _variant_config(template, "multi_span"),
            "citation_texts",
        )
        claim = case.evaluation_units[0].claims[0]
        target = claim.citation_requirements[0].alternatives[0]
        source_text = case.sources[0].text
        first_span, second_span = target.spans
        first_text = source_text[first_span.start : first_span.end]
        second_text = source_text[second_span.start : second_span.end]
        answer_prefix, cadence = case.answer.rsplit(" every ", maxsplit=1)
        period = cadence.removesuffix(" days.")

        assert claim.requires_non_contiguous_evidence is True
        assert first_text == expected_first
        assert second_text == expected_second
        assert first_text.removesuffix(".") == answer_prefix
        assert period in second_text
        assert any(
            character.isalpha()
            for character in source_text[first_span.end : second_span.start]
        )


@pytest.mark.parametrize("transformation_name", ALL_TRANSFORMATION_NAMES)
def test_different_seeds_change_identity_but_not_semantics(
    transformation_name: str,
) -> None:
    template = _second_catalog_template()
    seed_23_case = _generate_case(
        template=template,
        transformation_name=transformation_name,
        seed=23,
    )
    seed_24_case = _generate_case(
        template=template,
        transformation_name=transformation_name,
        seed=24,
    )

    _assert_case_derives_from_template(
        seed_23_case,
        template=template,
        transformation_name=transformation_name,
    )
    _assert_case_derives_from_template(
        seed_24_case,
        template=template,
        transformation_name=transformation_name,
    )
    assert seed_23_case.generation is not None
    assert seed_24_case.generation is not None
    assert seed_23_case.generation.seed == 23
    assert seed_24_case.generation.seed == 24
    assert seed_23_case.case_id == _authoritative_case_id(seed_23_case)
    assert seed_24_case.case_id == _authoritative_case_id(seed_24_case)
    assert seed_23_case.case_id != seed_24_case.case_id
    assert _case_dump_without_identity_and_generation(
        seed_23_case
    ) == _case_dump_without_identity_and_generation(seed_24_case)


def test_every_catalog_family_produces_positive_and_adversarial_siblings() -> None:
    authored_sources = _authored_sources_module()
    cases_module = _cases_module()
    for template in authored_sources.AUTHORED_FACT_TEMPLATES:
        cases = cases_module.generate_cases_for_template(template=template, seed=11)
        labels = {claim.label for case in cases for unit in case.evaluation_units for claim in unit.claims}

        assert "entailed" in labels
        assert "contradicted" in labels or "not_in_sources" in labels
        assert tuple(case.transformation_family_id for case in cases) == ALL_TRANSFORMATION_NAMES


def test_generate_cases_for_template_rejects_transformations_that_emit_no_cases() -> None:
    cases_module = _cases_module()

    with pytest.raises(
        ValueError,
        match="each transformation must generate exactly one case per template",
    ):
        cases_module.generate_cases_for_template(
            template=_fixture_template(),
            seed=23,
            transformations=_with_cardinality_override("missing"),
        )


def test_generate_cases_for_template_rejects_transformations_that_emit_two_cases() -> None:
    cases_module = _cases_module()

    with pytest.raises(
        ValueError,
        match="each transformation must generate exactly one case per template",
    ):
        cases_module.generate_cases_for_template(
            template=_fixture_template(),
            seed=23,
            transformations=_with_cardinality_override("duplicated"),
        )


def test_generate_cases_for_template_accepts_generator_transformations() -> None:
    cases_module = _cases_module()
    tuple_cases = cases_module.generate_cases_for_template(
        template=_fixture_template(),
        seed=23,
    )
    generator_cases = cases_module.generate_cases_for_template(
        template=_fixture_template(),
        seed=23,
        transformations=(
            transformation for transformation in _transformations_module().TRANSFORMATIONS
        ),
    )

    assert tuple(case.case_id for case in generator_cases) == tuple(
        case.case_id for case in tuple_cases
    )
    assert _canonical_case_digest(generator_cases) == _canonical_case_digest(tuple_cases)


def test_generate_all_authored_cases_accepts_generator_templates_and_transformations() -> None:
    authored_sources = _authored_sources_module()
    cases_module = _cases_module()
    tuple_cases = cases_module.generate_all_authored_cases(seed=31)
    generator_cases = cases_module.generate_all_authored_cases(
        seed=31,
        templates=(template for template in authored_sources.AUTHORED_FACT_TEMPLATES),
        transformations=(
            transformation for transformation in _transformations_module().TRANSFORMATIONS
        ),
    )

    assert [case.case_id for case in generator_cases] == [case.case_id for case in tuple_cases]
    assert _canonical_case_digest(generator_cases) == _canonical_case_digest(tuple_cases)


def test_generate_cases_for_template_rejects_duplicate_transformation_names() -> None:
    cases_module = _cases_module()

    with pytest.raises(
        ValueError,
        match=r"^duplicate transformation name 'negation' is not allowed$",
    ):
        cases_module.generate_cases_for_template(
            template=_fixture_template(),
            seed=23,
            transformations=(
                _transformation_by_name("negation"),
                _transformation_by_name("negation"),
            ),
        )


def test_generate_cases_for_template_rejects_wrong_document_family_id() -> None:
    cases_module = _cases_module()

    with pytest.raises(
        ValueError,
        match=r"^generated case document family does not match the input template$",
    ):
        cases_module.generate_cases_for_template(
            template=_fixture_template(),
            seed=23,
            transformations=(
                _StaticTransformation(
                    name="negation",
                    cases=(
                        _case_for("negation", seed=23).model_copy(
                            update={"document_family_id": "science-wrong-family"}
                        ),
                    ),
                ),
            ),
        )


def test_generate_cases_for_template_rejects_wrong_transformation_family_id() -> None:
    cases_module = _cases_module()

    with pytest.raises(
        ValueError,
        match=r"^generated case transformation family does not match the input transformation$",
    ):
        cases_module.generate_cases_for_template(
            template=_fixture_template(),
            seed=23,
            transformations=(
                _StaticTransformation(
                    name="negation",
                    cases=(
                        _case_for("negation", seed=23).model_copy(
                            update={"transformation_family_id": "number"}
                        ),
                    ),
                ),
            ),
        )


def test_generate_cases_for_template_rejects_non_authoritative_case_id() -> None:
    cases_module = _cases_module()

    with pytest.raises(
        ValueError,
        match=r"^generated case id must match the authoritative case id$",
    ):
        cases_module.generate_cases_for_template(
            template=_fixture_template(),
            seed=23,
            transformations=(
                _StaticTransformation(
                    name="negation",
                    cases=(
                        _case_for("negation", seed=23).model_copy(
                            update={"case_id": "case-not-authoritative"}
                        ),
                    ),
                ),
            ),
        )


def test_generate_all_authored_cases_rejects_duplicate_template_family_ids() -> None:
    cases_module = _cases_module()
    template = _fixture_template()

    with pytest.raises(
        ValueError,
        match=r"^template family ids must be unique$",
    ):
        cases_module.generate_all_authored_cases(
            seed=23,
            templates=(template, template),
            transformations=(_transformation_by_name("negation"),),
        )


def test_generate_cases_for_template_rejects_duplicate_output_case_ids() -> None:
    cases_module = _cases_module()

    with pytest.raises(
        ValueError,
        match=r"^duplicate case id 'case-[0-9a-f]{20}' in template 'science-orbital-archive'$",
    ):
        cases_module.generate_cases_for_template(
            template=_fixture_template(),
            seed=23,
            transformations=(
                _CollidingTransformation(
                    validation_name="alpha",
                    runtime_name="negation",
                    cases=(_case_for("negation", seed=23),),
                ),
                _CollidingTransformation(
                    validation_name="beta",
                    runtime_name="negation",
                    cases=(_case_for("negation", seed=23),),
                ),
            ),
        )


def test_catalog_generation_is_order_independent_after_stable_sorting() -> None:
    authored_sources = _authored_sources_module()
    cases_module = _cases_module()
    forward = cases_module.generate_all_authored_cases(seed=31)
    repeat = cases_module.generate_all_authored_cases(seed=31)
    reverse = cases_module.generate_all_authored_cases(
        templates=tuple(reversed(authored_sources.AUTHORED_FACT_TEMPLATES)),
        seed=31,
    )

    assert len(forward) == 720
    assert len({case.case_id for case in forward}) == 720
    assert Counter(case.document_family_id for case in forward) == {
        template.family_id: len(ALL_TRANSFORMATION_NAMES)
        for template in authored_sources.AUTHORED_FACT_TEMPLATES
    }
    assert Counter(case.transformation_family_id for case in forward) == {
        transformation_name: len(authored_sources.AUTHORED_FACT_TEMPLATES)
        for transformation_name in ALL_TRANSFORMATION_NAMES
    }
    assert [case.case_id for case in forward] == [case.case_id for case in repeat]
    assert [case.case_id for case in forward] == [case.case_id for case in reverse]
    assert _canonical_case_digest(forward) == _canonical_case_digest(repeat)
    assert _canonical_case_digest(forward) == _canonical_case_digest(reverse)
    assert _canonical_case_digest(forward) == "651142bd1bc5f40aa807ee59bd71fd642b11b80215b1cd1196a1423451cc7eeb"


def test_case_ids_are_authoritative_and_labels_do_not_depend_on_runtime_outputs() -> None:
    case = _generate_case(
        template=_fixture_template(),
        transformation_name="multi_source",
        seed=17,
    )

    assert case.case_id.startswith("case-")
    assert case.case_id == _authoritative_case_id(case)
    assert case.answer == "Mercury completes one orbit every 88 days. The mission launched in 1977."


def test_fact_validation_rejects_unresolved_answer_slots() -> None:
    authored_sources = _authored_sources_module()
    source_text = "Mercury completes one orbit every 88 days."

    with pytest.raises(ValueError, match="answer slots must reference defined slots"):
        authored_sources.Fact(
            fact_id="fact-orbit",
            claim_template="{planet} completes one orbit every {days} days.",
            slots={
                "planet": "Mercury",
                "days": "88",
            },
            answer_slots=("planet", "missing"),
            evidence=(
                {
                    "slot_id": "planet",
                    "text": "Mercury",
                    "span": _span(source_text, "Mercury"),
                },
            ),
            adversarial_variants={"negation": {"claim_template": "not used"}},
        )


def test_fact_validation_rejects_claim_template_with_missing_slot_reference() -> None:
    authored_sources = _authored_sources_module()
    source_text = "Mercury completes one orbit every 88 days."

    with pytest.raises(ValueError, match="claim templates must resolve using defined slots"):
        authored_sources.Fact(
            fact_id="fact-orbit",
            claim_template="{planet} completes one orbit every {duration} days.",
            slots={
                "planet": "Mercury",
                "days": "88",
            },
            answer_slots=("planet", "days"),
            evidence=(
                {
                    "slot_id": "planet",
                    "text": "Mercury",
                    "span": _span(source_text, "Mercury"),
                },
            ),
            adversarial_variants={"negation": {"claim_template": "not used"}},
        )


def test_fact_validation_rejects_evidence_out_of_source_order() -> None:
    authored_sources = _authored_sources_module()
    source_text = "Mercury completes one orbit every 88 days."

    with pytest.raises(
        ValueError,
        match="fact evidence must be ordered by source span",
    ):
        authored_sources.Fact(
            fact_id="fact-orbit",
            claim_template="{planet} completes one orbit every {days} days.",
            slots={
                "planet": "Mercury",
                "days": "88",
            },
            answer_slots=("planet", "days"),
            evidence=(
                {
                    "slot_id": "days",
                    "text": "88",
                    "span": _span(source_text, "88"),
                },
                {
                    "slot_id": "planet",
                    "text": "Mercury",
                    "span": _span(source_text, "Mercury"),
                },
            ),
            adversarial_variants={"negation": {"claim_template": "not used"}},
        )


def test_fact_is_deeply_immutable_and_round_trips_from_json() -> None:
    authored_sources = _authored_sources_module()
    source_text = "Mercury completes one orbit every 88 days."
    fact = authored_sources.Fact(
        fact_id="fact-orbit",
        claim_template="{subject} completes one orbit every {period} days.",
        slots={
            "subject": "Mercury",
            "period": "88",
        },
        answer_slots=("subject", "period"),
        evidence=(
            {
                "slot_id": "subject",
                "text": "Mercury",
                "span": _span(source_text, "Mercury"),
            },
            {
                "slot_id": "period",
                "text": "88",
                "span": _span(source_text, "88"),
            },
        ),
        adversarial_variants={
            "number": {
                "slots": {
                    "period": "89",
                }
            },
            "negation": {
                "claim_template": "{subject} does not complete one orbit every {period} days."
            },
        },
    )

    with pytest.raises(TypeError):
        fact.slots["subject"] = "Venus"

    with pytest.raises(TypeError):
        fact.adversarial_variants["new"] = {}

    with pytest.raises(TypeError):
        fact.adversarial_variants["number"]["slots"]["period"] = "90"

    dumped_once = fact.model_dump(mode="json")
    dumped_twice = fact.model_dump(mode="json")
    assert dumped_once == dumped_twice

    dumped_json_once = fact.model_dump_json()
    dumped_json_twice = fact.model_dump_json()
    assert dumped_json_once == dumped_json_twice

    round_tripped = authored_sources.Fact.model_validate_json(dumped_json_once)
    assert round_tripped == fact

    with pytest.raises(TypeError):
        round_tripped.slots["subject"] = "Venus"

    with pytest.raises(TypeError):
        round_tripped.adversarial_variants["number"]["slots"]["period"] = "90"


def test_fact_template_validation_rejects_out_of_bounds_evidence_span() -> None:
    authored_sources = _authored_sources_module()

    with pytest.raises(
        ValueError,
        match="evidence spans must stay within the source text bounds",
    ):
        authored_sources.FactTemplate(
            family_id="science-invalid-template",
            domain="science",
            source_text="Mercury completes one orbit every 88 days.",
            facts=(
                authored_sources.Fact(
                    fact_id="fact-orbit",
                    claim_template="{planet} completes one orbit every {days} days.",
                    slots={
                        "planet": "Mercury",
                        "days": "88",
                    },
                    answer_slots=("planet", "days"),
                    evidence=(
                        {
                            "slot_id": "planet",
                            "text": "Mercury",
                            "span": {"start": 0, "end": 99},
                        },
                    ),
                    adversarial_variants={"negation": {"claim_template": "not used"}},
                ),
            ),
            provenance_title="Invalid source",
            provenance_origin="internal-evaluation",
            provenance_publisher="Cite-Right",
            provenance_license="proprietary-draft",
            provenance_retrieval_date=date(2026, 7, 17),
        )


def test_fixture_template_fact_ids_are_sorted() -> None:
    template = _fixture_template()

    assert tuple(fact.fact_id for fact in template.facts) == tuple(
        sorted(fact.fact_id for fact in template.facts)
    )


def _fixture_template():
    authored_sources = _authored_sources_module()
    source_text = (
        "Mercury completes one orbit every 88 days. "
        "The mission launched in 1977. "
        "The report states the probe should remain active through 2030. "
        "Engineers call the guidance mode cafe\u0301-safe in internal notes."
    )
    return authored_sources.FactTemplate(
        family_id="science-orbital-archive",
        domain="science",
        source_text=source_text,
        facts=(
            authored_sources.Fact(
                fact_id="fact-conjunction",
                claim_template="{planet} completes one orbit every {days} days and the mission launched in {launch_year}.",
                slots={
                    "planet": "Mercury",
                    "days": "88",
                    "launch_year": "1977",
                },
                answer_slots=("planet", "days", "launch_year"),
                evidence=(
                    {
                        "slot_id": "planet",
                        "text": "Mercury",
                        "span": _span(source_text, "Mercury"),
                    },
                    {
                        "slot_id": "days",
                        "text": "88",
                        "span": _span(source_text, "88"),
                    },
                    {
                        "slot_id": "launch_year",
                        "text": "1977",
                        "span": _span(source_text, "1977"),
                    },
                ),
                adversarial_variants={
                    "multi_source": {
                        "answer_text": "Mercury completes one orbit every 88 days. The mission launched in 1977.",
                        "primary_claim_text": "Mercury completes one orbit every 88 days.",
                        "primary_source_text": "Mercury completes one orbit every 88 days.",
                        "secondary_claim_text": "The mission launched in 1977.",
                        "secondary_source_text": "Mission logs confirm the launch year was 1977.",
                    }
                },
            ),
            authored_sources.Fact(
                fact_id="fact-launch",
                claim_template="The mission launched in {launch_year}.",
                slots={
                    "launch_year": "1977",
                },
                answer_slots=("launch_year",),
                evidence=(
                    {
                        "slot_id": "launch_year",
                        "text": "1977",
                        "span": _span(source_text, "1977"),
                    },
                ),
                adversarial_variants={
                    "date": {"slots": {"launch_year": "1978"}},
                    "duplicate_distractor": {"distractor_source_text": "The mission launched in 1978."},
                },
            ),
            authored_sources.Fact(
                fact_id="fact-modality",
                claim_template="The report states the probe should remain active through {end_year}.",
                slots={
                    "end_year": "2030",
                },
                answer_slots=("end_year",),
                evidence=(
                    {
                        "slot_id": "end_year",
                        "text": "2030",
                        "span": _span(source_text, "2030"),
                    },
                ),
                adversarial_variants={
                    "modality": {
                        "claim_template": "The report states the probe will remain active through {end_year}."
                    },
                    "unsupported_clause": {
                        "unsupported_suffix": " It is powered by a thorium battery."
                    },
                },
            ),
            authored_sources.Fact(
                fact_id="fact-orbit",
                claim_template="{planet} completes one orbit every {days} days.",
                slots={
                    "planet": "Mercury",
                    "days": "88",
                },
                answer_slots=("planet", "days"),
                evidence=(
                    {
                        "slot_id": "planet",
                        "text": "Mercury",
                        "span": _span(source_text, "Mercury"),
                    },
                    {
                        "slot_id": "days",
                        "text": "88",
                        "span": _span(source_text, "88"),
                    },
                ),
                adversarial_variants={
                    "negation": {"claim_template": "{planet} does not complete one orbit every {days} days."},
                    "number": {"slots": {"days": "89"}},
                    "unit": {"claim_template": "{planet} completes one orbit every {days} weeks."},
                    "entity": {"slots": {"planet": "Venus"}},
                    "relation": {"claim_template": "{planet} documents one orbit every {days} days."},
                    "multi_span": {
                        "citation_texts": (
                            "Mercury completes one orbit.",
                            "That orbit lasts 88 days.",
                        ),
                        "primary_source_text": (
                            "Mercury completes one orbit. Archive staff track telescope windows separately. "
                            "That orbit lasts 88 days."
                        ),
                    },
                },
            ),
            authored_sources.Fact(
                fact_id="fact-unicode",
                claim_template="Engineers call the guidance mode {mode_name} in internal notes.",
                slots={
                    "mode_name": "cafe\u0301-safe",
                },
                answer_slots=("mode_name",),
                evidence=(
                    {
                        "slot_id": "mode_name",
                        "text": "cafe\u0301-safe",
                        "span": _span(source_text, "cafe\u0301-safe"),
                    },
                ),
                adversarial_variants={
                    "unicode": {"slots": {"mode_name": "café-safe"}},
                },
            ),
        ),
        provenance_title="Orbital archive bulletin",
        provenance_origin="internal-evaluation",
        provenance_publisher="Cite-Right",
        provenance_license="proprietary-draft",
        provenance_retrieval_date=date(2026, 7, 17),
    )


def _positive_claim(case: EvaluationCase):
    for claim in case.evaluation_units[0].claims:
        if claim.label == "entailed":
            return claim
    raise AssertionError("expected an entailed claim")


def _assert_positive_source_targets_slice_exact_text(
    case: EvaluationCase,
    *,
    template,
    transformation_name: str,
) -> None:
    source_text_by_id = {source.source_id: source.text for source in case.sources}
    expected_requirement_groups = _expected_positive_requirements(
        template=template,
        transformation_name=transformation_name,
    )
    entailed_claims = [
        claim for claim in case.evaluation_units[0].claims if claim.label == "entailed"
    ]

    assert len(entailed_claims) == len(expected_requirement_groups)
    for claim, expected_requirements in zip(
        entailed_claims,
        expected_requirement_groups,
        strict=True,
    ):
        assert len(claim.citation_requirements) == len(expected_requirements)
        for requirement, expected in zip(
            claim.citation_requirements,
            expected_requirements,
            strict=True,
        ):
            assert len(requirement.alternatives) == 1
            alternative = requirement.alternatives[0]
            assert alternative.source_id == expected.source_id
            assert alternative.spans == expected.spans
            for span, expected_text in zip(
                alternative.spans,
                expected.texts,
                strict=True,
            ):
                assert source_text_by_id[alternative.source_id][span.start : span.end] == expected_text


def _assert_answer_targets_are_authored(
    case: EvaluationCase,
    *,
    template,
    transformation_name: str,
) -> None:
    assert case.answer == _expected_answer_for_transformation(template, transformation_name)
    for unit in case.evaluation_units:
        assert case.answer[unit.answer_span.start : unit.answer_span.end] == unit.text
        for claim in unit.claims:
            assert case.answer[claim.answer_span.start : claim.answer_span.end] == claim.text
            assert claim.text in case.answer
            assert all(source.text != case.answer for source in case.sources)


def _assert_case_derives_from_template(
    case: EvaluationCase,
    *,
    template,
    transformation_name: str,
) -> None:
    assert case.document_family_id == template.family_id
    assert case.answer == _expected_answer_for_transformation(template, transformation_name)
    _assert_answer_targets_are_authored(
        case,
        template=template,
        transformation_name=transformation_name,
    )
    _assert_positive_source_targets_slice_exact_text(
        case,
        template=template,
        transformation_name=transformation_name,
    )


@dataclass(frozen=True)
class ExpectedRequirement:
    source_id: str
    spans: tuple[CharSpan, ...]
    texts: tuple[str, ...]


def _generate_case(*, template, transformation_name: str, seed: int) -> EvaluationCase:
    transformation = _transformation_by_name(transformation_name)
    cases = transformation.generate(template, seed)
    assert len(cases) == 1
    return cases[0]


def _case_for(transformation_name: str, *, seed: int) -> EvaluationCase:
    return _generate_case(
        template=_fixture_template(),
        transformation_name=transformation_name,
        seed=seed,
    )


def _transformation_by_name(transformation_name: str):
    transformations = _transformations_module()
    by_name = {
        transformation.name: transformation
        for transformation in transformations.TRANSFORMATIONS
    }
    return by_name[transformation_name]


def _fact_for_transformation(template, transformation_name: str):
    matches = [
        fact for fact in template.facts if transformation_name in fact.adversarial_variants
    ]
    assert len(matches) == 1
    return matches[0]


def _variant_config(template, transformation_name: str) -> Mapping[str, object]:
    config = _fact_for_transformation(template, transformation_name).adversarial_variants[
        transformation_name
    ]
    assert isinstance(config, Mapping)
    return config


def _formatted_claim_text(template, transformation_name: str, *, variant: bool) -> str:
    fact = _fact_for_transformation(template, transformation_name)
    config = _variant_config(template, transformation_name) if variant else {}
    claim_template = (
        config.get("claim_template", fact.claim_template)
        if isinstance(config, Mapping)
        else fact.claim_template
    )
    slots = dict(fact.slots)
    if isinstance(config, Mapping) and "slots" in config:
        slots.update(_string_mapping(config["slots"]))
    return str(claim_template).format(**slots)


def _formatted_variant_text(template, transformation_name: str, key: str) -> str:
    fact = _fact_for_transformation(template, transformation_name)
    config = _variant_config(template, transformation_name)
    raw_value = config[key]
    assert isinstance(raw_value, str)
    return raw_value.format(**dict(fact.slots))


def _expected_answer_for_transformation(template, transformation_name: str) -> str:
    if transformation_name == "multi_source":
        return _formatted_variant_text(template, transformation_name, "answer_text")
    if transformation_name == "unsupported_clause":
        suffix = str(_variant_config(template, transformation_name)["unsupported_suffix"])
        return _formatted_claim_text(template, transformation_name, variant=False) + suffix
    return _formatted_claim_text(template, transformation_name, variant=True)


def _expected_positive_requirements(
    *,
    template,
    transformation_name: str,
) -> tuple[tuple[ExpectedRequirement, ...], ...]:
    positive_transformations = {
        "unicode",
        "duplicate_distractor",
        "multi_span",
        "multi_source",
        "unsupported_clause",
    }
    if transformation_name not in positive_transformations:
        return ()

    if transformation_name == "multi_source":
        return (
            (
                ExpectedRequirement(
                    source_id="source-primary",
                    spans=(CharSpan(start=0, end=len(_formatted_variant_text(template, transformation_name, "primary_source_text"))),),
                    texts=(_formatted_variant_text(template, transformation_name, "primary_source_text"),),
                ),
            ),
            (
                ExpectedRequirement(
                    source_id="source-secondary",
                    spans=(CharSpan(start=0, end=len(_formatted_variant_text(template, transformation_name, "secondary_source_text"))),),
                    texts=(_formatted_variant_text(template, transformation_name, "secondary_source_text"),),
                ),
            ),
        )

    if transformation_name == "multi_span":
        primary_source_text = _formatted_variant_text(template, transformation_name, "primary_source_text")
        citation_texts = _string_tuple(_variant_config(template, transformation_name), "citation_texts")
        return (
            (
                ExpectedRequirement(
                    source_id="source-primary",
                    spans=tuple(CharSpan.model_validate(_span(primary_source_text, text)) for text in citation_texts),
                    texts=citation_texts,
                ),
            ),
        )

    proposition_text = _formatted_claim_text(template, transformation_name, variant=False)

    return (
        (
            ExpectedRequirement(
                source_id="source-primary",
                spans=(CharSpan.model_validate(_span(template.source_text, proposition_text)),),
                texts=(proposition_text,),
            ),
        ),
    )


def _string_mapping(value: object) -> dict[str, str]:
    assert isinstance(value, Mapping)
    return {str(key): str(item) for key, item in value.items()}


def _string_tuple(config: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw = config[key]
    assert isinstance(raw, tuple)
    return tuple(str(item) for item in raw)


def _entailed_target_texts(case: EvaluationCase) -> tuple[str, ...]:
    source_text_by_id = {source.source_id: source.text for source in case.sources}
    return tuple(
        source_text_by_id[alternative.source_id][span.start : span.end]
        for unit in case.evaluation_units
        for claim in unit.claims
        if claim.label == "entailed"
        for requirement in claim.citation_requirements
        for alternative in requirement.alternatives
        for span in alternative.spans
    )


def _canonical_case_digest(cases: tuple[EvaluationCase, ...]) -> str:
    payload = tuple(
        case.model_dump(mode="json")
        for case in sorted(cases, key=lambda case: case.case_id)
    )
    return sha256_hex(canonical_json_bytes({"cases": payload}))


def _authoritative_case_id(case: EvaluationCase) -> str:
    from evaluation.canonical import authoritative_case_id

    return authoritative_case_id(case)


def _case_dump_without_identity_and_generation(case: EvaluationCase) -> dict[str, object]:
    return case.model_copy(
        update={
            "case_id": "case-seed-agnostic",
            "generation": None,
        }
    ).model_dump(mode="json")


def _second_catalog_template():
    authored_sources = _authored_sources_module()
    template = next(
        template
        for template in authored_sources.AUTHORED_FACT_TEMPLATES
        if template.family_id == "finance-01-harbor-fund"
    )
    assert template.family_id != _fixture_template().family_id
    return template


def _span(text: str, fragment: str) -> dict[str, int]:
    start = text.index(fragment)
    return {"start": start, "end": start + len(fragment)}


def _authored_sources_module():
    return import_module("evaluation.builders.authored_sources")


def _cases_module():
    return import_module("evaluation.builders.cases")


REAL_CASE_CHALLENGE_TYPES = ("contradicted", "partial", "distractor")
REAL_CASE_DOMAINS = (
    "environment",
    "finance",
    "health",
    "history",
    "policy",
    "science",
    "technology",
)


def test_real_source_catalog_has_complete_local_provenance_and_balanced_domains() -> None:
    real_sources = _real_sources_module()
    families = real_sources.load_real_source_families()
    provenance = real_sources.load_real_source_provenance()

    assert len(families) == 15
    assert len(provenance) == 15
    assert tuple(family.family_id for family in families) == tuple(
        sorted(family.family_id for family in families)
    )
    assert tuple(item.family_id for item in provenance) == tuple(
        family.family_id for family in families
    )

    domain_counts = Counter(family.domain for family in families)
    assert tuple(sorted(domain_counts)) == REAL_CASE_DOMAINS
    assert all(domain_counts[domain] >= 1 for domain in REAL_CASE_DOMAINS)

    challenge_counts = Counter(family.challenge.kind for family in families)
    assert challenge_counts == {
        "contradicted": 5,
        "partial": 5,
        "distractor": 5,
    }

    for family, item in zip(families, provenance, strict=True):
        assert family.source_text
        assert family.supported_answer == family.source_text
        assert len(family.source_text.split()) <= 20
        assert family.snapshot_hash == item.snapshot_hash
        assert family.source_text == item.source_text

        assert item.origin_url.startswith("https://")
        assert item.policy_url.startswith("https://")
        assert item.statutory_url.startswith("https://")
        assert item.publisher
        assert item.page_title
        assert item.license_basis
        assert item.retrieval_date == date(2026, 7, 17)
        assert item.third_party_credit is False
        assert item.snapshot_hash == sha256_hex(item.source_text.encode("utf-8"))


def test_real_source_catalog_keeps_fdic_snapshot_byte_exact_with_unicode_apostrophe() -> None:
    real_sources = _real_sources_module()
    families = {
        family.family_id: family
        for family in real_sources.load_real_source_families()
    }

    fdic_text = families["finance-fdic-what-we-do"].source_text

    assert fdic_text == "maintain stability and public confidence in the nation’s financial system"
    assert "nation’s" in fdic_text
    assert "nation's" not in fdic_text
    assert fdic_text.encode("utf-8") == (
        "maintain stability and public confidence in the nation’s financial system".encode(
            "utf-8"
        )
    )


def test_real_source_models_reject_missing_metadata_local_text_hash_mismatch_and_third_party_credit() -> None:
    real_sources = _real_sources_module()
    source_text = "A black hole is a region in space where gravity is strong."

    base = {
        "family_id": "science-test-family",
        "domain": "science",
        "source_text": source_text,
        "origin_url": "https://www.nasa.gov/example",
        "page_title": "Example Page",
        "publisher": "NASA",
        "license_basis": "17 U.S.C. 105 public domain",
        "policy_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
        "statutory_url": "https://uscode.house.gov/view.xhtml?edition=2023&num=0&req=granuleid%3AUSC-2023-title17-section105",
        "retrieval_date": "2026-07-17",
        "snapshot_hash": sha256_hex(source_text.encode("utf-8")),
        "third_party_credit": False,
    }

    with pytest.raises(ValueError, match="license_basis must be non-empty"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "license_basis": " ",
            }
        )

    with pytest.raises(ValueError, match="real source text must be non-empty"):
        real_sources.RealSourceFamily.model_validate(
            {
                "family_id": "science-test-family",
                "domain": "science",
                "source_text": " ",
                "supported_answer": "Example",
                "snapshot_hash": base["snapshot_hash"],
                "provenance": base,
                "challenge": {
                    "kind": "contradicted",
                    "answer": "Example but false.",
                    "distractor_family_id": None,
                    "unsupported_suffix": None,
                },
            }
        )

    with pytest.raises(ValueError, match="snapshot_hash must equal sha256 of source_text"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "snapshot_hash": "0" * 64,
            }
        )

    with pytest.raises(ValueError, match="third_party_credit must be false"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "third_party_credit": True,
            }
        )

    with pytest.raises(ValueError, match="origin_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "origin_url": "http://example.com/not-official",
            }
        )

    with pytest.raises(ValueError, match="origin_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "origin_url": "file:///tmp/not-official",
            }
        )

    with pytest.raises(ValueError, match="policy_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "policy_url": "http://www.nasa.gov/not-secure",
            }
        )

    with pytest.raises(ValueError, match="policy_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "policy_url": "file:///tmp/policy",
            }
        )

    with pytest.raises(ValueError, match="statutory_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "statutory_url": "http://uscode.house.gov/not-secure",
            }
        )

    with pytest.raises(ValueError, match="statutory_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "statutory_url": "file:///tmp/statute",
            }
        )

    with pytest.raises(ValueError, match="policy_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "policy_url": " ",
            }
        )

    with pytest.raises(Exception, match="Field required"):
        real_sources.RealSourceProvenance.model_validate(
            {
                key: value
                for key, value in base.items()
                if key != "policy_url"
            }
        )

    with pytest.raises(ValueError, match="statutory_url must use https:// and point to an official source"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "statutory_url": " ",
            }
        )

    with pytest.raises(Exception, match="Field required"):
        real_sources.RealSourceProvenance.model_validate(
            {
                key: value
                for key, value in base.items()
                if key != "statutory_url"
            }
        )

    with pytest.raises(ValueError, match="page_title must be non-empty"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "page_title": " ",
            }
        )

    with pytest.raises(Exception, match="Field required"):
        real_sources.RealSourceProvenance.model_validate(
            {
                key: value
                for key, value in base.items()
                if key != "page_title"
            }
        )

    with pytest.raises(ValueError, match="publisher must be non-empty"):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "publisher": " ",
            }
        )

    with pytest.raises(Exception, match="Field required"):
        real_sources.RealSourceProvenance.model_validate(
            {
                key: value
                for key, value in base.items()
                if key != "publisher"
            }
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("origin_url", "https://attacker@www.nasa.gov/example"),
        ("origin_url", "https://user:pass@www.nasa.gov/example"),
        ("origin_url", "https://.gov/example"),
        ("origin_url", "https://gov/example"),
        ("origin_url", "https://www.nasa.gov:443/example"),
        ("origin_url", "https://www.nasa.gov.evil.example/example"),
        ("origin_url", "https://-bad.gov/example"),
        ("origin_url", "https://bad-.gov/example"),
        ("policy_url", "https://attacker@www.nasa.gov/example"),
        ("policy_url", "https://user:pass@www.nasa.gov/example"),
        ("policy_url", "https://.gov/example"),
        ("policy_url", "https://gov/example"),
        ("policy_url", "https://www.nasa.gov:443/example"),
        ("policy_url", "https://www.nasa.gov.evil.example/example"),
        ("policy_url", "https://-bad.gov/example"),
        ("policy_url", "https://bad-.gov/example"),
        ("statutory_url", "https://attacker@uscode.house.gov/view.xhtml?edition=2023"),
        ("statutory_url", "https://user:pass@uscode.house.gov/view.xhtml?edition=2023"),
        ("statutory_url", "https://.gov/view.xhtml?edition=2023"),
        ("statutory_url", "https://gov/view.xhtml?edition=2023"),
        ("statutory_url", "https://uscode.house.gov:443/view.xhtml?edition=2023"),
        ("statutory_url", "https://uscode.house.gov.evil.example/view.xhtml?edition=2023"),
        ("statutory_url", "https://-bad.gov/view.xhtml?edition=2023"),
        ("statutory_url", "https://bad-.gov/view.xhtml?edition=2023"),
    ),
)
def test_real_source_provenance_rejects_misleading_canonical_urls(
    field_name: str,
    value: str,
) -> None:
    real_sources = _real_sources_module()
    base = _real_source_provenance_fixture()

    message_by_field = {
        "origin_url": "origin_url must use https:// and point to an official source",
        "policy_url": "policy_url must use https:// and point to an official source",
        "statutory_url": "statutory_url must use https:// and point to an official source",
    }

    with pytest.raises(ValueError, match=message_by_field[field_name]):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                field_name: value,
            }
        )


@pytest.mark.parametrize(
    ("origin_url", "publisher"),
    (
        ("https://www.nasa.gov.evil.gov/example", "NASA"),
        ("https://www.xn--nasa-9o0a.gov/example", "NASA"),
        ("https://www.epa.gov/example", "NASA"),
        ("https://www.nasa.gov/example", "Unknown Federal Publisher"),
    ),
)
def test_real_source_provenance_binds_origin_host_to_declared_publisher(
    origin_url: str,
    publisher: str,
) -> None:
    real_sources = _real_sources_module()
    base = _real_source_provenance_fixture()

    with pytest.raises(
        ValueError,
        match="origin_url hostname must match the declared publisher",
    ):
        real_sources.RealSourceProvenance.model_validate(
            {
                **base,
                "origin_url": origin_url,
                "publisher": publisher,
            }
        )


def test_real_source_loaders_fail_closed_for_duplicate_records_and_join_gaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_sources = _real_sources_module()
    source_payload, provenance_payload = _read_real_source_payloads()

    duplicate_sources_path = tmp_path / "duplicate-real.json"
    duplicate_provenance_path = tmp_path / "duplicate-provenance.json"
    duplicate_sources_path.write_text(
        json.dumps([source_payload[0], source_payload[0]], ensure_ascii=False),
        encoding="utf-8",
    )
    duplicate_provenance_path.write_text(
        json.dumps(provenance_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(real_sources, "REAL_SOURCES_PATH", duplicate_sources_path)
    monkeypatch.setattr(real_sources, "PROVENANCE_PATH", duplicate_provenance_path)
    with pytest.raises(ValueError, match=r"duplicate family_id 'environment-doe-solar'"):
        real_sources.load_real_source_families()

    unique_sources_path = tmp_path / "unique-real.json"
    unique_provenance_path = tmp_path / "unique-provenance.json"
    unique_sources_path.write_text(
        json.dumps(source_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    duplicate_unique_provenance_path = tmp_path / "duplicate-only-provenance.json"
    duplicate_unique_provenance_path.write_text(
        json.dumps([provenance_payload[0], provenance_payload[0]], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(real_sources, "REAL_SOURCES_PATH", unique_sources_path)
    monkeypatch.setattr(real_sources, "PROVENANCE_PATH", duplicate_unique_provenance_path)
    with pytest.raises(
        ValueError,
        match=r"duplicate family_id 'environment-doe-solar' in provenance records",
    ):
        real_sources.load_real_source_provenance()

    unique_provenance_path.write_text(
        json.dumps(provenance_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    missing_provenance_path = tmp_path / "missing-provenance.json"
    missing_provenance_payload = [
        item
        for item in provenance_payload
        if item["family_id"] != "finance-fdic-what-we-do"
    ]
    missing_provenance_path.write_text(
        json.dumps(missing_provenance_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(real_sources, "REAL_SOURCES_PATH", unique_sources_path)
    monkeypatch.setattr(real_sources, "PROVENANCE_PATH", missing_provenance_path)
    with pytest.raises(
        ValueError,
        match=r"missing provenance record for family_id 'finance-fdic-what-we-do'",
    ):
        real_sources.load_real_source_families()

    missing_source_path = tmp_path / "missing-source.json"
    missing_source_payload = [
        item
        for item in source_payload
        if item["family_id"] != "finance-fdic-what-we-do"
    ]
    missing_source_path.write_text(
        json.dumps(missing_source_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(real_sources, "REAL_SOURCES_PATH", missing_source_path)
    monkeypatch.setattr(real_sources, "PROVENANCE_PATH", unique_provenance_path)
    with pytest.raises(
        ValueError,
        match="provenance.json contains family_ids that are missing from real.json",
    ):
        real_sources.load_real_source_families()


def test_real_sources_module_has_no_cite_right_or_align_citations_dependency() -> None:
    real_sources = _real_sources_module()
    assert real_sources.__file__ is not None
    source_path = Path(real_sources.__file__)
    source_text = source_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source_text, filename=str(source_path))

    imported_modules = {
        alias.name
        for node in ast.walk(module_ast)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from_modules = {
        node.module
        for node in ast.walk(module_ast)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "cite_right" not in source_text
    assert "align_citations" not in source_text
    assert all(
        name != "cite_right" and not name.startswith("cite_right.")
        for name in imported_modules
    )
    assert all(
        name != "cite_right" and not name.startswith("cite_right.")
        for name in imported_from_modules
    )


def test_real_case_generation_is_offline_deterministic_and_balanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sources = _real_sources_module()
    cases_module = _cases_module()

    _disable_network(monkeypatch)
    first = real_sources.generate_real_cases()
    second = real_sources.generate_real_cases()
    combined = cases_module.generate_all_authored_cases(seed=31) + first

    assert len(first) == 30
    assert len({case.case_id for case in first}) == 30
    assert tuple(case.case_id for case in first) == tuple(case.case_id for case in second)
    assert _canonical_case_digest(first) == _canonical_case_digest(second)
    assert _canonical_case_digest(first) == real_sources.REAL_CASES_CANONICAL_DIGEST
    assert _canonical_case_digest(first) == "7d5bc92d9020a35ff730c720bae70b632d87d74d1d21b788e0f024a03b851637"

    assert len(combined) == 750
    assert len({case.case_id for case in combined}) == 750
    assert _canonical_case_digest(combined) == real_sources.ALL_CASES_CANONICAL_DIGEST
    assert _canonical_case_digest(combined) == "d82c674362634225fcedd45dfb0b68daa8c6d4a6b91ae00ef9d27e9796591f28"

    challenge_counts = Counter(case.transformation_family_id for case in first if case.transformation_family_id != "real-positive")
    assert challenge_counts == {
        "real-contradicted": 5,
        "real-partial": 5,
        "real-distractor": 5,
    }


def test_real_cases_slice_exact_local_snapshots_and_match_expected_semantics() -> None:
    real_sources = _real_sources_module()
    families = {
        family.family_id: family
        for family in real_sources.load_real_source_families()
    }
    cases = real_sources.generate_real_cases()

    assert len(cases) == 30

    for case in cases:
        family = families[case.document_family_id]
        assert case.provenance.kind == "public_domain"
        assert case.generation is None
        assert case.review is None
        assert case.split == "train"

        primary_source = case.sources[0]
        assert primary_source.source_id == "source-primary"
        assert primary_source.text == family.source_text

        if case.transformation_family_id == "real-positive":
            assert case.answer == family.supported_answer
            assert len(case.evaluation_units) == 1
            assert case.evaluation_units[0].expected_status == "supported"
            claim = case.evaluation_units[0].claims[0]
            assert claim.label == "entailed"
            target = claim.citation_requirements[0].alternatives[0]
            assert target.source_id == "source-primary"
            assert target.spans == (CharSpan(start=0, end=len(primary_source.text)),)
            continue

        challenge = family.challenge
        if challenge.kind == "contradicted":
            assert case.transformation_family_id == "real-contradicted"
            assert case.answer == challenge.answer
            assert case.evaluation_units[0].expected_status == "unsupported"
            claim = case.evaluation_units[0].claims[0]
            assert claim.label == "contradicted"
            assert claim.citation_requirements == ()
        elif challenge.kind == "partial":
            assert case.transformation_family_id == "real-partial"
            assert case.answer == challenge.answer
            assert case.evaluation_units[0].expected_status == "partial"
            first_claim, second_claim = case.evaluation_units[0].claims
            assert first_claim.label == "entailed"
            assert second_claim.label == "not_in_sources"
            assert second_claim.text == second_claim.text.lstrip(" \t\n\r.,;:!?-")
            target = first_claim.citation_requirements[0].alternatives[0]
            assert target.source_id == "source-primary"
            assert target.spans == (CharSpan(start=0, end=len(primary_source.text)),)
        else:
            assert challenge.kind == "distractor"
            assert case.transformation_family_id == "real-distractor"
            assert case.answer == family.supported_answer
            assert case.evaluation_units[0].expected_status == "supported"
            assert len(case.sources) == 2
            assert case.sources[1].source_id == "source-distractor"
            claim = case.evaluation_units[0].claims[0]
            assert claim.label == "entailed"
            assert claim.acceptable_retrieval_source_ids == ("source-primary",)
            target = claim.citation_requirements[0].alternatives[0]
            assert target.source_id == "source-primary"
            assert target.spans == (CharSpan(start=0, end=len(primary_source.text)),)


def test_real_source_json_artifacts_exist_and_round_trip() -> None:
    real_sources = _real_sources_module()

    families_path = Path("evaluation/data/v1/sources/real.json")
    provenance_path = Path("evaluation/data/v1/provenance.json")

    family_payload = json.loads(families_path.read_text(encoding="utf-8"))
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert isinstance(family_payload, list)
    assert isinstance(provenance_payload, list)
    assert len(family_payload) == 15
    assert len(provenance_payload) == 15
    assert tuple(item["family_id"] for item in family_payload) == tuple(
        family.family_id for family in real_sources.load_real_source_families()
    )


def _real_sources_module():
    return import_module("evaluation.builders.real_sources")


def _read_real_source_payloads() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    family_payload = json.loads(
        Path("evaluation/data/v1/sources/real.json").read_text(encoding="utf-8")
    )
    provenance_payload = json.loads(
        Path("evaluation/data/v1/provenance.json").read_text(encoding="utf-8")
    )
    assert isinstance(family_payload, list)
    assert isinstance(provenance_payload, list)
    return family_payload, provenance_payload


def _real_source_provenance_fixture() -> dict[str, object]:
    source_text = "A black hole is a region in space where gravity is strong."
    return {
        "family_id": "science-test-family",
        "domain": "science",
        "source_text": source_text,
        "origin_url": "https://www.nasa.gov/example",
        "page_title": "Example Page",
        "publisher": "NASA",
        "license_basis": "17 U.S.C. 105 public domain",
        "policy_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
        "statutory_url": "https://uscode.house.gov/view.xhtml?edition=2023&num=0&req=granuleid%3AUSC-2023-title17-section105",
        "retrieval_date": "2026-07-17",
        "snapshot_hash": sha256_hex(source_text.encode("utf-8")),
        "third_party_credit": False,
    }


def _disable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked_socket(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network access is forbidden for real source loading")

    monkeypatch.setattr(socket, "create_connection", _blocked_socket)
    monkeypatch.setattr(socket.socket, "connect", _blocked_socket, raising=False)


def _transformations_module():
    return import_module("evaluation.builders.transformations")


@dataclass(frozen=True)
class _CardinalityOverrideTransformation:
    name: str
    cardinality: str

    def generate(self, template, seed: int) -> tuple[EvaluationCase, ...]:
        real_cases = _transformation_by_name(self.name).generate(template, seed)
        assert len(real_cases) == 1
        if self.cardinality == "missing":
            return ()
        if self.cardinality == "duplicated":
            second_cases = _transformation_by_name(self.name).generate(template, seed + 1)
            assert len(second_cases) == 1
            return (real_cases[0], second_cases[0])
        raise AssertionError(f"unexpected cardinality override {self.cardinality!r}")


def _with_cardinality_override(cardinality: str):
    transformations = _transformations_module()
    return tuple(
        _CardinalityOverrideTransformation(name="negation", cardinality=cardinality)
        if transformation.name == "negation"
        else transformation
        for transformation in transformations.TRANSFORMATIONS
    )


@dataclass(frozen=True)
class _StaticTransformation:
    name: str
    cases: tuple[EvaluationCase, ...]

    def generate(self, template, seed: int) -> tuple[EvaluationCase, ...]:
        return self.cases


@dataclass
class _CollidingTransformation:
    validation_name: str
    runtime_name: str
    cases: tuple[EvaluationCase, ...]
    _name_reads: int = 0

    @property
    def name(self) -> str:
        self._name_reads += 1
        if self._name_reads == 1:
            return self.validation_name
        return self.runtime_name

    def generate(self, template, seed: int) -> tuple[EvaluationCase, ...]:
        return self.cases
