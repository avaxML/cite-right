from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from importlib import import_module

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
        claim_label="contradicted",
        expected_answer="Venus completes one orbit every 88 days.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "relation": ExpectedFamilyBehavior(
        claim_label="contradicted",
        expected_answer="Mercury begins one orbit every 88 days.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "modality": ExpectedFamilyBehavior(
        claim_label="not_in_sources",
        expected_answer="The probe will remain active through 2030.",
        requires_citations=False,
        expected_status="unsupported",
        expected_source_count=1,
        expected_claim_count=1,
    ),
    "unsupported_clause": ExpectedFamilyBehavior(
        claim_label="not_in_sources",
        expected_answer="The probe should remain active through 2030. It is powered by a thorium battery.",
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
        expected_answer="Mercury completes one orbit every 88 days and the mission launched in 1977.",
        requires_citations=True,
        expected_status="supported",
        expected_source_count=2,
        expected_claim_count=1,
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
    elif transformation_name == "multi_source":
        positive_claim = _positive_claim(case)
        assert len(positive_claim.citation_requirements) == 2
        assert all(len(requirement.alternatives) >= 1 for requirement in positive_claim.citation_requirements)


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
    assert _canonical_case_digest(forward) == "568df690f1248d0ad56fcebda8f8d45222a7e7a44e7747bf3311c707d64cc74e"


def test_case_ids_are_authoritative_and_labels_do_not_depend_on_runtime_outputs() -> None:
    case = _generate_case(
        template=_fixture_template(),
        transformation_name="multi_source",
        seed=17,
    )

    assert case.case_id.startswith("case-")
    assert case.case_id == _authoritative_case_id(case)
    assert case.answer == "Mercury completes one orbit every 88 days and the mission launched in 1977."


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
                        "secondary_source_text": "Mission logs confirm the launch year was 1977."
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
                claim_template="The probe should remain active through {end_year}.",
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
                        "claim_template": "The probe will remain active through {end_year}."
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
                    "relation": {"claim_template": "{planet} begins one orbit every {days} days."},
                    "multi_span": {"evidence_slot_ids": ("planet", "days")},
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
    expected_requirements = _expected_positive_requirements(
        template=template,
        transformation_name=transformation_name,
    )

    for claim in case.evaluation_units[0].claims:
        if claim.label != "entailed":
            continue

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


def _expected_answer_for_transformation(template, transformation_name: str) -> str:
    if transformation_name == "unsupported_clause":
        suffix = str(_variant_config(template, transformation_name)["unsupported_suffix"])
        return _formatted_claim_text(template, transformation_name, variant=False) + suffix
    return _formatted_claim_text(template, transformation_name, variant=True)


def _expected_positive_requirements(
    *,
    template,
    transformation_name: str,
) -> tuple[ExpectedRequirement, ...]:
    positive_transformations = {
        "unicode",
        "duplicate_distractor",
        "multi_span",
        "multi_source",
        "unsupported_clause",
    }
    if transformation_name not in positive_transformations:
        return ()

    fact = _fact_for_transformation(template, transformation_name)
    evidence_by_slot = {evidence.slot_id: evidence for evidence in fact.evidence}

    if transformation_name == "multi_source":
        secondary_text = str(_variant_config(template, transformation_name)["secondary_source_text"])
        secondary_evidence = [
            evidence
            for evidence in fact.evidence
            if evidence.text in secondary_text
        ]
        assert len(secondary_evidence) == 1
        secondary = secondary_evidence[0]
        primary_evidence = tuple(
            evidence for evidence in fact.evidence if evidence.slot_id != secondary.slot_id
        )
        secondary_span = _span(secondary_text, secondary.text)
        return (
            ExpectedRequirement(
                source_id="source-primary",
                spans=tuple(evidence.span for evidence in primary_evidence),
                texts=tuple(evidence.text for evidence in primary_evidence),
            ),
            ExpectedRequirement(
                source_id="source-secondary",
                spans=(CharSpan.model_validate(secondary_span),),
                texts=(secondary.text,),
            ),
        )

    if transformation_name == "multi_span":
        slot_ids = _string_tuple(_variant_config(template, transformation_name), "evidence_slot_ids")
        evidence = tuple(evidence_by_slot[slot_id] for slot_id in slot_ids)
    else:
        evidence = fact.evidence

    return (
        ExpectedRequirement(
            source_id="source-primary",
            spans=tuple(item.span for item in evidence),
            texts=tuple(item.text for item in evidence),
        ),
    )


def _string_mapping(value: object) -> dict[str, str]:
    assert isinstance(value, Mapping)
    return {str(key): str(item) for key, item in value.items()}


def _string_tuple(config: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw = config[key]
    assert isinstance(raw, tuple)
    return tuple(str(item) for item in raw)


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
