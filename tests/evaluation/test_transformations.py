from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from importlib import import_module

import pytest

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase

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
    cases_module = _cases_module()

    first = cases_module.generate_cases_for_template(template=template, seed=23)
    second = cases_module.generate_cases_for_template(template=template, seed=23)

    first_cases = _cases_by_transformation(first)
    second_cases = _cases_by_transformation(second)

    assert first_cases[transformation_name].model_dump(mode="json") == second_cases[
        transformation_name
    ].model_dump(mode="json")


@pytest.mark.parametrize("transformation_name", ALL_TRANSFORMATION_NAMES)
def test_transformation_family_semantics_and_lineage(
    transformation_name: str,
) -> None:
    template = _fixture_template()
    cases_module = _cases_module()
    cases = _cases_by_transformation(
        cases_module.generate_cases_for_template(template=template, seed=23)
    )
    case = cases[transformation_name]
    expected = EXPECTED_BEHAVIOR_BY_TRANSFORMATION[transformation_name]

    assert case.document_family_id == template.family_id
    assert case.transformation_family_id == transformation_name
    assert case.provenance.kind == "authored"
    assert case.generation is not None
    assert case.generation.seed == 23
    assert case.generation.generator_name == "evaluation.builders.cases"
    assert case.generation.prompt_version == "authored-v1"
    assert case.generation.notes is not None
    assert f"family={template.family_id}" in case.generation.notes
    assert f"transformation={transformation_name}" in case.generation.notes

    assert case.case_id
    assert case.case_id != ""
    assert case.answer == expected.expected_answer

    assert tuple(source.source_id for source in case.sources[:1]) == ("source-primary",)
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

    _assert_positive_source_targets_slice_exact_text(case)
    _assert_answer_targets_are_authored(case, template)

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


def test_every_catalog_family_produces_positive_and_adversarial_siblings() -> None:
    authored_sources = _authored_sources_module()
    cases_module = _cases_module()
    for template in authored_sources.AUTHORED_FACT_TEMPLATES:
        cases = cases_module.generate_cases_for_template(template=template, seed=11)
        labels = {claim.label for case in cases for unit in case.evaluation_units for claim in unit.claims}

        assert "entailed" in labels
        assert "contradicted" in labels or "not_in_sources" in labels
        assert {case.transformation_family_id for case in cases} == set(ALL_TRANSFORMATION_NAMES)


def test_catalog_generation_is_order_independent_after_stable_sorting() -> None:
    authored_sources = _authored_sources_module()
    cases_module = _cases_module()
    forward = cases_module.generate_all_authored_cases(seed=31)
    reverse = cases_module.generate_all_authored_cases(
        templates=tuple(reversed(authored_sources.AUTHORED_FACT_TEMPLATES)),
        seed=31,
    )

    assert [case.case_id for case in forward] == [case.case_id for case in reverse]
    assert _canonical_case_digest(forward) == _canonical_case_digest(reverse)
    assert _canonical_case_digest(forward) == "22db781dd7e899d1da6fd55076c6382b5540a155d483f7155b7f9686a8308b84"


def test_case_ids_are_authoritative_and_labels_do_not_depend_on_runtime_outputs() -> None:
    cases_module = _cases_module()
    case = _cases_by_transformation(
        cases_module.generate_cases_for_template(template=_fixture_template(), seed=17)
    )["multi_source"]

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


def _cases_by_transformation(cases: tuple[EvaluationCase, ...]) -> dict[str, EvaluationCase]:
    return {case.transformation_family_id: case for case in cases}


def _positive_claim(case: EvaluationCase):
    for claim in case.evaluation_units[0].claims:
        if claim.label == "entailed":
            return claim
    raise AssertionError("expected an entailed claim")


def _assert_positive_source_targets_slice_exact_text(case: EvaluationCase) -> None:
    source_text_by_id = {source.source_id: source.text for source in case.sources}

    for claim in case.evaluation_units[0].claims:
        for requirement in claim.citation_requirements:
            for alternative in requirement.alternatives:
                for span in alternative.spans:
                    assert (
                        source_text_by_id[alternative.source_id][span.start : span.end]
                        != ""
                    )


def _assert_answer_targets_are_authored(case: EvaluationCase, template) -> None:
    for unit in case.evaluation_units:
        assert case.answer[unit.answer_span.start : unit.answer_span.end] == unit.text
        for claim in unit.claims:
            assert case.answer[claim.answer_span.start : claim.answer_span.end] == claim.text
            assert claim.text in case.answer
            assert all(source.text != case.answer for source in case.sources)
    expected_answers = {
        behavior.expected_answer for behavior in EXPECTED_BEHAVIOR_BY_TRANSFORMATION.values()
    }
    assert case.answer in expected_answers


def _canonical_case_digest(cases: tuple[EvaluationCase, ...]) -> str:
    payload = tuple(
        case.model_dump(mode="json")
        for case in sorted(cases, key=lambda case: case.case_id)
    )
    return sha256_hex(canonical_json_bytes({"cases": payload}))


def _authoritative_case_id(case: EvaluationCase) -> str:
    from evaluation.canonical import authoritative_case_id

    return authoritative_case_id(case)


def _span(text: str, fragment: str) -> dict[str, int]:
    start = text.index(fragment)
    return {"start": start, "end": start + len(fragment)}


def _authored_sources_module():
    return import_module("evaluation.builders.authored_sources")


def _cases_module():
    return import_module("evaluation.builders.cases")


def _transformations_module():
    return import_module("evaluation.builders.transformations")
