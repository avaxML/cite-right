from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from evaluation.schema import (
    CharSpan,
    ClaimAnnotation,
    EvaluationCase,
    EvaluationUnit,
    ReviewRecord,
)


def test_evaluation_package_is_not_public_library_api() -> None:
    import cite_right
    import evaluation

    assert evaluation.DATASET_VERSION == "1.0.0"
    assert not hasattr(cite_right, "evaluation")


def test_entailed_claim_requires_citation_requirement() -> None:
    case_data = _make_valid_case_data()
    case_data["evaluation_units"][0]["claims"][0]["citation_requirements"] = ()

    with pytest.raises(
        ValidationError, match="entailed claims must define at least one citation requirement"
    ):
        EvaluationCase.model_validate(case_data)


def test_schema_rejects_python_input_coercion_but_accepts_json_round_trip() -> None:
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        CharSpan.model_validate({"start": "0", "end": 1})

    with pytest.raises(ValidationError, match="Input should be a valid tuple"):
        ClaimAnnotation.model_validate(
            {
                "claim_id": "claim-coercion",
                "answer_span": {"start": 0, "end": 5},
                "text": "Paris",
                "label": "contradicted",
                "acceptable_retrieval_source_ids": ["source-a"],
            }
        )

    claim = ClaimAnnotation.model_validate(
        {
            "claim_id": "claim-valid",
            "answer_span": {"start": 0, "end": 5},
            "text": "Paris",
            "label": "contradicted",
            "acceptable_retrieval_source_ids": ("source-a",),
        }
    )
    claim_from_json = ClaimAnnotation.model_validate_json(
        json.dumps(
            {
                "claim_id": "claim-json",
                "answer_span": {"start": 0, "end": 5},
                "text": "Paris",
                "label": "contradicted",
                "acceptable_retrieval_source_ids": ["source-a"],
            }
        )
    )

    assert claim.acceptable_retrieval_source_ids == ("source-a",)
    assert claim_from_json.acceptable_retrieval_source_ids == ("source-a",)


def test_negative_claim_forbids_citation_requirements() -> None:
    case_data = _make_valid_case_data()
    case_data["evaluation_units"][0]["claims"][0]["label"] = "contradicted"

    with pytest.raises(
        ValidationError, match="negative claims must not define citation requirements"
    ):
        EvaluationCase.model_validate(case_data)


@pytest.mark.parametrize("state", ["approved", "rejected"])
def test_review_record_requires_audit_fields_for_completed_decisions(
    state: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match="completed review records require reviewer and reviewed_at",
    ):
        ReviewRecord.model_validate({"state": state})

    with pytest.raises(
        ValidationError,
        match="completed review records require a non-empty reviewer",
    ):
        ReviewRecord.model_validate(
            {
                "state": state,
                "reviewer": "   ",
                "reviewed_at": date(2026, 7, 17),
            }
        )

    pending = ReviewRecord.model_validate({"state": "pending"})

    assert pending.reviewer is None
    assert pending.reviewed_at is None


def test_target_spans_are_ordered_non_overlapping_and_in_bounds() -> None:
    case_data = _make_valid_case_data()
    alternatives = case_data["evaluation_units"][0]["claims"][0]["citation_requirements"][0][
        "alternatives"
    ]

    invalid_alternatives = (
        (
            (
                {"start": 9, "end": 15},
                {"start": 0, "end": 5},
            ),
            "citation target spans must be strictly ordered",
        ),
        (
            (
                {"start": 0, "end": 5},
                {"start": 4, "end": 10},
            ),
            "citation target spans must not overlap",
        ),
        (
            (
                {"start": 0, "end": 5},
                {"start": 999, "end": 1004},
            ),
            "citation target spans must stay within the referenced source text",
        ),
    )

    for spans, message in invalid_alternatives:
        invalid_case = deepcopy(case_data)
        invalid_case["evaluation_units"][0]["claims"][0]["citation_requirements"][0][
            "alternatives"
        ] = (
            {
                "source_id": alternatives[0]["source_id"],
                "spans": spans,
            },
        )

        with pytest.raises(ValidationError, match=message):
            EvaluationCase.model_validate(invalid_case)


def test_evaluation_unit_status_is_derived_from_claim_labels() -> None:
    entailed_claim = ClaimAnnotation.model_validate(
        {
            "claim_id": "claim-entailed",
            "answer_span": {"start": 0, "end": 5},
            "text": "Paris",
            "label": "entailed",
            "citation_requirements": (
                {
                    "requirement_id": "req-1",
                    "alternatives": (
                        {
                            "source_id": "source-a",
                            "spans": ({"start": 0, "end": 5},),
                        },
                    ),
                },
            ),
        }
    )
    contradicted_claim = ClaimAnnotation.model_validate(
        {
            "claim_id": "claim-contradicted",
            "answer_span": {"start": 6, "end": 12},
            "text": "Berlin",
            "label": "contradicted",
        }
    )
    not_in_sources_claim = ClaimAnnotation.model_validate(
        {
            "claim_id": "claim-missing",
            "answer_span": {"start": 13, "end": 20},
            "text": "Madrid",
            "label": "not_in_sources",
        }
    )

    supported = EvaluationUnit.model_validate(
        {
            "unit_id": "unit-supported",
            "answer_span": {"start": 0, "end": 5},
            "text": "Paris",
            "claims": (entailed_claim.model_dump(mode="python"),),
        }
    )
    partial = EvaluationUnit.model_validate(
        {
            "unit_id": "unit-partial",
            "answer_span": {"start": 0, "end": 12},
            "text": "Paris Berlin",
            "claims": (
                entailed_claim.model_dump(mode="python"),
                contradicted_claim.model_dump(mode="python"),
            ),
        }
    )
    unsupported = EvaluationUnit.model_validate(
        {
            "unit_id": "unit-unsupported",
            "answer_span": {"start": 6, "end": 20},
            "text": "Berlin Madrid",
            "claims": (
                contradicted_claim.model_dump(mode="python"),
                not_in_sources_claim.model_dump(mode="python"),
            ),
        }
    )

    assert supported.expected_status == "supported"
    assert partial.expected_status == "partial"
    assert unsupported.expected_status == "unsupported"


def test_multi_source_claim_requires_all_requirements() -> None:
    case = EvaluationCase.model_validate(_make_valid_case_data())
    claim = case.evaluation_units[0].claims[0]

    assert case.evaluation_units[0].expected_status == "supported"
    assert case.generation is not None
    assert case.generation.recipe_id == "recipe-001"
    assert case.review is not None
    assert case.review.state == "approved"
    assert "generation" in case.model_dump(mode="json")
    assert "review" in case.model_dump(mode="json")
    assert "generation_recipe" not in case.model_dump(mode="json")
    assert "review_record" not in case.model_dump(mode="json")
    assert len(claim.citation_requirements) == 2
    assert claim.citation_requirements[0].requirement_id == "req-paris"
    assert claim.citation_requirements[1].requirement_id == "req-berlin"
    assert len(claim.citation_requirements[0].alternatives) == 2
    assert len(claim.citation_requirements[1].alternatives) == 2

    invalid_case = _make_valid_case_data()
    invalid_case["evaluation_units"][0]["expected_status"] = "unsupported"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvaluationCase.model_validate(invalid_case)

    deprecated_name_case = _make_valid_case_data()
    deprecated_name_case["generation_recipe"] = deprecated_name_case["generation"]
    deprecated_name_case["review_record"] = deprecated_name_case["review"]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvaluationCase.model_validate(deprecated_name_case)


def test_case_offsets_slice_exact_answer_and_source_text() -> None:
    case = EvaluationCase.model_validate(_make_valid_case_data())
    unit = case.evaluation_units[0]
    claim = unit.claims[0]
    paris_target = claim.citation_requirements[0].alternatives[0]
    berlin_target = claim.citation_requirements[1].alternatives[0]
    source_texts = {source.source_id: source.text for source in case.sources}

    assert case.answer[unit.answer_span.start : unit.answer_span.end] == unit.text
    assert case.answer[claim.answer_span.start : claim.answer_span.end] == claim.text
    assert (
        source_texts[paris_target.source_id][
            paris_target.spans[0].start : paris_target.spans[0].end
        ]
        == "Paris is in France"
    )
    assert (
        source_texts[berlin_target.source_id][
            berlin_target.spans[0].start : berlin_target.spans[0].end
        ]
        == "Berlin is in Germany"
    )


def _make_valid_case_data() -> dict[str, Any]:
    source_paris = "Paris is in France. It is the capital city."
    source_berlin = "Berlin is in Germany. It is a major European city."
    answer = "Paris is in France, and Berlin is in Germany."
    claim_text = answer

    return {
        "case_id": "case-001",
        "dataset_version": "1.0.0",
        "split": "dev",
        "document_family_id": "family-europe-capitals",
        "transformation_family_id": "composed-facts",
        "provenance": {
            "kind": "authored",
            "title": "European capitals worksheet",
            "origin": "internal",
            "publisher": "Cite-Right",
            "license": "proprietary-draft",
            "retrieval_date": date(2026, 7, 17),
            "snapshot_hash": "snapshot-001",
        },
        "sources": (
            {
                "source_id": "source-paris",
                "text": source_paris,
                "chunk_id": "chunk-1",
                "chunk_char_start": 0,
                "chunk_char_end": len(source_paris),
            },
            {
                "source_id": "source-berlin",
                "text": source_berlin,
                "chunk_id": "chunk-2",
                "chunk_char_start": 0,
                "chunk_char_end": len(source_berlin),
            },
        ),
        "answer": answer,
        "evaluation_units": (
            {
                "unit_id": "unit-1",
                "answer_span": _span_dict(answer, answer),
                "text": answer,
                "claims": (
                    {
                        "claim_id": "claim-1",
                        "answer_span": _span_dict(answer, claim_text),
                        "text": claim_text,
                        "label": "entailed",
                        "citation_requirements": (
                            {
                                "requirement_id": "req-paris",
                                "alternatives": (
                                    {
                                        "source_id": "source-paris",
                                        "spans": (
                                            _span_dict(
                                                source_paris, "Paris is in France"
                                            ),
                                        ),
                                    },
                                    {
                                        "source_id": "source-paris",
                                        "spans": (
                                            _span_dict(source_paris, "Paris"),
                                            _span_dict(source_paris, "France"),
                                        ),
                                    },
                                ),
                            },
                            {
                                "requirement_id": "req-berlin",
                                "alternatives": (
                                    {
                                        "source_id": "source-berlin",
                                        "spans": (
                                            _span_dict(
                                                source_berlin, "Berlin is in Germany"
                                            ),
                                        ),
                                    },
                                    {
                                        "source_id": "source-berlin",
                                        "spans": (
                                            _span_dict(source_berlin, "Berlin"),
                                            _span_dict(source_berlin, "Germany"),
                                        ),
                                    },
                                ),
                            },
                        ),
                        "acceptable_retrieval_source_ids": (
                            "source-paris",
                            "source-berlin",
                        ),
                        "requires_non_contiguous_evidence": True,
                    },
                ),
            },
        ),
        "difficulty_tags": ("multi_source", "alternative_targets"),
        "generation": {
            "recipe_id": "recipe-001",
            "generator_name": "hand-authored",
            "prompt_version": "v1",
            "seed": 7,
            "notes": "Two-source conjunction with alternative targets.",
        },
        "review": {
            "state": "approved",
            "reviewer": "schema-task",
            "reviewed_at": date(2026, 7, 17),
            "notes": "Checked for exact spans.",
        },
    }


def _span_dict(text: str, fragment: str) -> dict[str, int]:
    start = text.index(fragment)
    return {"start": start, "end": start + len(fragment)}
