from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from evaluation.matching import (
    EmittedCitation,
    MatchingThreshold,
    MatchResult,
    match_citations,
)
from evaluation.schema import CharSpan, CitationRequirement, CitationTarget


def test_matching_threshold_is_the_three_way_literal_contract() -> None:
    assert get_args(MatchingThreshold) == ("exact", "0.9", "0.5")


def test_emitted_citation_is_frozen_and_uses_schema_spans() -> None:
    emission = _emission("source-a", (0, 4), (10, 14))

    with pytest.raises((ValidationError, TypeError, AttributeError)):
        emission.source_id = "source-b"  # type: ignore[misc]

    assert emission.spans == (_span(0, 4), _span(10, 14))


def test_match_citations_returns_match_result_for_exact_source_and_span_set() -> None:
    result = match_citations(
        emissions=(_emission("source-a", (0, 4), (10, 14)),),
        requirements=(_requirement("req-1", _target("source-a", (0, 4), (10, 14))),),
        threshold="exact",
    )

    assert isinstance(result, MatchResult)
    assert _match_summary(result) == (
        _match("req-1", 0, "source-a", ((0, 4), (10, 14)), 1.0),
    )
    assert result.unmatched_emission_indices == ()
    assert result.unmatched_requirement_ids == ()
    assert result.errors == ()


def test_match_citations_exact_preserves_adjacent_span_boundaries() -> None:
    result = match_citations(
        emissions=(_emission("source-a", (0, 5), (5, 10)),),
        requirements=(_requirement("req-1", _target("source-a", (0, 10))),),
        threshold="exact",
    )

    assert _match_summary(result) == ()
    assert result.unmatched_emission_indices == (0,)
    assert result.unmatched_requirement_ids == ("req-1",)


def test_match_citations_exact_ignores_span_order() -> None:
    result = match_citations(
        emissions=(_emission("source-a", (10, 14), (0, 4)),),
        requirements=(_requirement("req-1", _target("source-a", (0, 4), (10, 14))),),
        threshold="exact",
    )

    assert _match_summary(result) == (
        _match("req-1", 0, "source-a", ((0, 4), (10, 14)), 1.0),
    )
    assert result.unmatched_emission_indices == ()
    assert result.unmatched_requirement_ids == ()


def test_match_citations_enforces_union_iou_boundary_at_point_nine() -> None:
    requirement = _requirement("req-1", _target("source-a", (0, 10)))

    passing = match_citations(
        emissions=(_emission("source-a", (0, 9)),),
        requirements=(requirement,),
        threshold="0.9",
    )
    failing = match_citations(
        emissions=(_emission("source-a", (0, 8)),),
        requirements=(requirement,),
        threshold="0.9",
    )

    # [0, 9) over [0, 10) gives IoU = 9 / 10 exactly.
    assert _match_summary(passing) == (_match("req-1", 0, "source-a", ((0, 10),), 0.9),)
    assert passing.unmatched_emission_indices == ()
    assert passing.unmatched_requirement_ids == ()
    # [0, 8) over [0, 10) gives IoU = 8 / 10, which must miss the 0.9 gate.
    assert _match_summary(failing) == ()
    assert failing.unmatched_emission_indices == (0,)
    assert failing.unmatched_requirement_ids == ("req-1",)


def test_match_citations_enforces_union_iou_boundary_at_point_five() -> None:
    requirement = _requirement("req-1", _target("source-a", (0, 10)))

    passing = match_citations(
        emissions=(_emission("source-a", (0, 5)),),
        requirements=(requirement,),
        threshold="0.5",
    )
    failing = match_citations(
        emissions=(_emission("source-a", (0, 4)),),
        requirements=(requirement,),
        threshold="0.5",
    )

    # [0, 5) over [0, 10) gives IoU = 5 / 10 exactly.
    assert _match_summary(passing) == (_match("req-1", 0, "source-a", ((0, 10),), 0.5),)
    assert passing.unmatched_emission_indices == ()
    assert passing.unmatched_requirement_ids == ()
    # [0, 4) over [0, 10) gives IoU = 4 / 10, below the 0.5 floor.
    assert _match_summary(failing) == ()
    assert failing.unmatched_emission_indices == (0,)
    assert failing.unmatched_requirement_ids == ("req-1",)


def test_match_citations_rejects_wrong_source_even_for_exact_spans() -> None:
    result = match_citations(
        emissions=(_emission("source-b", (0, 10)),),
        requirements=(_requirement("req-1", _target("source-a", (0, 10))),),
        threshold="exact",
    )

    assert _match_summary(result) == ()
    assert result.unmatched_emission_indices == (0,)
    assert result.unmatched_requirement_ids == ("req-1",)
    assert result.errors == ()


def test_malformed_offsets_are_rejected_by_model_or_reported_as_evaluator_errors() -> None:
    invalid_payload = {
        "source_id": "source-a",
        "spans": (
            {
                "start": 5,
                "end": 5,
            },
        ),
    }

    try:
        EmittedCitation.model_validate(invalid_payload)
    except ValidationError as exc:
        assert "start < end" in str(exc)
    else:
        result = match_citations(
            emissions=(invalid_payload,),  # type: ignore[arg-type]
            requirements=(_requirement("req-1", _target("source-a", (0, 10))),),
            threshold="exact",
        )

        assert _match_summary(result) == ()
        assert result.unmatched_emission_indices == ()
        assert result.unmatched_requirement_ids == ("req-1",)
        assert len(result.errors) == 1


def test_match_citations_accepts_any_alternative_target() -> None:
    result = match_citations(
        emissions=(_emission("source-b", (5, 10)),),
        requirements=(
            _requirement(
                "req-1",
                _target("source-a", (0, 5)),
                _target("source-b", (5, 10)),
            ),
        ),
        threshold="exact",
    )

    assert _match_summary(result) == (_match("req-1", 0, "source-b", ((5, 10),), 1.0),)
    assert result.unmatched_emission_indices == ()
    assert result.unmatched_requirement_ids == ()
    assert result.errors == ()


def test_match_citations_uses_union_iou_for_multi_span_targets_without_double_counting() -> None:
    result = match_citations(
        emissions=(_emission("source-a", (0, 4), (10, 13)),),
        requirements=(_requirement("req-1", _target("source-a", (0, 4), (10, 14))),),
        threshold="0.5",
    )

    # Target length is 8 chars, emission length is 7 chars, intersection is 7 chars.
    # IoU is therefore 7 / 8 = 0.875, not 7 / 11 or any double-counted variant.
    assert _match_summary(result) == (
        _match("req-1", 0, "source-a", ((0, 4), (10, 14)), 0.875),
    )
    assert result.unmatched_emission_indices == ()
    assert result.unmatched_requirement_ids == ()


def test_match_citations_requires_each_conjunctive_requirement_to_be_satisfied() -> None:
    result = match_citations(
        emissions=(_emission("source-a", (0, 5)),),
        requirements=(
            _requirement("req-a", _target("source-a", (0, 5))),
            _requirement("req-b", _target("source-b", (10, 15))),
        ),
        threshold="exact",
    )

    assert _match_summary(result) == (_match("req-a", 0, "source-a", ((0, 5),), 1.0),)
    assert result.unmatched_emission_indices == ()
    assert result.unmatched_requirement_ids == ("req-b",)
    assert result.errors == ()


def test_match_citations_enforces_one_to_one_matching_for_duplicate_emissions() -> None:
    result = match_citations(
        emissions=(
            _emission("source-a", (0, 5)),
            _emission("source-a", (0, 5)),
        ),
        requirements=(_requirement("req-1", _target("source-a", (0, 5))),),
        threshold="exact",
    )

    assert _match_summary(result) == (_match("req-1", 0, "source-a", ((0, 5),), 1.0),)
    assert result.unmatched_emission_indices == (1,)
    assert result.unmatched_requirement_ids == ()
    assert result.errors == ()


def test_match_citations_uses_maximum_matching_when_greedy_choice_loses_a_match() -> None:
    result = match_citations(
        emissions=(
            _emission("source-a", (3, 13)),
            _emission("source-a", (0, 10)),
        ),
        requirements=(
            _requirement("req-1", _target("source-a", (0, 10))),
            _requirement("req-2", _target("source-a", (6, 16))),
        ),
        threshold="0.5",
    )

    # Greedy "best local score first" would spend emission 1 on req-1 (score 1.0)
    # and strand req-2. Maximum cardinality keeps both by pairing:
    # req-1 <- emission 1 with IoU 1.0, req-2 <- emission 0 with IoU 7 / 13.
    assert _match_summary(result) == (
        _match("req-1", 1, "source-a", ((0, 10),), 1.0),
        _match("req-2", 0, "source-a", ((6, 16),), 7 / 13),
    )
    assert result.unmatched_emission_indices == ()
    assert result.unmatched_requirement_ids == ()
    assert result.errors == ()


def test_match_citations_is_stable_under_input_permutations_and_ties() -> None:
    requirements = (
        _requirement("req-2", _target("source-a", (6, 16))),
        _requirement("req-1", _target("source-a", (0, 10))),
        _requirement("req-3", _target("source-b", (0, 5))),
    )
    emissions = (
        _emission("source-b", (0, 5)),
        _emission("source-a", (3, 13)),
        _emission("source-a", (0, 10)),
        _emission("source-c", (1, 4)),
    )

    forward = match_citations(
        emissions=emissions,
        requirements=requirements,
        threshold="0.5",
    )
    reversed_inputs = match_citations(
        emissions=tuple(reversed(emissions)),
        requirements=tuple(reversed(requirements)),
        threshold="0.5",
    )

    assert _stable_view(forward, emissions) == _stable_view(
        reversed_inputs, tuple(reversed(emissions))
    )


def _requirement(
    requirement_id: str,
    *alternatives: CitationTarget,
) -> CitationRequirement:
    return CitationRequirement(requirement_id=requirement_id, alternatives=alternatives)


def _target(source_id: str, *spans: tuple[int, int]) -> CitationTarget:
    return CitationTarget(
        source_id=source_id,
        spans=tuple(_span(start, end) for start, end in spans),
    )


def _emission(source_id: str, *spans: tuple[int, int]) -> EmittedCitation:
    return EmittedCitation(
        source_id=source_id,
        spans=tuple(_span(start, end) for start, end in spans),
    )


def _span(start: int, end: int) -> CharSpan:
    return CharSpan(start=start, end=end)


def _match(
    requirement_id: str,
    emission_index: int,
    alternative_source_id: str,
    alternative_spans: tuple[tuple[int, int], ...],
    score: float,
) -> tuple[object, ...]:
    return (
        requirement_id,
        emission_index,
        alternative_source_id,
        alternative_spans,
        score,
    )


def _match_summary(result: MatchResult) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            match.requirement_id,
            match.emission_index,
            match.alternative.source_id,
            tuple((span.start, span.end) for span in match.alternative.spans),
            match.score,
        )
        for match in result.matches
    )


def _stable_view(
    result: MatchResult,
    emissions: tuple[EmittedCitation, ...],
) -> tuple[object, ...]:
    return (
        tuple(
            sorted(
                (
                    (
                        match.requirement_id,
                        emissions[match.emission_index].source_id,
                        tuple((span.start, span.end) for span in emissions[match.emission_index].spans),
                        match.alternative.source_id,
                        tuple((span.start, span.end) for span in match.alternative.spans),
                        match.score,
                    )
                    for match in result.matches
                )
            )
        ),
        tuple(
            sorted(
                (
                    emissions[index].source_id,
                    tuple((span.start, span.end) for span in emissions[index].spans),
                )
                for index in result.unmatched_emission_indices
            )
        ),
        tuple(sorted(result.unmatched_requirement_ids)),
        tuple(result.errors),
    )
