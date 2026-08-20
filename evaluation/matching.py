"""Deterministic matching between emitted citations and citation requirements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from evaluation.schema import CharSpan, CitationRequirement, CitationTarget

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
MatchingThreshold = Literal["exact", "0.9", "0.5"]


class EmittedCitation(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    source_id: str
    spans: tuple[CharSpan, ...]

    @model_validator(mode="after")
    def _validate_spans(self) -> EmittedCitation:
        if not self.spans:
            raise ValueError("emitted citations must define at least one span")

        previous_span: CharSpan | None = None
        for span in sorted(self.spans, key=_char_span_key):
            if previous_span is not None and span.start < previous_span.end:
                raise ValueError("emitted citation spans must not overlap")
            previous_span = span
        return self


EmissionInput: TypeAlias = EmittedCitation | Mapping[str, object]


class EvaluatorError(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    code: Literal["invalid_emitted_citation"]
    input_index: int
    path: str
    message: str


class CitationMatch(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    requirement_id: str
    emission_index: int
    alternative: CitationTarget
    score: float


class MatchResult(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    threshold: MatchingThreshold
    matches: tuple[CitationMatch, ...]
    unmatched_emission_indices: tuple[int, ...]
    unmatched_requirement_ids: tuple[str, ...]
    errors: tuple[EvaluatorError, ...]


class _NormalizedEmission(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    input_index: int
    emission: EmittedCitation
    canonical_spans: tuple[CharSpan, ...]


class _MatchEdge(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    requirement_index: int
    emission_input_index: int
    requirement_id: str
    alternative: CitationTarget
    alternative_canonical_spans: tuple[CharSpan, ...]
    emission_key: tuple[str, tuple[tuple[int, int], ...]]
    score_numerator: int
    score_denominator: int

    @property
    def score_fraction(self) -> Fraction:
        return Fraction(self.score_numerator, self.score_denominator)

    @property
    def score_float(self) -> float:
        return float(self.score_fraction)


def match_citations(
    *,
    emissions: Sequence[EmissionInput],
    requirements: Sequence[CitationRequirement],
    threshold: MatchingThreshold,
) -> MatchResult:
    ordered_requirements = tuple(sorted(requirements, key=_requirement_key))
    normalized_emissions, errors = _normalize_emissions(emissions)
    candidate_edges = _build_candidate_edges(
        requirements=ordered_requirements,
        emissions=normalized_emissions,
        threshold=threshold,
    )
    assignment = _choose_assignment(
        requirement_count=len(ordered_requirements),
        candidate_edges=candidate_edges,
    )

    matched_emission_indices = {edge.emission_input_index for edge in assignment}
    matched_requirement_ids = {edge.requirement_id for edge in assignment}

    return MatchResult(
        threshold=threshold,
        matches=tuple(
            CitationMatch(
                requirement_id=edge.requirement_id,
                emission_index=edge.emission_input_index,
                alternative=edge.alternative,
                score=edge.score_float,
            )
            for edge in assignment
        ),
        unmatched_emission_indices=tuple(
            emission.input_index
            for emission in normalized_emissions
            if emission.input_index not in matched_emission_indices
        ),
        unmatched_requirement_ids=tuple(
            requirement.requirement_id
            for requirement in ordered_requirements
            if requirement.requirement_id not in matched_requirement_ids
        ),
        errors=errors,
    )


def _normalize_emissions(
    emissions: Sequence[EmissionInput],
) -> tuple[tuple[_NormalizedEmission, ...], tuple[EvaluatorError, ...]]:
    normalized: list[_NormalizedEmission] = []
    errors: list[EvaluatorError] = []

    for input_index, payload in enumerate(emissions):
        try:
            emission = (
                payload
                if isinstance(payload, EmittedCitation)
                else EmittedCitation.model_validate(payload)
            )
        except ValidationError as exc:
            errors.extend(_validation_errors(exc, input_index=input_index))
            continue

        normalized.append(
            _NormalizedEmission(
                input_index=input_index,
                emission=emission,
                canonical_spans=_canonicalize_spans(emission.spans),
            )
        )

    return tuple(normalized), tuple(errors)


def _validation_errors(
    exc: ValidationError, *, input_index: int
) -> list[EvaluatorError]:
    return [
        EvaluatorError(
            code="invalid_emitted_citation",
            input_index=input_index,
            path=".".join(str(part) for part in detail["loc"]),
            message=detail["msg"],
        )
        for detail in exc.errors()
    ]


def _build_candidate_edges(
    *,
    requirements: tuple[CitationRequirement, ...],
    emissions: tuple[_NormalizedEmission, ...],
    threshold: MatchingThreshold,
) -> tuple[tuple[_MatchEdge, ...], ...]:
    return tuple(
        tuple(
            edge
            for emission in emissions
            for edge in [
                _best_edge_for_requirement(
                    requirement=requirement,
                    requirement_index=requirement_index,
                    emission=emission,
                    threshold=threshold,
                )
            ]
            if edge is not None
        )
        for requirement_index, requirement in enumerate(requirements)
    )


def _best_edge_for_requirement(
    *,
    requirement: CitationRequirement,
    requirement_index: int,
    emission: _NormalizedEmission,
    threshold: MatchingThreshold,
) -> _MatchEdge | None:
    best_alternative: CitationTarget | None = None
    best_canonical_spans: tuple[CharSpan, ...] | None = None
    best_score: Fraction | None = None

    for alternative in sorted(requirement.alternatives, key=_target_key):
        alternative_exact_spans = _sort_spans(alternative.spans)
        alternative_canonical_spans = _canonicalize_spans(alternative.spans)
        score = _match_score(
            alternative_source_id=alternative.source_id,
            alternative_exact_spans=alternative_exact_spans,
            alternative_spans=alternative_canonical_spans,
            emission=emission,
            threshold=threshold,
        )
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_alternative = alternative
            best_canonical_spans = alternative_canonical_spans
            best_score = score
            continue
        if score == best_score and best_alternative is not None:
            if _target_key(alternative) < _target_key(best_alternative):
                best_alternative = alternative
                best_canonical_spans = alternative_canonical_spans

    if best_alternative is None or best_canonical_spans is None or best_score is None:
        return None

    return _MatchEdge(
        requirement_index=requirement_index,
        emission_input_index=emission.input_index,
        requirement_id=requirement.requirement_id,
        alternative=best_alternative,
        alternative_canonical_spans=best_canonical_spans,
        emission_key=_emission_key(emission.emission),
        score_numerator=best_score.numerator,
        score_denominator=best_score.denominator,
    )


def _match_score(
    *,
    alternative_source_id: str,
    alternative_exact_spans: tuple[CharSpan, ...],
    alternative_spans: tuple[CharSpan, ...],
    emission: _NormalizedEmission,
    threshold: MatchingThreshold,
) -> Fraction | None:
    if alternative_source_id != emission.emission.source_id:
        return None
    if threshold == "exact":
        emission_exact_spans = _sort_spans(emission.emission.spans)
        if _span_key(alternative_exact_spans) == _span_key(emission_exact_spans):
            return Fraction(1, 1)
        return None

    iou = _character_iou(alternative_spans, emission.canonical_spans)
    return iou if iou >= _threshold_floor(threshold) else None


def _character_iou(
    left_spans: tuple[CharSpan, ...], right_spans: tuple[CharSpan, ...]
) -> Fraction:
    intersection = _intersection_length(left_spans, right_spans)
    union = _total_length(left_spans) + _total_length(right_spans) - intersection
    return Fraction(intersection, union)


def _intersection_length(
    left_spans: tuple[CharSpan, ...], right_spans: tuple[CharSpan, ...]
) -> int:
    left_index = 0
    right_index = 0
    total = 0

    while left_index < len(left_spans) and right_index < len(right_spans):
        left = left_spans[left_index]
        right = right_spans[right_index]
        overlap_start = max(left.start, right.start)
        overlap_end = min(left.end, right.end)
        if overlap_start < overlap_end:
            total += overlap_end - overlap_start
        if left.end <= right.end:
            left_index += 1
        else:
            right_index += 1

    return total


def _total_length(spans: tuple[CharSpan, ...]) -> int:
    return sum(span.end - span.start for span in spans)


def _threshold_floor(threshold: MatchingThreshold) -> Fraction:
    if threshold == "0.9":
        return Fraction(9, 10)
    if threshold == "0.5":
        return Fraction(1, 2)
    if threshold == "exact":
        raise ValueError("exact threshold does not use an IoU floor")
    raise ValueError(f"unsupported threshold {threshold!r}")


def _choose_assignment(
    *,
    requirement_count: int,
    candidate_edges: tuple[tuple[_MatchEdge, ...], ...],
) -> tuple[_MatchEdge, ...]:
    all_emission_indices = sorted(
        {edge.emission_input_index for row in candidate_edges for edge in row}
    )
    emission_positions = {
        input_index: position
        for position, input_index in enumerate(all_emission_indices)
    }

    @lru_cache(maxsize=None)
    def solve(requirement_index: int, used_mask: int) -> tuple[_MatchEdge, ...]:
        if requirement_index == requirement_count:
            return ()

        best_assignment = solve(requirement_index + 1, used_mask)
        for edge in candidate_edges[requirement_index]:
            bit = 1 << emission_positions[edge.emission_input_index]
            if used_mask & bit:
                continue
            candidate = (edge,) + solve(requirement_index + 1, used_mask | bit)
            if _assignment_is_better(candidate, best_assignment):
                best_assignment = candidate
        return best_assignment

    return tuple(sorted(solve(0, 0), key=_match_sort_key))


def _assignment_is_better(
    candidate: tuple[_MatchEdge, ...], current: tuple[_MatchEdge, ...]
) -> bool:
    if len(candidate) != len(current):
        return len(candidate) > len(current)

    candidate_score = _assignment_score(candidate)
    current_score = _assignment_score(current)
    if candidate_score != current_score:
        return candidate_score > current_score

    return _assignment_signature(candidate) < _assignment_signature(current)


def _assignment_score(assignment: tuple[_MatchEdge, ...]) -> Fraction:
    return sum((edge.score_fraction for edge in assignment), start=Fraction(0, 1))


def _assignment_signature(
    assignment: tuple[_MatchEdge, ...],
) -> tuple[
    tuple[
        str,
        tuple[str, tuple[tuple[int, int], ...]],
        tuple[str, tuple[tuple[int, int], ...]],
        int,
    ],
    ...,
]:
    return tuple(
        (
            edge.requirement_id,
            _target_key(edge.alternative),
            edge.emission_key,
            edge.emission_input_index,
        )
        for edge in sorted(assignment, key=_match_sort_key)
    )


def _requirement_key(
    requirement: CitationRequirement,
) -> tuple[str, tuple[tuple[str, tuple[tuple[int, int], ...]], ...]]:
    return (
        requirement.requirement_id,
        tuple(
            sorted(
                (_target_key(alternative) for alternative in requirement.alternatives)
            )
        ),
    )


def _target_key(target: CitationTarget) -> tuple[str, tuple[tuple[int, int], ...]]:
    return (target.source_id, _span_key(_canonicalize_spans(target.spans)))


def _emission_key(emission: EmittedCitation) -> tuple[str, tuple[tuple[int, int], ...]]:
    return (emission.source_id, _span_key(_canonicalize_spans(emission.spans)))


def _match_sort_key(
    match: _MatchEdge,
) -> tuple[str, int, tuple[str, tuple[tuple[int, int], ...]], int]:
    return (
        match.requirement_id,
        match.emission_input_index,
        _target_key(match.alternative),
        match.emission_input_index,
    )


def _canonicalize_spans(spans: Sequence[CharSpan]) -> tuple[CharSpan, ...]:
    ordered = _sort_spans(spans)
    if not ordered:
        return ()

    merged: list[CharSpan] = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start <= previous.end:
            merged[-1] = CharSpan(start=previous.start, end=max(previous.end, span.end))
            continue
        merged.append(span)
    return tuple(merged)


def _sort_spans(spans: Sequence[CharSpan]) -> tuple[CharSpan, ...]:
    return tuple(sorted(spans, key=_char_span_key))


def _span_key(spans: Sequence[CharSpan]) -> tuple[tuple[int, int], ...]:
    return tuple((span.start, span.end) for span in spans)


def _char_span_key(span: CharSpan) -> tuple[int, int]:
    return (span.start, span.end)


__all__ = [
    "CitationMatch",
    "EmittedCitation",
    "EvaluatorError",
    "MatchResult",
    "MatchingThreshold",
    "match_citations",
]
