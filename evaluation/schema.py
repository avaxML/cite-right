"""Canonical schema models for attribution evaluation cases."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

SupportLabel = Literal["entailed", "contradicted", "not_in_sources"]
Split = Literal["train", "dev", "holdout"]
ProvenanceKind = Literal["authored", "public_domain", "permissive_license"]
ReviewState = Literal["pending", "approved", "rejected"]
ExpectedStatus = Literal["supported", "partial", "unsupported"]


class CharSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int
    end: int

    @model_validator(mode="after")
    def _validate_bounds(self) -> CharSpan:
        if self.start < 0 or self.end < 0:
            raise ValueError("char spans must be non-negative")
        if self.start >= self.end:
            raise ValueError("char spans must satisfy start < end")
        return self


class CitationTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    spans: tuple[CharSpan, ...]

    @model_validator(mode="after")
    def _validate_spans(self) -> CitationTarget:
        if not self.spans:
            raise ValueError("citation targets must define at least one span")
        return self


class CitationRequirement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requirement_id: str
    alternatives: tuple[CitationTarget, ...]

    @model_validator(mode="after")
    def _validate_alternatives(self) -> CitationRequirement:
        if not self.alternatives:
            raise ValueError(
                "citation requirements must define at least one alternative target"
            )
        return self


class ClaimAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str
    answer_span: CharSpan
    text: str
    label: SupportLabel
    citation_requirements: tuple[CitationRequirement, ...] = Field(default_factory=tuple)
    acceptable_retrieval_source_ids: tuple[str, ...] = Field(default_factory=tuple)
    requires_non_contiguous_evidence: bool = False

    @model_validator(mode="after")
    def _validate_label_contract(self) -> ClaimAnnotation:
        if self.label == "entailed" and not self.citation_requirements:
            raise ValueError(
                "entailed claims must define at least one citation requirement"
            )
        if self.label != "entailed" and self.citation_requirements:
            raise ValueError("negative claims must not define citation requirements")
        return self


class EvaluationUnit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    unit_id: str
    answer_span: CharSpan
    text: str
    claims: tuple[ClaimAnnotation, ...]

    @model_validator(mode="after")
    def _validate_claims(self) -> EvaluationUnit:
        if not self.claims:
            raise ValueError("evaluation units must define at least one claim")
        return self

    @computed_field(return_type=ExpectedStatus)
    @property
    def expected_status(self) -> ExpectedStatus:
        labels = {claim.label for claim in self.claims}
        if labels == {"entailed"}:
            return "supported"
        if "entailed" in labels:
            return "partial"
        return "unsupported"


class Source(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    text: str
    chunk_id: str | None = None
    chunk_char_start: int | None = None
    chunk_char_end: int | None = None

    @model_validator(mode="after")
    def _validate_chunk_metadata(self) -> Source:
        if (self.chunk_char_start is None) != (self.chunk_char_end is None):
            raise ValueError(
                "source chunk_char_start and chunk_char_end must be provided together"
            )
        if self.chunk_char_start is None or self.chunk_char_end is None:
            return self
        if self.chunk_char_start < 0 or self.chunk_char_end <= self.chunk_char_start:
            raise ValueError(
                "source chunk offsets must define a valid non-negative range"
            )
        return self


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ProvenanceKind
    title: str | None = None
    origin: str | None = None
    publisher: str | None = None
    license: str | None = None
    retrieval_date: date | None = None
    snapshot_hash: str | None = None


class GenerationRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe_id: str
    generator_name: str
    prompt_version: str
    seed: int | None = None
    notes: str | None = None


class ReviewRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: ReviewState
    reviewer: str | None = None
    reviewed_at: date | None = None
    notes: str | None = None


class EvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    dataset_version: str
    split: Split
    document_family_id: str
    transformation_family_id: str
    provenance: Provenance
    sources: tuple[Source, ...]
    answer: str
    evaluation_units: tuple[EvaluationUnit, ...]
    difficulty_tags: tuple[str, ...] = Field(default_factory=tuple)
    generation: GenerationRecipe | None = None
    review: ReviewRecord | None = None

    @model_validator(mode="after")
    def _validate_case(self) -> EvaluationCase:
        if not self.sources:
            raise ValueError("evaluation cases must define at least one source")
        if not self.evaluation_units:
            raise ValueError("evaluation cases must define at least one evaluation unit")

        source_ids = _ensure_unique(
            [source.source_id for source in self.sources],
            "source ids must be unique within a case",
        )
        _ensure_unique(
            [unit.unit_id for unit in self.evaluation_units],
            "evaluation unit ids must be unique within a case",
        )

        previous_unit_end = -1
        source_map = {source.source_id: source for source in self.sources}

        for unit in self.evaluation_units:
            _validate_answer_span(
                span=unit.answer_span,
                answer_text=self.answer,
                exact_text=unit.text,
                bounds_message="evaluation unit answer spans must stay within answer bounds",
                text_message="evaluation unit text must equal the referenced answer slice",
            )
            if unit.answer_span.start < previous_unit_end:
                raise ValueError(
                    "evaluation units must be ordered and non-overlapping"
                )
            previous_unit_end = unit.answer_span.end

            _ensure_unique(
                [claim.claim_id for claim in unit.claims],
                "claim ids must be unique within an evaluation unit",
            )
            for claim in unit.claims:
                _validate_answer_span(
                    span=claim.answer_span,
                    answer_text=self.answer,
                    exact_text=claim.text,
                    bounds_message="claim answer spans must stay within answer bounds",
                    text_message="claim text must equal the referenced answer slice",
                )
                if claim.answer_span.start < unit.answer_span.start:
                    raise ValueError(
                        "claim answer spans must stay within their evaluation unit"
                    )
                if claim.answer_span.end > unit.answer_span.end:
                    raise ValueError(
                        "claim answer spans must stay within their evaluation unit"
                    )

                _ensure_unique(
                    [
                        requirement.requirement_id
                        for requirement in claim.citation_requirements
                    ],
                    "citation requirement ids must be unique within a claim",
                )

                for retrieval_source_id in claim.acceptable_retrieval_source_ids:
                    if retrieval_source_id not in source_ids:
                        raise ValueError(
                            "acceptable retrieval source ids must reference case sources"
                        )

                for requirement in claim.citation_requirements:
                    for target in requirement.alternatives:
                        if target.source_id not in source_ids:
                            raise ValueError(
                                "citation target source ids must reference case sources"
                            )
                        _validate_target_spans(
                            target=target,
                            source_text=source_map[target.source_id].text,
                        )

        return self


def _ensure_unique(values: list[str], message: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(message)
        seen.add(value)
    return seen


def _validate_answer_span(
    *,
    span: CharSpan,
    answer_text: str,
    exact_text: str,
    bounds_message: str,
    text_message: str,
) -> None:
    if span.end > len(answer_text):
        raise ValueError(bounds_message)
    if answer_text[span.start : span.end] != exact_text:
        raise ValueError(text_message)


def _validate_target_spans(
    *,
    target: CitationTarget,
    source_text: str,
) -> None:
    previous_span: CharSpan | None = None
    for span in target.spans:
        if span.end > len(source_text):
            raise ValueError(
                "citation target spans must stay within the referenced source text"
            )
        if previous_span is not None:
            if span.start <= previous_span.start:
                raise ValueError("citation target spans must be strictly ordered")
            if span.start < previous_span.end:
                raise ValueError("citation target spans must not overlap")
        previous_span = span
