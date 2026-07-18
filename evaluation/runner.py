"""Execution boundary between evaluation cases and cite-right alignment."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, field_validator

from cite_right import align_citations
from cite_right.core.citation_config import CitationConfig
from cite_right.core.results import SourceChunk, SourceDocument, SpanCitations
from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase, Source

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
Backend: TypeAlias = Literal["python", "rust"]
RunErrorCode: TypeAlias = Literal[
    "exception",
    "timeout",
    "invalid_answer_span",
    "ambiguous_answer_span",
    "unmappable_answer_span",
]
Clock: TypeAlias = Callable[[], int]
class _FrozenModel(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__class__.__pydantic_fields__:
            raise TypeError(f"{self.__class__.__name__} is immutable")
        super().__setattr__(name, value)


class CanonicalConfig(_FrozenModel):
    payload: object
    canonical_json: str
    sha256: str

    @field_validator("payload")
    @classmethod
    def _freeze_payload(cls, value: object) -> object:
        return _freeze_json_like(value)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("sha256 must be a 64-character lowercase hex digest")
        return value


class RunError(_FrozenModel):
    code: RunErrorCode
    message: str
    output_index: int | None = None
    exception_type: str | None = None


class CaseRun(_FrozenModel):
    case_id: str
    backend: Backend
    config: CanonicalConfig
    outputs: tuple[SpanCitations, ...] = ()
    output_unit_ids: tuple[tuple[str, ...], ...] = ()
    duration_ns: int
    error: RunError | None = None


def canonicalize_config(
    config: CitationConfig | Mapping[str, object],
) -> CanonicalConfig:
    canonical_bytes = canonical_json_bytes(config)
    normalized = _freeze_json_like(_config_payload(config))
    return CanonicalConfig(
        payload=normalized,
        canonical_json=canonical_bytes.decode("utf-8"),
        sha256=sha256_hex(canonical_bytes),
    )


def execute_case(
    *,
    case: EvaluationCase,
    backend: Backend,
    config: CitationConfig | Mapping[str, object],
    clock: Clock = time.perf_counter_ns,
    timeout_ns: int | None = None,
) -> CaseRun:
    canonical_config = canonicalize_config(config)
    resolved_config = _resolve_config(config)
    sources = tuple(_to_citation_source(source, index) for index, source in enumerate(case.sources))

    started_at = clock()
    try:
        outputs = tuple(
            align_citations(
                case.answer,
                sources,
                config=resolved_config,
                backend=backend,
            )
        )
    except Exception as exc:
        finished_at = clock()
        return CaseRun(
            case_id=case.case_id,
            backend=backend,
            config=canonical_config,
            duration_ns=finished_at - started_at,
            error=RunError(
                code="exception",
                message=str(exc),
                exception_type=type(exc).__name__,
            ),
        )

    finished_at = clock()
    duration_ns = finished_at - started_at
    if timeout_ns is not None and duration_ns > timeout_ns:
        return CaseRun(
            case_id=case.case_id,
            backend=backend,
            config=canonical_config,
            duration_ns=duration_ns,
            error=RunError(
                code="timeout",
                message=(
                    f"case execution exceeded timeout: {duration_ns}ns > {timeout_ns}ns"
                ),
            ),
        )

    output_unit_ids, mapping_error = _map_output_unit_ids(case=case, outputs=outputs)
    return CaseRun(
        case_id=case.case_id,
        backend=backend,
        config=canonical_config,
        outputs=outputs,
        output_unit_ids=output_unit_ids,
        duration_ns=duration_ns,
        error=mapping_error,
    )


def _config_payload(
    config: CitationConfig | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(config, CitationConfig):
        return cast(Mapping[str, object], config.model_dump(mode="json"))
    return config


def _resolve_config(config: CitationConfig | Mapping[str, object]) -> CitationConfig:
    if isinstance(config, CitationConfig):
        return config
    return CitationConfig.model_validate(config)


def _freeze_json_like(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return _freeze_json_like(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return tuple(
            (key, _freeze_json_like(item))
            for key, item in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_like(item) for item in value)
    raise TypeError("config payload must be JSON-like")


def _to_citation_source(source: Source, source_index: int) -> SourceDocument | SourceChunk:
    if source.chunk_char_start is None or source.chunk_char_end is None:
        return SourceDocument(id=source.source_id, text=source.text)
    return SourceChunk(
        source_id=source.source_id,
        text=source.text,
        doc_char_start=source.chunk_char_start,
        doc_char_end=source.chunk_char_end,
        source_index=source_index,
    )


def _map_output_unit_ids(
    *,
    case: EvaluationCase,
    outputs: Sequence[SpanCitations],
) -> tuple[tuple[tuple[str, ...], ...], RunError | None]:
    mapped: list[tuple[str, ...]] = []
    answer_length = len(case.answer)

    for output_index, output in enumerate(outputs):
        span = output.answer_span
        start = span.char_start
        end = span.char_end

        if start < 0 or end < 0 or start >= end or end > answer_length:
            return tuple(mapped), RunError(
                code="invalid_answer_span",
                message=(
                    "answer span offsets must define a valid in-bounds half-open range"
                ),
                output_index=output_index,
            )

        expected_text = case.answer[start:end]
        if span.text != expected_text:
            return tuple(mapped), RunError(
                code="ambiguous_answer_span",
                message="answer span text must equal the referenced answer slice",
                output_index=output_index,
            )

        overlapping_units = tuple(
            unit.unit_id
            for unit in case.evaluation_units
            if max(start, unit.answer_span.start) < min(end, unit.answer_span.end)
        )
        if not overlapping_units:
            return tuple(mapped), RunError(
                code="unmappable_answer_span",
                message="answer span did not overlap any evaluation unit",
                output_index=output_index,
            )

        mapped.append(overlapping_units)

    return tuple(mapped), None
