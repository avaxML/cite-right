from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class TokenizedText:
    text: str
    token_ids: list[int]
    token_spans: list[tuple[int, int]]


@dataclass(frozen=True)
class Segment:
    text: str
    doc_char_start: int
    doc_char_end: int


@dataclass(frozen=True)
class Alignment:
    score: int
    token_start: int
    token_end: int
    query_start: int = 0
    query_end: int = 0
    matches: int = 0


@dataclass(frozen=True)
class SourceDocument:
    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceChunk:
    source_id: str
    text: str
    doc_char_start: int
    doc_char_end: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_text: str | None = None
    source_index: int | None = None


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    char_start: int
    char_end: int
    kind: Literal["sentence", "clause", "paragraph"] = "sentence"
    paragraph_index: int | None = None
    sentence_index: int | None = None


@dataclass(frozen=True)
class Citation:
    score: float
    source_id: str
    source_index: int
    candidate_index: int
    char_start: int
    char_end: int
    evidence: str
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SpanCitations:
    answer_span: AnswerSpan
    citations: list[Citation]
    status: Literal["supported", "partial", "unsupported"]
