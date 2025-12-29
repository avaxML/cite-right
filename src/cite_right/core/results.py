from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True, slots=True)
class TokenizedText:
    text: str
    token_ids: list[int]
    token_spans: list[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    doc_char_start: int
    doc_char_end: int


@dataclass(frozen=True, slots=True)
class Alignment:
    score: int
    token_start: int
    token_end: int
    query_start: int = 0
    query_end: int = 0
    matches: int = 0
    match_blocks: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceChunk:
    source_id: str
    text: str
    doc_char_start: int
    doc_char_end: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_text: str | None = None
    source_index: int | None = None


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    text: str
    char_start: int
    char_end: int
    kind: Literal["sentence", "clause", "paragraph"] = "sentence"
    paragraph_index: int | None = None
    sentence_index: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """A contiguous evidence slice in a source document.

    Attributes:
        char_start: Absolute 0-based start offset (inclusive) in the source document.
        char_end: Absolute 0-based end offset (exclusive) in the source document.
        evidence: Exact substring `source_text[char_start:char_end]`.
    """

    char_start: int
    char_end: int
    evidence: str


@dataclass(frozen=True, slots=True)
class Citation:
    score: float
    source_id: str
    source_index: int
    candidate_index: int
    char_start: int
    char_end: int
    evidence: str
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpanCitations:
    answer_span: AnswerSpan
    citations: list[Citation]
    status: Literal["supported", "partial", "unsupported"]
