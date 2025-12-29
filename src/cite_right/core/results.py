from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


class TokenizedText(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    token_ids: list[int]
    token_spans: list[tuple[int, int]]


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    doc_char_start: int
    doc_char_end: int


class Alignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: int
    token_start: int
    token_end: int
    query_start: int = 0
    query_end: int = 0
    matches: int = 0
    match_blocks: list[tuple[int, int]] = Field(default_factory=list)


class SourceDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class SourceChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    text: str
    doc_char_start: int
    doc_char_end: int
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    document_text: str | None = None
    source_index: int | None = None


class AnswerSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    char_start: int
    char_end: int
    kind: Literal["sentence", "clause", "paragraph"] = "sentence"
    paragraph_index: int | None = None
    sentence_index: int | None = None


class EvidenceSpan(BaseModel):
    """A contiguous evidence slice in a source document.

    Attributes:
        char_start: Absolute 0-based start offset (inclusive) in the source document.
        char_end: Absolute 0-based end offset (exclusive) in the source document.
        evidence: Exact substring `source_text[char_start:char_end]`.
    """

    model_config = ConfigDict(frozen=True)

    char_start: int
    char_end: int
    evidence: str


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float
    source_id: str
    source_index: int
    candidate_index: int
    char_start: int
    char_end: int
    evidence: str
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    components: Mapping[str, float] = Field(default_factory=dict)


class SpanCitations(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer_span: AnswerSpan
    citations: list[Citation]
    status: Literal["supported", "partial", "unsupported"]
