"""Results of the citation alignment pipeline."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class TokenizedText(BaseModel):
    """Result of tokenizing a text string.

    Attributes:
        text: The original input text.
        token_ids: List of integer token IDs produced by the tokenizer.
        token_spans: List of (start, end) character offsets for each token,
            where offsets are 0-based and half-open [start, end).
    """

    model_config = ConfigDict(frozen=True)

    text: str
    token_ids: list[int]
    token_spans: list[tuple[int, int]]

    @model_validator(mode="after")
    def _validate_token_spans(self) -> "TokenizedText":
        """Enforce the tokenizer contract used by downstream offset mapping."""
        if len(self.token_ids) != len(self.token_spans):
            raise ValueError("token_ids and token_spans must have the same length")

        text_length = len(self.text)
        previous_end = 0
        for index, (start, end) in enumerate(self.token_spans):
            if start < 0 or end > text_length:
                raise ValueError("token_spans must stay within text bounds")
            if start >= end:
                raise ValueError("token_spans must satisfy start < end")
            if index > 0 and start < previous_end:
                if start < self.token_spans[index - 1][0]:
                    raise ValueError("token_spans must be monotonic")
                raise ValueError("token_spans must not overlap")
            previous_end = end

        return self


class Segment(BaseModel):
    """A segment of text from a document (typically a sentence).

    Attributes:
        text: The segment text content.
        doc_char_start: Absolute 0-based start offset (inclusive) in the document.
        doc_char_end: Absolute 0-based end offset (exclusive) in the document.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    doc_char_start: int
    doc_char_end: int


class Alignment(BaseModel):
    """Result of Smith-Waterman sequence alignment.

    Attributes:
        score: Raw alignment score (sum of match/mismatch/gap penalties).
        token_start: Start index (inclusive) of best matching span in candidate.
        token_end: End index (exclusive) of best matching span in candidate.
        query_start: Start index (inclusive) of matching span in query sequence.
        query_end: End index (exclusive) of matching span in query sequence.
        matches: Number of exact token matches in the alignment.
        match_blocks: List of (start, end) token spans for non-contiguous matches
            when multi-span evidence is enabled.
    """

    model_config = ConfigDict(frozen=True)

    score: int
    token_start: int
    token_end: int
    query_start: int = 0
    query_end: int = 0
    matches: int = 0
    match_blocks: list[tuple[int, int]] = Field(default_factory=list)


class SourceDocument(BaseModel):
    """A complete source document for citation alignment.

    Use this when passing full documents to `align_citations()`.

    Attributes:
        id: Unique identifier for the document (returned in Citation.source_id).
        text: Full text content of the document.
        metadata: Optional key-value metadata (not used by alignment).

    Example:
        >>> doc = SourceDocument(id="report-2024", text="Revenue grew 15%...")
        >>> results = align_citations(answer, [doc])
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class SourceChunk(BaseModel):
    """A chunk (excerpt) from a larger source document.

    Use this when you have pre-chunked documents and want citation offsets
    to be computed relative to the original full document.

    Attributes:
        source_id: Identifier for the source document this chunk came from.
        text: The chunk text content.
        doc_char_start: Offset where this chunk starts in the original document.
        doc_char_end: Offset where this chunk ends in the original document.
        metadata: Optional key-value metadata (not used by alignment).
        document_text: Full original document text. If provided, Citation and
            EvidenceSpan evidence strings can be re-sliced directly from the
            original document using absolute offsets.
        source_index: Index of this source in the sources list. If None,
            uses the position in the list passed to align_citations().

    Example:
        >>> chunk = SourceChunk(
        ...     source_id="doc1",
        ...     text="Revenue grew 15%.",
        ...     doc_char_start=100,
        ...     doc_char_end=117,
        ...     document_text=full_document_text,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    text: str
    doc_char_start: int
    doc_char_end: int
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    document_text: str | None = None
    source_index: int | None = None

    @model_validator(mode="after")
    def _validate_document_text_alignment(self) -> "SourceChunk":
        """Ensure provided full-document metadata matches the chunk text."""
        if self.document_text is None:
            return self
        if self.doc_char_start < 0 or self.doc_char_end < self.doc_char_start:
            raise ValueError(
                "document_text offsets must define a valid non-negative range"
            )
        if self.doc_char_end > len(self.document_text):
            raise ValueError("document_text is shorter than doc_char_end")
        if self.document_text[self.doc_char_start : self.doc_char_end] != self.text:
            raise ValueError("document_text slice must exactly match SourceChunk.text")
        return self


class AnswerSpan(BaseModel):
    """A segment of the generated answer (sentence, clause, or paragraph).

    Returned as part of SpanCitations to identify which part of the answer
    each set of citations corresponds to.

    Attributes:
        text: The answer segment text content.
        char_start: 0-based start offset (inclusive) in the full answer string.
        char_end: 0-based end offset (exclusive) in the full answer string.
        kind: Type of segment - "sentence", "clause", or "paragraph".
        paragraph_index: 0-based index of the paragraph containing this span.
        sentence_index: 0-based index of this sentence within the answer.

    Note:
        Character offsets satisfy: `answer[char_start:char_end] == text`
    """

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
        evidence: Exact substring for the absolute document range
            `[char_start:char_end]`, re-sliced from the full document when
            available or from chunk-local text after subtracting the chunk's
            base offset.
    """

    model_config = ConfigDict(frozen=True)

    char_start: int
    char_end: int
    evidence: str


class Citation(BaseModel):
    """A citation linking an answer span to evidence in a source document.

    Attributes:
        score: Combined citation quality score (0.0 to ~2.0+). Higher is better.
            Computed as weighted sum of components based on CitationWeights.
        source_id: Identifier of the source document (from SourceDocument.id
            or SourceChunk.source_id).
        source_index: 0-based index of this source in the input sources list.
        candidate_index: Internal index of the passage candidate (for debugging).
        char_start: 0-based start offset (inclusive) of the legacy contiguous
            evidence view in the source.
        char_end: 0-based end offset (exclusive) of the legacy contiguous
            evidence view in the source.
        evidence: Extracted legacy contiguous evidence text for the absolute
            document range `[char_start:char_end]`. Satisfies the same rebased
            slicing rule used internally by `_slice_source_text()`.
            In multi-span mode, this enclosing span may include bridge text that
            was not matched directly. Use ``exact_evidence`` or
            ``evidence_spans`` for precise attribution.
        evidence_spans: List of EvidenceSpan for multi-span evidence. When
            multi_span_evidence is enabled in CitationConfig, may contain
            multiple non-contiguous spans. Otherwise, contains a single span
            matching char_start/char_end.
        exact_evidence: Canonical exact evidence text derived from
            ``evidence_spans``. Multi-span citations join exact matched spans
            with ``" ... "`` so omitted bridge text is visible.
        components: Breakdown of score components. Keys include:

            - ``alignment_score``: Raw Smith-Waterman alignment score.
            - ``normalized_alignment``: Alignment score / max possible score.
            - ``matches``: Number of exact token matches.
            - ``answer_coverage``: Fraction of answer tokens matched (0.0-1.0).
            - ``evidence_coverage``: Fraction of evidence tokens matched.
            - ``lexical_score``: IDF-weighted lexical overlap score (0.0-1.0).
            - ``embedding_score``: Cosine similarity from embeddings (-1.0-1.0).
            - ``num_evidence_spans``: Count of evidence spans.
            - ``evidence_chars_total``: Total characters across evidence spans.
            - ``passage_char_start``: Start offset of source passage window.
            - ``passage_char_end``: End offset of source passage window.

    Example:
        >>> for citation in span_citations.citations:
        ...     print(f"Source: {citation.source_id}")
        ...     print(f"Score: {citation.score:.2f}")
        ...     print(f"Evidence: {citation.evidence!r}")
        ...     print(f"Coverage: {citation.components['answer_coverage']:.1%}")
    """

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

    @computed_field(return_type=str)
    @property
    def exact_evidence(self) -> str:
        """Return the canonical exact evidence text for precise attribution."""
        if not self.evidence_spans:
            return self.evidence

        ordered_spans = sorted(
            self.evidence_spans, key=lambda span: (span.char_start, span.char_end)
        )
        return " ... ".join(span.evidence for span in ordered_spans)


class RetrievalSupport(BaseModel):
    """A retrieval-only support signal without localized evidence spans."""

    model_config = ConfigDict(frozen=True)

    retrieval_score: float
    source_id: str
    source_index: int
    candidate_index: int
    passage_char_start: int
    passage_char_end: int
    passage_text: str
    embedding_score: float
    lexical_score: float


class SpanCitations(BaseModel):
    """Citations for a single answer span.

    Returned by `align_citations()` - one SpanCitations per answer segment.

    Attributes:
        answer_span: The answer segment these citations correspond to.
        citations: List of Citation objects, ranked by score (best first).
            May be empty if no citations met the minimum thresholds.
        retrieval_support: List of retrieval-only support signals for passages
            selected during lexical and/or embedding candidate search but not
            localized into an exact citation.
        status: Overall citation status for this span:

            - ``"supported"``: Top-ranked citation has answer_coverage >= threshold
              (default 0.6). The claim is well-grounded in sources.
            - ``"partial"``: Has citations but below supported threshold.
              Some evidence exists but coverage is incomplete.
            - ``"unsupported"``: No citations found. This span may be
              hallucinated or paraphrased beyond recognition.

    Example:
        >>> results = align_citations(answer, sources)
        >>> for span_citations in results:
        ...     print(f"{span_citations.status}: {span_citations.answer_span.text}")
        ...     if span_citations.citations:
        ...         best = span_citations.citations[0]
        ...         print(f"  Evidence: {best.evidence!r}")
    """

    model_config = ConfigDict(frozen=True)

    answer_span: AnswerSpan
    citations: list[Citation]
    retrieval_support: list[RetrievalSupport] = Field(default_factory=list)
    status: Literal["supported", "partial", "unsupported"]

    @model_validator(mode="after")
    def _validate_status_matches_exact_citations(self) -> "SpanCitations":
        has_exact_citations = bool(self.citations)
        if has_exact_citations and self.status == "unsupported":
            raise ValueError("unsupported status requires citations to be empty")
        if not has_exact_citations and self.status != "unsupported":
            raise ValueError(
                "supported or partial status requires at least one citation"
            )
        return self
