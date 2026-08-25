---
type: concept
title: Result data model
description: "Reference Pydantic models returned by the cite-right API: SpanCitations, Citation, EvidenceSpan, RetrievalSupport, AnswerSpan, SourceDocument, SourceChunk, Alignment, TokenizedText, and Segment."
tags: [result-types, api-reference, citation-model]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The cite-right library defines a suite of frozen Pydantic models in `src/cite_right/core/results.py` that represent the output of the citation alignment pipeline. These models form a layered data model: `SpanCitations` is the top-level container returned by `align_citations()`, and it aggregates a list of `Citation` objects for each answer span, where each `Citation` may carry one or more `EvidenceSpan` slices from the source.

## TokenizedText

`TokenizedText` is the result of tokenizing a text string. It holds the original text, a list of integer token IDs, and a parallel list of half-open character offsets for each token.

```python
class TokenizedText(BaseModel):
    text: str
    token_ids: list[int]
    token_spans: list[tuple[int, int]]  # half-open [start, end)
```

A model validator enforces three invariants:

1. `token_ids` and `token_spans` must have the same length.
2. Each `(start, end)` pair must be within `0 ≤ start < end ≤ len(text)`.
3. `token_spans` must be monotonic and non-overlapping.

These guarantees are required by downstream offset mapping logic that maps answer token positions back to source character positions.

## Segment

`Segment` represents a contiguous piece of a source document, typically a sentence. It uses absolute 0-based character offsets that are inclusive at `doc_char_start` and exclusive at `doc_char_end`:

```python
class Segment(BaseModel):
    text: str
    doc_char_start: int  # inclusive
    doc_char_end: int    # exclusive
```

Unlike `SourceChunk`, `Segment` does not carry metadata or rebasing context; it is a plain result of document segmentation.

## Alignment

`Alignment` is the output of the Smith-Waterman sequence alignment between answer tokens and evidence tokens. It records the optimal local alignment and, when multi-span evidence is enabled, the disjoint match blocks.

```python
class Alignment(BaseModel):
    score: int
    token_start: int  # start of best span in candidate
    token_end: int    # exclusive end of best span in candidate
    query_start: int = 0
    query_end: int = 0
    matches: int = 0
    match_blocks: list[tuple[int, int]] = Field(default_factory=list)
```

When `multi_span_evidence` is enabled in `CitationConfig`, the `match_blocks` field contains a list of `(start, end)` token indices for non-contiguous matched regions. These are used to construct `EvidenceSpan` objects.

## SourceDocument

`SourceDocument` wraps a complete source document for citation alignment. It is the simplest source input when no chunking is needed.

```python
class SourceDocument(BaseModel):
    id: str                           # returned as Citation.source_id
    text: str                         # full document text
    metadata: Mapping[str, Any] = {}  # passthrough, not used by alignment
```

When a `SourceDocument` is passed to `align_citations()`, all `Citation.char_start` and `char_end` offsets are absolute positions within that document text.

## SourceChunk

`SourceChunk` represents a pre-chunked excerpt from a larger document. Its primary role is to carry rebasing information so that `Citation` offsets can be expressed as absolute positions in the original document, even when alignment is performed against the chunk text.

```python
class SourceChunk(BaseModel):
    source_id: str
    text: str                         # chunk text (alignment is done against this)
    doc_char_start: int              # offset where chunk starts in original document
    doc_char_end: int                # offset where chunk ends in original document
    metadata: Mapping[str, Any] = {}
    document_text: str | None = None  # full original document text
    source_index: int | None = None
```

`SourceChunk` carries a model validator that checks the rebasing invariants: if `document_text` is provided, the slice `document_text[doc_char_start:doc_char_end]` must equal `text` exactly. This ensures that offsets computed during alignment can be rebased to the original document.

The rebasing function `_slice_source_text()` in `src/cite_right/citations.py` (lines 1376–1381) implements this behavior:

- If `full_text` is present, it slices directly: `full_text[abs_start:abs_end]`.
- Otherwise, it subtracts the chunk's base offset: `source.text[abs_start - base_doc_offset : abs_end - base_doc_offset]`.

This means `Citation.evidence` and `Citation.evidence_spans[*].evidence` are always correct relative to either the original document or the chunk, depending on which text was provided.

## AnswerSpan

`AnswerSpan` identifies a segment of the generated answer. Each span corresponds to one unit of citation grouping: `align_citations()` returns one `SpanCitations` object per answer span.

```python
class AnswerSpan(BaseModel):
    text: str
    char_start: int          # inclusive in full answer
    char_end: int            # exclusive in full answer
    kind: Literal["sentence", "clause", "paragraph"] = "sentence"
    paragraph_index: int | None = None
    sentence_index: int | None = None
```

The invariant `answer[char_start:char_end] == text` holds. The `kind` field reflects the segmentation strategy used.

## EvidenceSpan

`EvidenceSpan` is a contiguous evidence slice in a source document. When multi-span evidence is enabled, a single `Citation` may carry multiple `EvidenceSpan` objects.

```python
class EvidenceSpan(BaseModel):
    char_start: int  # absolute offset in source document
    char_end: int    # exclusive
    evidence: str    # exact substring for [char_start:char_end]
```

The `evidence` field is always the result of slicing either the full document (when `document_text` was provided) or the chunk (after rebasing). This is determined by `_slice_source_text()`.

## Citation

`Citation` is the primary result object. It links an answer span to evidence in a source document.

```python
class Citation(BaseModel):
    score: float
    source_id: str
    source_index: int
    candidate_index: int
    char_start: int      # legacy contiguous view, may include bridge text
    char_end: int        # exclusive
    evidence: str        # contiguous evidence text for [char_start:char_end]
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    components: Mapping[str, float] = Field(default_factory=dict)

    @computed_field
    @property
    def exact_evidence(self) -> str: ...
```

### char_start / char_end semantics

`Citation.char_start` and `char_end` form a Python half-open interval: the evidence is `source[char_start:char_end]`. This is always true after chunk rebasing, whether slicing from the full document or from the chunk-local text.

In **single-span mode** (the default), `char_start` and `char_end` point to the exact matched span, and `evidence` equals `exact_evidence`. In **multi-span mode**, these fields describe an enclosing span that may include unmatched bridge text between matched segments.

### exact_evidence and evidence_spans

`evidence_spans` is the authoritative list of matched evidence regions. In single-span mode it contains exactly one `EvidenceSpan` matching `char_start`/`char_end`. In multi-span mode it contains all non-contiguous matched segments.

`exact_evidence` is a computed property that returns the canonical evidence string by joining the evidence spans in order:

```python
return " ... ".join(span.evidence for span in ordered_spans)
```

This join pattern makes omitted bridge text visible to readers. The `evidence` field, by contrast, is the raw contiguous slice from the enclosing span and may contain text that was not matched.

### components

The `components` mapping provides a breakdown of the citation score. Keys include:

| Key | Description |
|---|---|
| `alignment_score` | Raw Smith-Waterman alignment score |
| `normalized_alignment` | Alignment score divided by max possible score |
| `matches` | Count of exact token matches |
| `answer_coverage` | Fraction of answer tokens matched (0.0–1.0) |
| `evidence_coverage` | Fraction of evidence tokens matched |
| `lexical_score` | IDF-weighted lexical overlap (0.0–1.0) |
| `embedding_score` | Cosine similarity from embeddings (−1.0–1.0) |
| `num_evidence_spans` | Count of evidence spans |
| `evidence_chars_total` | Total characters across evidence spans |
| `passage_char_start` | Start offset of the source passage window |
| `passage_char_end` | End offset of the source passage window |

## RetrievalSupport

`RetrievalSupport` represents a retrieval-only hint. It is returned for passages that were selected during lexical and/or embedding candidate search but did not meet the minimum alignment thresholds to become a `Citation`.

```python
class RetrievalSupport(BaseModel):
    retrieval_score: float
    source_id: str
    source_index: int
    candidate_index: int
    passage_char_start: int
    passage_char_end: int
    passage_text: str
    embedding_score: float
    lexical_score: float
```

`RetrievalSupport` is **not** a `Citation`. It does not carry evidence spans or evidence text, and it does not affect the `SpanCitations.status`. A passage that appears in `retrieval_support` but not in `citations` contributes no grounding to the answer span.

## SpanCitations

`SpanCitations` is the top-level result returned by `align_citations()` and the primary entry point for consumers of the API.

```python
class SpanCitations(BaseModel):
    answer_span: AnswerSpan
    citations: list[Citation]               # ranked by score, best first
    retrieval_support: list[RetrievalSupport] = Field(default_factory=list)
    status: Literal["supported", "partial", "unsupported"]
```

### Status validation

`SpanCitations` carries a model validator `_validate_status_matches_exact_citations` that enforces an invariant:

```python
# repo://src/cite_right/core/results.py#L339-L347
if has_exact_citations and self.status == "unsupported":
    raise ValueError("unsupported status requires citations to be empty")
if not has_exact_citations and self.status != "unsupported":
    raise ValueError(
        "supported or partial status requires at least one citation"
    )
```

Consequences:

- `unsupported` requires that `citations` is empty (no citations at all).
- `supported` or `partial` requires at least one `Citation` in `citations`.

This validator is the definitive source of truth for the consistency of `status` and `citations`. Attempting to construct an inconsistent `SpanCitations` raises a `ValueError`.

### Status semantics

The `status` field reflects the quality of citation grounding for the answer span:

- **`supported`**: The top-ranked citation has `answer_coverage >= supported_answer_coverage` (default 0.6). The claim is well-grounded.
- **`partial`**: Citations exist but `answer_coverage` falls below the supported threshold. Some evidence is present but coverage is incomplete.
- **`unsupported`**: No citations were found. This span may be hallucinated or paraphrased beyond recognition.

`retrieval_support` does not appear in this decision. A passage that appears only in `retrieval_support` does not elevate a span from `unsupported`.

## Data flow summary

```
align_citations(answer, sources)
  ├── TokenizedText for answer spans
  ├── NormalizedSource (SourceDocument or SourceChunk normalized)
  │     └── Alignment (Smith-Waterman with optional match_blocks)
  └── SpanCitations per answer span
        ├── AnswerSpan
        ├── Citation[]
        │     ├── evidence_spans: EvidenceSpan[]
        │     ├── char_start / char_end (half-open)
        │     ├── evidence = _slice_source_text(...)
        │     └── exact_evidence = " ... ".join([...])
        ├── RetrievalSupport[]
        │     └── (no Citation status; retrieval-only hint)
        └── status (validated against citations list)
```

## Alignment

`Alignment` is defined in `src/cite_right/core/results.py` (lines 64–86). It is the result of Smith-Waterman local alignment between answer tokens and evidence tokens. The `match_blocks` field captures disjoint matching regions for multi-span mode.

## AlignmentMetrics

`AlignmentMetrics` (`src/cite_right/core/prepared_corpus.py`, lines 52–60) tracks pipeline performance:

```python
class AlignmentMetrics(BaseModel):
    total_time_ms: float
    num_answer_spans: int
    num_candidates: int
    num_alignments: int
    embedding_time_ms: float = 0.0
    alignment_time_ms: float = 0.0
```

This is a diagnostic object, not part of the citation result tree. It is passed to a `MetricsCallback` during `align_citations()` when a Rust backend is available.
