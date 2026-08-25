---
type: concept
title: Citation model and offsets
description: "Offset invariants for citations: half-open intervals, chunk rebasing, evidence string equality with source slices, and the multi-span representation."
tags: [citation-model, offsets, invariants, cite-right]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The cite-right citation model uses a set of consistent conventions for character offsets across all types: every offset range is **half-open** (`[start, end)`), offsets are **rebased to absolute positions** when `SourceChunk` inputs are used, and the evidence string is always a direct slice of the source text at the reported offsets. These invariants hold regardless of whether the Python or Rust alignment backend is used.

## Half-open interval convention

All character offsets in the citation model are **0-based and half-open**: `start` is inclusive, `end` is exclusive. This applies uniformly to every model that carries offsets:

| Model | Fields | Meaning |
|---|---|---|
| `AnswerSpan` | `char_start`, `char_end` | Offsets in the full answer string |
| `EvidenceSpan` | `char_start`, `char_end` | Absolute offsets in the source document |
| `Citation` | `char_start`, `char_end` | Enclosing span in the source (legacy contiguous view) |
| `RetrievalSupport` | `passage_char_start`, `passage_char_end` | Absolute offsets of the passage window |
| `SourceChunk` | `doc_char_start`, `doc_char_end` | Position of the chunk in the original document |

The `TokenizedText` model in `results.py` (lines 26–45) enforces the half-open contract with a model validator that checks token spans are monotonic, non-overlapping, and within text bounds. The same convention appears in `slice_tokenized_text()` in `prepared_corpus.py` (lines 486–519), which uses `bisect_right` / `bisect_left` to select token indices that fall inside a passage range.

```python
start_index = bisect_right(token_ends, start)   # first token ending after start
end_index = bisect_left(token_starts, end, lo=start_index)  # first token starting at or after end
```

## Chunk rebasing and absolute offsets

When a `SourceDocument` is passed to `align_citations()`, all citation offsets are absolute positions within that document text. When a `SourceChunk` is used instead, offsets must be rebased to positions in the **original document**, not the chunk text.

The rebasing mechanism uses `NormalizedSource.base_doc_offset` in `prepared_corpus.py` (lines 399–435). For a `SourceChunk`, this field is set to `item.doc_char_start`:

```python
normalized.append(
    NormalizedSource(
        source_id=item.source_id,
        source_index=source_index,
        text=item.text,              # chunk text (alignment is done against this)
        base_doc_offset=item.doc_char_start,  # offset of chunk in original document
        full_text=item.document_text,  # original full document (may be None)
    )
)
```

All absolute offsets in `Citation` and `EvidenceSpan` are computed by adding `base_doc_offset` to the local passage offset. This is visible in `_create_evidence_span()` in `citations.py` (lines 1450–1458):

```python
abs_start = (
    candidate.source.base_doc_offset
    + candidate.passage.doc_char_start
    + seg_char_start
)
abs_end = (
    candidate.source.base_doc_offset
    + candidate.passage.doc_char_start
    + seg_char_end
)
```

The `base_doc_offset` is `0` for `SourceDocument` inputs and `doc_char_start` for `SourceChunk` inputs, so the math is identical in both cases.

## Evidence string equality with source slices

The central invariant is: after rebasing, `SourceDocument.text[citation.char_start:citation.char_end] == citation.evidence`.

This is guaranteed by `_slice_source_text()` in `citations.py` (lines 1376–1381), which is called for every `Citation` and `RetrievalSupport`:

```python
def _slice_source_text(source: NormalizedSource, abs_start: int, abs_end: int) -> str:
    if source.full_text is not None:
        return source.full_text[abs_start:abs_end]
    local_start = abs_start - source.base_doc_offset
    local_end = abs_end - source.base_doc_offset
    return source.text[local_start:local_end]
```

Two cases exist:

- **Full document available** (`full_text` is not `None`): slices directly from the original document.
- **Chunk only** (`full_text` is `None`): subtracts `base_doc_offset` to convert absolute offsets to chunk-local positions, then slices `source.text`.

The same logic applies to `EvidenceSpan.evidence`. The `SourceChunk` model validator (lines 149–162 in `results.py`) enforces that the provided `document_text` slice matches the chunk text exactly, so rebasing is always safe.

The docstring on `Citation` in `results.py` (line 228) explicitly documents this: *"Satisfies the same rebased slicing rule used internally by `_slice_source_text()`."*

## Multi-span evidence representation

When `CitationConfig.multi_span_evidence` is `True` (default `False`), the model can represent non-contiguous evidence in a single citation via `Citation.evidence_spans`. This is useful when a source passage has a gap that was not matched but is still relevant context.

### Fields

- **`Citation.evidence_spans`** (`list[EvidenceSpan]`): Each `EvidenceSpan` has its own `char_start`, `char_end`, and `evidence` string. Offsets are absolute and rebased.
- **`Citation.exact_evidence`** (computed property): Canonical exact evidence derived from `evidence_spans`. Spans are sorted by position and joined with `" ... "` so omitted bridge text is visible to readers.
- **`Citation.evidence`**: Legacy contiguous enclosing span. May include bridge text not directly matched. Use `exact_evidence` or `evidence_spans` for precise attribution.

### Alignment to spans

The path from alignment to `EvidenceSpan` objects runs through `_extract_evidence()` in `citations.py` (lines 930–943), which calls `_alignment_to_evidence_spans()`:

```python
def _alignment_to_evidence_spans(candidate, alignment, cfg):
    spans = _extract_multi_span_evidence(candidate, alignment, cfg)
    if cfg.multi_span_evidence and _alignment_has_disjoint_match_blocks(alignment) and spans:
        return spans
    # fallback to single span
    spans = _extract_single_span_evidence(candidate, alignment)
    return spans if spans else None
```

`multi_span_merge_gap_chars` controls whether neighboring spans separated by small gaps are merged into a single span. `multi_span_max_spans` caps the number of spans; if exceeded, the citation falls back to a single contiguous span.

### Legacy enclosing span

`Citation.char_start` / `char_end` always describe the **enclosing span** from the minimum start to the maximum end across all `evidence_spans`. This maintains backward compatibility:

```python
abs_start = min(span.char_start for span in evidence_spans)
abs_end = max(span.char_end for span in evidence_spans)
evidence = _slice_source_text(candidate.source, abs_start, abs_end)
```

## Alignment model

`Alignment` (lines 64–86 in `results.py`) represents the Smith-Waterman alignment result:

```python
class Alignment(BaseModel):
    score: int
    token_start: int   # inclusive start in candidate tokens
    token_end: int     # exclusive end in candidate tokens
    matches: int       # exact token matches
    match_blocks: list[tuple[int, int]] = Field(default_factory=list)
```

`match_blocks` is populated when the aligner detects disjoint matched regions. The `return_match_blocks` flag on the aligner must be `True` for this to work.

## Configuration

`CitationConfig` (lines 25–83 in `citation_config.py`) controls multi-span behavior:

| Field | Default | Purpose |
|---|---|---|
| `multi_span_evidence` | `False` | Enable multi-span citations |
| `multi_span_merge_gap_chars` | `16` | Merge spans with gaps ≤ N characters |
| `multi_span_max_spans` | `5` | Cap on spans before fallback to contiguous |

The `strict()` and `permissive()` presets adjust scoring thresholds but do not change multi-span settings.
