---
type: concept
title: Source input shapes
description: The three accepted input types for source documents, how they are normalized into `NormalizedSource`, and how citation offsets are rebased relative to the original document.
tags: [source-inputs, input-types, offsets, citations, cite-right]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
  - id: openwiki-source-6bdbae01887b73d94b8375db
    resource: repo://src/cite_right/integrations.py
  - id: openwiki-source-81dc541c73d5fbfa6a7e1947
    resource: repo://tests/test_citations_multi_span.py
  - id: openwiki-source-e7ea83755eccd303a72448ec
    resource: repo://tests/test_error_conditions.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

cite-right accepts three distinct shapes when you pass a `sources` list to [`align_citations()`](repo://src/cite_right/citations.py#L83-L146). All three are immediately normalised into an internal [`NormalizedSource`](repo://src/cite_right/core/prepared_corpus.py#L63-L70) representation before any tokenisation or candidate generation occurs. Understanding how offsets are handled at this layer explains the entire re-basing contract.

## The three accepted types

### Plain `str`

The simplest form — cite-right auto-assigns an integer source ID and treats the string as a complete, self-contained document:

```python
results = align_citations(answer, ["Revenue grew 15% in Q4."])
```

Internally this becomes:

| Field | Value |
|---|---|
| `source_id` | `str(index)` (e.g. `"0"`) |
| `source_index` | list position |
| `text` | the string itself |
| `base_doc_offset` | `0` |
| `full_text` | the string itself |

### `SourceDocument`

Use this when you want a named document ID that propagates to every returned [`Citation.source_id`](repo://src/cite_right/core/results.py#L218-L219):

```python
from cite_right import SourceDocument, align_citations

doc = SourceDocument(id="annual-report-2024", text="Revenue grew 15% in Q4.")
results = align_citations(answer, [doc])
```

Fields map as follows:

| Field | Value |
|---|---|
| `source_id` | `SourceDocument.id` |
| `source_index` | list position |
| `text` | `SourceDocument.text` |
| `base_doc_offset` | `0` |
| `full_text` | `SourceDocument.text` |

Because `base_doc_offset` is zero, all citation character offsets (`Citation.char_start`, `Citation.char_end`) are direct indexes into `SourceDocument.text`.

### `SourceChunk`

Use this when you have **pre-chunked** documents — for example, chunks produced by a RAG chunker — and you want citation offsets to be expressed relative to the **original full document**, not the chunk:

```python
from cite_right import SourceChunk

chunk = SourceChunk(
    source_id="annual-report-2024",
    text="Revenue grew 15%.",
    doc_char_start=120,       # position in the ORIGINAL document
    doc_char_end=138,
    document_text=full_doc,    # optional but recommended
)
```

| Field | Value |
|---|---|
| `source_id` | `SourceChunk.source_id` |
| `source_index` | `SourceChunk.source_index` if set, else list position |
| `text` | `SourceChunk.text` |
| `base_doc_offset` | `SourceChunk.doc_char_start` |
| `full_text` | `SourceChunk.document_text` (or `None`) |

The `base_doc_offset` is the single mechanism that shifts every downstream citation offset back to the original document origin.

## The `normalize_sources()` pipeline

[`normalize_sources()`](repo://src/cite_right/core/prepared_corpus.py#L399-L435) is the single function responsible for converting the heterogeneous input list into a uniform `list[NormalizedSource]`. It is called from [`PreparedCitationCorpus.from_sources()`](repo://src/cite_right/core/prepared_corpus.py#L126-L190) before any other preparation step.

For a `SourceChunk`, the critical line is:

```python
# prepared_corpus.py  lines 425–434
source_index = item.source_index if item.source_index is not None else index
normalized.append(
    NormalizedSource(
        source_id=item.source_id,
        source_index=source_index,
        text=item.text,
        base_doc_offset=item.doc_char_start,
        full_text=item.document_text,
    )
)
```

`doc_char_start` becomes `base_doc_offset`, anchoring all subsequent evidence slicing to the original document's coordinate system.

## The `document_text` alignment validator

[`SourceChunk._validate_document_text_alignment()`](repo://src/cite_right/core/results.py#L149-L162) enforces the invariant that `document_text[doc_char_start:doc_char_end]` must **exactly equal** `text`. This validator runs automatically during Pydantic model construction:

```python
# results.py  lines 149–162
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
```

Three checks are performed when `document_text` is supplied:

1. **Range validity** — `doc_char_start >= 0` and `doc_char_end >= doc_char_start`.
2. **Bounds check** — `doc_char_end <= len(document_text)`.
3. **Slice equality** — `document_text[doc_char_start:doc_char_end] == text` (the core invariant).

Passing a mismatched slice (e.g. a `doc_char_start` that points at the wrong paragraph) raises `ValueError` immediately, before any alignment work begins. This catches misconfiguration at construction time rather than producing silent wrong offsets downstream.

## The chunk-rebasing rule in evidence slicing

Once a `NormalizedSource` is built, every evidence extraction passes through [`_slice_source_text()`](repo://src/cite_right/citations.py#L1376-L1381):

```python
# citations.py  lines 1376–1381
def _slice_source_text(source: NormalizedSource, abs_start: int, abs_end: int) -> str:
    if source.full_text is not None:
        return source.full_text[abs_start:abs_end]
    local_start = abs_start - source.base_doc_offset
    local_end = abs_end - source.base_doc_offset
    return source.text[local_start:local_end]
```

Two branches:

- **`full_text` is available** (`document_text` was supplied): the evidence string is re-sliced **directly** from the original full document using the absolute offsets returned by the alignment. No offset arithmetic is needed.
- **`full_text` is `None`: the absolute offsets are **relative to the original document start**, but only `text` (the chunk) is available. The function subtracts `base_doc_offset` to convert them back to local chunk coordinates before slicing `text`.

The absolute offsets themselves are computed in [`_create_evidence_span()`](repo://src/cite_right/citations.py#L1441-L1468) and [`_extract_evidence()`](repo://src/cite_right/citations.py#L930-L943):

```python
# citations.py  lines 1450–1458
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

Because `candidate.passage.doc_char_start` is always relative to `source.text`, and `source.base_doc_offset` equals `SourceChunk.doc_char_start` (or `0` for `SourceDocument`), the sum is an **absolute offset in the original document coordinate system**.

## Invariant guarantees

| Input type | `Citation.char_start/char_end` are relative to | Evidence re-sliced from |
|---|---|---|
| `str` | the string itself | the string (`full_text` branch) |
| `SourceDocument` | `SourceDocument.text` | `SourceDocument.text` (`full_text` branch) |
| `SourceChunk` + `document_text` | the original full document | `document_text` (`full_text` branch, direct slice) |
| `SourceChunk` (no `document_text`) | the original full document | `SourceChunk.text` (`local` branch, re-based) |

In every case, `full_doc[citation.char_start:citation.char_end] == citation.evidence` holds **when `full_text` is available**; when it is not, `chunk.text[citation.char_start - base_doc_offset : citation.char_end - base_doc_offset] == citation.evidence` holds instead.

## Relationship to `PreparedCitationCorpus`

[`PreparedCitationCorpus.from_sources()`](repo://src/cite_right/core/prepared_corpus.py#L126-L190) is the public factory that consumes the heterogeneous `sources` list. It calls `normalize_sources()` first, then either dispatches to a Rust fast path (when `use_rust=True` and the tokenizer/segmenter are the simple defaults) or falls back to a pure-Python pipeline. Both paths share the same `NormalizedSource` contract.

## Framework integration helpers

The three integration helpers in [`cite_right.integrations`](repo://src/cite_right/integrations.py) convert external framework types into cite-right's native shapes:

- [`from_langchain_documents()`](repo://src/cite_right/integrations.py#L138-L180) → `list[SourceDocument]`
- [`from_langchain_chunks()`](repo://src/cite_right/integrations.py#L183-L239) → `list[SourceChunk]` (reads `start_index`/`end_index` from LangChain document metadata)
- [`from_llamaindex_nodes()`](repo://src/cite_right/integrations.py#L242-L284) → `list[SourceDocument]`
- [`from_llamaindex_chunks()`](repo://src/cite_right/integrations.py#L287-L342) → `list[SourceChunk]` (reads `start_char_idx`/`end_char_idx` from LlamaIndex metadata)
- [`from_dicts()`](repo://src/cite_right/integrations.py#L345-L381) → `list[SourceDocument]` for arbitrary dict-based pipelines

## Representative tests

| Test | What it verifies |
|---|---|
| [`test_sourcechunk_document_text_must_match_chunk_text`](repo://tests/test_error_conditions.py#L192-L200) | The validator raises `ValueError` when `document_text[doc_char_start:doc_char_end] != text` |
| [`test_align_citations_multi_span_evidence_respects_sourcechunk_offsets`](repo://tests/test_citations_multi_span.py#L83-L123) | Multi-span citations produce correct absolute offsets in `full_doc` when `document_text` is supplied |
| [`test_align_citations_sourcechunk_without_document_text_slices_locally`](repo://tests/test_citations_multi_span.py#L220-L248) | When `document_text` is `None`, evidence is correctly re-sliced from the chunk text using `base_doc_offset` subtraction |
| [`test_citations_api_mixed_sources_with_sourcechunk`](repo://tests/test_citations_api.py#L488-L550) | A mixed list of `SourceDocument` and `SourceChunk` entries is processed correctly, with chunk offsets verified against the full document text |
