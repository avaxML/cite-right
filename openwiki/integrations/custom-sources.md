---
type: integration-guide
title: Custom Sources
description: How to feed non-framework retrieval into Cite-Right — build SourceDocument and SourceChunk directly, use from_dicts for plain dictionaries, and pass the result to align_citations. Covers chunk-rebase offsets, the document_text validation, and the evidence equality invariant.
tags: [custom-sources, source-document, source-chunk, from-dicts, align-citations, char-offsets, chunk-rebase, evidence-equality, retrieval]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Custom Sources

Not every retrieval layer is LangChain or LlamaIndex. If your pipeline already produces documents, chunks, or plain dictionaries, you can hand them to Cite-Right directly. The public inputs are `SourceDocument` and `SourceChunk`; the convenience adapter for dictionaries is `from_dicts`. Once you have a list of those, you call `align_citations` as usual.

This page covers the three integration shapes (whole documents, pre-chunked excerpts, raw dicts), the chunk-rebase offset contract that makes citation highlights point at the original document, and the validation that protects the offsets from drifting. The end-to-end pipeline and the offset convention are in [Citation Alignment](../concepts/citation-alignment.md). The data-model fields that come back are in [Citation Alignment](../concepts/citation-alignment.md) and the API reference.

## The Three Input Shapes

`align_citations` (in `src/cite_right/citations.py`) accepts a sequence of any of:

- `SourceDocument(id, text, metadata=...)` — a full document.
- `SourceChunk(source_id, text, doc_char_start, doc_char_end, ...)` — a pre-chunked excerpt with offsets into a parent document.
- plain `str` — auto-assigned an id of `"source_0"`, `"source_1"`, and so on.

Mixing all three in a single call is fine. The pipeline normalizes each into a `NormalizedSource` (`base_doc_offset`, `full_text`) before any segmentation, indexing, or Smith-Waterman runs, so the candidates that come out the other side are uniform.

## Building SourceDocument Directly

When you have whole documents, build `SourceDocument` and call `align_citations`. This is the simplest path and the one you use for any retrieval system that already returns complete texts (a database, a search index, an object store, an in-memory list).

```python
from cite_right import SourceDocument, align_citations

sources = [
    SourceDocument(
        id="annual_report_2024",
        text=(
            "Acme Corporation reported revenue of 5.2 billion dollars in 2024, "
            "representing a 12% increase over the previous year."
        ),
        metadata={"year": 2024, "type": "financial"},
    ),
    SourceDocument(
        id="press_release",
        text=(
            "Acme Corporation today announced fourth quarter revenue of "
            "5.2 billion dollars, a new company record."
        ),
    ),
]

answer = "Acme Corporation reported revenue of 5.2 billion dollars in 2024."
results = align_citations(answer, sources)
for result in results:
    print(result.status, result.answer_span.text)
    for citation in result.citations:
        print(citation.source_id, citation.evidence)
```

`id` is the stable identifier that will be returned on every `Citation.source_id` and `RetrievalSupport.source_id` that resolves to that document. Use IDs that are meaningful in your application — database keys, file paths, URLs, or hashes — so the citation metadata can be tied back to the source row.

`text` is the full document text. The pipeline will segment it into passage windows, build the inverted index, and run Smith-Waterman against the answer.

`metadata` is optional, defaults to `{}`, and is preserved on the `SourceDocument`. Cite-Right does not read it during alignment, but your application code can read it back off the `SourceDocument` you built, or attach it to the citation through your own lookup.

## Building SourceChunk For Pre-Chunked Content

If your retrieval system already divides documents into chunks and stores each chunk's position in the parent (most vector databases do this), build `SourceChunk` instead of `SourceDocument`. The pipeline uses the offsets to rebase every resulting `Citation` back onto the original document.

```python
from cite_right import SourceChunk, align_citations

sources = [
    SourceChunk(
        source_id="report_2024",
        text="This is the text of chunk 1.",
        doc_char_start=0,
        doc_char_end=28,
    ),
    SourceChunk(
        source_id="report_2024",
        text="This is the text of chunk 2.",
        doc_char_start=29,
        doc_char_end=57,
    ),
]

results = align_citations(answer, sources)
for result in results:
    for citation in result.citations:
        # char_start and char_end are absolute in the parent document
        print(citation.source_id, citation.char_start, citation.char_end)
        print(citation.evidence)
```

`source_id` identifies the parent document. Multiple chunks that came from the same parent share a `source_id`; the citation offsets will be absolute in the same parent.

`doc_char_start` and `doc_char_end` are the chunk's half-open character offsets in the parent. They are added to whatever the local alignment produces so the public `Citation.char_start` / `Citation.char_end` are absolute in the parent, not in the chunk.

### When To Use Chunks Versus Whole Documents

Use `SourceChunk` when you already know each excerpt's position in the original — vector databases, text splitters that emit offsets, sliding-window retrievers. The pipeline does not re-window or re-segment, so the candidate passages are the chunks themselves, and the offsets come out already aligned with the parent.

Use `SourceDocument` when you have complete texts and you want the pipeline to handle passage creation internally. The default `SimpleSegmenter` and `SimpleTokenizer` will window, tokenize, and index the document. The `PreparedCitationCorpus` built from `SourceDocument`s still works the same way; the only thing `SourceChunk` saves is the windowing step on the source side.

## Chunk Rebase And The Evidence Equality Invariant

The central contract when feeding `SourceChunk` is the **evidence equality invariant**: after chunk rebasing,

```
source.text[citation.char_start:citation.char_end] == citation.evidence
```

`source.text` here is the parent document's text. `citation.char_start` and `citation.char_end` are absolute in the parent, regardless of which chunk the evidence was located in. The same rule applies to each `EvidenceSpan` in multi-span mode.

The rebase happens in two places. In `src/cite_right/core/prepared_corpus.py`, `normalize_sources` sets `base_doc_offset = item.doc_char_start` for chunks (and `0` for whole documents or plain strings) and stores `full_text = item.document_text` when supplied. In `src/cite_right/citations.py`, `_slice_source_text` and the alignment builders add `base_doc_offset` to every candidate's local offset before the result lands on a `Citation` or `EvidenceSpan`. `_create_evidence_span` shows the same formula:

```python
abs_start = candidate.source.base_doc_offset + candidate.passage.doc_char_start + seg_char_start
abs_end   = candidate.source.base_doc_offset + candidate.passage.doc_char_end   + seg_char_end
return EvidenceSpan(char_start=abs_start, char_end=abs_end, evidence=_slice_source_text(...))
```

Because `Citation.evidence` is re-sliced from `source.full_text` (when you pass it) or from the chunk text minus the chunk's `base_doc_offset`, the equality holds even though the alignment itself ran on chunk-local coordinates.

You do not have to pass the full parent document for the rebase to work — the chunk range alone is enough. If you do pass `document_text`, you also get a slice-equality check at construction time.

### Optional document_text Validation

`SourceChunk.document_text` is optional. When you pass it, the model validator on `SourceChunk` (`_validate_document_text_alignment` in `src/cite_right/core/results.py`) enforces three things:

1. `doc_char_start >= 0` and `doc_char_end >= doc_char_start`.
2. `doc_char_end <= len(document_text)`.
3. `document_text[doc_char_start:doc_char_end] == text`.

If the parent text is known (you have it in memory, your retrieval API returned it, you are about to feed it back to a frontend), pass it as `document_text` to catch drift between the chunk text and the parent slice at construction. If you do not have it on hand, omit `document_text` and the rebase still produces correct absolute offsets — `_slice_source_text` falls back to chunk-local slicing.

```python
full_text = open("report_2024.txt").read()
chunk = SourceChunk(
    source_id="report_2024",
    text=full_text[1500:1548],
    doc_char_start=1500,
    doc_char_end=1548,
    document_text=full_text,  # enables the slice-equality check
)
```

The LangChain and LlamaIndex chunk adapters use the same field: `from_langchain_chunks` reads `full_text_key` from metadata and passes it through as `document_text`; `from_llamaindex_chunks` omits `document_text` because LlamaIndex nodes do not always carry the parent.

## Using from_dicts For Plain Dictionaries

When the upstream system hands you plain dictionaries (an API response, a JSON file, a search index's wire format), `from_dicts` converts each one to a `SourceDocument`. It is exported from the top-level `cite_right` package alongside the framework adapters.

```python
from cite_right import align_citations
from cite_right.integrations import from_dicts

api_response = [
    {"id": "result_1", "text": "First document text.", "score": 0.95},
    {"id": "result_2", "text": "Second document text.", "score": 0.87},
]

sources = from_dicts(api_response)
results = align_citations(answer, sources)
```

`from_dicts` is defined in `src/cite_right/integrations.py` and takes two keyword-only knobs:

- `text_key` (default `"text"`) — the dictionary key that holds the document text. The value is read with `doc.get(text_key, "")` and coerced to `str` before being put on the `SourceDocument`.
- `id_key` (default `"id"`) — the dictionary key that holds the document ID. If the key is missing, the index (`"0"`, `"1"`, ...) is used.

Any other key in the dictionary is copied into the `SourceDocument.metadata` mapping, except for the `text_key` and `id_key` keys themselves.

```python
# Renaming fields is one explicit call
sources = from_dicts(
    api_response,
    text_key="content",
    id_key="doc_id",
)
```

`from_dicts` does not do a multi-key fallback like the older draft of this page claimed. If your payload uses `"body"` or `"page_content"`, either rename the key in Python or pass the right `text_key`. That is the whole contract.

```python
# Standardize first, then convert
standardized = [
    {"text": d["body"], "id": d["url"]}
    for d in raw_payload
]
sources = from_dicts(standardized)
```

If you also need chunk offsets, build `SourceChunk` directly with the offsets your retrieval layer already knows. There is no `from_dicts` variant that emits `SourceChunk`s; the dict adapter is for whole documents.

## Mixing Source Types

You can mix `SourceDocument` and `SourceChunk` in one call. The pipeline normalizes each, so candidates from a whole document and candidates from a pre-chunked excerpt sit in the same list and compete on equal footing.

```python
sources = [
    SourceDocument(id="full_doc", text="Complete document text..."),
    SourceChunk(
        source_id="chunked_doc",
        text="Chunk text...",
        doc_char_start=100,
        doc_char_end=200,
    ),
]

results = align_citations(answer, sources)
```

This is useful for hybrid retrieval: one set of results comes back from a vector store with offset metadata (build `SourceChunk`s), another set comes back as complete blobs from a database (build `SourceDocument`s). Pass them both in one call and the citations will carry the right `source_id` and the right absolute offsets in each parent.

## Custom Sources In A RAG Pipeline

The typical shape: retrieve a set of documents or chunks, build the right input type for each, call `align_citations` on the generated answer. Here is the pattern with a search index that returns whole documents.

```python
from cite_right import SourceDocument, align_citations


def search_and_cite(query, answer, index):
    hits = index.search(query, top_k=10)
    sources = [
        SourceDocument(
            id=hit["id"],
            text=hit["text"],
            metadata={
                "score": hit["score"],
                "title": hit.get("title"),
            },
        )
        for hit in hits
    ]
    return align_citations(answer, sources)
```

If your store already splits into chunks with offsets, swap `SourceDocument` for `SourceChunk` and add the offsets your store tracks. If your store returns JSON, `from_dicts` does the conversion in one line — `sources = from_dicts(json_response)` and the rest of the pipeline is unchanged.

## What Cite-Right Does Not Touch

`metadata` is opaque to Cite-Right. Cite-Right does not index it, does not score on it, and does not emit it on `Citation` results. It is your handle for round-tripping the original document back to the user. The same is true of any framework-specific extras (retrieval scores, file paths, page numbers) — they belong in `SourceDocument.metadata` or `SourceChunk.metadata` and you read them off after the call returns.

`from_dicts` makes the round-trip automatic: every dictionary key that is not the `text_key` or the `id_key` ends up in `metadata`. That is the only place Cite-Right looks at your data outside the text.

## Common Pitfalls

- **Do not pass plain `str` in production.** Plain strings get auto-assigned ids `"source_0"`, `"source_1"`, and so on, and you will lose the ability to map citations back to your real source identifiers. Build `SourceDocument` or `SourceChunk` so the `source_id` is meaningful.
- **Do not forget the chunk range when building `SourceChunk`.** Without `doc_char_start` and `doc_char_end`, the model validator rejects the construction, and the pipeline has no way to rebase the offsets.
- **Do not pass `document_text` that does not actually contain the chunk text.** The slice-equality check will raise and you will know immediately, which is the point. If the parent text is not available, omit the field and the rebase still produces correct absolute offsets.
- **Do not expect `from_dicts` to fall through to `content` or `body`.** It only reads the configured `text_key` (default `"text"`). If the key is missing, the text defaults to `""` and the document is effectively empty.
- **Do not pass a chunk that overlaps another chunk from the same parent.** Cite-Right does not deduplicate; two overlapping chunks can each produce a citation, and the offsets will both be valid but the user will see the same evidence twice. Deduplicate at the retrieval layer.

## See Also

- [Citation Alignment](../concepts/citation-alignment.md) — full I/O contract, the half-open offset convention, the evidence equality invariant, and the rules behind `status`.
- [LangChain Integration](langchain.md) — `from_langchain_documents` and `from_langchain_chunks` for LangChain retrievers.
- [LlamaIndex Integration](llamaindex.md) — `from_llamaindex_nodes` and `from_llamaindex_chunks` for LlamaIndex retrievers.
- [Quickstart](../getting-started/quickstart.md) — the basic `align_citations` call and `PreparedCitationCorpus` reuse.
