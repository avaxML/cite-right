
# Citation Alignment

Citation alignment is the operation that takes a generated answer and a set of source documents, and returns character-accurate citations showing where each span of the answer is grounded. This page covers the public I/O contract: what you hand to `align_citations`, what comes back, and how to interpret the offsets.

For the end-to-end pipeline (index-first retrieval, Smith-Waterman localization, contradiction checks, embedder path) see [How It Works](how-it-works.md). For the precise attribution view with non-contiguous evidence, see [Multi-Span Evidence](../advanced/multi-span-evidence.md).

## Quick Example

```python
from cite_right import SourceDocument, align_citations

answer = "The company reported record revenue in Q4."
sources = [
    SourceDocument(
        id="earnings_call",
        text="During the earnings call, the CEO announced that the company reported record revenue in Q4 of 2024.",
    )
]

results = align_citations(answer, sources)
for result in results:
    print(result.answer_span.text, result.status)
    for citation in result.citations:
        print(citation.evidence, citation.char_start, citation.char_end)
```

The entry point is `align_citations` in `src/cite_right/citations.py`. It returns a list of `SpanCitations`, one per answer segment, with localized citations and a per-span status.

## Input Types

The `sources` argument accepts three forms, reflecting different retrieval patterns: plain `str`, `SourceDocument`, or `SourceChunk`. Mixing them in one call is fine; each is normalized into the same internal candidate list.

### SourceDocument

`SourceDocument(id, text, metadata=...)` is a full document. Use it when your retrieval system returns whole documents and you want Cite-Right to handle passage creation internally.

```python
from cite_right import SourceDocument

doc = SourceDocument(
    id="annual_report_2024",
    text="The full text of the annual report...",
    metadata={"year": 2024, "type": "financial"},
)
```

The `id` becomes the `source_id` on every `Citation` and `RetrievalSupport` that resolves to that document.

### SourceChunk

`SourceChunk(source_id, text, doc_char_start, doc_char_end, ...)` is a pre-chunked excerpt. Use it when you have already chunked your documents and you want citation offsets to point at positions in the original full document, not the chunk itself.

```python
from cite_right import SourceChunk

chunk = SourceChunk(
    source_id="annual_report_2024",
    text="This is a specific passage from the document.",
    doc_char_start=1500,
    doc_char_end=1548,
)
```

`doc_char_start` and `doc_char_end` are absolute offsets into the parent document (half-open, like every other offset in the system). Internally, Cite-Right rebases chunk-local alignment offsets back onto the original document so that `source.text[citation.char_start:citation.char_end] == citation.evidence` still holds. The rebasing only needs the chunk range — you do not have to pass the full document text. If you do pass `document_text`, `SourceChunk` validates that the slice matches `text` at construction time.

### Plain Strings

A bare `str` is treated as a `SourceDocument` with an auto-assigned id (`"source_0"`, `"source_1"`, ...). Use it for ad hoc tests and quick experiments. Real pipelines should pass named `SourceDocument` or `SourceChunk` objects so citations carry stable `source_id` values.

## Output Structure

`align_citations` returns `list[SpanCitations]`, one entry per answer segment. The structure is defined in `src/cite_right/core/results.py`.

### SpanCitations

A `SpanCitations` has four fields.

`answer_span` is the `AnswerSpan` this set of citations belongs to. It carries the segment text plus `char_start` / `char_end` in the full answer string, and a `kind` of `"sentence"`, `"clause"`, or `"paragraph"`.

`citations` is a list of `Citation` objects, ranked best first. Empty when no citation met the minimum thresholds. Each `Citation` is exact and localized: a `source_id`, a half-open `char_start` / `char_end` into the source, and an `evidence` string that slices cleanly from the source.

`retrieval_support` is a list of `RetrievalSupport` objects. These are passages the candidate selector picked up (by inverted index, lexical prefilter, or embedding similarity) that did not localize into a `Citation`. They are evidence-of-interest, not a grounded citation.

`status` is one of `"supported"`, `"partial"`, or `"unsupported"`. It comes from the best exact `Citation`, never from embedding similarity. A high embedding score that never localizes is a `RetrievalSupport`, not a `Citation`, and it does not flip the status. The model validator on `SpanCitations` enforces that `"unsupported"` requires an empty `citations` list, and that `"supported"` or `"partial"` requires at least one.

### Status Rules

`"supported"` means the top-ranked `Citation` has `answer_coverage >= supported_answer_coverage` (default 0.6) and the cheap contradiction check did not fire.

`"partial"` means citations exist but the supported threshold was not met, **or** the contradiction check fired. The literal is `"partial"`, never `"partially_supported"`. Contradiction (negation, number mismatch, leftover n-gram slot, entity swap) downgrades to `"partial"`, not `"unsupported"`, because the evidence exists — it just conflicts with the claim.

`"unsupported"` means no citation survived filtering. The span may be hallucinated, or it may be content from outside the provided sources. `"unsupported"` is "no localized citation survived," not a high-precision hallucination label.

### Citation

Each `Citation` carries the matched text and where it came from.

```python
for result in results:
    for citation in result.citations:
        print(citation.source_id, citation.evidence)
        print(citation.char_start, citation.char_end)
        print(citation.score, citation.components)
```

`source_id` is the document identifier. `source_index` is the position in the input list. `char_start` and `char_end` are the half-open offsets of the legacy contiguous evidence view in the source. `evidence` is the text sliced by that range. `score` is a weighted sum of `components` (`alignment_score`, `normalized_alignment`, `matches`, `answer_coverage`, `evidence_coverage`, `lexical_score`, `embedding_score`).

When `CitationConfig(multi_span_evidence=True)`, `evidence_spans` carries the precise non-contiguous regions and `exact_evidence` joins them with `" ... "`. The legacy `char_start` / `char_end` / `evidence` stay a contiguous enclosing span in that mode. See [Multi-Span Evidence](../advanced/multi-span-evidence.md).

## Character Offset Convention

All character offsets in Cite-Right are Python half-open intervals: `char_start` is inclusive, `char_end` is exclusive, and `text[char_start:char_end]` is the slice the offset pair refers to. This holds for `AnswerSpan`, `Citation`, and `EvidenceSpan`.

The rule that ties inputs and outputs together is the evidence equality invariant: after chunk rebasing, `source.text[citation.char_start:citation.char_end] == citation.evidence`. This is what makes the offsets safe to feed directly to a Python slicer, a highlighter, or an offset map in your UI. The same applies to `EvidenceSpan` in multi-span mode.

`SourceChunk` keeps the same rule. Internally, alignment runs against the chunk text; the chunk's `doc_char_start` is added to the resulting offsets before they are exposed on the `Citation`, so the public offsets are absolute in the parent document.

## What Status Comes From

Status is driven by the best exact citation's `answer_coverage` and the contradiction check. Embedding similarity, lexical overlap, and the inverted index are used to **select** candidate passages; they are not what flips a span to `"supported"`. A passage that scores well on embedding but never localizes into a `Citation` lands in `retrieval_support` and does not change the status.

Concretely: the per-span logic in `_span_status` looks at `citations[0].components["answer_coverage"]`. If that coverage meets `supported_answer_coverage` and `check_contradiction` is clean, the status is `"supported"`. If contradiction fires, the status is `"partial"` (not `"unsupported"`). If coverage is below the threshold, the status is `"partial"`. If `citations` is empty, the status is `"unsupported"`.

## How Candidates Get Chosen

`align_citations` does not run Smith-Waterman over every passage window. On the default / Rust path, prepare builds an inverted index over source windows and rare-token intersect chooses the windows worth aligning. Smith-Waterman still localizes the citation; the index only chooses which windows to localize on. If the optional Rust extension is missing, or you supply a custom tokenizer or segmenter, `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and `_select_candidates` falls back to lexical selection.

When an embedder is provided, `_add_embedding_candidates` can add non-index windows to the candidate set. Those extras still need Smith-Waterman to produce a `Citation`. Rust prepare still runs in that case — the embedding index is built on the prepared candidates. The full pipeline is in [How It Works](how-it-works.md). The embedder-specific behavior is in [Embedding Retrieval](../advanced/embedding-retrieval.md).

## When Citations Stay Exact

Two paraphrasing paths let spans that would otherwise overflag to `"unsupported"` still emit a `Citation`.

Content-word overlap on the candidate passage can emit a citation when sequential Smith-Waterman coverage is low. Grounded how-to and news paraphrases that share content words in different order are not dropped just because the alignment score is low.

Structured field:value sources (Data2txt hours, amenities, and similar) get a second Smith-Waterman pass per matching candidate with `gap_score=0`. Faithful rewrites of known fields can land on `"supported"` or `"partial"`. Invented fields stay `"unsupported"`.

Both paths still require an exact, localized evidence slice. They do not produce embedding-only citations.

## Backend, Tokenizer, and Segmenter

`align_citations` exposes the same backend, tokenizer, and segmenter knobs as the underlying pipeline. Defaults are Rust when available (`backend="auto"`), `SimpleTokenizer`, and `SimpleAnswerSegmenter` / `SimpleSegmenter`.

```python
results = align_citations(
    answer,
    sources,
    backend="auto",            # "auto" | "python" | "rust"
    tokenizer=HuggingFaceTokenizer.from_pretrained("bert-base-uncased"),
    answer_segmenter=SpacyAnswerSegmenter(split_clauses=True),
    source_segmenter=SpacySegmenter(),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)
```

A custom tokenizer or segmenter disables the Rust inverted-index fast path: `from_sources` leaves `inverted_index=None` and candidate selection falls back to lexical prefilter. Smith-Waterman still runs on whatever the fallback picks.

## Custom Sources

If you have a non-standard retrieval layer, you can build `SourceDocument` and `SourceChunk` directly from your own data, or use the dictionary adapter. The offsets you pass to `SourceChunk.doc_char_start` / `doc_char_end` are the contract that lets citation offsets rebase onto your full document. See [Custom Sources](../integrations/custom-sources.md).
