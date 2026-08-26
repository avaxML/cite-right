---
type: getting-started-guide
title: Quickstart
description: Build a complete citation pipeline in a few lines — call align_citations on a generated answer plus sources, read per-span status and evidence, handle multiple sources, and reuse a PreparedCitationCorpus for repeated queries.
tags: [quickstart, getting-started, align-citations, prepared-citation-corpus, source-document, status, supported, partial, unsupported, answer-coverage, evidence, char-offsets]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-ad97420c2a0900f616ed0fef
    resource: repo://src/cite_right/contradiction.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Quickstart

This page walks through the basic Cite-Right pattern, the shape of what `align_citations` returns, what the three status values mean, how to feed multiple sources, and how to reuse a prepared corpus when you have a fixed source set and many answers to score against it.

For installation and the optional extras, see [Installation](installation.md). For the I/O contract and what each field on `SpanCitations` and `Citation` means, see [Citation Alignment](../concepts/citation-alignment.md). For the end-to-end pipeline, see [How It Works](../concepts/how-it-works.md). For tuning, see [Citation Config](../configuration/citation-config.md).

## The Basic Pattern

Cite-Right takes a generated answer and a collection of source documents, and returns character-accurate citations for each span of the answer. The default tokenizer is `SimpleTokenizer`, the default answer segmenter is `SimpleAnswerSegmenter`, and the default source segmenter is `SimpleSegmenter`.

```python
from cite_right import SourceDocument, align_citations

answer = "Acme Corporation reported revenue of 5.2 billion dollars in 2024."

sources = [
    SourceDocument(
        id="annual_report",
        text=(
            "Acme Corporation reported revenue of 5.2 billion dollars in 2024, "
            "representing a 12% increase over the previous year."
        ),
    )
]

results = align_citations(answer, sources)
print(results[0].status)
print(results[0].citations[0].evidence)
```

`sources` also accepts plain `str` and `SourceChunk` objects; mixing forms in one call is fine. Plain strings are auto-assigned ids like `"source_0"`. Real pipelines should pass named `SourceDocument` or `SourceChunk` so citations carry stable `source_id` values.

## Reading The Results

`align_citations` returns a `list[SpanCitations]`, one per answer segment. Each result carries the answer span, a ranked list of citations, optional `retrieval_support`, and a `status`.

```python
for result in results:
    span = result.answer_span
    print(f"Answer text: {span.text!r}")
    print(f"Position: characters {span.char_start} to {span.char_end}")
    print(f"Status: {result.status}")

    for citation in result.citations:
        print(f"  Source: {citation.source_id}")
        print(f"  Evidence: {citation.evidence!r}")
        print(f"  Score: {citation.score:.3f}")
```

`result.answer_span.char_start` and `result.answer_span.char_end` are the half-open character offsets of the segment inside the full answer string. The same half-open convention applies to `citation.char_start` and `citation.char_end` inside the source document, and the invariant `source.text[citation.char_start:citation.char_end] == citation.evidence` always holds after chunk rebasing. The same applies to `EvidenceSpan` in multi-span mode.

`citation.score` is a weighted sum of components (`alignment_score`, `answer_coverage`, `evidence_coverage`, `lexical_score`, `embedding_score`, and others). `citation.components["answer_coverage"]` is the fraction of answer tokens that the alignment matched, and it is the value that drives `status`.

## Status Values

`status` is one of three literals: `"supported"`, `"partial"`, or `"unsupported"`. The literal is exactly `"partial"`, never `"partially_supported"`. Status comes from the best exact citation's `answer_coverage` and the contradiction check, not from embedding similarity.

`"supported"` means the top-ranked citation's `answer_coverage` is at least `supported_answer_coverage` (default `0.6`) and the contradiction check did not fire. The claim is well-grounded in sources.

`"partial"` means citations exist but coverage is below `supported_answer_coverage`, **or** the contradiction check fired. The literal is `"partial"`, never `"partially_supported"`. The evidence exists; the span just does not clear the supported threshold, or the claim conflicts with what the source actually says. Source `"The vaccine is safe and effective."` paired with answer `"The vaccine is not safe."` resolves to `"partial"` with citations, not to `"unsupported"`.

`"unsupported"` means no citation survived filtering. The span may be hallucinated, paraphrased beyond recognition, or simply outside the provided sources. `"unsupported"` is "no localized citation survived," not a high-precision hallucination label. On RAGTruth test (2,675 answers), unsupported precision is about 14%, so `"unsupported"` overflags relative to gold hallucination labels.

The thresholds live on `CitationConfig`. The supported threshold is `supported_answer_coverage` (default `0.6`); the lower gate that decides whether a candidate even becomes a `Citation` is `min_answer_coverage` (default `0.2`).

```python
from cite_right import CitationConfig, align_citations

results = align_citations(
    answer,
    sources,
    config=CitationConfig(supported_answer_coverage=0.6, top_k=3),
)
```

## Working With Multiple Sources

Real applications retrieve several documents. Cite-Right scores against every source and returns the best matches across the whole list.

```python
sources = [
    SourceDocument(
        id="earnings_call",
        text=(
            "During the Q4 earnings call, CEO Jane Smith noted that revenue "
            "reached 5.2 billion dollars, exceeding analyst expectations."
        ),
    ),
    SourceDocument(
        id="press_release",
        text=(
            "Acme Corporation today announced fourth quarter revenue of "
            "5.2 billion dollars, a new company record."
        ),
    ),
    SourceDocument(
        id="market_analysis",
        text=(
            "Industry analysts had predicted Acme would report between 4.8 "
            "and 5.0 billion in revenue for the quarter."
        ),
    ),
]

answer = "Revenue reached 5.2 billion dollars, exceeding expectations."
results = align_citations(answer, sources)

for result in results:
    print(f"{result.answer_span.text!r} -> {result.status}")
    for citation in result.citations:
        print(f"  From {citation.source_id}: {citation.evidence!r}")
```

By default, up to `top_k` citations are returned per answer span (default `top_k=3`). Use `max_citations_per_source` (default `2`) to cap how many citations any one source can contribute to a single span. `retrieval_support` is the list of passages the index or embedder selected that did not localize into an exact citation; it is informational and does not change the status.

## Handling Multi-Sentence Answers

Multi-sentence answers are segmented first, and each segment is processed independently. The default `SimpleAnswerSegmenter` splits on sentence boundaries; `SpacyAnswerSegmenter` can split coordinated clauses further.

```python
answer = """Acme Corporation reported record revenue in Q4.
The company attributed growth to its new product line.
European sales exceeded expectations."""

sources = [
    SourceDocument(
        id="financial",
        text="Q4 revenue hit a record high at 5.2 billion dollars.",
    ),
    SourceDocument(
        id="products",
        text="The new product line launched in March drove significant growth.",
    ),
    SourceDocument(
        id="regional",
        text="Sales in Europe surpassed all projections by 15%.",
    ),
]

results = align_citations(answer, sources)

for result in results:
    print(f"\n{result.answer_span.text}")
    print(f"  Status: {result.status}")
    if result.citations:
        best = result.citations[0]
        print(f"  Best match from {best.source_id!r}: {best.evidence!r}")
```

Each segment carries its own `answer_span` with `char_start` and `char_end` back into the full answer, and its own status. The `kind` field on `AnswerSpan` reports whether the segmenter produced a `"sentence"`, `"clause"`, or `"paragraph"`.

## Reusing A Prepared Corpus

`align_citations` rebuilds the inverted index, IDF, and passage windows on every call. When the same source set is queried many times, build a `PreparedCitationCorpus` once and call `corpus.align(answer)` against it.

```python
from cite_right import (
    CitationConfig,
    PreparedCitationCorpus,
    SourceDocument,
)

sources = [
    SourceDocument(
        id="annual_report",
        text=(
            "Acme Corporation reported revenue of 5.2 billion dollars in 2024, "
            "representing a 12% increase over the previous year."
        ),
    ),
    SourceDocument(
        id="press_release",
        text=(
            "Acme Corporation today announced fourth quarter revenue of "
            "5.2 billion dollars, a new company record."
        ),
    ),
]

corpus = PreparedCitationCorpus.from_sources(
    sources, config=CitationConfig(top_k=3)
)

answers = [
    "Acme Corporation reported revenue of 5.2 billion dollars in 2024.",
    "Revenue increased during fiscal year 2024.",
    "The press release announced record quarterly revenue.",
]

for answer in answers:
    for result in corpus.align(answer):
        print(f"{result.answer_span.text!r} -> {result.status}")
```

`PreparedCitationCorpus.from_sources` returns the same corpus whether the optional Rust extension is present or not. On the default / Rust path (with `SimpleTokenizer` and `SimpleSegmenter`), the corpus builds an inverted index and rare-token intersect chooses which source windows are worth aligning. When the Rust extension is missing, or you supply a custom tokenizer or segmenter, `from_sources` leaves `inverted_index=None` and the candidate selector falls back to lexical prefilter. In both cases, `align` returns the same `SpanCitations` shape and the same status values.

If an embedder is set on `from_sources`, the corpus also builds an embedding index on top of the prepared candidates. `_add_embedding_candidates` can add non-index windows before alignment; those extras still go through Smith-Waterman, and any passage that does not localize into a `Citation` lands in `retrieval_support`. Embedding-only `retrieval_support` still respects `min_embedding_similarity`. The same prepared corpus can be reused for every answer in a session.

## What Status Comes From

Status is driven by the top-ranked exact `Citation` and the contradiction check, not by embedding similarity. Concretely, the per-span logic in `_span_status` looks at `citations[0].components["answer_coverage"]`. If that coverage meets `supported_answer_coverage` and `check_contradiction` is clean, the status is `"supported"`. If contradiction fires, the status is `"partial"`, never `"unsupported"` — the evidence exists, it just conflicts with the claim. If coverage is below the threshold, the status is `"partial"`. If `citations` is empty, the status is `"unsupported"`. A high embedding score that never localizes is `retrieval_support`, not a `Citation`, and it does not change the status.

## Next Steps

[Citation Alignment](../concepts/citation-alignment.md) covers the full I/O contract: `SourceDocument` and `SourceChunk` inputs, `SpanCitations` and `Citation` outputs, and the half-open offset convention. [How It Works](../concepts/how-it-works.md) walks the pipeline end to end, including the index-first candidate selector, the Smith-Waterman localizer, and the contradiction check. [Citation Config](../configuration/citation-config.md) is the reference for every knob on `CitationConfig`, including the status thresholds, candidate caps, multi-span evidence, and the `strict`, `permissive`, `fast`, and `balanced` presets. [Embedding Retrieval](../advanced/embedding-retrieval.md) and [Rust Acceleration](../advanced/rust-acceleration.md) cover the embedder path and the Rust extension that powers the default pipeline.
