# Citation Alignment

Citation alignment is the core operation in Cite-Right. This page provides a detailed look at the algorithm, its configuration options, and the structure of its output.

## The align_citations Function

The primary entry point for citation extraction is the `align_citations` function defined in `src/cite_right/citations.py`. This function accepts an answer string and a collection of source documents, returning a list of `SpanCitations` objects that describe how each part of the answer relates to the sources.

```python
from cite_right import SourceDocument, align_citations, CitationConfig

answer = "The study found a 30% reduction in emissions."
sources = [
    SourceDocument(
        id="paper",
        text="Research indicates a 30% reduction in emissions over the study period.",
    )
]

config = CitationConfig(top_k=3)
results = align_citations(answer, sources, config=config)
```

The function signature provides considerable flexibility through optional parameters. The public API is the same as before 0.4.0: `align_citations` and `PreparedCitationCorpus`.

## Input Types

The `sources` parameter accepts two types of input, reflecting different retrieval patterns.

`SourceDocument` represents a complete document with an identifier and full text. Use this type when your retrieval system returns whole documents or when you want Cite-Right to handle passage creation internally.

```python
from cite_right import SourceDocument

doc = SourceDocument(
    id="annual_report_2024",
    text="The full text of the annual report goes here...",
    metadata={"year": 2024, "type": "financial"},
)
```

`SourceChunk` represents a pre-chunked excerpt with offsets indicating its position within a parent document. Use this type when your retrieval system already performs chunking and you want citations to reference the original document positions.

```python
from cite_right import SourceChunk

chunk = SourceChunk(
    source_id="annual_report_2024",
    text="This is a specific passage from the document.",
    doc_char_start=1500,
    doc_char_end=1548,
)
```

When using `SourceChunk`, the `doc_char_start` and `doc_char_end` values are added to the alignment offsets, producing character positions in the original document rather than the chunk.

## Output Structure

The function returns a list of `SpanCitations` objects, one for each segment of the answer. The structure is defined in `src/cite_right/core/results.py`.

### SpanCitations

Each `SpanCitations` object contains three fields.

The `answer_span` field is an `AnswerSpan` object describing the text segment being cited. It includes the text itself along with `char_start` and `char_end` offsets within the original answer string.

```python
for result in results:
    span = result.answer_span
    print(f"Text: {span.text}")
    print(f"Position: {span.char_start} to {span.char_end}")
    print(f"Kind: {span.kind}")  # "sentence", "clause", or "paragraph"
```

The `citations` field is a list of `Citation` objects, ranked from best to worst match. Each citation represents exact localized evidence. The number of citations depends on the `top_k` configuration parameter.

The `retrieval_support` field is a list of retrieval-only support candidates. These are passages selected during index, lexical, and/or embedding candidate search that did not produce an exact localized citation.

The `status` field is a string indicating overall support level. It takes one of three values based on the best citation's answer coverage after contradiction checking.

A status of `"supported"` means the best citation has `answer_coverage >= supported_answer_coverage` and the cheap contradiction check did not fire.

A status of `"partial"` means at least one citation was produced but the supported thresholds were not met, or a contradiction (negation, number, leftover n-gram slot, or entity swap) downgraded the span. This often occurs with paraphrased or only partly covered content. The literal is `"partial"`, never `"partially_supported"`.

A status of `"unsupported"` indicates no citations met the minimum thresholds. This may indicate hallucination or content derived from knowledge outside the provided sources. Contradiction does not produce this status when evidence exists.

### Citation

Each `Citation` object contains detailed information about the match.

```python
for result in results:
    for citation in result.citations:
        print(f"Source: {citation.source_id}")
        print(f"Index: {citation.source_index}")
        print(f"Score: {citation.score}")
        print(f"Evidence: {citation.evidence}")
        print(f"Char range: {citation.char_start} to {citation.char_end}")
        print(f"Components: {citation.components}")
```

The `source_id` field identifies the source document by its ID string. The `source_index` field provides the integer index in the original sources list, useful for array access.

The `score` field is a floating-point value indicating match quality. Higher values indicate better matches. The score combines multiple components according to the configured weights.

The `evidence` field contains the matched text extracted from the source document. The `char_start` and `char_end` fields specify the exact character offsets in the source.

The `components` dictionary breaks down the score into its constituent parts.

## How Candidates Are Chosen

0.4.0 does not run Smith-Waterman over every passage window.

Prepare builds an inverted index over source windows. For each answer span, rare-token intersect selects which windows are worth aligning. Smith-Waterman still localizes; the index only chooses the windows.

When an embedder is provided, high-similarity passages can join that set. Rust prepare still runs. The embedding index is built on the prepared candidates. Embedding-only `retrieval_support` still respects `min_embedding_similarity`.

See [How It Works](how-it-works.md) for the full pipeline and [Embedding Retrieval](../advanced/embedding-retrieval.md) for the embedder path.

## Paraphrase, Fields, and Contradiction

Grounded how-to and news paraphrases can emit a citation from content-word overlap on the candidate passage when Smith-Waterman sequential coverage is low. They are no longer overflagged as `unsupported` for reordered content words.

Data2txt field:value paraphrases get a second Smith-Waterman pass per structured-field candidate with `gap_score=0`. Faithful rewrites of hours, amenities, and similar fields can be `"supported"` or `"partial"`. Invented fields stay `"unsupported"`.

Leftover n-gram conflicts check contradiction against the full candidate passage, not only the truncated Smith-Waterman evidence span. Cheap contradiction (negation, number, leftover n-gram slot, entity swap) downgrades to `"partial"`, not `"unsupported"`.

## Character Offset Convention

All character offsets in Cite-Right follow Python's standard half-open interval convention. The start position is inclusive and the end position is exclusive.

For a source document containing the text "Hello world", the word "world" would have `char_start=6` and `char_end=11`. Slicing with these offsets as `text[6:11]` produces "world".

This convention ensures that offsets can be used directly with Python string slicing and that adjacent spans can be identified by comparing the end of one with the start of the next.

## Backend Selection

The `backend` parameter controls which alignment implementation is used.

```python
results = align_citations(answer, sources, backend="auto")  # Default
results = align_citations(answer, sources, backend="python")
results = align_citations(answer, sources, backend="rust")
```

The "auto" setting uses the Rust extension if available, falling back to pure Python otherwise. The "python" setting forces the pure Python implementation even when Rust is available. The "rust" setting requires the Rust extension and raises an error if it is not installed.

Both implementations produce identical results. The Rust extension is significantly faster for large workloads due to parallel processing. Released 0.4.0 wheels ship the extension as abi3 (`abi3-py311`), including linux/aarch64, plus an sdist.

## Tokenizer Selection

The `tokenizer` parameter specifies how text is converted to token sequences for alignment.

```python
from cite_right import SimpleTokenizer, HuggingFaceTokenizer, align_citations

# Default tokenizer
results = align_citations(answer, sources)

# Custom tokenizer
tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
results = align_citations(answer, sources, tokenizer=tokenizer)
```

The choice of tokenizer affects how text is compared. Using a tokenizer that matches your language model's tokenization scheme may improve alignment quality for content generated by that model.

## Segmenter Selection

The `answer_segmenter` and `source_segmenter` parameters control how text is divided into segments.

```python
from cite_right import SpacyAnswerSegmenter, SpacySegmenter, align_citations

results = align_citations(
    answer,
    sources,
    answer_segmenter=SpacyAnswerSegmenter(split_clauses=True),
    source_segmenter=SpacySegmenter(),
)
```

Finer segmentation produces more spans with more specific citations but may split logically connected content. Coarser segmentation groups related sentences but may miss cases where only part of a segment is supported.

## Embedding Retrieval

The `embedder` parameter enables semantic retrieval of candidate passages.

```python
from cite_right import SentenceTransformerEmbedder, align_citations

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
results = align_citations(answer, sources, embedder=embedder)
```

When an embedder is provided, candidate selection still starts from the inverted index. Semantic similarity can add extra windows. This improves recall for paraphrased content where the answer uses different words than the source but conveys the same meaning.

Rust prepare is not skipped when an embedder is set. Embedding-only `retrieval_support` still respects `min_embedding_similarity`.

## Score Interpretation

The `score` field is a weighted sum of several components (normalized alignment, answer coverage, evidence coverage, lexical overlap, and optional embedding similarity). Because the weights are not normalized, the absolute scale of `score` depends on your configuration. Changing `CitationWeights` or alignment scoring parameters will shift the range of scores you observe.

For tuning, rely on the component values in `citation.components` (for example `answer_coverage` and `normalized_alignment`) and calibrate `min_final_score` against your own data. Comparing scores across runs is only meaningful if the configuration is unchanged.

## Deterministic Ordering

Citations are ordered deterministically to ensure reproducible results. When scores are equal, the following tie-breakers apply in order (with `prefer_source_order=True`, the default).

Lower source index takes precedence, preserving the order in which sources were provided. This is useful when retrieval systems return sources in relevance order.

Earlier character positions take precedence within the same source. This favors evidence appearing earlier in the document.

Longer evidence spans take precedence when positions are equal. This provides more context for the user.

If `prefer_source_order=False`, the ordering favors earlier character positions first and uses source order as a later tie-breaker.
