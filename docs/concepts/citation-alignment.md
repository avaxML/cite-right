# Citation Alignment

Citation alignment is the core operation in Cite-Right. This page provides a detailed look at the algorithm, its configuration options, and the structure of its output.

## The align_citations Function

The primary entry point for citation extraction is the `align_citations` function defined in `src/cite_right/citations.py`. This function accepts an answer string and a collection of source documents, returning a list of `SpanCitations` objects that describe how each part of the answer relates to the sources.

```python
from cite_right import SourceDocument, align_citations, CitationConfig

answer = "The study found a 30% reduction in emissions."
sources = [
    SourceDocument(id="paper", text="Research indicates a 30% reduction in emissions over the study period.")
]

config = CitationConfig(top_k=3)
results = align_citations(answer, sources, config=config)
```

The function signature provides considerable flexibility through optional parameters.

## Input Types

The `sources` parameter accepts two types of input, reflecting different retrieval patterns.

`SourceDocument` represents a complete document with an identifier and full text. Use this type when your retrieval system returns whole documents or when you want Cite-Right to handle passage creation internally.

```python
from cite_right import SourceDocument

doc = SourceDocument(
    id="annual_report_2024",
    text="The full text of the annual report goes here...",
    metadata={"year": 2024, "type": "financial"}
)
```

`SourceChunk` represents a pre-chunked excerpt with offsets indicating its position within a parent document. Use this type when your retrieval system already performs chunking and you want citations to reference the original document positions.

```python
from cite_right import SourceChunk

chunk = SourceChunk(
    source_id="annual_report_2024",
    text="This is a specific passage from the document.",
    doc_char_start=1500,
    doc_char_end=1548
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

The `citations` field is a list of `Citation` objects, ranked from best to worst match. The number of citations depends on the `top_k` configuration parameter.

The `status` field is a string indicating overall support level. It takes one of three values based on the quality of the best citation.

A status of "supported" indicates the answer span has a high-quality match in the sources. The threshold for this status is configurable.

A status of "partial" indicates some support was found but the match quality is below the full support threshold. This often occurs with paraphrased content.

A status of "unsupported" indicates no adequate match was found. This may indicate hallucination or content derived from knowledge outside the provided sources.

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

The `evidence` field contains the matched text extracted from the source document. The `char_start` and `char_end` fields specify the exact byte positions in the source.

The `components` dictionary breaks down the score into its constituent parts.

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

Both implementations produce identical results. The Rust extension is significantly faster for large workloads due to parallel processing.

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
    source_segmenter=SpacySegmenter()
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

When an embedder is provided, candidate selection considers both lexical overlap and semantic similarity. This improves recall for paraphrased content where the answer uses different words than the source but conveys the same meaning.

## Score Interpretation

Understanding what scores mean helps with threshold tuning and quality assessment.

Scores above 0.7 typically indicate strong verbatim or near-verbatim matches. The answer closely mirrors the source text.

Scores between 0.4 and 0.7 often indicate paraphrased content or partial matches. The meaning is similar but the wording differs.

Scores below 0.4 suggest weak support. The match may be coincidental overlap rather than genuine citation.

These ranges are approximate and depend on configuration settings. Applications should tune thresholds based on empirical testing with representative data.

## Deterministic Ordering

Citations are ordered deterministically to ensure reproducible results. When scores are equal, the following tie-breakers apply in order.

Lower source index takes precedence, preserving the order in which sources were provided. This is useful when retrieval systems return sources in relevance order.

Earlier character positions take precedence within the same source. This favors evidence appearing earlier in the document.

Longer evidence spans take precedence when positions are equal. This provides more context for the user.
