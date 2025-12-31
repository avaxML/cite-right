# Citation Configuration

The `CitationConfig` class controls the citation alignment pipeline end to end. It determines how passages are windowed, how candidates are selected, how alignment is scored, and how citations are filtered and ranked. The configuration is defined in `src/cite_right/core/citation_config.py`.

## Creating a Configuration

The simplest approach uses the default configuration, which provides balanced settings suitable for most applications.

```python
from cite_right import align_citations, CitationConfig

config = CitationConfig()
results = align_citations(answer, sources, config=config)
```

Each parameter can be customized by passing keyword arguments to the constructor.

```python
config = CitationConfig(
    top_k=3,
    min_final_score=0.25,
    window_size_sentences=2
)
```

## Output Filtering and Status

### top_k

Maximum citations to return per answer span. Default is 3.

```python
config = CitationConfig(top_k=5)
```

### min_final_score

Minimum final citation score required for inclusion. This score is a weighted sum of alignment, coverage, lexical, and embedding components (see Citation Weights). Default is 0.0.

```python
config = CitationConfig(min_final_score=0.3)
```

### min_alignment_score

Minimum raw Smith-Waterman alignment score required to use alignment evidence. Default is 0.

```python
config = CitationConfig(min_alignment_score=10)
```

### min_answer_coverage

Minimum fraction of answer tokens that must match for alignment evidence to be used. Default is 0.2.

```python
config = CitationConfig(min_answer_coverage=0.3)
```

### supported_answer_coverage

Answer coverage threshold for a span to be marked `supported`. Default is 0.6.

```python
config = CitationConfig(supported_answer_coverage=0.7)
```

### allow_embedding_only

Allow citations based solely on embedding similarity when alignment evidence is insufficient. Default is False.

```python
config = CitationConfig(allow_embedding_only=True)
```

When this is enabled, the evidence span is the entire passage window rather than a token-level alignment span.

### min_embedding_similarity

Minimum embedding similarity required for embedding-only citations. Default is 0.3.

```python
config = CitationConfig(min_embedding_similarity=0.4)
```

### supported_embedding_similarity

Embedding similarity threshold for a span to be marked `supported` when `allow_embedding_only=True`. Default is 0.6.

```python
config = CitationConfig(supported_embedding_similarity=0.7)
```

### max_citations_per_source

Maximum citations returned from a single source document for each answer span. Default is 2.

```python
config = CitationConfig(max_citations_per_source=1)
```

### prefer_source_order

Tie-breaker preference when citation scores are equal. If True (default), ties prefer earlier sources, then earlier character positions, then longer evidence. If False, ties prefer earlier character positions first.

```python
config = CitationConfig(prefer_source_order=False)
```

## Passage Windowing

### window_size_sentences

Number of sentences in each source passage window. Default is 1.

```python
config = CitationConfig(window_size_sentences=3)
```

Larger windows provide more context during alignment, which can help match content that spans multiple sentences. Smaller windows produce more precise evidence spans but may miss cross-sentence patterns.

### window_stride_sentences

Step size between consecutive passage windows. Default is 1.

```python
config = CitationConfig(window_stride_sentences=2)
```

A stride of 1 creates overlapping windows and maximizes recall. Larger strides reduce the number of passages and improve performance at the cost of potentially missing matches.

## Candidate Selection

Candidate selection combines lexical overlap and (optionally) embedding similarity.

### max_candidates_lexical

Maximum number of lexical candidates to consider per answer span. Default is 200.

```python
config = CitationConfig(max_candidates_lexical=100)
```

### max_candidates_embedding

Maximum number of embedding candidates to consider per answer span when an embedder is provided. Default is 200.

```python
config = CitationConfig(max_candidates_embedding=100)
```

### max_candidates_total

Maximum total candidates after combining lexical and embedding candidates. Default is 400.

```python
config = CitationConfig(max_candidates_total=200)
```

Candidates are ranked by the stronger of their lexical or embedding score, then capped by `max_candidates_total` before full alignment.

## Alignment Scoring

### match_score

Score awarded for matching tokens during Smith-Waterman alignment. Default is 2.

### mismatch_score

Penalty applied when tokens do not match. Default is -1.

### gap_score

Penalty for gaps (insertions/deletions) in alignment. Default is -1.

```python
config = CitationConfig(
    match_score=2,
    mismatch_score=-1,
    gap_score=-2
)
```

Higher gap penalties produce more compact evidence spans with fewer skipped tokens. Lower penalties allow bridging gaps between matching regions.

## Citation Weights

The `CitationWeights` class controls how score components combine into the final citation score. These weights are summed directly (they are not normalized), so their absolute values matter.

```python
from cite_right.core.citation_config import CitationConfig, CitationWeights

weights = CitationWeights(
    alignment=1.0,
    answer_coverage=1.0,
    evidence_coverage=0.0,
    lexical=0.5,
    embedding=0.0
)

config = CitationConfig(weights=weights)
```

- `alignment`: Influence of normalized Smith-Waterman alignment score.
- `answer_coverage`: Influence of matched answer token fraction.
- `evidence_coverage`: Influence of matched evidence token fraction.
- `lexical`: Influence of IDF-weighted lexical overlap.
- `embedding`: Influence of embedding similarity (when enabled).

## Multi-Span Evidence

### multi_span_evidence

Enable extraction of non-contiguous evidence spans. Default is False.

```python
config = CitationConfig(multi_span_evidence=True)
```

### multi_span_merge_gap_chars

Maximum character gap between evidence spans before they are merged. Default is 16.

```python
config = CitationConfig(
    multi_span_evidence=True,
    multi_span_merge_gap_chars=30
)
```

### multi_span_max_spans

Maximum number of evidence spans to return per citation after merging. Default is 5. If exceeded, the citation falls back to a single contiguous span.

```python
config = CitationConfig(multi_span_evidence=True, multi_span_max_spans=3)
```

## Complete Example

Here is a configuration tuned for high-precision fact-checking where only strong matches should be considered valid.

```python
from cite_right import CitationConfig, align_citations
from cite_right.core.citation_config import CitationWeights

config = CitationConfig(
    top_k=1,
    min_final_score=0.3,
    min_answer_coverage=0.4,
    supported_answer_coverage=0.7,
    window_size_sentences=1,
    max_candidates_lexical=150,
    max_candidates_total=200,
    weights=CitationWeights(
        alignment=1.0,
        answer_coverage=1.0,
        evidence_coverage=0.0,
        lexical=0.3,
        embedding=0.0
    )
)

results = align_citations(answer, sources, config=config)
```

This configuration returns only the single best citation per span, requires high alignment quality for inclusion, and uses a high answer-coverage threshold for `supported` status.
