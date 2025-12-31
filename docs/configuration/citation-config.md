# Citation Configuration

The `CitationConfig` class provides fine-grained control over the citation alignment process. Understanding these options helps you tune the library for your specific requirements. The configuration is defined in `src/cite_right/core/citation_config.py`.

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
    min_score_threshold=0.25,
    window_size_sentences=5
)
```

## Core Parameters

### top_k

This parameter controls how many citations are returned for each answer span. The default value is 1, returning only the best match. Increasing this value provides alternative citations that users can explore.

```python
config = CitationConfig(top_k=5)
```

Setting a higher value does not significantly impact performance since the alignment work has already been done. The additional citations are simply included in the ranking.

### min_score_threshold

This parameter sets the minimum alignment score required for a citation to be included in results. Citations below this threshold are discarded even if they are the best match available. The default is 0.2.

```python
config = CitationConfig(min_score_threshold=0.3)
```

Increasing this value produces stricter filtering, showing citations only when alignment quality is high. Decreasing it allows weaker matches to appear, which may be useful when source overlap is expected to be low.

### supported_threshold

This parameter determines when an answer span receives "supported" status. If the best citation score meets or exceeds this threshold, the span is marked as fully supported. The default is 0.5.

```python
config = CitationConfig(supported_threshold=0.6)
```

Raising this threshold makes the "supported" designation more exclusive, reserving it for high-quality matches.

### partial_threshold

This parameter sets the boundary between "partial" and "unsupported" status. Spans with best citation scores above the minimum threshold but below the supported threshold are marked as partial. The default aligns with the minimum score threshold.

Adjusting this value changes how middle-quality matches are categorized.

## Passage Windowing

### window_size_sentences

This parameter controls how many sentences are grouped into each passage window when processing source documents. The default is 3.

```python
config = CitationConfig(window_size_sentences=5)
```

Larger windows provide more context during alignment, which can help match content that spans multiple sentences. Smaller windows produce more precise evidence spans but may miss cross-sentence patterns.

### window_stride_sentences

This parameter controls the step size between consecutive passage windows. The default is 1, meaning windows overlap substantially.

```python
config = CitationConfig(window_stride_sentences=2)
```

A stride of 1 ensures every sentence appears in multiple windows, maximizing the chance of finding good alignments. Larger strides reduce the number of passages and improve performance at the cost of potentially missing some matches.

## Candidate Selection

### max_candidates

This parameter limits how many passage candidates undergo full Smith-Waterman alignment for each answer span. The default is 50.

```python
config = CitationConfig(max_candidates=100)
```

The candidate selection phase identifies promising passages using efficient lexical and embedding-based scoring. Only the top candidates proceed to expensive alignment. Increasing this value improves recall at the cost of performance. Decreasing it speeds up processing but may miss some matches.

### lexical_weight

This parameter controls the influence of lexical overlap in candidate scoring. Higher values prioritize passages with more word overlap. The default provides balanced weighting.

```python
config = CitationConfig(lexical_weight=0.7)
```

For content with high verbatim overlap, increasing lexical weight improves candidate selection. For heavily paraphrased content, reducing it allows embedding similarity to have more influence.

### embedding_weight

This parameter controls the influence of semantic similarity in candidate scoring when an embedder is provided. The default balances lexical and semantic signals.

```python
config = CitationConfig(embedding_weight=0.5)
```

When using embeddings for paraphrase detection, increasing this weight prioritizes semantically similar passages over lexically similar ones.

## Alignment Scoring

### match_score

This parameter sets the score awarded when two tokens match during Smith-Waterman alignment. The default is 2.

### mismatch_penalty

This parameter sets the penalty applied when two tokens do not match. The default is -1.

### gap_penalty

This parameter sets the penalty for introducing gaps (insertions or deletions) in the alignment. The default is -1.

```python
config = CitationConfig(
    match_score=2,
    mismatch_penalty=-1,
    gap_penalty=-2
)
```

Adjusting these values changes how the algorithm trades off between matched regions and gaps. A higher gap penalty produces more compact evidence spans with fewer skipped tokens. A lower penalty allows the algorithm to bridge gaps between matching regions.

## Multi-Span Evidence

### multi_span_evidence

This boolean parameter enables extraction of non-contiguous evidence spans. When enabled, a single citation may point to multiple separate regions in the source that together support the answer. The default is False.

```python
config = CitationConfig(multi_span_evidence=True)
```

With multi-span evidence enabled, each citation includes an `evidence_spans` list containing individual span objects. The `evidence` field continues to provide a single contiguous span for backward compatibility.

### multi_span_merge_gap_chars

This parameter specifies the maximum gap in characters between evidence regions before they are kept separate. Regions closer than this threshold are merged into a single span. The default is 50 characters.

```python
config = CitationConfig(
    multi_span_evidence=True,
    multi_span_merge_gap_chars=30
)
```

Smaller values produce more separate spans. Larger values merge nearby regions for cleaner presentation.

## Embedding Behavior

### allow_embedding_only

This boolean parameter controls whether citations can be returned based solely on embedding similarity when alignment scores are low. The default is False.

```python
config = CitationConfig(allow_embedding_only=True)
```

Enabling this option helps with heavily paraphrased content where alignment fails but semantic similarity indicates a genuine match. The returned evidence will be the entire passage window rather than a precise span.

## Citation Weights

The `CitationWeights` class provides additional control over how score components combine.

```python
from cite_right.core.citation_config import CitationConfig, CitationWeights

weights = CitationWeights(
    alignment_score=0.4,
    answer_coverage=0.3,
    evidence_coverage=0.2,
    embedding_similarity=0.1
)

config = CitationConfig(weights=weights)
```

The `alignment_score` weight controls the influence of the raw Smith-Waterman score.

The `answer_coverage` weight controls the influence of what fraction of the answer span was matched.

The `evidence_coverage` weight controls the influence of what fraction of the evidence span was matched, penalizing overly long evidence.

The `embedding_similarity` weight controls the influence of semantic similarity when embeddings are enabled.

These weights are normalized during scoring, so their relative values matter more than absolute values.

## Complete Example

Here is a configuration tuned for high-precision fact-checking where only strong matches should be considered valid.

```python
from cite_right import CitationConfig, align_citations
from cite_right.core.citation_config import CitationWeights

config = CitationConfig(
    top_k=1,
    min_score_threshold=0.4,
    supported_threshold=0.7,
    window_size_sentences=3,
    max_candidates=100,
    weights=CitationWeights(
        alignment_score=0.5,
        answer_coverage=0.3,
        evidence_coverage=0.2
    )
)

results = align_citations(answer, sources, config=config)
```

This configuration returns only the single best citation per span, requires high alignment quality for inclusion, and uses a high threshold for supported status.
