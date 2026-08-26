---
type: configuration
title: Citation Config
description: "CitationConfig and CitationWeights knobs for the citation alignment pipeline: status thresholds, candidate selection caps, multi-span evidence, embedder interaction, contradiction behavior, and presets."
tags: [configuration, citation-config, citation-weights, status, threshold, candidate-selection, multi-span, embedder, contradiction, presets, smith-waterman, scoring]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Citation Config

`CitationConfig` is the single object that controls how the citation alignment pipeline behaves end to end. It lives in `src/cite_right/core/citation_config.py` and is the third positional-style argument accepted by `align_citations(answer, sources, *, config=...)` and `PreparedCitationCorpus.from_sources(sources, config=...)`. Both classes are public, immutable Pydantic models, so you can subclass or rebuild a config from a base preset without mutating it.

This page is a reference for every knob and how it interacts with the pipeline. For the high-level orientation, see [How It Works](../concepts/how-it-works.md). For ready-made starting points, see [Presets](./presets.md). For multi-span evidence specifics, see [Multi-Span Evidence](../advanced/multi-span-evidence.md). For the embedder path, see [Embedding Retrieval](../advanced/embedding-retrieval.md).

## A Small Run

The default config is balanced. Pass an explicit one to tune.

```python
from cite_right import CitationConfig, align_citations

config = CitationConfig(top_k=3, min_final_score=0.0)
results = align_citations(answer, sources, config=config)
```

Every field can be overridden by keyword argument. Presets are just class methods that return a pre-tuned `CitationConfig`.

```python
config = CitationConfig.strict()           # or .permissive() / .fast() / .balanced()
config = CitationConfig.balanced().model_copy(update={"top_k": 5})
```

## Status And Thresholds

Status is one of `"supported"`, `"partial"`, or `"unsupported"` and is computed from the best exact citation's `answer_coverage`, not from the overall `score`. The contradiction check runs against the **full candidate passage**, not the truncated Smith-Waterman evidence, and downgrades the span to `"partial"` (never `"unsupported"`) when it fires. So source `"The vaccine is safe and effective."` paired with answer `"The vaccine is not safe."` resolves to `"partial"` with citations.

The thresholds below decide where the line falls.

| Field | Default | Effect |
|-------|---------|--------|
| `supported_answer_coverage` | `0.6` | A span is `"supported"` only when its best citation's `answer_coverage` clears this. Below it (or any contradiction) is `"partial"`. No surviving citation is `"unsupported"`. |
| `min_answer_coverage` | `0.2` | Lower gate on a candidate before it becomes a `Citation`. A candidate must clear it on sequential Smith-Waterman coverage **or** on content-word coverage over the same passage. Stopword-only SW hits never clear it. |
| `min_alignment_score` | `0` | Minimum raw Smith-Waterman score required before alignment evidence is used. |
| `min_final_score` | `0.0` | Final weighted score must clear this for a `Citation` to be emitted at all. |
| `min_embedding_similarity` | `0.3` | Cosine-similarity threshold for an embedding-only entry to surface as `retrieval_support`. Anything below is dropped. |
| `top_k` | `3` | Maximum exact citations returned per answer span. |
| `max_citations_per_source` | `2` | Maximum citations from a single source per span. |
| `max_retrieval_support` | `3` | Maximum `retrieval_support` entries per span. Independent of `top_k`. |

Two removed options, `allow_embedding_only` and `supported_embedding_similarity`, raise on construction; the supported path is `SpanCitations.retrieval_support`, not an embedding-only `Citation`.

```python
from cite_right import CitationConfig

config = CitationConfig(
    supported_answer_coverage=0.6,
    min_answer_coverage=0.2,
    min_final_score=0.0,
    min_embedding_similarity=0.3,
    top_k=3,
    max_citations_per_source=2,
    max_retrieval_support=3,
)
```

## Passage Windowing

Sources are split into overlapping sentence windows before indexing. Both knobs default to `1`, so each sentence is its own window.

| Field | Default | Effect |
|-------|---------|--------|
| `window_size_sentences` | `1` | Sentences per window. Larger windows add context for cross-sentence alignments. |
| `window_stride_sentences` | `1` | Stride between consecutive windows. Stride `1` produces overlapping windows. |

```python
config = CitationConfig(window_size_sentences=3, window_stride_sentences=1)
```

Larger windows improve recall on multi-sentence claims at the cost of more candidates. Stride larger than `1` skips windows and trades recall for speed.

## Candidate Selection

Candidate selection is index-first on the default / Rust path: an inverted index over source windows plus a rare-token intersect chooses which windows move on to Smith-Waterman. Smith-Waterman still localizes `char_start` / `char_end`; the index only chooses the windows.

A custom tokenizer or segmenter takes the lexical fallback path. `PreparedCitationCorpus.from_sources` then leaves `inverted_index=None` and `_select_candidates` uses the older lexical prefilter. With an embedder set, `_add_embedding_candidates` may add non-index windows to the candidate set on either path. Those extras still need Smith-Waterman.

Lexical scores are filled only for inverted-index seeds, so embedding-only extras keep `lexical_score == 0.0` and rely on `min_embedding_similarity` to surface as `retrieval_support`.

| Field | Default | Effect |
|-------|---------|--------|
| `max_candidates_lexical` | `200` | Maximum inverted-index seeds (or lexical prefilter hits) per answer span. If the index returns nothing, the lexical prefilter is the fallback and this cap still applies. |
| `max_candidates_embedding` | `200` | Maximum embedding candidates added on top of the index seeds. |
| `max_candidates_total` | `400` | Maximum total candidates after combining index seeds and embedding extras. Ranking uses `max(embedding_score, lexical_score)`, then this cap applies before Smith-Waterman. |
| `min_embedding_similarity` | `0.3` | Embedding-only `retrieval_support` threshold. Preset `permissive()` lowers this to `0.25`. |

```python
config = CitationConfig(
    max_candidates_lexical=200,
    max_candidates_embedding=100,
    max_candidates_total=300,
    min_embedding_similarity=0.4,
)
```

`CitationConfig.fast()` reduces all three caps to `50` / `50` / `100` for latency-bound workloads.

## Ranking And Tie-Breaking

After Smith-Waterman, surviving citations are sorted by `_citation_sort_key`. The primary key is `-citation.score`. Ties break on `prefer_source_order`:

- `prefer_source_order=True` (default): earlier `source_index`, then earlier `char_start`, then longer evidence span, then `candidate_index`.
- `prefer_source_order=False`: earlier `char_start`, then longer evidence span, then `source_index`, then `candidate_index`.

After sorting, citations are deduplicated by `(source_id, evidence_spans)` tuple, capped by `max_citations_per_source`, and trimmed to `top_k`. `retrieval_support` is ranked independently by `retrieval_score` and trimmed to `max_retrieval_support`.

```python
config = CitationConfig(prefer_source_order=True, top_k=3, max_citations_per_source=2)
```

## Alignment Scoring

The Smith-Waterman step scores local alignments between an answer span and a candidate window. Three integer knobs control that score; they are direct Smith-Waterman parameters, not weighted.

| Field | Default | Effect |
|-------|---------|--------|
| `match_score` | `2` | Reward for matching tokens. |
| `mismatch_score` | `-1` | Penalty when tokens differ. |
| `gap_score` | `-1` | Penalty for insertions or deletions. |

```python
config = CitationConfig(match_score=2, mismatch_score=-1, gap_score=-1)
```

Higher (less negative) gap penalties produce more compact evidence with fewer skipped tokens. Lower gap penalties allow bridging across short gaps between matching regions. The structured-field pass over Data2txt hours, amenities, and similar field:value sources runs a second Smith-Waterman pass with `gap_score=0` only on the answer text; prose candidates keep the configured gap penalty.

## Citation Weights

`CitationWeights` is a frozen Pydantic model that holds the per-component weights combined into the final citation `score`. Weights are summed directly and are not normalized, so their absolute values matter.

| Field | Default | Effect |
|-------|---------|--------|
| `alignment` | `1.0` | Influence of normalized Smith-Waterman alignment score. |
| `answer_coverage` | `1.0` | Influence of matched answer token fraction. |
| `evidence_coverage` | `0.0` | Influence of matched evidence token fraction. Penalizes over-long evidence that merely contains the answer. |
| `lexical` | `0.5` | Influence of IDF-weighted lexical overlap. |
| `embedding` | `0.5` | Influence of embedding similarity (when enabled). Negative embeddings are clipped to `0.0` before the weight is applied. |

```python
from cite_right import CitationConfig
from cite_right.core.citation_config import CitationWeights

config = CitationConfig(
    weights=CitationWeights(
        alignment=1.0,
        answer_coverage=1.0,
        evidence_coverage=0.0,
        lexical=0.5,
        embedding=0.5,
    )
)
```

High-precision tuning typically raises `alignment` and `answer_coverage`; paraphrase-heavy workloads lean on `embedding` and a higher `top_k`.

## Multi-Span Evidence

Three fields control the multi-span evidence feature. The feature is off by default; the numeric knobs only matter when it is on.

| Field | Default | Effect |
|-------|---------|--------|
| `multi_span_evidence` | `False` | Master switch. When off, `Citation.evidence_spans` is a one-element list equal to the enclosing slice. |
| `multi_span_merge_gap_chars` | `16` | Maximum source-character gap between two match regions before they are merged into one span. `<= 0` disables merging. |
| `multi_span_max_spans` | `5` | Maximum number of spans per citation after merging. If exceeded, the citation falls back to a single contiguous span. |

```python
config = CitationConfig(
    multi_span_evidence=True,
    multi_span_merge_gap_chars=16,
    multi_span_max_spans=5,
)
```

The Rust and Python backends produce identical `evidence_spans` and `exact_evidence` when the flag is on. The aligner runs `return_match_blocks=True` only when this option is set. See [Multi-Span Evidence](../advanced/multi-span-evidence.md) for the full flow.

## Contradiction Interaction

Status is decided on the best-ranked citation. The cheap contradiction check in `check_contradiction` runs over the full candidate passage (not the truncated Smith-Waterman evidence) and fires on:

- Negation mismatch: one side has a negation marker, the other does not.
- Number mismatch: numbers differ between the answer and the candidate.
- Leftover n-gram slot: a shared number attaches to different content words in each side.
- Entity swap: capitalized entities differ between the two sides.
- Temporal or polarity mismatch: BC vs ago, oppose vs support, and similar paired markers.

When any of those fire, the span is forced to `"partial"` even if `answer_coverage` clears `supported_answer_coverage`. Contradiction never produces `"unsupported"`; the citation survived, it just conflicts with the claim.

## Embedder-Aware Behavior

When an embedder is passed, the candidate selector queries the index first, then `_add_embedding_candidates` adds high-cosine-similarity passages the index may have missed. Those extras still go through Smith-Waterman. Rust prepare still runs with the embedder on the default `SimpleTokenizer` + `SimpleSegmenter` path. The embedding index is built on the prepared candidates; lexical scores are filled only for inverted-index seeds.

A passage the index or embedder selected but Smith-Waterman could not localize is exposed on `SpanCitations.retrieval_support`, not as a `Citation`, and never flips `status`. The relevant configuration levers are `min_embedding_similarity`, `max_candidates_embedding`, `max_candidates_total`, and `weights.embedding`.

```python
from cite_right import CitationConfig, SentenceTransformerEmbedder, align_citations
from cite_right.core.citation_config import CitationWeights

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
config = CitationConfig(
    min_embedding_similarity=0.4,
    max_candidates_embedding=100,
    max_candidates_total=250,
    weights=CitationWeights(lexical=0.3, embedding=0.7),
)
results = align_citations(answer, sources, embedder=embedder, config=config)
```

See [Embedding Retrieval](../advanced/embedding-retrieval.md) for the full embedder pipeline and `CitationConfig.permissive()` for the lower-similarity variant.

## Custom Tokenizer Or Segmenter

A custom tokenizer or segmenter takes the lexical fallback path. `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and `rust_corpus=None`, and `_select_candidates` uses the lexical prefilter plus optional embedding extras. Smith-Waterman still localizes. The public API is unchanged: `align_citations(answer, sources, config=...)` and `PreparedCitationCorpus.from_sources(sources, config=...)` work either way.

## Complete Defaults

For quick reference, the default `CitationConfig()` is equivalent to:

```python
from cite_right import CitationConfig
from cite_right.core.citation_config import CitationWeights

CitationConfig(
    top_k=3,
    min_final_score=0.0,
    min_alignment_score=0,
    min_answer_coverage=0.2,
    supported_answer_coverage=0.6,
    min_embedding_similarity=0.3,
    window_size_sentences=1,
    window_stride_sentences=1,
    max_candidates_lexical=200,
    max_candidates_embedding=200,
    max_candidates_total=400,
    max_citations_per_source=2,
    max_retrieval_support=3,
    require_all_answer_tokens_in_evidence=False,
    weights=CitationWeights(
        alignment=1.0,
        answer_coverage=1.0,
        evidence_coverage=0.0,
        lexical=0.5,
        embedding=0.5,
    ),
    match_score=2,
    mismatch_score=-1,
    gap_score=-1,
    prefer_source_order=True,
    multi_span_evidence=False,
    multi_span_merge_gap_chars=16,
    multi_span_max_spans=5,
)
```

## Presets

Four class methods return pre-tuned configs. `balanced()` is identical to the default constructor; the other three are described in detail on [Presets](./presets.md).

| Preset | `top_k` | `min_answer_coverage` | `supported_answer_coverage` | `min_final_score` | `min_embedding_similarity` | `max_candidates_lexical` | `max_citations_per_source` | `max_retrieval_support` | `require_all_answer_tokens_in_evidence` |
|--------|---------|------------------------|------------------------------|--------------------|------------------------------|----------------------------|-------------------------------|--------------------------|------------------------------------------|
| `balanced()` | 3 | 0.2 | 0.6 | 0.0 | 0.3 | 200 | 2 | 3 | `False` |
| `strict()` | 2 | 0.4 | 0.7 | 0.3 | 0.3 | 200 | 1 | 2 | `True` |
| `permissive()` | 5 | 0.15 | 0.4 | 0.0 | 0.25 | 200 | 3 | 5 | `False` |
| `fast()` | 1 | 0.2 | 0.6 | 0.0 | 0.3 | 50 | 1 | 1 | `False` |

`permissive()` still requires localized Smith-Waterman evidence. It does not emit embedding-only citations; `retrieval_support` is the only embedding-driven output.
