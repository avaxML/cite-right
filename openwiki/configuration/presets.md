---
type: configuration
title: Configuration Presets
description: CitationConfig preset tradeoffs — balanced() default, strict() high-precision, permissive() paraphrase-friendly, fast() latency-bound. Permissive still requires localized Smith-Waterman evidence and does not emit embedding-only citations.
tags: [configuration, presets, citation-config, balanced, strict, permissive, fast, smith-waterman, retrieval-support, candidate-selection, threshold]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Configuration Presets

`CitationConfig` ships with four class methods that return pre-tuned configurations. They are the fastest way to move between recall, precision, and latency budgets without reasoning about every individual knob. All four live in `src/cite_right/core/citation_config.py` and are plain class-method factories; you can read any field off the returned object and override individual values with `model_copy(update=...)`.

```python
from cite_right import CitationConfig, align_citations

config = CitationConfig.strict()           # or .permissive() / .fast() / .balanced()
results = align_citations(answer, sources, config=config)
```

`balanced()` is identical to the default constructor `CitationConfig()`. The other three change specific fields. The comparison table at the bottom of this page lists every preset-touched field side by side; everything else (window sizes, alignment scoring, weights, ranking) stays at the defaults.

This page is the preset orientation. For every field on `CitationConfig` and `CitationWeights`, see [Citation Config](./citation-config.md). For the cost model and which knobs actually move steady-state latency, see [Performance Tuning](../advanced/performance-tuning.md). For the embedder channel that `permissive()` leans on, see [Embedding Retrieval](../advanced/embedding-retrieval.md).

## Balanced

`CitationConfig.balanced()` is the default. Use it when you do not have a strong reason to bias toward precision or recall.

```python
from cite_right import CitationConfig, align_citations

config = CitationConfig.balanced()
results = align_citations(answer, sources, config=config)
```

Balanced uses the default thresholds: `top_k=3`, `min_answer_coverage=0.2`, `supported_answer_coverage=0.6`, `min_embedding_similarity=0.3`, and the full candidate caps (`max_candidates_lexical=200`, `max_candidates_embedding=200`, `max_candidates_total=400`). It works well for typical RAG applications where sources and answers have reasonable lexical overlap. If you are not sure which preset to start from, start here.

## Strict

`CitationConfig.strict()` is the high-precision preset. Use it when false positives are the dominant cost: fact-checking, legal or medical document review, or any context where a wrong citation is worse than a missing one.

```python
config = CitationConfig.strict()
results = align_citations(answer, sources, config=config)
```

Strict raises every gate that distinguishes a strong citation from a marginal one:

- `top_k=2` (default 3) — fewer citations per span.
- `min_answer_coverage=0.4` (default 0.2) — a candidate must cover a larger fraction of the answer to qualify.
- `supported_answer_coverage=0.7` (default 0.6) — a span is `"supported"` only when its best citation clears a higher coverage bar.
- `min_final_score=0.3` (default 0.0) — a non-zero final-score floor rejects near-miss alignments.
- `max_citations_per_source=1` (default 2) — at most one citation per source document.
- `max_retrieval_support=2` (default 3) — fewer embedding-driven support passages.
- `require_all_answer_tokens_in_evidence=True` (default `False`) — every answer token has to be reachable in the evidence, otherwise the citation is dropped. This is the strictest single field in the preset; it filters out paraphrases where some content words are reordered or replaced, even when the rest of the evidence lines up.

The trade is precision for recall: more spans land on `"partial"` or `"unsupported"` because the bar is higher. `require_all_answer_tokens_in_evidence=True` in particular is the field that turns this preset into a precision tool. If you want strictness on numerical or entity-based adversarial inputs without losing recall on faithful paraphrases, do not just paste the preset — read what each field does in [Citation Config](./citation-config.md) and tune the ones relevant to your adversarial class.

## Permissive

`CitationConfig.permissive()` is the recall-friendly preset. Use it when answers are heavily paraphrased from sources: summarization outputs, translated content, creative rewrites, or any task where high recall matters more than precision.

```python
config = CitationConfig.permissive()
results = align_citations(answer, sources, config=config)
```

Permissive lowers every gate that suppresses a borderline citation:

- `top_k=5` (default 3) — more citations per span.
- `min_answer_coverage=0.15` (default 0.2) — candidates with smaller answer-token coverage still qualify.
- `supported_answer_coverage=0.4` (default 0.6) — a span is `"supported"` at a much lower coverage threshold.
- `min_embedding_similarity=0.25` (default 0.3) — a lower cosine bar lets the embedder surface more recall candidates.
- `max_citations_per_source=3` (default 2) — more citations from the same source.
- `max_retrieval_support=5` (default 3) — more embedding-driven support passages.

What permissive does not change matters as much as what it does:

- **Permissive still requires localized Smith-Waterman evidence.** Lower thresholds let more candidates through, but every emitted `Citation` still has to be localized by Smith-Waterman, with `char_start` / `char_end` rebaseable onto the source text. Permissive does not skip alignment; it just lets weaker alignments through.
- **Permissive does not emit embedding-only citations.** The removed `allow_embedding_only` flag is still rejected at construction. A passage that the index or embedder selected but Smith-Waterman could not localize surfaces on `SpanCitations.retrieval_support`, never as a `Citation`, and never flips `status`. The embedder channel in permissive is a recall expander that runs through the same alignment step, not a shortcut around it.

Permissive is a recall preset, not a precision preset. It will let through borderline alignments that strict rejects. For workloads where the index alone misses most of the right window because vocabulary does not match, pair permissive with the embedder path described in [Embedding Retrieval](../advanced/embedding-retrieval.md).

## Fast

`CitationConfig.fast()` is the latency-bound preset. Use it for batch processing, interactive previews, or any workload where wall time dominates.

```python
config = CitationConfig.fast()
results = align_citations(answer, sources, config=config)
```

Fast reduces every candidate cap and tightens every per-span limit:

- `top_k=1` (default 3) — one citation per span.
- `max_candidates_lexical=50` (default 200) — one quarter of the index seeds.
- `max_candidates_embedding=50` (default 200) — one quarter of the embedding extras.
- `max_candidates_total=100` (default 400) — one quarter of the merged candidate pool.
- `max_citations_per_source=1` (default 2).
- `max_retrieval_support=1` (default 3).

Coverage thresholds (`min_answer_coverage`, `supported_answer_coverage`, `min_final_score`, `min_embedding_similarity`) stay at the balanced defaults, so the citations that survive are still the ones that would have survived balanced — there are just fewer of them. The trade is throughput for recall: `fast()` will drop real matches that the index found because the candidate cap is too small to keep them.

Fast is not a recall preset and not a precision preset. It is a way to spend less compute per span. On the 50-case pack with no embedder, balanced 0.4.0 lands around p50 wall of 12.4ms; tightening caps with `fast()` reduces that further at the cost of missing some matches. For the full cost-model breakdown, see [Performance Tuning](../advanced/performance-tuning.md).

For the highest throughput on workloads that align the same sources against many answers, pair `fast()` with `PreparedCitationCorpus.from_sources(sources, config=fast_config)` so the inverted index and passage windows are built once and reused.

## Choosing A Preset

The right preset depends on the cost of being wrong in each direction.

Reach for **strict** when a wrong citation is more expensive than a missing one: medical, legal, financial, or any "I will act on this answer" context. The `require_all_answer_tokens_in_evidence=True` field is what makes it strict; the other fields are coarser levers around the same idea.

Reach for **permissive** when sources and answers are expected to differ substantially in wording: summarization, translation, paraphrase-heavy generation. The lowered coverage thresholds and higher `top_k` let weak-but-real alignments through, and the lowered `min_embedding_similarity` widens the embedder recall channel. Remember it still requires localized Smith-Waterman evidence and still does not emit embedding-only citations — it widens the funnel before Smith-Waterman, it does not bypass it.

Reach for **fast** when wall time dominates: batch pipelines, interactive previews, anything where you would rather have a noisy answer in milliseconds than a clean one in seconds. Fast trades recall for latency, not precision for recall. A real match that the index found can still be dropped because the cap is too small.

Reach for **balanced** for everything else. The defaults are tuned for typical RAG workloads with reasonable lexical overlap and no extreme precision or latency budget.

If your workload straddles these, mix the preset with explicit overrides on a single field rather than a wholesale re-derivation.

```python
from cite_right import CitationConfig

# Strict except keep balanced's recall-friendly max_retrieval_support
config = CitationConfig.strict().model_copy(update={"max_retrieval_support": 3})
```

## Customizing Presets

Since `CitationConfig` is a frozen Pydantic model, mutating a preset directly is not possible. The supported customization is `model_copy(update={...})`, which returns a new config with the named fields overridden and the rest inherited from the preset.

```python
from cite_right import CitationConfig

base = CitationConfig.strict()
config = base.model_copy(update={"supported_answer_coverage": 0.65})
```

The same pattern works from a plain `CitationConfig()`:

```python
config = CitationConfig().model_copy(update={"top_k": 5, "min_final_score": 0.1})
```

For a custom high-recall configuration with no embedder, a useful starting point is permissive minus the wider candidate caps:

```python
config = CitationConfig.permissive().model_copy(
    update={"max_candidates_total": 200, "max_citations_per_source": 2}
)
```

## Runtime Preset Selection

For applications that need different citation behavior based on context, presets can be selected dynamically.

```python
from cite_right import CitationConfig


def get_config_for_context(context) -> CitationConfig:
    if context.requires_high_precision:
        return CitationConfig.strict()
    if context.is_summarization_task:
        return CitationConfig.permissive()
    if context.has_latency_constraint:
        return CitationConfig.fast()
    return CitationConfig.balanced()


config = get_config_for_context(current_context)
results = align_citations(answer, sources, config=config)
```

This pattern lets a single pipeline serve diverse use cases without rebuilding the source configuration at every call site. When the same sources are reused, hand the chosen config to `PreparedCitationCorpus.from_sources(sources, config=config)` so prepare runs once and every subsequent call reuses the prepared corpus.

## Preset Comparison

The table below lists every field a preset touches. Anything not listed stays at the balanced default.

| Field | `balanced()` | `strict()` | `permissive()` | `fast()` |
|-------|--------------|------------|----------------|----------|
| `top_k` | 3 | 2 | 5 | 1 |
| `min_answer_coverage` | 0.2 | 0.4 | 0.15 | 0.2 |
| `supported_answer_coverage` | 0.6 | 0.7 | 0.4 | 0.6 |
| `min_final_score` | 0.0 | 0.3 | 0.0 | 0.0 |
| `min_embedding_similarity` | 0.3 | 0.3 | 0.25 | 0.3 |
| `max_candidates_lexical` | 200 | 200 | 200 | 50 |
| `max_candidates_embedding` | 200 | 200 | 200 | 50 |
| `max_candidates_total` | 400 | 400 | 400 | 100 |
| `max_citations_per_source` | 2 | 1 | 3 | 1 |
| `max_retrieval_support` | 3 | 2 | 5 | 1 |
| `require_all_answer_tokens_in_evidence` | `False` | `True` | `False` | `False` |

Reading the table: `strict()` is the only preset that changes the final-score floor and turns on `require_all_answer_tokens_in_evidence`. `permissive()` is the only preset that lowers the embedder similarity threshold. `fast()` is the only preset that touches the candidate caps. The `top_k` / `max_citations_per_source` / `max_retrieval_support` row shows the per-span budget each preset commits to: strict is conservative, permissive is generous, fast is minimal.

The other `CitationConfig` fields — `window_size_sentences`, `window_stride_sentences`, `match_score`, `mismatch_score`, `gap_score`, `weights`, `prefer_source_order`, `multi_span_evidence`, `multi_span_merge_gap_chars`, `multi_span_max_spans`, `min_alignment_score` — are not touched by any preset and stay at their balanced defaults.

## What Permissive Does Not Change

Worth repeating because it is the most-misread preset:

- **Permissive does not skip Smith-Waterman.** Every `Citation` it emits is still a localized alignment with `char_start` / `char_end` rebaseable onto the source text. The lower thresholds let more candidates through, but each surviving candidate still goes through the same alignment step.
- **Permissive does not emit embedding-only citations.** The removed `allow_embedding_only` and `supported_embedding_similarity` fields are still rejected at construction; the supported path for embedding-only signals is `SpanCitations.retrieval_support`, which is a separate channel that does not produce `Citation` objects and does not flip `status`. The lower `min_embedding_similarity` widens that channel, it does not bypass Smith-Waterman to make it a citation.
- **Permissive is a recall preset, not a precision preset.** It will let through borderline alignments that strict rejects. For workloads where you want both wide recall and tight numerical/entity matching, tune individual fields rather than picking the preset wholesale.
