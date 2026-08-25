---
type: concept
title: Embedding-backed recall
description: How semantic similarity via `embedder=` expands candidate recall, how `_add_embedding_candidates` merges embedding scores with lexical candidates, and when embedding-only passages become RetrievalSupport instead of Citations.
tags: [retrieval, embeddings, cosine-similarity, candidate-selection, semantic-recall]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
  - id: openwiki-source-81b6e35be1922824d5712143
    resource: repo://src/cite_right/models/embedding_index.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

# Embedding-backed recall

This page documents the semantic recall mechanism: how passing an `embedder` to `align_citations()` or `PreparedCitationCorpus` enables similarity-based candidate expansion beyond lexical overlap. Embeddings improve recall for paraphrased content, but only alignment-backed matches produce character-accurate `Citation` objects; high-similarity passages without localized alignment surface as `RetrievalSupport`.

## Enabling semantic recall

Pass an `Embedder` instance to `align_citations()`:

```python
from cite_right import align_citations, SourceDocument
from cite_right.models.sbert_embedder import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

results = align_citations(
    answer,
    sources,
    embedder=embedder,
)
```

The embedder is forwarded to `PreparedCitationCorpus.from_sources()` (repo://src/cite_right/core/prepared_corpus.py#L126-L158), which calls `build_embedding_index()` to create an `EmbeddingIndex` from all source passage texts:

```python
embedding_index = EmbeddingIndex.build(
    embedder, [candidate.passage.text for candidate in candidates]
)
```

Query vectors for each answer span are computed lazily via `EmbeddingCache` and stored for reuse (repo://src/cite_right/core/prepared_corpus.py#L84-L98). When `_process_answer_span()` runs, it retrieves the cached query vector and passes it to `_add_embedding_candidates()` (repo://src/cite_right/citations.py#L260-L277).

## `EmbeddingIndex.top_k` cosine scoring

`EmbeddingIndex.top_k()` (repo://src/cite_right/models/embedding_index.py#L43-L78) finds the top-k most similar passages using cosine similarity:

```
score = dot(query, passage) / (||query|| × ||passage||)
```

The implementation:
1. Converts the query vector to a NumPy array (repo://src/cite_right/models/embedding_index.py#L60)
2. Returns an empty list if the query norm is zero (repo://src/cite_right/models/embedding_index.py#L61-L63)
3. Computes dot products against all stored passage vectors (repo://src/cite_right/models/embedding_index.py#L65)
4. Masks zero-norm passages to avoid division by zero (repo://src/cite_right/models/embedding_index.py#L66-L68)
5. Sorts indices by descending score using `np.lexsort` (repo://src/cite_right/models/embedding_index.py#L72)
6. Returns up to k `(index, score)` pairs where passage norm > 0 (repo://src/cite_right/models/embedding_index.py#L75-L77)

Scores range from -1 to 1, where 1 indicates identical direction. The default `min_embedding_similarity` threshold is 0.3 (repo://src/cite_right/core/citation_config.py#L60).

## `_add_embedding_candidates` merge logic

`_add_embedding_candidates()` (repo://src/cite_right/citations.py#L1338-L1355) runs after the inverted index and lexical stages complete, merging embedding candidates into the shared `selected` dict:

```python
def _add_embedding_candidates(
    selected: dict[int, tuple[float, float]],
    embedding_index: EmbeddingIndex | None,
    query_vector: list[float] | None,
    cfg: CitationConfig,
) -> None:
    if (
        cfg.max_candidates_embedding <= 0
        or query_vector is None
        or embedding_index is None
    ):
        return
    for idx, score in embedding_index.top_k(query_vector, cfg.max_candidates_embedding):
        prev = selected.get(idx)
        lexical_score = 0.0 if prev is None else prev[1]
        selected[idx] = (score, lexical_score)
```

Key behaviors:
- If a candidate was already added by the index or lexical stage, its **existing lexical score is preserved** and the embedding score is merged in
- If the candidate is new to `selected`, it gets `lexical_score = 0.0`
- No `min_embedding_similarity` filtering happens here; that gate is applied later when building `RetrievalSupport`

## From embedding candidate to Citation vs. RetrievalSupport

All selected candidates — whether seeded by index, lexical, or embedding — proceed to Smith-Waterman alignment. Two outcomes are possible:

### Outcome 1: Exact citation with localized evidence

If alignment succeeds and meets quality thresholds, `_build_exact_citation()` (repo://src/cite_right/citations.py#L680-L725) produces a `Citation` with character-accurate `char_start`, `char_end`, and `evidence` spans. The embedding score contributes to the final score alongside alignment quality and lexical overlap.

### Outcome 2: Retrieval-only support without localization

If alignment fails to localize evidence (low alignment score, insufficient coverage), the candidate falls through to `_build_retrieval_support_for_candidate()` (repo://src/cite_right/citations.py#L728-L747):

```python
def _build_retrieval_support_for_candidate(
    *,
    embed_score: float,
    lexical_score: float,
    cfg: CitationConfig,
) -> RetrievalSupport | None:
    if lexical_score <= 0.0 and embed_score < cfg.min_embedding_similarity:
        return None
    # ...
```

This is the `min_embedding_similarity` filter in practice: embedding-only candidates (lexical = 0.0) **must** exceed this threshold to become `RetrievalSupport`. Candidates that fail both lexical and embedding thresholds are silently discarded.

The `RetrievalSupport` object records (repo://src/cite_right/core/results.py):
- `passage_char_start`, `passage_char_end`, `passage_text` — the **full passage**, not a localized span
- `embedding_score` and `lexical_score` — the raw signals that selected this candidate
- `retrieval_score` — the combined signal strength

## Score inheritance pattern summary

The `(embed_score, lexical_score)` tuple in `selected` follows a specific merge convention (repo://src/cite_right/citations.py#L1193-L1232):

| Candidate source | Embed score | Lexical score |
|---|---|---|
| Inverted index seed | 0.0 | Precomputed or filled on-demand |
| Lexical fallback | 0.0 | IDF-weighted overlap |
| Embedding extra (new) | cosine similarity | 0.0 |
| Embedding extra (already selected) | cosine similarity | Preserved from prior stage |

Lexical scores stay 0.0 for embedding-only candidates. This is intentional: `retrieval_support` still respects `min_embedding_similarity` because the guard checks the **embedding score directly**, not whether it was the primary selection signal.

## Candidate limits

Three independent limits control pool size (repo://src/cite_right/core/citation_config.py#L65-L67):

| Limit | Default | Role |
|---|---|---|
| `max_candidates_lexical` | 200 | Index-seeded and lexical-fallback candidates |
| `max_candidates_embedding` | 200 | Embedding-only extras added after lexical stage |
| `max_candidates_total` | 400 | Hard cap on total candidates sent to alignment |

Embedding extras run last and may add candidates beyond the lexical limit, up to `max_candidates_total`.

## Relationship to retrieval pipeline

Embedding-backed recall is the fourth and final stage in candidate selection (documented in `/openwiki/architecture/retrieval-pipeline`). The full ordering is:

1. **Rust corpus index** or **Python InvertedIndex** — conjunctive token query
2. **Lexical fallback** — IDF-weighted overlap when index misses
3. **Lexical score fill** — on-demand IDF scoring for index seeds (Rust corpus path)
4. **Embedding top-k** — semantic expansion for candidates missed by lexical signals

Embedding extras can rescue candidates that share meaning but not tokens. However, they provide **semantic retrieval signal only**; without Smith-Waterman localization, they remain `RetrievalSupport` entries indicating "this passage is topically related" rather than "here is the exact evidence."
