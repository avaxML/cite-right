---
type: workflow
title: Prepared corpus workflow
description: When and how to use PreparedCitationCorpus.from_sources(...).align(answer) to amortize prepare cost across many answers.
tags: [cite-right, prepared-corpus, workflow, citation-alignment, rust-prepare]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-61cbfe170d8c82a627f10456
    resource: repo://tests/test_inverted_index.py
  - id: openwiki-source-5ddf2e3b4fca9c3c6270fdcf
    resource: repo://tests/test_rust_prepare_with_embeddings.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

# Prepared corpus workflow

`PreparedCitationCorpus` separates the expensive source-preparation phase from the per-answer alignment phase. When you have a fixed corpus of sources and need to align many answers against it, prepare once and call `align()` repeatedly to avoid redundant tokenization, segmentation, and indexing work.

## The cost-amortization pattern

```python
from cite_right import PreparedCitationCorpus, CitationConfig
from cite_right.models.sbert_embedder import SentenceTransformerEmbedder

# Prepare once — this is the expensive step
sources = [...]  # your document list
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

corpus = PreparedCitationCorpus.from_sources(
    sources,
    config=CitationConfig(),
    embedder=embedder,
)
# corpus.tokenizer, corpus.candidates, corpus.idf, corpus.embedding_index,
# corpus.rust_corpus are all populated here

# Align many answers against the same corpus
for answer in answers:
    results = corpus.align(answer)  # fast: reuses prepared state
```

The `align_citations` convenience function internally follows this same pattern — it calls `PreparedCitationCorpus.from_sources()` and then `corpus.align()` in one shot. The workflow documented here is for callers who need to align multiple answers against the same source set without re-preparing each time.

## `from_sources` entrypoint

```python
@classmethod
def from_sources(
    cls,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: CitationConfig | None = None,
    source_segmenter: Segmenter | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    use_rust: bool = True,
) -> "PreparedCitationCorpus"
```

Sources are normalized to `NormalizedSource`, segmented into passages by the `source_segmenter`, and converted into `Candidate` objects with token IDs and IDF weights.

### Rust vs. Python prepare decision

The `from_sources` method tries a Rust fast path first (repo://src/cite_right/core/prepared_corpus.py#L144-L162):

```python
if (
    use_rust
    and RUST_PREPARE_AVAILABLE
    and isinstance(tokenizer, SimpleTokenizer)
    and isinstance(source_segmenter, SimpleSegmenter)
):
    try:
        return cls._from_sources_rust(...)
    except Exception:
        # Fall back to Python
        pass
```

- **`RUST_PREPARE_AVAILABLE`**: set to `True` when `cite_right._core` loads successfully (repo://src/cite_right/core/prepared_corpus.py#L31-L42). Falls back to `False` if the Rust extension is absent.
- **Custom tokenizers** (e.g., `TiktokenTokenizer`, `HuggingFaceTokenizer`) or **custom segmenters** (e.g., `SpacySegmenter`) bypass the Rust path and use the Python fallback even when `use_rust=True`.
- The Python fallback silently degrades: it still produces a correct corpus but without Rust-level token storage.

### What gets built during `from_sources`

| Component | Python path | Rust path |
|---|---|---|
| `normalized_sources` | ✅ | ✅ |
| `source_passages` | ✅ | ✅ (metadata only) |
| `candidates` | ✅ full `token_ids`/`token_spans`/`token_set` | ✅ empty `token_ids`/`token_spans`/`token_set` (populated on-demand) |
| `idf` | ✅ computed in Python | ✅ computed in Rust |
| `embedding_index` | ✅ built from embedder | ✅ built from embedder (same code path) |
| `rust_corpus` | `None` | ✅ `PreparedCorpus` object |
| `_embedding_build_time_ms` | ✅ tracked | ✅ tracked |

### Rust corpus and on-demand token fetching

When the Rust path is used, `rust_corpus` (repo://src/cite_right/core/prepared_corpus.py#L116-L118) holds token data in Rust memory. The Python-side `Candidate` objects are created with empty `token_ids`, `token_spans`, and `token_set` — these are fetched on-demand at alignment time:

```python
# repo://src/cite_right/core/prepared_corpus.py#L219-L260
candidates.append(
    Candidate(
        ...
        token_ids=[],  # Empty — will fetch from rust_corpus on demand
        token_spans=[],
        token_set=frozenset(),
    )
)
```

At alignment time, tokens are retrieved via `rust_corpus.get_candidate_tokens(...)` (repo://src/cite_right/citations.py#L301-L303) or the faster `rust_corpus.build_citations(...)` fast path (repo://src/cite_right/citations.py#L384-L397). This design avoids materializing token data for all candidates when only a subset is selected per answer span.

## `align` method

```python
def align(
    self,
    answer: str,
    *,
    backend: str = "auto",
    answer_segmenter: AnswerSegmenter | None = None,
    aligner: Aligner | None = None,
    on_metrics: MetricsCallback | None = None,
    process_answer_span: Callable[..., tuple[SpanCitations, int, float, float]] | None = None,
) -> list[SpanCitations]
```

For each answer span, `align`:
1. Segments the answer with `answer_segmenter`.
2. Tokenizes the span with `self.tokenizer`.
3. Selects candidates via `_select_candidates` — using `rust_corpus.query_index` for index seeds when available, falling back to IDF-weighted lexical scoring (repo://src/cite_right/citations.py#L1180-L1232).
4. Runs Smith-Waterman alignment with the provided or default `aligner`.
5. Builds `Citation` objects with character offsets and computes status (`supported`/`partial`/`unsupported`).

The `rust_corpus` fast path in `_process_answer_span` (repo://src/cite_right/citations.py#L332-L397) avoids serializing token data to Python when both the Rust corpus and `RustSmithWatermanAligner` are available.

## Embedder compatibility with Rust prepare

An `Embedder` (e.g., `SentenceTransformerEmbedder`) is fully compatible with the Rust prepare path:

```python
# repo://src/cite_right/core/prepared_corpus.py#L270-L276
if embedder is not None:
    embedding_start = time.perf_counter()
    embedding_index = build_embedding_index(embedder, candidates)
    embedding_build_time_ms = (time.perf_counter() - embedding_start) * 1000
```

The embedder is applied **after** the Rust or Python prepare step. The `embedding_index` is always built in Python from the candidate passage texts, regardless of how candidates were prepared. This is confirmed by `test_rust_prepare_with_embeddings.py` (repo://tests/test_rust_prepare_with_embeddings.py#L35-L42) and `test_rust_corpus_with_embedder` (repo://tests/test_inverted_index.py#L175-L196).

## Embedding build time budget signal

`embedding_build_time_ms` (repo://src/cite_right/core/prepared_corpus.py#L119-L124) records wall-clock time spent building the source embedding index during `from_sources`. It is **not** included in per-span alignment metrics — it belongs to the prepare phase. When `on_metrics` is provided to `align_citations`, the total embedding time is the sum of `corpus.embedding_build_time_ms` (prepare) plus the per-span embedding time (repo://src/cite_right/citations.py#L180-L182):

```python
embedding_time_ms=(
    corpus.embedding_build_time_ms + align_metrics.embedding_time_ms
)
```

## Multiple `align` calls: what is reused vs. recomputed

| State | Reused across `align` calls? | Notes |
|---|---|---|
| `normalized_sources`, `source_passages`, `candidates`, `idf` | ✅ Yes | Prepared once in `from_sources` |
| `embedding_index` | ✅ Yes | Built once; `EmbeddingCache` is rebuilt per `align` call |
| `rust_corpus` | ✅ Yes | Stays in Rust; queried on-demand per span |
| `answer_segmenter`, `aligner` | ✅ Yes (if not provided per-call) | Defaults cached via closure |
| Per-span `EmbeddingCache` | ❌ Rebuilt | `build_answer_embedding_cache` runs each `align` call |
| Per-span `answer_tokens` | ❌ Rebuilt | Tokenized fresh per answer span |

## Relationship to `align_citations`

`align_citations` (repo://src/cite_right/citations.py#L72-L187) is a convenience wrapper that calls `PreparedCitationCorpus.from_sources()` and then `corpus.align()` in sequence. For single-answer use cases, `align_citations` is simpler:

```python
results = align_citations(answer, sources, embedder=embedder)
```

For batch alignment against the same corpus, use `PreparedCitationCorpus` directly:

```python
corpus = PreparedCitationCorpus.from_sources(sources, embedder=embedder)
for answer in answers:
    results = corpus.align(answer)
```

## Relevant tests

| Test | File | What it verifies |
|---|---|---|
| `test_rust_prepare_with_dummy_embedder_dim8` | `tests/test_rust_prepare_with_embeddings.py` | Rust path taken with embedder; `embedding_index` and `rust_corpus` populated |
| `test_rust_prepare_candidate_count_close_to_python` | `tests/test_rust_prepare_with_embeddings.py` | Rust and Python produce similar candidate counts (within 20%) |
| `test_custom_tokenizer_falls_back_to_python` | `tests/test_rust_prepare_with_embeddings.py` | Custom tokenizer triggers Python fallback; `rust_corpus is None` |
| `test_rust_prepare_embedding_build_time_tracked` | `tests/test_rust_prepare_with_embeddings.py` | `embedding_build_time_ms >= 0.0` after Rust prepare with embedder |
| `test_prepare_does_not_fetch_all_tokens` | `tests/test_inverted_index.py` | Candidates have empty `token_ids` after Rust prepare; tokens fetched on-demand during alignment |
| `test_python_fallback_without_index` | `tests/test_inverted_index.py` | `use_rust=False` produces correct corpus with `rust_corpus is None`; alignment still works |
| `test_prepared_citation_corpus_matches_align_citations` | `tests/test_citations_api.py` | `corpus.align(answer)` returns identical results to `align_citations(answer, sources)` |
