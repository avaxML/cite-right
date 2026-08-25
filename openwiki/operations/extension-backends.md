---
type: operations
title: Backend selection and fallbacks
description: How the align_citations backend parameter selects between Rust and Python execution paths, what forces the Python fallback, and the three Rust fast paths during citation building.
tags: [backend, rust, python, citation, extension]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-edff227b75f84c03f46d0ad0
    resource: repo://src/cite_right/core/aligner_rust.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The `align_citations()` function and `PreparedCitationCorpus.align()` accept a `backend` parameter that controls whether citation alignment uses the compiled Rust extension or the pure-Python Smith-Waterman implementation. The system is designed to run without the Rust extension while gaining its performance when the extension is present.

## `backend` parameter

The `backend` parameter accepts three literal values, all declared in `src/cite_right/citations.py`:

```python
backend: Literal["auto", "python", "rust"] = "auto"
```

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Attempts Rust; silently falls back to Python if the extension is unavailable or incomplete |
| `"python"` | Forces the pure-Python Smith-Waterman aligner; Rust extension is never used |
| `"rust"` | Requires the Rust extension; raises `RuntimeError` if it cannot be loaded or lacks required functions |

## `_default_aligner` selection logic

The `_default_aligner` function in `src/cite_right/citations.py` (lines 1105–1135) resolves the active `Aligner` implementation. It is called from two call sites:

1. `align_citations()` at line 205: `active_aligner = aligner or _default_aligner(cfg, backend=backend)`
2. `PreparedCitationCorpus.align()` at line 323: `resolved_aligner = _default_aligner(self.config, backend=backend)`

The selection logic branches on `backend`:

- `"python"` always returns `SmithWatermanAligner` with scoring parameters from `cfg`.
- `"rust"` attempts to construct `RustSmithWatermanAligner`, which raises `RuntimeError` on import failure or missing exports. Callers that pass `"rust"` explicitly expect this to succeed.
- `"auto"` wraps the Rust attempt in `try/except RuntimeError` and returns `SmithWatermanAligner` as the silent fallback.

The scoring parameters (`match_score`, `mismatch_score`, `gap_score`, `return_match_blocks`) are always forwarded from `CitationConfig`, ensuring identical alignment behavior regardless of backend.

## Exception-driven Rust → Python fallback

The fallback is not conditional on a feature flag or environment variable. It is driven by exceptions raised during `RustSmithWatermanAligner` initialization in `src/cite_right/core/aligner_rust.py`:

```python
try:
    from cite_right import _core  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Rust extension is not available. Build it with: uv run maturin develop"
    ) from exc
```

After the import succeeds, the constructor performs a second guard check:

```python
if not hasattr(self._core, "align_pair_details") or not hasattr(
    self._core, "align_batch_details"
):
    raise RuntimeError(
        "Rust extension is missing detailed alignment outputs required for "
        "citation scoring; rebuild it or use backend='python'"
    )
```

The `align_pair_details` and `align_batch_details` functions provide the detailed traceback output (`score`, `token_start`, `token_end`, `query_start`, `query_end`, `matches`) required for citation scoring. If the installed Rust extension was built without these exports, the wrapper raises before any alignment work begins. The `"auto"` path catches this `RuntimeError` and transparently uses `SmithWatermanAligner`.

## Three Rust fast paths in `_process_answer_span`

`src/cite_right/citations.py` defines `_process_answer_span` (lines 227–677), which is called per answer span. Within a single span, three distinct execution paths are attempted in order:

### Fast path 1: `PreparedCorpus.build_citations`

This is the most optimized path, available when all of the following hold:

- `HAS_RUST_CORE` is `True` (Rust extension loaded successfully)
- `rust_corpus is not None` (corpus was prepared via `rust_tokenize_and_prepare`)
- `hasattr(rust_corpus, "build_citations")` (extension exposes the method)
- `isinstance(aligner, RustSmithWatermanAligner)` (the aligner is Rust-backed)

When all conditions are met, the code constructs a config tuple and calls `rust_corpus.build_citations(...)` directly at line 384. This keeps all token data in Rust, avoiding the overhead of copying token IDs and metadata to Python and back. If this call raises any exception, the code falls through to the next path.

### Fast path 2: `rust_build_citations_fast` (JSON)

The second path attempts the legacy JSON-based fast path when all of the following hold:

- `HAS_RUST_CORE` is `True`
- The first fast path was not taken (`not use_corpus_fast_path`)
- `isinstance(aligner, RustSmithWatermanAligner)`
- `hasattr(_core, "rust_build_citations_fast")`

This path requires fetching token data from the Rust corpus (`get_candidate_tokens`, `get_candidate_metadata`) and serializing candidates as Python tuples. The result is returned as a JSON string and decoded at line 537:

```python
result_json = _core.rust_build_citations_fast(...)
result = json.loads(result_json)
```

Marshalling overhead makes this path slower than the first, but it still avoids running Smith-Waterman in Python. If this call raises an exception, the code falls through to the pure-Python fallback.

### Fallback: Pure-Python citation building

When neither Rust fast path succeeds, `_process_answer_span` aligns candidates using the provided `aligner` (Python or Rust) and builds citations in Python. The code at lines 586–640 dispatches:

```python
if not use_corpus_fast_path and not use_rust_fast_path:
    align_batch = getattr(aligner, "align_batch", None)
    if align_batch is None:
        alignments = [
            aligner.align(answer_tokens, token_ids)
            for token_ids in candidate_token_ids
        ]
    else:
        alignments = align_batch(answer_tokens, candidate_token_ids)
```

If `aligner` is `SmithWatermanAligner`, the alignment runs in pure Python. If it is `RustSmithWatermanAligner`, the wrapper delegates to `align_batch_details` in the Rust extension while citation building remains in Python.

## When `inverted_index` and `rust_corpus` stay `None`

The `PreparedCitationCorpus` fields `inverted_index` and `rust_corpus` are declared as nullable:

```python
inverted_index: InvertedIndex | None = None
rust_corpus: RustPreparedCorpus | None = None
```

`rust_corpus` is `None` when the Rust prepare fast path is not taken. This happens under three conditions:

1. **Custom tokenizer** — `use_rust=True` but `tokenizer` is not an instance of `SimpleTokenizer`. The Rust path requires the tokenization vocabulary to match what `rust_tokenize_and_prepare` produces.
2. **Custom segmenter** — `use_rust=True` but `source_segmenter` is not an instance of `SimpleSegmenter`. The windowing logic in Rust must match the Python passage generation.
3. **Missing Rust extension or failed preparation** — `RUST_PREPARE_AVAILABLE` is `False` (extension not importable) or `_from_sources_rust` raised an exception. The `except` block at lines 154–161 in `prepared_corpus.py` catches failures and falls through to the Python path, setting `rust_corpus=None`.

When `rust_corpus` is `None`, the `inverted_index` field is also left as `None`. The `inverted_index` field was historically used to hold a standalone `InvertedIndex` Rust object, but with the current design the index lives inside the `rust_corpus` object and is queried via `rust_corpus.query_index()` instead.

Candidate selection then falls back to lexical prefiltering. The `_select_candidates` function in `citations.py` (line 1196) checks for `rust_corpus` first and only calls `_add_index_candidates_from_corpus` when it is available. Otherwise it calls `_add_lexical_candidates` using Python-set operations over `candidate.token_set`.

## Summary of state transitions

```
align_citations(backend="auto")
  └── PreparedCitationCorpus.from_sources()
        ├── use_rust=True + SimpleTokenizer + SimpleSegmenter + extension available
        │     └── _from_sources_rust() → rust_corpus set, inverted_index stays None
        └── otherwise
              └── Python fallback → rust_corpus = None, inverted_index = None

PreparedCitationCorpus.align(backend="auto")
  └── _default_aligner(backend="auto")
        ├── RustSmithWatermanAligner() succeeds → Rust aligner
        └── RustSmithWatermanAligner() raises → SmithWatermanAligner (silent)

_process_answer_span() per span
  ├── use_corpus_fast_path → rust_corpus.build_citations() (no marshalling)
  ├── use_rust_fast_path → _core.rust_build_citations_fast() (JSON marshalling)
  └── fallback → aligner.align() in Python or Rust wrapper
```
