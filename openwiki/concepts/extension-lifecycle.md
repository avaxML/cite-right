---
type: concept
title: Rust extension lifecycle
description: How the optional cite_right._core Rust extension is built, imported, probed, and what its presence or absence means for each code path.
tags: [architecture, extension, rust, backend]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-b774a369d680d48a9a4648c3
    resource: repo://rust_core/Cargo.toml
  - id: openwiki-source-5e3251d7fd54ced7f7fb97fd
    resource: repo://rust_core/src/lib.rs
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-edff227b75f84c03f46d0ad0
    resource: repo://src/cite_right/core/aligner_rust.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The `cite_right._core` Rust extension is an optional high-performance backend for tokenization, corpus preparation, candidate indexing, and Smith-Waterman alignment. When it is absent, all functionality falls back to pure Python with no change in API behavior.

The extension is built as a Python native extension module using [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/), producing a cdylib loaded as `cite_right._core`. The `[tool.maturin]` section in `pyproject.toml` configures the build:

```toml
[tool.maturin]
bindings = "pyo3"
module-name = "cite_right._core"
python-source = "src"
manifest-path = "rust_core/Cargo.toml"
```

Build command:

```bash
uv run maturin develop
```

## Import probe: `HAS_RUST_CORE`

`src/cite_right/citations.py` performs a one-time import probe at module load time:

```python
try:
    from cite_right import _core
    from cite_right._core import InvertedIndex
    from cite_right._core import PreparedCorpus as RustPreparedCorpus
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False
    _core = None
```

`HAS_RUST_CORE` is a module-level boolean that gates every Rust-involving code path. The stub file `src/cite_right/_core.pyi` provides type annotations for IDE support and static analysis when the extension is unavailable.

## Three backend modes

The `align_citations()` function and `PreparedCitationCorpus.align()` accept a `backend` parameter with three modes:

| Mode | Behavior |
|------|----------|
| `"auto"` (default) | Attempts `RustSmithWatermanAligner`; silently falls back to `SmithWatermanAligner` if the extension is missing or lacks required functions. |
| `"python"` | Forces pure Python `SmithWatermanAligner`, ignoring the Rust extension entirely. |
| `"rust"` | Forces `RustSmithWatermanAligner`; raises `RuntimeError` if the extension cannot be imported or is missing required exports. |

The backend selection happens in `_default_aligner()`:

```python
def _default_aligner(cfg: CitationConfig, *, backend: str) -> Aligner:
    if backend == "python":
        return SmithWatermanAligner(...)
    if backend == "rust":
        return RustSmithWatermanAligner(...)  # raises RuntimeError on failure
    # auto
    try:
        return RustSmithWatermanAligner(...)
    except RuntimeError:
        return SmithWatermanAligner(...)  # silent fallback
```

## `RustSmithWatermanAligner` initialization requirements

`src/cite_right/core/aligner_rust.py` validates the extension at construction time. If `return_match_blocks=False` (the default), the extension must export `align_pair_details` and `align_batch_details`. If `return_match_blocks=True` (required for `multi_span_evidence`), it must additionally export `align_pair_blocks_details` and `align_batch_blocks_details`. Missing exports cause a `RuntimeError`:

```python
if not hasattr(self._core, "align_pair_details"):
    raise RuntimeError(
        "Rust extension is missing detailed alignment outputs required for "
        "citation scoring; rebuild it or use backend='python'"
    )
```

The Rust module exports these functions via `#[pyfunction]` decorators in `rust_core/src/lib.rs`.

## `PreparedCorpus` fast path with `SimpleTokenizer` + `SimpleSegmenter`

When `PreparedCitationCorpus.from_sources()` is called with the default `SimpleTokenizer` and `SimpleSegmenter`, it attempts the Rust prepare fast path:

```python
if (
    use_rust
    and RUST_PREPARE_AVAILABLE
    and isinstance(tokenizer, SimpleTokenizer)
    and isinstance(source_segmenter, SimpleSegmenter)
):
    return cls._from_sources_rust(...)
```

`RUST_PREPARE_AVAILABLE` is a parallel import probe in `src/cite_right/core/prepared_corpus.py` that checks for `rust_tokenize_and_prepare` availability.

### What happens in the Rust fast path

`rust_tokenize_and_prepare()` in `rust_core/src/lib.rs` orchestrates the entire preparation phase in Rust:

1. **Tokenization** (sequential, to maintain consistent vocabulary): `SimpleTokenizer::tokenize()` for each source text.
2. **Segmentation and passage generation** (parallel via Rayon): `simple_segment()` + `generate_passages()`.
3. **Index building** (sequential): all candidates, token data, and postings are accumulated in Rust.
4. **IDF and vocabulary** computation.

The function returns a `prepared_corpus::PreparedCorpus` Python object—an opaque Rust struct registered with `#[pyclass]`:

```python
# Python side receives an opaque Rust object
rust_corpus = rust_tokenize_and_prepare(source_texts, window_size, stride)
```

### Data kept in Rust vs. Python

When the fast path succeeds, `PreparedCitationCorpus` holds:

- `rust_corpus`: The Rust `PreparedCorpus` object.
- `candidates`: A Python list of `Candidate` objects with **empty** `token_ids`/`token_spans` fields.
- `inverted_index`: `None` (the index lives inside `rust_corpus`).

Token data is **not** copied to Python at preparation time. Instead, it is fetched on-demand at alignment time via `rust_corpus.get_candidate_tokens()` and `rust_corpus.get_candidate_metadata()`.

The Rust vocabulary is synchronized back to the Python tokenizer:

```python
rust_vocab = rust_corpus.get_vocab()
tokenizer._vocab = {normalized: int(token_id) for normalized, token_id in rust_vocab}
```

## Candidate selection with `rust_corpus`

When `rust_corpus` is present, candidate selection follows a tiered strategy in `_select_candidates()`:

1. **Index seeding** (Rust-side): `rust_corpus.query_index()` runs a conjunctive token overlap query entirely in Rust, returning candidate indices.
2. **On-demand lexical scoring**: If index seeds were found but no pre-computed lexical scores exist, `_fill_rust_lexical_scores()` fetches token IDs from Rust for the selected candidates and computes IDF-weighted overlap scores.
3. **Embedding fallback**: `embedding_index.top_k()` adds semantically similar candidates if an embedder is present.

When `rust_corpus` is absent, candidate selection falls back to `_lexical_prefilter()` using Python-side IDF scores and token sets.

## Three citation-building fast paths

At alignment time, `_process_answer_span()` attempts three progressively less-optimized paths:

### 1. Corpus fast path (most efficient)

Used when:
- `HAS_RUST_CORE` is `True`
- `rust_corpus` is present with a `build_citations` method
- `aligner` is a `RustSmithWatermanAligner`

This path calls `rust_corpus.build_citations()` directly, keeping all alignment, scoring, and citation construction in Rust. No token data is marshalled between languages:

```python
result = rust_corpus.build_citations(
    answer_tokens,
    candidate_indices_orig,
    lexical_scores_list,
    embed_scores,
    source_id_map,
    base_offset_map,
    config_tuple,
    multi_span_config,
    ...
)
```

### 2. Legacy Rust fast path

Used when the corpus fast path is unavailable but `_core.rust_build_citations_fast` exists. This path fetches token data from `rust_corpus`, marshals it to Rust via a tuple of all candidate data, performs alignment in Rust, and returns results as a JSON string. Slower than the corpus fast path due to data marshalling overhead.

### 3. Standard path (Python fallback)

Used when neither Rust fast path is available. Alignment runs via `aligner.align_batch()` on Python-side token lists, and citation construction happens in Python via `_build_exact_citation()`.

## Extension presence affects all phases

| Phase | Without Rust | With Rust |
|-------|-------------|-----------|
| **Import** | `HAS_RUST_CORE = False` | `HAS_RUST_CORE = True` |
| **Preparation** | Python tokenization, segmentation, candidate building | `rust_tokenize_and_prepare()` keeps all data in Rust |
| **Index** | `inverted_index: None` or Python `InvertedIndex` | Rust `PreparedCorpus` with built-in index |
| **Candidate selection** | Python IDF prefilter | Rust index query + on-demand token fetching |
| **Alignment** | `SmithWatermanAligner` | `RustSmithWatermanAligner` |
| **Citation building** | Python `_build_exact_citation()` | `rust_corpus.build_citations()` or `rust_build_citations_fast()` |

## Failure modes

- **Missing extension at import**: All `HAS_RUST_CORE`-gated paths take the Python fallback. No error is raised.
- **`backend="rust"` with missing extension**: `RustSmithWatermanAligner.__init__()` raises `RuntimeError`.
- **Extension present but missing detailed outputs**: `RustSmithWatermanAligner.__init__()` raises `RuntimeError` listing the missing function.
- **Rust prepare path fails at runtime**: Falls back to Python preparation with a warning printed to stderr.
- **Corpus fast path fails at alignment time**: Silently falls back to legacy Rust fast path or standard path.

## Tests

| Test file | Purpose |
|-----------|---------|
| `tests/test_alignment_rust_parity.py` | Verifies Rust and Python aligners produce identical results for all modes |
| `tests/test_rust_prepare_with_embeddings.py` | End-to-end test of Rust prepare path with embedder |
| `tests/test_citations_api.py::test_align_citations_auto_falls_back_when_rust_core_lacks_details` | Verifies `auto` backend falls back when extension lacks `align_pair_details` |
| `tests/test_citations_api.py::test_align_citations_rust_backend_requires_detailed_core` | Verifies `backend="rust"` raises `RuntimeError` without detailed outputs |
| `tests/conftest.py::rust_core` | Pytest fixture that skips Rust-dependent tests if extension is absent |
| `tests/conftest.py::rust_core_with_blocks` | Fixture requiring `align_pair_blocks_details` |

## Related pages

- [/openwiki/architecture/smith-waterman.md](/openwiki/architecture/smith-waterman.md) — Smith-Waterman algorithm details and Rust implementation
- [/openwiki/operations/extension-backends.md](/openwiki/operations/extension-backends.md) — Operational guide for building and configuring the Rust extension
- [/openwiki/workflows/align-citations.md](/openwiki/workflows/align-citations.md) — Workflow for using `align_citations()` with backend selection
