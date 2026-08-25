---
type: testing
title: Test Markers and Fixtures
description: Custom pytest markers and skipif helpers that gate tests to optional dependencies, the Rust extension, and specific feature availability.
tags: [testing, pytest, rust, spacy, optional-dependencies, markers]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-f0a6e7dc03522b2682f88655
    resource: repo://tests/conftest.py
  - id: openwiki-source-811d84c9631d27a47d6421e0
    resource: repo://tests/test_alignment_rust_parity.py
  - id: openwiki-source-81dc541c73d5fbfa6a7e1947
    resource: repo://tests/test_citations_multi_span.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The cite-right test suite uses custom pytest markers to gate tests to optional dependencies and the Rust extension. All markers and skipif helpers are defined in `tests/conftest.py` and registered via `pytest_configure`. This allows a single test suite to run across environments with varying dependency availability without requiring all optional packages to be installed.

## Registered Markers

Markers are declared to pytest via `pytest_configure` in `tests/conftest.py`. Each marker documents its requirement so pytest can filter tests and report skipped ones clearly.

| Marker | Requirement | Typical reason to skip |
|--------|-------------|------------------------|
| `rust` | `cite_right._core` extension built via `maturin develop` | Rust toolchain not available; extension not rebuilt after source change |
| `spacy` | `spacy` installed and `en_core_web_sm` model downloaded | Full spaCy ecosystem not installed |
| `embeddings` | `sentence-transformers` installed | Heavy ML dependency not installed |
| `tiktoken` | `tiktoken` installed | Fast OpenAI-compatible tokenizer not installed |
| `huggingface` | `transformers` and `tokenizers` installed | HuggingFace stack not installed |
| `pysbd` | `pysbd` installed | Sentence-boundary detection library not installed |
| `slow` | No dependency; marked manually | Test takes significant time (e.g., model downloads) |

## Skipif Helpers

The module-level skipif decorators in `tests/conftest.py` are the primary mechanism for conditional test skipping. They are preferred over inline `pytest.skip()` calls because they produce consistent skip messages and keep availability logic centralized.

### `requires_rust`

```python
requires_rust = pytest.mark.skipif(
    not _rust_available(),
    reason="Rust extension not built",
)
```

Skips when `cite_right._core` cannot be imported. Used for tests that verify Rust/Python parity or measure Rust speedups. The `_rust_available()` check attempts a direct import of `_core` and returns `False` on `ImportError`.

**Used by:** `test_alignment_rust_parity.py::test_rust_parity`, `test_alignment_rust_parity.py::test_rust_align_best_matches_python_selection`

### `requires_rust_blocks`

```python
requires_rust_blocks = pytest.mark.skipif(
    not _rust_has_blocks_details(),
    reason="Rust extension missing align_pair_blocks_details",
)
```

Skips when the Rust extension lacks the `align_pair_blocks_details` function. This function provides the match-block traceback required for multi-span evidence extraction (non-contiguous citations). The `_rust_has_blocks_details()` check verifies `hasattr(_core, "align_pair_blocks_details")` after confirming `_rust_available()`.

**Used by:** `test_alignment_rust_parity.py` (parity tests for block-level traceback), `test_citations_multi_span.py::test_align_citations_multi_span_python_and_rust_backends_match`

### Additional skipif helpers

| Helper | Condition | Reason |
|--------|-----------|--------|
| `requires_spacy` | `spacy` spec not found | spaCy not installed |
| `requires_spacy_model` | spaCy installed but `en_core_web_sm` unavailable | Model not downloaded |
| `requires_embeddings` | `sentence_transformers` spec not found | sentence-transformers not installed |
| `requires_tiktoken` | `tiktoken` spec not found | tiktoken not installed |
| `requires_huggingface` | `transformers` or `tokenizers` spec not found | HuggingFace stack incomplete |
| `requires_pysbd` | `pysbd` spec not found | pysbd not installed |

## Fixtures

### `rust_core`

Provides the `cite_right._core` module directly. Skips via `pytest.skip()` if the import fails.

```python
@pytest.fixture
def rust_core() -> ModuleType:
    try:
        from cite_right import _core
        return _core
    except ImportError:
        pytest.skip("Rust extension not built")
```

### `rust_core_with_blocks`

Provides `cite_right._core` only when `align_pair_blocks_details` is available. Used by tests that specifically exercise multi-span alignment in Rust.

```python
@pytest.fixture
def rust_core_with_blocks() -> ModuleType:
    try:
        from cite_right import _core
    except ImportError:
        pytest.skip("Rust extension not built")

    if not hasattr(_core, "align_pair_blocks_details"):
        pytest.skip("Rust extension is missing align_pair_blocks_details (rebuild required)")
    return _core
```

### `spacy_nlp`

Provides a spaCy `Language` object loaded with `en_core_web_sm`. Skips if the model is not installed.

## Configuration

Markers are referenced in `pyproject.toml` only as a comment:

```toml
[tool.pytest.ini_options]
# Markers are registered in conftest.py via pytest_configure
```

The actual marker declarations live in `tests/conftest.py` to keep them close to the skip logic.

## Running Tests by Marker

Select tests using `-m`:

```bash
# Run only Rust-dependent tests
pytest -m rust

# Run Rust tests except slow ones
pytest -m "rust and not slow"

# Skip all optional-dependency tests
pytest -m "not (spacy or embeddings or tiktoken or huggingface or pysbd)"
```
