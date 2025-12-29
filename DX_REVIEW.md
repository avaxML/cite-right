# Developer Experience Review: Cite-Right

This document provides a comprehensive analysis of the Cite-Right library from a developer experience perspective, identifying strengths and areas for improvement.

---

## Executive Summary

**Overall Assessment: Strong (8/10)**

Cite-Right is a well-designed, production-ready library with excellent documentation, clean API design, and thoughtful architecture. The library successfully balances power with simplicity, offering sensible defaults while allowing deep customization. Key strengths include comprehensive type hints, clear documentation, and a pluggable architecture.

---

## Strengths

### 1. Excellent Documentation (README)

The README is exemplary:
- Clear problem statement and value proposition upfront
- Installation instructions with optional extras clearly explained
- Working quickstart example that demonstrates the core use case
- Comprehensive "How it works" section explaining concepts
- Pipeline breakdown for advanced users
- Multiple tokenizer/segmenter examples
- Configuration reference with sensible defaults explained

**Notable:** The documentation explains *why* certain choices were made, not just *how* to use the API.

### 2. Clean, Intuitive API Design

The main entry point `align_citations()` demonstrates excellent API ergonomics:

```python
# Simple case - just works
spans = align_citations(answer, sources)

# Advanced case - full control
spans = align_citations(
    answer,
    sources,
    config=CitationConfig(top_k=1),
    tokenizer=HuggingFaceTokenizer.from_pretrained("bert-base-uncased"),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
)
```

**Key API design wins:**
- Sensible defaults allow zero-config usage
- All customization via keyword arguments (no positional args beyond required)
- Sources accept multiple types (`str | SourceDocument | SourceChunk`) for flexibility
- Return type is a simple list of dataclasses, easy to iterate

### 3. Strong Type System

The library leverages Python's type system effectively:
- All public classes use Pydantic `BaseModel` with `frozen=True` for immutability
- Full type annotations throughout the codebase
- Type stubs for the Rust extension (`_core.pyi`)
- Runtime-checkable `Protocol` classes for extensibility
- Works well with IDE autocomplete and static analysis tools

### 4. Pluggable Architecture

The Protocol-based design allows easy extension:

```python
# src/cite_right/core/interfaces.py
@runtime_checkable
class Tokenizer(Protocol):
    def tokenize(self, text: str) -> TokenizedText: ...
```

This enables:
- Custom tokenizers (3 built-in implementations)
- Custom segmenters (simple, spaCy, PySBD)
- Custom aligners
- Custom embedders

### 5. Optional Dependencies Done Right

The library uses extras to keep core installs lightweight:
- `pip install cite-right` - just numpy + pydantic
- `pip install "cite-right[spacy]"` - adds spaCy support
- `pip install "cite-right[embeddings]"` - adds sentence-transformers

This approach reduces install time and avoids dependency conflicts for basic use cases.

### 6. Consistent Data Model

All result types follow consistent patterns:
- Character offsets are always 0-based, half-open `[start, end)`
- All models are immutable (frozen)
- Clear naming conventions (`char_start`/`char_end`, `source_id`/`source_index`)
- Evidence strings are always extractable: `source[char_start:char_end] == evidence`

### 7. Deterministic Output

The library guarantees deterministic results with documented tie-breaking rules. This is critical for testing and reproducibility in production systems.

### 8. Comprehensive Test Suite

- 16 test files with ~3555 LOC
- Clear test organization by feature
- Pytest markers for optional dependencies (`@pytest.mark.spacy`, `@pytest.mark.embeddings`)
- Edge case coverage (`test_error_conditions.py`)
- Real-world scenarios (`test_dspy_paper_scenarios.py`)

### 9. Error Handling

The library handles edge cases gracefully:
- Empty sources return `unsupported` status (not an exception)
- Empty/whitespace answers return empty results
- Invalid backend parameter raises `ValueError` with clear message
- No silent failures

### 10. Observability

The `on_metrics` callback provides visibility into the pipeline:

```python
def log_metrics(metrics: AlignmentMetrics) -> None:
    print(f"Aligned {metrics.num_answer_spans} spans in {metrics.total_time_ms:.1f}ms")

align_citations(answer, sources, on_metrics=log_metrics)
```

---

## Areas for Improvement

### 1. Import Ergonomics

**Issue:** `CitationConfig` must be imported from a nested module:

```python
# Current - requires knowledge of internal structure
from cite_right.core.citation_config import CitationConfig

# vs. other config classes that are exported at top level
from cite_right import HallucinationConfig  # This works!
```

**Recommendation:** Export `CitationConfig` and `CitationWeights` from `__init__.py` for consistency.

### 2. Missing `CitationConfig` Documentation in README

**Issue:** The README references `CitationConfig` tuning knobs but points to the source file rather than documenting them inline:

> "Common tuning knobs live in `CitationConfig` (see `src/cite_right/core/citation_config.py`)"

**Recommendation:** Add a configuration reference table in the README similar to the `HallucinationMetrics` table.

### 3. Incomplete Docstrings on Data Models

**Issue:** Some Pydantic models lack comprehensive docstrings:

```python
# src/cite_right/core/results.py
class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float          # What does this score mean? Range?
    source_id: str
    source_index: int
    candidate_index: int  # What is a "candidate"?
    char_start: int
    char_end: int
    evidence: str
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    components: Mapping[str, float] = Field(default_factory=dict)  # What keys exist?
```

**Recommendation:** Add docstrings explaining:
- Score ranges and interpretation
- What each component key means
- When `evidence_spans` differs from the single `evidence` field

### 4. TokenizerConfig Not Exported

**Issue:** `TokenizerConfig` must be imported from a deep path:

```python
from cite_right.text.tokenizer import TokenizerConfig
```

**Recommendation:** Either export at top level or document the full import path in the tokenizer section.

### 5. Backend Selection Could Be an Enum

**Issue:** Backend is a string literal, which IDE autocomplete doesn't help with:

```python
backend: Literal["auto", "python", "rust"] = "auto"
```

**Recommendation:** Consider an `AlignmentBackend` enum for better discoverability, while still accepting string literals for backward compatibility.

### 6. No CHANGELOG

**Issue:** No CHANGELOG.md to track version history and breaking changes.

**Recommendation:** Add a CHANGELOG following [Keep a Changelog](https://keepachangelog.com/) format.

### 7. Limited Inline Examples in Docstrings

**Issue:** While the README has good examples, function docstrings have minimal examples:

```python
def align_citations(...) -> list[SpanCitations]:
    """Align answer spans to source citations.

    Args:
        answer: The answer text to find citations for.
        ...

    Returns:
        List of SpanCitations, one per answer span.
    """
```

**Recommendation:** Add `Examples:` section to main API functions (like `compute_hallucination_metrics` already has).

### 8. No Convenience Methods on Result Types

**Issue:** Result types are pure data containers. Common operations require manual iteration:

```python
# Getting all unsupported spans requires:
unsupported = [r for r in results if r.status == "unsupported"]

# Could be:
# results.get_unsupported()  # or similar
```

**Recommendation:** Consider adding helper methods while keeping models immutable:
- `SpanCitations.is_supported` property
- A results wrapper class with filtering methods

### 9. Rust Extension Build Documentation

**Issue:** The README mentions Rust is "only needed when building from source" but doesn't explain:
- How to check if Rust extension is active
- Performance difference between Python/Rust backends
- How to force pure-Python mode if Rust causes issues

**Recommendation:** Add a troubleshooting section for Rust extension issues.

### 10. No Interactive Examples (Notebooks)

**Issue:** No Jupyter notebook examples for interactive exploration.

**Recommendation:** Add an `examples/` directory with:
- `quickstart.ipynb` - basic usage
- `advanced_config.ipynb` - configuration tuning
- `hallucination_detection.ipynb` - analyzing RAG outputs

---

## Minor Suggestions

1. **Add py.typed marker visibility** - The `py.typed` file exists but isn't mentioned in docs for typed package consumers.

2. **Consider lazy imports** for optional dependencies to speed up `import cite_right` when only using core features.

3. **Add `__version__` attribute** to the package for programmatic version access.

4. **Link to API reference** - Consider adding auto-generated API docs (Sphinx/MkDocs).

5. **Add performance benchmarks** - Would help users choose between tokenizers/backends.

---

## Comparison to Similar Libraries

| Aspect | Cite-Right | Typical RAG Libraries |
|--------|------------|----------------------|
| Type hints | Complete | Often partial |
| Documentation | Comprehensive | Variable |
| Optional deps | Well-isolated | Often bundled |
| Extensibility | Protocol-based | Inheritance or none |
| Immutability | Frozen models | Often mutable |
| Determinism | Guaranteed | Rarely documented |

---

## Conclusion

Cite-Right demonstrates mature library design with strong attention to developer experience. The main areas for improvement are relatively minor polish items (export consistency, documentation gaps) rather than fundamental design issues.

**Priority recommendations:**
1. Export `CitationConfig` from top-level `__init__.py`
2. Add docstrings to `Citation` and other result models
3. Add CHANGELOG.md
4. Add `Examples:` sections to main function docstrings

The library is ready for production use and provides an excellent foundation for citation extraction in RAG systems.
