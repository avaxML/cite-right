---
type: advanced-guide
title: Rust Acceleration
description: How the optional cite_right._core extension accelerates prepare, inverted-index retrieval, and Smith-Waterman alignment, how to select a backend, and what the Python fallback path does when the extension is missing.
tags: [rust, _core, backend, prepare, inverted-index, smith-waterman, performance, fallback, thread-safety, abi3]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-b774a369d680d48a9a4648c3
    resource: repo://rust_core/Cargo.toml
  - id: openwiki-source-91f3630f8a21f16b6af8a13e
    resource: repo://rust_core/src/inverted_index.rs
  - id: openwiki-source-5e3251d7fd54ced7f7fb97fd
    resource: repo://rust_core/src/lib.rs
  - id: openwiki-source-547a8756cc6b03449d99ed3f
    resource: repo://rust_core/src/prepared_corpus.rs
  - id: openwiki-source-8c2260658c1a4514202dea35
    resource: repo://rust_core/src/smith_waterman.rs
  - id: openwiki-source-0b1b3279f2fdef17b4081691
    resource: repo://src/cite_right/_core.pyi
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-565dc547e636f5aa89fb94bd
    resource: repo://src/cite_right/core/aligner_py.py
  - id: openwiki-source-edff227b75f84c03f46d0ad0
    resource: repo://src/cite_right/core/aligner_rust.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Rust Acceleration

Cite-Right 0.4.0 ships an optional native extension, `cite_right._core`, that reimplements the prepare phase, the inverted index, and Smith-Waterman alignment in Rust. The extension is published as abi3 wheels (`abi3-py311`), including linux/aarch64, plus an sdist. When the extension is importable, the library uses it automatically. When it is not, the same pipeline runs on a pure-Python fallback path so the public API is unchanged.

The hot path stays the same shape either way: prepare tokenizes sources, builds passage windows, computes IDF, and (on the default path) fills an inverted index; per answer span the pipeline picks candidate windows, then Smith-Waterman localizes the citation. The Rust extension accelerates each of those stages; it does not change which stages run. The inverted index only chooses which windows to align. Smith-Waterman still localizes `char_start` / `char_end`.

## What The Extension Covers

The Rust extension exposes three layers over the existing Python API.

`rust_tokenize_and_prepare(source_texts, window_size, stride) -> PreparedCorpus` tokenizes the source corpus, generates passage windows, computes IDF, and builds the inverted index in one Rust call. The Python `SimpleTokenizer` vocabulary is then synchronized from the Rust side so the answer-side tokenizer maps tokens to the same IDs. Per-passage `token_ids` and `token_spans` are kept on the Rust `PreparedCorpus` and fetched on demand at alignment time, avoiding the cost of copying large token arrays back to Python after prepare.

`query_index(query_tokens, max_candidates) -> list[int]` and `InvertedIndex.query(query_tokens, max_candidates) -> list[int]` both implement the same conjunctive rare-token intersect: tokens are sorted by posting count, the intersection of the rarest tokens is taken first, and the result is capped at `max_candidates`. The result is a list of candidate global indices that survive to Smith-Waterman.

`align_pair`, `align_pair_details`, `align_pair_blocks_details`, `align_batch_details`, `align_batch_blocks_details`, `align_best`, `align_best_details`, and `align_topk_details` are the alignment entry points. `align_pair_blocks_details` and `align_batch_blocks_details` additionally return the per-alignment `match_blocks` needed by `CitationConfig(multi_span_evidence=True)`. The Rust `RustSmithWatermanAligner` selects between these depending on the configuration; the construction-time check requires the relevant entry point to be present, so a mismatched extension fails fast rather than silently producing empty `evidence_spans`.

The legacy JSON-shaped `rust_build_citations_fast` is also exposed on `_core` and is used by `citations.py` to build a `Citation` / `RetrievalSupport` list without round-tripping source text through Python. The newer `PreparedCorpus.build_citations` path on the Rust `PreparedCorpus` object eliminates that round-trip further: source texts and offsets stay in Rust and the result returns only the citation and support structs.

## When Rust Prepare Runs

`PreparedCitationCorpus.from_sources` takes the Rust prepare path when all four of the following hold:

- `use_rust=True` (the default).
- `cite_right._core` imports successfully.
- The tokenizer is a `SimpleTokenizer`.
- The source segmenter is a `SimpleSegmenter`.

When any of those is false, the Python prepare path runs: source passages are built, candidates are built from `SimpleTokenizer`, and IDF is computed in Python. The resulting `PreparedCitationCorpus` has `inverted_index=None` and `rust_corpus=None`, and `_select_candidates` falls back to lexical prefilter on each span.

When the Rust prepare path runs, an embedder does not disable it. The 0.3.x skip of Rust prepare on the embedder path is gone. With an embedder set, the embedding index is built on the Rust-prepared candidates; lexical scores are filled only for inverted-index seeds; embedding-only extras keep `lexical_score == 0.0`; and `_add_embedding_candidates` can still add non-index windows to the candidate set before Smith-Waterman runs. Those extras still need Smith-Waterman. `retrieval_support` is not a `Citation` and does not flip status.

If the Rust path raises during `from_sources`, the corpus is rebuilt on the Python fallback and a message is written to `sys.stderr`. The public API still returns a `PreparedCitationCorpus`.

## Backend Selection

`align_citations` and `PreparedCitationCorpus.align` both accept a `backend` argument with three values: `"auto"`, `"python"`, `"rust"`.

```python
from cite_right import align_citations

# Default. Uses Rust if cite_right._core is importable, else pure Python.
results = align_citations(answer, sources, backend="auto")

# Force the pure-Python aligner (SmithWatermanAligner).
results = align_citations(answer, sources, backend="python")

# Require Rust. Raises RuntimeError if the extension is missing.
results = align_citations(answer, sources, backend="rust")
```

`"auto"` first tries `RustSmithWatermanAligner`; if construction raises `RuntimeError` (the extension is missing or an older extension is missing the `*_details` entry points), it falls back to `SmithWatermanAligner`. `"rust"` constructs `RustSmithWatermanAligner` directly and propagates the `RuntimeError` so the caller sees the failure. `"python"` skips the construction attempt entirely. Both aligners are interchangeable on the same prepared corpus; status, offsets, and `evidence_spans` are the same.

The `RustSmithWatermanAligner` constructor takes the same `match_score`, `mismatch_score`, `gap_score`, and `return_match_blocks` arguments as `SmithWatermanAligner`. The `multi_span_evidence` field on `CitationConfig` is the switch that decides whether `return_match_blocks` is set.

## Fallback When `_core` Is Missing

If `cite_right._core` is not importable, every step that the extension would do has a Python equivalent:

- **Prepare.** `PreparedCitationCorpus.from_sources` skips the Rust `_from_sources_rust` branch and runs the Python prepare path. The corpus has `rust_corpus=None` and `inverted_index=None`. The same `PreparedCitationCorpus` type, the same `align` API.
- **Inverted index.** `_select_candidates` calls `_add_lexical_candidates` directly. There is no conjunctive rare-token intersect, but the same `max_candidates_lexical` cap applies. For paraphrase-heavy content where the index is the recall win, the Python path is the more likely failure mode.
- **Alignment.** `_default_aligner` returns a `SmithWatermanAligner` (pure Python). The aligner computes the same alignment matrix and supports `return_match_blocks=True`, so multi-span evidence is still available on the fallback path.
- **Citation building.** The Python `process_alignment_to_citation` path is used; the Rust `rust_build_citations_fast` and `PreparedCorpus.build_citations` paths are skipped silently.

```python
try:
    from cite_right._core import align_pair
    print("Rust extension is available")
except ImportError:
    print("Rust extension is not available, using pure Python")
```

The library handles the fallback automatically. You do not need to check availability unless you are debugging.

## Parallelism And GIL

The Rust entry points use `py.detach(...)` from PyO3 to release the Python GIL while doing work, so other Python threads can run while Rust is computing. The alignment and citation-building paths additionally use `rayon` to parallelize across candidate windows and across candidates, respectively. Combined, this means a `ThreadPoolExecutor` of N workers can use up to N cores for the alignment hot path when the Rust extension is active; the pure-Python aligner is GIL-bound and Python threads overlap but do not parallelize the alignment matrix.

```python
from concurrent.futures import ThreadPoolExecutor
from cite_right import PreparedCitationCorpus

corpus = PreparedCitationCorpus.from_sources(sources)

def align(answer):
    return corpus.align(answer)

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(align, answer_workload))
```

## Memory

The Rust extension allocates outside Python's managed heap. `tracemalloc` will not see Rust memory; use system tools such as `/usr/bin/time -v`, `ps`, or a Rust-aware profiler to see the full picture. The per-hit memory cost is dominated by the Smith-Waterman alignment matrix at O(M × N) in the lengths of the answer span and the candidate passage. With the default one-sentence windows, each matrix is small, and index-first retrieval caps how many matrices are built per span. Parallel alignment multiplies peak memory by the number of concurrent candidates; on a 16-core system, peak memory is roughly 16× a single alignment at saturation.

## Thread Safety

`align_citations` and `PreparedCitationCorpus.align` are thread-safe; multiple threads can call them on the same prepared corpus concurrently. The Rust aligner releases the GIL during the hot path, so a thread pool can use multiple cores. The pure-Python aligner is GIL-bound. The `PreparedCitationCorpus` itself is treated as read-only across threads; rebuild it on the writer side if the underlying sources change.

## Correctness Guarantees

The Rust implementation is required to match Python outputs exactly. That is enforced by tests, not just aimed for. The same `Citation` struct comes out of either backend: same `status`, same `char_start` / `char_end`, same `evidence`, same `evidence_spans` (when `multi_span_evidence=True`), same `components`, same `retrieval_support` (when present). Deterministic tie-breaking and identical score normalization are part of the contract, not aspirational. Edge cases including empty sequences, single-token sequences, and sequences with no matches are handled identically.

If a call must match between the two backends (audit, regression check), the simplest pattern is to run both and compare:

```python
python_results = align_citations(answer, sources, backend="python")
rust_results = align_citations(answer, sources, backend="rust")

for py, rs in zip(python_results, rust_results):
    assert py.status == rs.status
    assert len(py.citations) == len(rs.citations)
    for py_cite, rs_cite in zip(py.citations, rs.citations):
        assert py_cite.char_start == rs_cite.char_start
        assert py_cite.char_end == rs_cite.char_end
        assert py_cite.evidence == rs_cite.evidence
```

A mismatched extension (too old to expose the `*_details` entry points) raises at `RustSmithWatermanAligner` construction time with a clear message and the `align` call falls back to the Python aligner.

## Building The Extension From Source

Pre-built abi3 wheels cover common platforms including linux/aarch64. To build from source, install a Rust toolchain and the project's build dependencies.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build the extension
git clone https://github.com/avaxML/cite-right.git
cd cite-right
uv sync --frozen
uv run maturin develop --release
```

`--release` enables optimizations; development builds without it are significantly slower. The Rust extension requires a C compiler in addition to Rust. On Linux install `build-essential` or the platform equivalent; on macOS install the Xcode command line tools; on Windows install Visual Studio Build Tools with the C++ workload. When you do not need a source build, install from PyPI with `pip install cite-right==0.4.0`.

## When To Use The Pure-Python Backend

Forcing `backend="python"` makes sense in a few specific cases:

- **Debugging alignment behavior.** It is easier to add prints or step through a Smith-Waterman call in Python than in Rust.
- **Minimal environments.** A wheel is not available, Rust cannot be installed, and the abi3 extension cannot be built. The pure-Python path has no extra dependencies beyond NumPy.
- **Cross-backend verification.** Running both backends on the same input and comparing `status`, `char_start`, `char_end`, and `evidence` is the cleanest way to confirm a release has not regressed the Rust path.

For all other workloads, `backend="auto"` is the right default.

## Performance

On the 50-case pack with no embedder, 0.4.0 p50 wall time is about 12.4ms versus about 175.8ms in 0.3.1, roughly 14×. spp is 81.3% versus 83.4%. RAGTruth test quality on 2,675 answers matched 0.3.1. The speedup comes from index-first retrieval plus the native aligner; Smith-Waterman is still run on every selected candidate. Embedder encoding is extra cost on top of the no-embedder numbers above. For per-span metrics and the configuration levers that move latency, see [Performance Tuning](./performance-tuning.md). For how the index and the embedder interact, see [Embedding Retrieval](./embedding-retrieval.md).

## Background Reading

- [How It Works](../concepts/how-it-works.md) — the index-first pipeline end to end.
- [Performance Tuning](./performance-tuning.md) — what the per-span metrics tell you and which configuration levers move latency.
- [Embedding Retrieval](./embedding-retrieval.md) — when to add semantic recall on top of the lexical index.
- [Multi-Span Evidence](./multi-span-evidence.md) — how the `*_blocks_details` Rust entry points populate `Citation.evidence_spans`.
