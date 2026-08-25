# Rust Acceleration

Cite-Right includes an optional Rust extension that accelerates prepare, inverted-index retrieval, and Smith-Waterman alignment while matching Python results. Released 0.4.0 wheels ship that extension as abi3 (`abi3-py311`), including linux/aarch64, plus an sdist.

## How the Extension Works

The Rust extension reimplements the hot path in Rust, compiled to a Python module using PyO3. When available, the library automatically uses this faster implementation.

Prepare (tokenization, passage windows, IDF, inverted index) still runs in Rust when an embedder is set. The embedding index is then built on those prepared candidates. 0.3.x skipped Rust prepare in the embedder path. That skip is gone.

The extension also exposes lower-level alignment helpers.

The `align_pair` function computes alignment between two token sequences.

The `align_best` function finds the best match from one query sequence against multiple candidate sequences. It uses Rayon for parallel processing across candidates.

The `align_topk_details` function returns the top-k alignments with full scoring details. This supports the multi-citation feature.

All functions release the Python GIL during computation, allowing other Python threads to run concurrently.

0.4.0 retrieval is index-first. The inverted index lives on the prepared Rust corpus. Rare-token intersect chooses which windows get Smith-Waterman. Smith-Waterman still localizes citations. The index only chooses the windows.

## Checking Availability

You can check whether the Rust extension is available.

```python
try:
    from cite_right._core import align_pair

    print("Rust extension is available")
except ImportError:
    print("Rust extension is not available, using pure Python")
```

The library handles this automatically. You do not need to check availability unless debugging.

## Backend Selection

The `backend` parameter controls which implementation is used.

```python
from cite_right import align_citations

# Use Rust if available, fall back to Python (default)
results = align_citations(answer, sources, backend="auto")

# Force pure Python implementation
results = align_citations(answer, sources, backend="python")

# Require Rust (raises error if not available)
results = align_citations(answer, sources, backend="rust")
```

The "auto" setting is recommended for production. It provides Rust performance when available while ensuring the library works everywhere.

## Performance Characteristics

On the 50-case pack with no embedder, 0.4.0 p50 wall time is about 12.4ms versus about 175.8ms in 0.3.1, roughly 14×. spp is 81.3% versus 83.4%. RAGTruth test quality on 2,675 answers matched 0.3.1.

Those numbers are the published 0.4.0 measurements. They include index-first retrieval, not Smith-Waterman over every window.

Native code execution removes Python interpreter overhead from prepare, index query, and alignment. Rayon parallelizes work across CPU cores where the extension already did so. GIL release lets other Python threads run during those calls.

## Building the Extension

Pre-built abi3 wheels are available for common platforms, including linux/aarch64. If you need to build from source, ensure you have a Rust toolchain installed.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/avaxML/cite-right.git
cd cite-right
uv sync --frozen
uv run maturin develop --release
```

The `--release` flag enables optimizations. Development builds without this flag are significantly slower.

### Build Requirements

The Rust extension requires a C compiler in addition to Rust. On Linux, install build-essential or equivalent. On macOS, install Xcode command line tools. On Windows, install Visual Studio Build Tools with the C++ workload.

Install from PyPI with `pip install cite-right==0.4.0` when you do not need a source build.

## Correctness Guarantees

The Rust implementation is required to match Python outputs exactly. This is not just a quality goal but a design constraint enforced by tests.

Deterministic tie-breaking ensures that when multiple alignments have equal scores, both implementations choose the same one. The ordering considers score, source index, character position, and evidence length in that order.

Floating-point handling ensures consistent rounding. Both implementations use the same formulas for score normalization.

Edge cases including empty sequences, single-token sequences, and sequences with no matches are handled identically.

The test suite runs alignment operations on both implementations and verifies output equality. Any divergence fails the build.

## When to Use Pure Python

There are scenarios where forcing the pure Python backend makes sense.

Debugging alignment behavior is easier in Python. You can add print statements or step through with a debugger.

Minimal environments without a wheel or Rust build capability can still use the library. The pure Python implementation has no external dependencies beyond NumPy.

Verification of results in high-stakes applications may benefit from running both backends and confirming agreement.

```python
# Verify both backends agree
python_results = align_citations(answer, sources, backend="python")
rust_results = align_citations(answer, sources, backend="rust")

for py, rs in zip(python_results, rust_results):
    assert py.status == rs.status
    assert len(py.citations) == len(rs.citations)
    for py_cite, rs_cite in zip(py.citations, rs.citations):
        assert py_cite.char_start == rs_cite.char_start
        assert py_cite.char_end == rs_cite.char_end
```

Python backend still uses the same public API. It does not run Smith-Waterman over every window either when the inverted index is available. If the Rust extension is missing entirely, candidate selection falls back to the lexical prefilter.

## Memory Considerations

The Rust extension allocates memory outside Python's managed heap. For very large workloads, monitor system memory rather than relying solely on Python memory profiling.

The alignment matrix is the main memory consumer per hit. For sequences of length M and N, the matrix requires O(M × N) memory. With passage windowing limiting sequence lengths, this is typically a few megabytes per alignment. Index-first retrieval limits how many of those matrices are built.

Parallelization multiplies memory usage by the number of concurrent alignments. On a 16-core system, peak memory is roughly 16x a single alignment.

## Thread Safety

The Rust extension is thread-safe. Multiple Python threads can call alignment functions concurrently without synchronization issues. The GIL is released during Rust computation, enabling true parallelism.

```python
from concurrent.futures import ThreadPoolExecutor


def process_batch(items):
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(align_citations, answer, sources)
            for answer, sources in items
        ]
        results = [f.result() for f in futures]
    return results
```

This pattern allows processing multiple answers concurrently, with each using Rust parallelization internally for index query and passage alignment.
