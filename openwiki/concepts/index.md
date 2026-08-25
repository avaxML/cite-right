# Files

- [Citation model and offsets](citation-model.md) - Offset invariants for citations: half-open intervals, chunk rebasing, evidence string equality with source slices, and the multi-span representation.
- [Contradiction detection](contradiction-detection.md) - Lightweight post-alignment check that detects when a cited answer contradicts the source, downgrading span status to partial rather than unsupported.
- [Rust extension lifecycle](extension-lifecycle.md) - How the optional cite_right._core Rust extension is built, imported, probed, and what its presence or absence means for each code path.
- [Citation status semantics](status-semantics.md) - The rules that determine whether an answer span is marked supported, partial, or unsupported — and why retrieval_support never flips status.
