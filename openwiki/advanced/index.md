# Files

- [Embedding Retrieval](embedding-retrieval.md) - How to enable semantic candidate expansion in Cite-Right using sentence-transformers, and how it interacts with index-first retrieval and Smith-Waterman localization.
- [Multi-Span Evidence](multi-span-evidence.md) - How CitationConfig(multi_span_evidence=True) exposes non-contiguous evidence regions on a Citation via evidence_spans, with gap merging and a max-spans fallback to the legacy contiguous span.
- [Performance Tuning](performance-tuning.md) - How Cite-Right's index-first pipeline scales, which configuration levers actually move steady-state latency, and how to reuse PreparedCitationCorpus for high-volume workloads.
- [Rust Acceleration](rust-acceleration.md) - How the optional cite_right._core extension accelerates prepare, inverted-index retrieval, and Smith-Waterman alignment, how to select a backend, and what the Python fallback path does when the extension is missing.
