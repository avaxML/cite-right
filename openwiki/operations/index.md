# Files

- [CitationConfig and weights](citation-config.md) - Configuration classes controlling citation alignment thresholds, scoring weights, candidate selection limits, and named presets for common use cases.
- [Convenience helpers](convenience-helpers.md) - High-level helper functions in cite-right for common RAG post-processing workflows: groundedness checks, answer annotation, citation formatting, and summary generation.
- [Backend selection and fallbacks](extension-backends.md) - How the align_citations backend parameter selects between Rust and Python execution paths, what forces the Python fallback, and the three Rust fast paths during citation building.
- [Fact-level verification](fact-verification.md) - The verify_facts function decomposes RAG answers into atomic claims and verifies each claim independently against source documents using citation alignment.
- [Hallucination metrics](hallucination-metrics.md) - The `compute_hallucination_metrics` function, `HallucinationConfig` knobs, and how per-span status rolls up into `groundedness_score`, `hallucination_rate`, and ratio fields.
