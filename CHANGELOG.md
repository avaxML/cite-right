# Changelog

All notable changes to cite-right will be documented in this file.

## [0.3.1] - 2026-08-22

PyPI republish of 0.3.0 due to wheel filename reuse restriction.

## [0.3.0] - 2026-08-22

### Added
- **Rust-Accelerated Citation Pipeline**: Complete rewrite of the hot path in Rust
  - Phase 1: Tokenization, segmentation, passage generation, IDF computation in Rust (2.2x speedup on prepare phase)
  - Phase 2: Smith-Waterman batch alignment with Rayon parallelization and GIL release
  - Phase 3: Citation building and evidence extraction in Rust with JSON serialization
- JSON-based Rust↔Python data exchange to avoid PyO3 type complexity
- Parallel citation building with Rayon
- `serde` and `serde_json` dependencies for efficient serialization

### Performance
- **4.3x end-to-end speedup** on realistic RAG workloads (30 sources, 3-sentence answer)
  - Original baseline (no Rust): ~16ms p50
  - Current Rust pipeline: ~3.7ms p50
- Rust prepare: ~1.5ms (40% of runtime)
- Rust SW + citation building: ~2.2ms (60% of runtime)

### Changed
- Backend architecture now uses Rust for all hot-path operations
- Maintains same public API (`align_citations`, `PreparedCitationCorpus`, etc.)
- Automatic fallback to Python implementation on errors

### Fixed
- CI failures: stale setup-uv, click dependency for spacy, cryptography CVE, tiktoken unicode test
- Windows-specific bugs in file/path handling and os.fsync usage
- Coverage badge reliability issues
- Clippy warnings in Rust code

## [0.2.0] - 2026-08-20

Initial release with performance and reliability improvements.
