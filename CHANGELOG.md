# Changelog

All notable changes to cite-right will be documented in this file.

## [0.4.0] - 2026-08-25

### Added
- **Index-first retrieval**: inverted index over sources, rare-token intersect, Smith-Waterman only on hits (#47)
  - SW still localizes citations; the index only chooses which windows get SW
  - Same public API (`align_citations`, `PreparedCitationCorpus`)
  - Cheap contradiction check (negation / number / leftover n-gram slot / entity swap) downgrades to status `partial`, never `unsupported`
- Grounded how-to and news paraphrases can emit a citation from content-word overlap on the candidate passage when SW sequential coverage is low, instead of being tagged `unsupported`. (#51, #49)
- Second Smith-Waterman pass per structured-field candidate (`gap_score=0`) so Data2txt field:value paraphrases (hours, amenities, etc.) can be `supported` or `partial` without blessing invented fields. (#53, #50)
- abi3 wheels (pyo3 `abi3-py311`), linux/aarch64 wheels, and an sdist so arm64 Docker installs work. (#55)

### Performance
- **~14× faster** on the 50-case pack with no embedder
  - 0.3.1: ~175.8ms p50
  - 0.4.0: ~12.4ms p50
- spp 81.3% vs 83.4%
- RAGTruth test (2,675 answers): quality matched 0.3.1 (false-supported on gold hallu ~1.6%; unsupported precision ~14%)

### Changed
- Rust prepare still runs when an embedder is set (it previously skipped). The embedding index is built on those candidates. Embedding-only `retrieval_support` still respects `min_embedding_similarity`; lexical scores are filled only for index seeds. (#46)
- Leftover n-gram conflicts now check contradiction against the full candidate passage, not only the truncated SW evidence span. Shared tokens that would bless a contradictory statement as `supported` still become `partial`, not `unsupported`. (#52, #48)
- Faster CI: lint extracted, matrix 9→5, rust/uv caches, concurrency cancel. (#55)

### Fixed
- Leftover n-gram false-supported (#48)
- Paraphrase overflag (#49)
- Data2txt field:value overflag (#50)

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
