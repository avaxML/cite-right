# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

- **Hallucination detection metrics** - New `compute_hallucination_metrics()` function
  to measure how well generated answers are grounded in source documents
  - `HallucinationMetrics` with groundedness score, hallucination rate, and per-span confidence
  - `HallucinationConfig` for customizing detection thresholds
  - `SpanConfidence` for detailed per-span analysis
- **Fact-level verification** - New `verify_facts()` function for claim-level verification
  - `ClaimVerification` result type with verification status per claim
  - `SimpleClaimDecomposer` and `SpacyClaimDecomposer` for breaking answers into atomic claims
  - `FactVerificationConfig` and `FactVerificationMetrics` for configuration and results
- **PySBDSegmenter** - Faster sentence boundary detection using pysbd library
  - Available via `pip install "cite-right[pysbd]"`
- **Top-level exports** - `CitationConfig`, `CitationWeights`, and `TokenizerConfig` now
  exported from `cite_right` package root for easier imports
- **Package version** - `cite_right.__version__` attribute for programmatic version access
- **Comprehensive docstrings** - Added detailed documentation to all result types
  (`Citation`, `SpanCitations`, `AnswerSpan`, `SourceDocument`, `SourceChunk`, etc.)
- **Examples in docstrings** - Added usage examples to `align_citations()` and result types

### Changed

- Improved test suite organization following pytest best practices
- Enhanced type annotations throughout the codebase

## [0.2.0]

### Added

- **Multi-span evidence** - Support for non-contiguous evidence via `Citation.evidence_spans`
  - Enable with `CitationConfig(multi_span_evidence=True)`
  - Configurable merging with `multi_span_merge_gap_chars` and `multi_span_max_spans`
- **HuggingFace tokenizer** - Support for BERT, RoBERTa, and other transformer tokenizers
  - `HuggingFaceTokenizer.from_pretrained("bert-base-uncased")`
  - Available via `pip install "cite-right[huggingface]"`
- **Tiktoken tokenizer** - Support for OpenAI GPT tokenization
  - `TiktokenTokenizer("cl100k_base")` for GPT-4/GPT-3.5-turbo
  - Available via `pip install "cite-right[tiktoken]"`
- **Observability metrics** - `on_metrics` callback for performance monitoring
  - `AlignmentMetrics` with timing and operation counts
- **Test coverage** - Integrated coverage tracking with badge

### Changed

- Migrated all dataclasses to Pydantic models for better validation
- All models now frozen (immutable) for thread safety
- Improved embedding index performance with numpy optimizations

## [0.1.0]

### Added

- Initial release of Cite-Right
- **Core citation alignment** - `align_citations()` for RAG-style citation extraction
  - Smith-Waterman local sequence alignment
  - Character-accurate citation offsets
  - IDF-weighted lexical prefiltering
- **Tokenization** - `SimpleTokenizer` with configurable normalization
  - Number, percent, and currency normalization options
  - Unicode NFKC normalization and casefolding
- **Segmentation** - `SimpleSegmenter` and `SpacySegmenter` for sentence splitting
  - `SimpleAnswerSegmenter` and `SpacyAnswerSegmenter` for answer segmentation
  - Paragraph-aware sentence detection
- **Embeddings support** - Optional semantic similarity retrieval
  - `SentenceTransformerEmbedder` wrapper
  - Available via `pip install "cite-right[embeddings]"`
- **Rust acceleration** - Optional Rust extension for faster alignment
  - GIL-released parallel alignment with Rayon
  - Automatic fallback to pure Python
- **spaCy integration** - Clause-level splitting and better sentence boundaries
  - Available via `pip install "cite-right[spacy]"`
- **Source types** - `SourceDocument` and `SourceChunk` for flexible input
- **Citation configuration** - `CitationConfig` with 20+ tuning parameters
- **Deterministic output** - Guaranteed reproducible results with documented tie-breaking

[0.3.0]: https://github.com/avaxML/cite-right/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/avaxML/cite-right/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/avaxML/cite-right/releases/tag/v0.1.0
