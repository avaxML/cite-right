---
type: agent-routing
title: Cite-Right Wiki Quickstart
description: Agent-only routing map for the documentation tree under openwiki/. Indexes the 16 public pages and the 2 agent-only pages under openwiki/testing/, lists the public-vs-agent split, and points new agents at the right entry page (openwiki/index.md for reader questions, openwiki/concepts/how-it-works.md for pipeline work, openwiki/testing/contract-tests.md for Rust parity work).
tags: [quickstart, agent-routing, public-pages, agent-only, testing, index, how-it-works, contract-tests, pytest-markers, mkdocs, public-paths, instructions, page-map]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-5e3251d7fd54ced7f7fb97fd
    resource: repo://rust_core/src/lib.rs
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-ad97420c2a0900f616ed0fef
    resource: repo://src/cite_right/contradiction.py
  - id: openwiki-source-edff227b75f84c03f46d0ad0
    resource: repo://src/cite_right/core/aligner_rust.py
  - id: openwiki-source-376cf838d701ccf9f40efc03
    resource: repo://src/cite_right/core/citation_config.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-f0a6e7dc03522b2682f88655
    resource: repo://tests/conftest.py
  - id: openwiki-source-811d84c9631d27a47d6421e0
    resource: repo://tests/test_alignment_rust_parity.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Cite-Right Wiki Quickstart

This page is the routing map for the Cite-Right documentation tree under `openwiki/`. It is for coding agents, not for readers. There are two parallel audiences in this tree:

- **Public pages** under `openwiki/` that mirror what `mkdocs.yml` ships to GitHub Pages. There are 16 of them, plus the auto-generated Home at `openwiki/index.md`. Readers see only these.
- **Agent-only pages** under `openwiki/testing/`. There are 2 of them. They are denser, may mention `openwiki/` paths, and are not published.

The job of this page is to point a new agent at the right entry page for the question it actually has. Public-to-public links use relative paths (for example `../concepts/how-it-works.md`). Public-to-agent links are forbidden: no public page may link to `testing/`. This page itself is the one and only place that names the split.

## How The Tree Is Wired

`openwiki/INSTRUCTIONS.md` is the user-authored brief and is not generated. The public page paths are listed there under "Public Paths" and match `mkdocs.yml` exactly. The 16 public pages plus the auto-generated Home feed GitHub Pages. After each successful update, `scripts/publish_openwiki_to_docs.py` copies the public paths into `docs/`. The two pages under `openwiki/testing/` and this routing page are agent-only and never get published.

There is one documentation site. Do not invent a second nav, a `/wiki/` URL, a Wiki tab, or a "see the wiki" cross-link. Public pages never mention `wiki`, `MkDocs`, or `OpenWiki`. This page is allowed to name those because it is not on the public site.

## The Public Page Index

The Home (`openwiki/index.md`) is auto-generated and not counted in the 16 below. Each entry gives the path, a one-line role, and the canonical source the page documents.

### Getting Started (2)

- `openwiki/getting-started/installation.md` — install name `cite-right`, import package `cite_right`, Python 3.11+, 0.4.0 abi3 wheels (`abi3-py311`), linux/aarch64 wheels, sdist, and the `embeddings` / `spacy` / `huggingface` / `tiktoken` extras. Source: `pyproject.toml`, `docs/getting-started/installation.md`.
- `openwiki/getting-started/quickstart.md` — the basic `align_citations` pattern, reading per-span `status` and `evidence`, multiple sources, and `PreparedCitationCorpus` for repeated queries. Defines `"supported"`, `"partial"`, `"unsupported"` at the default `supported_answer_coverage = 0.6`. Source: `src/cite_right/citations.py`, `docs/getting-started/quickstart.md`.

### Core Concepts (4)

- `openwiki/concepts/how-it-works.md` — the pipeline orientation. Segment answer, prepare source passage windows, tokenize with one instance, index-first candidate selection, Smith-Waterman localization, offset rebase, ranking, contradiction check, status assignment. Covers the default / Rust path, the fallback path, the embedder path, content-word overlap, and the Data2txt second pass. Entry point for any "how does the pipeline work?" question. Source: `src/cite_right/citations.py`, `docs/concepts/how-it-works.md`.
- `openwiki/concepts/citation-alignment.md` — the I/O contract for `align_citations`. `SourceDocument` vs `SourceChunk` with chunk-offset rebasing; `SpanCitations` (`answer_span`, ranked `citations`, `retrieval_support`, `status`); half-open `char_start` / `char_end`; the evidence equality invariant; status from the best exact citation, not embedding similarity. Source: `src/cite_right/models/`, `docs/concepts/citation-alignment.md`.
- `openwiki/concepts/hallucination-detection.md` — the groundedness tagger in `src/cite_right/hallucination.py`. `"unsupported"` is "no localized citation survived," not a high-precision hallucination label. RAGTruth test (2,675 answers) numbers: false-supported on gold hallucinations about 1.6%, unsupported precision about 14%. Demonstrates `compute_hallucination_metrics`, `HallucinationConfig.include_partial_in_grounded`, and the `is_grounded` / `is_hallucinated` / `check_groundedness` convenience helpers. Source: `src/cite_right/hallucination.py`, `docs/concepts/hallucination-detection.md`.
- `openwiki/concepts/fact-verification.md` — `verify_facts` and claim decomposition in `src/cite_right/fact_verification.py`. Sentence-level tagging can hide a mixed sentence; `SimpleClaimDecomposer` keeps one claim per sentence; `SpacyClaimDecomposer` splits coordinated clauses and needs the spacy extra. Source: `src/cite_right/fact_verification.py`, `docs/concepts/fact-verification.md`.

### Configuration (4)

- `openwiki/configuration/citation-config.md` — `CitationConfig` and `CitationWeights` in `src/cite_right/core/citation_config.py`. Defaults: `supported_answer_coverage=0.6`, `top_k=3`, `min_final_score=0.0`, `min_answer_coverage=0.2`, `min_embedding_similarity=0.3`, `max_candidates_lexical=200`. Covers embedder-aware behavior and contradiction interaction. Source: `src/cite_right/core/citation_config.py`.
- `openwiki/configuration/presets.md` — the `CitationConfig` presets: `balanced()`, `strict()`, `permissive()`, `fast()`. Permissive still requires localized Smith-Waterman evidence and does not emit embedding-only citations. Source: `src/cite_right/core/citation_config.py`.
- `openwiki/configuration/tokenizers.md` — tokenizer options. `SimpleTokenizer` (default, Unicode NFKC, case-fold, original offsets kept), `HuggingFaceTokenizer`, `TiktokenTokenizer`. A custom tokenizer forces the lexical fallback path with no inverted index. Source: `src/cite_right/models/`.
- `openwiki/configuration/segmenters.md` — segmenter options. `SimpleSegmenter` / `SimpleAnswerSegmenter` (default), `SpacySegmenter` / `SpacyAnswerSegmenter`, `PySBDSegmenter`. A custom segmenter takes the lexical fallback path with no inverted index. Source: `src/cite_right/models/`.

### Integrations (3)

- `openwiki/integrations/langchain.md` — `from_langchain_documents` and `from_langchain_chunks` in `src/cite_right/integrations.py`; then call `align_citations` as usual. Notes on `LANGCHAIN_AVAILABLE` and the optional `langchain` extra. Source: `src/cite_right/integrations.py`.
- `openwiki/integrations/llamaindex.md` — `from_llamaindex_nodes` and `from_llamaindex_chunks` in `src/cite_right/integrations.py`; then call `align_citations` as usual. Notes on `LLAMAINDEX_AVAILABLE` and the optional `llama-index` extra. Source: `src/cite_right/integrations.py`.
- `openwiki/integrations/custom-sources.md` — non-framework source inputs. Build `SourceDocument` / `SourceChunk` directly, or use `from_dicts` in `src/cite_right/integrations.py`. Then call `align_citations` as usual. Covers chunk rebasing and offset invariants. Source: `src/cite_right/integrations.py`.

### Advanced Topics (4)

- `openwiki/advanced/multi-span-evidence.md` — `CitationConfig(multi_span_evidence=True)`. Prefer `evidence_spans` or `exact_evidence` for precise attribution. Legacy `evidence` / `char_start` / `char_end` stay a contiguous enclosing span. Off by default. Source: `src/cite_right/citations.py`, `docs/advanced/multi-span-evidence.md`.
- `openwiki/advanced/embedding-retrieval.md` — `pip install "cite-right[embeddings]==0.4.0"` then `SentenceTransformerEmbedder("all-MiniLM-L6-v2")`. Index-first still chooses lexical seeds; `_add_embedding_candidates` may add non-index windows. They still need Smith-Waterman. `retrieval_support` is not a `Citation` and does not flip status. Rust prepare still runs with the embedder on the simple tokenizer/segmenter path. Source: `src/cite_right/citations.py`, `src/cite_right/models/embedding_index.py`, `src/cite_right/models/sbert_embedder.py`.
- `openwiki/advanced/rust-acceleration.md` — the optional `cite_right._core` extension. `backend="auto" | "python" | "rust"`. Prepare, inverted index, and alignment stay on the hot path. Rust must match Python outputs. If `_core` is missing, candidate selection falls back to lexical prefilter. Smith-Waterman is not skipped for speed. Source: `src/cite_right/core/aligner_rust.py`, `src/cite_right/core/aligner_py.py`, `rust_core/`, `docs/advanced/rust-acceleration.md`.
- `openwiki/advanced/performance-tuning.md` — the index-first pipeline. Smith-Waterman runs on index hits plus optional embedding extras, not on every window. On the 50-case pack with no embedder, 0.4.0 p50 wall is about 12.4ms versus about 175.8ms in 0.3.1, roughly 14×. spp is 81.3% versus 83.4%. RAGTruth test quality on 2,675 answers matched 0.3.1. Embedder encoding is extra cost on top. Source: `bench_rust.py`, `docs/advanced/performance-tuning.md`.

## The Agent-Only Page Index

The pages under `openwiki/testing/` are for coding agents only. They never reach the public site. They are denser, and they may mention `openwiki/` paths and source-file paths.

- `openwiki/testing/pytest-markers.md` — the seven pytest markers registered in `tests/conftest.py` (`rust`, `spacy`, `embeddings`, `tiktoken`, `huggingface`, `pysbd`, `slow`), the `rust_core` and `rust_core_with_blocks` fixtures, and the `requires_rust` / `requires_rust_blocks` / `requires_spacy` / `requires_spacy_model` / `requires_embeddings` / `requires_tiktoken` / `requires_huggingface` / `requires_pysbd` skip decorators. Points at `src/cite_right/__init__.py` for the public surface, `src/cite_right/citations.py` for the pipeline, `src/cite_right/core/prepared_corpus.py` for prepare, `src/cite_right/contradiction.py` for the cheap contradiction check, and `rust_core/` for the extension. Source: `tests/conftest.py`, `pyproject.toml`.
- `openwiki/testing/contract-tests.md` — the Python vs Rust parity contract enforced by `tests/test_alignment_rust_parity.py`. Compares status, offsets, scores, matches, `match_blocks`, and best-candidate selection between `SmithWatermanAligner` and the `cite_right._core` extension. Covers `align_pair_details`, `align_pair_blocks_details`, and `align_best_details` parity, the equal-score coverage regression, the tie-breaker key, and the `_core` capability check. Points at `src/cite_right/core/aligner_py.py` and `src/cite_right/core/aligner_rust.py`. Source: `tests/test_alignment_rust_parity.py`, `src/cite_right/core/aligner_rust.py`, `src/cite_right/core/aligner_py.py`.

## Picking The Entry Page

Use this table to pick the right starting point for a new task. Reader questions and pipeline questions take different paths.

| You need to… | Open |
| --- | --- |
| Answer a reader / user question, or write a public page | `openwiki/index.md` (Home) and the 16 public pages above |
| Understand or modify the pipeline (segment, tokenize, index, align, rank, contradict, status) | `openwiki/concepts/how-it-works.md` |
| Change a config default, add a config field, or update a preset | `openwiki/configuration/citation-config.md` then `openwiki/configuration/presets.md` |
| Add or change a tokenizer / segmenter, or wire a custom one | `openwiki/configuration/tokenizers.md` and `openwiki/configuration/segmenters.md` |
| Add a LangChain / LlamaIndex / custom-source integration | `openwiki/integrations/langchain.md`, `openwiki/integrations/llamaindex.md`, `openwiki/integrations/custom-sources.md` |
| Change the Rust extension, the `backend=` switch, or the fallback | `openwiki/advanced/rust-acceleration.md` and `openwiki/testing/contract-tests.md` |
| Change the inverted index, multi-span evidence, embedder recall, or any "advanced" feature | `openwiki/advanced/multi-span-evidence.md`, `openwiki/advanced/embedding-retrieval.md`, `openwiki/advanced/performance-tuning.md` |
| Touch a test that depends on an optional extra (Rust, spaCy, embeddings, tiktoken, huggingface, pysbd) | `openwiki/testing/pytest-markers.md` |
| Touch the Python / Rust parity contract | `openwiki/testing/contract-tests.md` |
| Update the public voice or the invariants list | `openwiki/INSTRUCTIONS.md` (not under `openwiki/`) |

## What You Must Not Do On Public Pages

These are the rules every public page has to honor. The full list is in `openwiki/INSTRUCTIONS.md`; the abbreviated version is below.

- Use public-to-public relative links only. No `/openwiki/` or `openwiki/` prefixes in reader-facing hrefs.
- Never link a public page to `openwiki/testing/` or to this routing page.
- Never use the words `wiki`, `MkDocs`, or `OpenWiki` in prose. Never contrast two documentation sites. Never write "see the wiki" or "see MkDocs".
- Use the install name `cite-right` and the import package `cite_right` exactly. The product is Cite-Right in prose.
- Address the reader as `you`. One H1, short opening paragraph, `##` Title Case sections, small runnable example before theory.
- Status literals are exactly `"supported"`, `"partial"`, `"unsupported"`. The literal is `"partial"`, never `"partially_supported"`. `"partial"` means low answer coverage **or** contradiction.
- The only allowed measured numbers are: 50-case pack with no embedder, p50 wall about 12.4ms versus about 175.8ms in 0.3.1 (roughly 14×), spp 81.3% versus 83.4%; RAGTruth test 2,675 answers, false-supported on gold hallucinations about 1.6%, unsupported precision about 14%. Anything else is omitted.
- Do not write that the pipeline skips Smith-Waterman for speed. Do not write that Rust prepare is skipped when an embedder is set.
- Do not document `evaluation/`, hill-climb, or RAGTruth tables beyond the two allowed measured lines.
- Do not generate a second public tree (`architecture/`, `operations/`, `workflows/`). Extra depth for agents belongs under `openwiki/testing/` or not at all.

## Invariants To Preserve

These must stay true on every generated page that touches them.

- Public API: `align_citations`, `PreparedCitationCorpus`. Default `supported_answer_coverage` is 0.6.
- Half-open `char_start` / `char_end`. Evidence equals the sliced source after chunk rebasing: `source.text[citation.char_start:citation.char_end] == citation.evidence`.
- Status is exactly `"supported"`, `"partial"`, or `"unsupported"`. Never `"partially_supported"`. `"partial"` is low coverage or contradiction.
- Default / Rust path: inverted index, rare-token intersect, Smith-Waterman localizes on hits. The index chooses windows. Smith-Waterman still localizes.
- Fallback: no Rust, or a custom tokenizer / segmenter, then `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and `_select_candidates` uses lexical selection.
- Embedder: `_add_embedding_candidates` may add non-index windows. Those still need Smith-Waterman. `retrieval_support` is not a `Citation` and does not flip status.
- Rust prepare still runs with an embedder when `SimpleTokenizer` and `SimpleSegmenter` are in use.
- Contradiction (negation, number, leftover n-gram slot, entity swap) downgrades to `"partial"` with citations, never `"unsupported"`. The check uses the full candidate passage, not only truncated Smith-Waterman evidence. Source "The vaccine is safe and effective." and answer "The vaccine is not safe." is `"partial"` with citations.
- Content-word overlap can emit a citation when sequential Smith-Waterman coverage is low.
- Data2txt `field:value` gets a second Smith-Waterman pass with `gap_score=0`. Invented fields stay `"unsupported"`.

## Pointers

- `openwiki/INSTRUCTIONS.md` — the user-authored brief. Public Paths list, invariants, voice rules, testing-page rules. Not generated.
- `mkdocs.yml` — the nav that the public site renders. The 16 public pages plus the auto-generated Home and the `docs/api/` API reference map exactly onto `openwiki/` paths.
- `docs/index.md` — the source of truth for the public Home voice. Match the voice of the current public pages.
- `docs/concepts/how-it-works.md`, `docs/concepts/hallucination-detection.md`, `docs/advanced/rust-acceleration.md` — additional voice anchors named in `openwiki/INSTRUCTIONS.md`.
- `src/cite_right/__init__.py` — the public surface. `align_citations`, `PreparedCitationCorpus`, `SourceDocument`, `SourceChunk`, `CitationConfig`, `CitationWeights`, the segmenter / tokenizer / embedder classes, the presets, and the helpers used on public pages.
- `src/cite_right/citations.py` — the pipeline. Per-span segmentation, candidate selection, Smith-Waterman, ranking, contradiction, status.
- `src/cite_right/core/prepared_corpus.py` — prepare. Windowing, tokenization, IDF, inverted index. The Rust-vs-Python prepare branch lives here.
- `src/cite_right/core/citation_config.py` — `CitationConfig` and `CitationWeights`. Defaults and presets.
- `src/cite_right/core/aligner_py.py` and `src/cite_right/core/aligner_rust.py` — the two Smith-Waterman backends. The Rust wrapper performs the `align_pair_blocks_details` capability check.
- `src/cite_right/contradiction.py` — the cheap contradiction check.
- `rust_core/` — the Rust extension source. `Cargo.toml` and `rust_core/src/`. The entry points `align_pair`, `align_pair_details`, `align_pair_blocks_details`, `align_best`, `align_best_details`, `align_topk_details`, `align_batch_details`, `align_batch_blocks_details`, `rust_tokenize_and_prepare`, `InvertedIndex`, `PreparedCorpus`, `align_batch_with_match_blocks`, `rust_build_citations_fast` are declared in `src/cite_right/_core.pyi`.
- `tests/conftest.py` — the seven pytest markers, the `_rust_available` / `_rust_has_blocks_details` / `_spacy_available` / `_spacy_model_available` / `_embeddings_available` / `_tiktoken_available` / `_huggingface_available` / `_pysbd_available` probes, and the `rust_core` / `rust_core_with_blocks` / `spacy_nlp` fixtures.
- `tests/test_alignment_rust_parity.py` — the parity contract enforced between `SmithWatermanAligner` and `RustSmithWatermanAligner`. Documented in `openwiki/testing/contract-tests.md`.
- `pyproject.toml` — `[project.optional-dependencies]` for `spacy`, `embeddings`, `tiktoken`, `huggingface`, `pysbd`, `langchain`, `llamaindex`; `[tool.pytest.ini_options]` notes that markers are registered in conftest rather than re-declared.
- `scripts/publish_openwiki_to_docs.py` — copies the public paths from `openwiki/` into `docs/` after a successful update. The two `openwiki/testing/` pages and this routing page are not published.
- `openwiki/advanced/rust-acceleration.md` — public-facing Rust extension guide, including the `backend="auto" | "python" | "rust"` switch and the fallback when `_core` is missing.
- `openwiki/concepts/how-it-works.md` — where Smith-Waterman sits in the pipeline.
- `openwiki/testing/contract-tests.md` — the parity contract test reference.
- `openwiki/testing/pytest-markers.md` — the other testing reference page.
