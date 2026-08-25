---
type: Repository guide
title: cite-right wiki brief
description: Scope and invariants for the cite-right code wiki. Written for coding agents.
tags: [cite-right, citations, agents]
---

Write this wiki for coding agents that need to call cite-right correctly, not for a marketing tour. Prefer `src/cite_right/__init__.py` and the types it re-exports over evaluation code, benches, or internal helpers.

## Product

cite-right aligns generated answer text to source documents and returns character-accurate citations. The Python package `cite_right` is the public API. An optional Rust extension (`cite_right._core`) is the fast path; `backend="auto"` uses it when present.

## Public API to document

Primary:

- `align_citations(answer, sources, *, config=None, tokenizer=None, answer_segmenter=None, source_segmenter=None, embedder=None, backend="auto")` → `list[SpanCitations]`
- `PreparedCitationCorpus.from_sources(...)` then `.align(answer)` when many answers share the same sources
- `SourceDocument(id, text)` for full docs; `SourceChunk(source_id, text, doc_char_start, doc_char_end)` when retrieval already chunked (offsets are rebased onto the original document)
- `CitationConfig` / `CitationWeights`; default `supported_answer_coverage` is `0.6`

Also public, document only as needed: `compute_hallucination_metrics`, `verify_facts`, convenience helpers (`annotate_answer`, `check_groundedness`, `format_with_citations`, `get_citation_summary`, `is_grounded`, `is_hallucinated`), tokenizers, segmenters, `SentenceTransformerEmbedder`, LangChain/LlamaIndex adapters.

Do not treat `evaluation/`, hill-climb search spaces, or RAGTruth-style bench tables as product documentation. Those numbers drift and they are not the API.

## Citation statuses

Each `SpanCitations.status` is exactly one of `supported`, `partial`, `unsupported`. Status comes from the best exact citation, not from embedding similarity.

- `supported`: citations exist and the best citation's `answer_coverage` is `>= supported_answer_coverage`
- `partial`: citations exist but coverage is below that threshold, **or** a contradiction was detected (see below)
- `unsupported`: no citation survived filtering

`retrieval_support` is a retrieval-only hint. A high embedding score that never localizes with Smith-Waterman is **not** a `Citation` and must not flip status.

Every `Citation` has `char_start` / `char_end` as a Python half-open interval. After chunk rebasing, slicing the logical source text at those offsets equals `citation.evidence`.

## Index-first retrieval, then Smith-Waterman localization

0.4.0 retrieves with an inverted index over source windows (rare-token intersect). Smith-Waterman runs only on those hits. The index chooses which windows get alignment; SW still localizes the evidence span. The public API is unchanged (`align_citations`, `PreparedCitationCorpus`).

## Rust prepare still runs with an embedder

`PreparedCitationCorpus.from_sources(..., embedder=...)` still takes the Rust prepare path when the simple tokenizer/segmenter are in use. The embedding index is built on those prepared candidates. Embedding-only `retrieval_support` still respects `min_embedding_similarity`. Lexical scores are filled only for inverted-index seeds.

## Contradiction stays `partial`

A cheap contradiction check (negation, number mismatch, leftover n-gram slot, entity swap) downgrades a span to `partial`, never `unsupported`. The check uses the full candidate passage, not only the truncated SW evidence span. Shared tokens that would otherwise bless a contradictory claim as `supported` become `partial` and still keep the citation.

Example: source "The vaccine is safe and effective." and answer "The vaccine is not safe." → `partial` with citations, not `unsupported`.

## CI model rotation (not wiki content)

Wiki generation in GitHub Actions uses OpenRouter free models (`:free` in the model id). `scripts/openwiki_pick_model.py` ranks them (prefer `tools`, then Artificial Analysis `coding_index` if present, else `context_length`, then `created`). `scripts/openwiki_update.sh` retries the next model on 429/402/rate-limit. Do not document a single paid model id as required.
