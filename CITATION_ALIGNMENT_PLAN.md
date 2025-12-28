# Citation Alignment Improvement Plan (RAG)

Goal: align **multi-paragraph generated answers** to **retrieved sources** and return **usable citations** with **deterministic**, **character-accurate** evidence spans (plus optional semantic matching via embeddings).

All offsets are **0-based half-open**: `[start, end)`.

---

## 0) Scope + invariants

- [x] Keep Python as the correctness oracle; Rust must match outputs exactly given the same inputs/candidates.
- [x] Every citation includes **absolute** `char_start/char_end` into the **original source document** (not just a chunk).
- [x] Deterministic ordering/tie-breaks are defined and implemented identically across Python/Rust.
- [x] Core install stays lightweight; heavy NLP/ML lives behind extras.

---

## 1) Public API + data model (citations are first-class)

- [x] Add `SourceDocument` type: `id`, `text`, `metadata`.
- [x] Add `SourceChunk` type: `source_id`, `text`, `doc_char_start`, `doc_char_end`, `metadata`.
- [x] Add `AnswerSpan` type: `text`, `char_start`, `char_end`, `kind` (`sentence|clause|paragraph`).
- [x] Add `Citation` type:
  - [x] `score` (final score)
  - [x] `source_id` (stable id)
  - [x] `source_index` (for backward compatibility / convenience)
  - [x] `candidate_index` (which passage/chunk/window)
  - [x] `char_start/char_end` (absolute in source doc)
  - [x] `evidence` (source slice)
  - [x] `components` (debug: alignment score, coverage, embed sim, etc.)
- [x] Add `SpanCitations` type: `answer_span` + `citations` + `status` (`supported|partial|unsupported`).
- [x] Add a new public entry point:
  - [x] `align_citations(answer: str, sources: Sequence[str|SourceDocument|SourceChunk], *, config=..., answer_segmenter=..., source_segmenter=..., tokenizer=..., embedder=..., backend="auto") -> list[SpanCitations]`
- [x] Remove legacy `cite_right.align()` API to avoid ambiguity.

**Acceptance**
- [x] For chunk inputs, returned offsets map to the original doc via `doc_char_start + local_offset`.
- [x] `align_citations` returns per-answer-span citations (not just global top-k).

---

## 2) Answer segmentation (what gets cited)

- [x] Implement `AnswerSegmenter` returning `list[AnswerSpan]` with absolute offsets.
- [x] Default behavior: sentence segmentation; preserve paragraph boundaries for grouping.
- [x] Optional: clause segmentation (spaCy or heuristic), but never required for core usage.

**Acceptance**
- [x] Works on multi-paragraph answers with mixed punctuation and newlines.
- [x] Offsets slice back to exact answer text.

---

## 3) Source passage generation (what we match against)

- [x] Support two input modes:
  - [x] **Documents**: segment into sentences/clauses, then create passage windows.
  - [x] **Retrieved chunks**: treat each chunk as a base candidate; optionally window inside the chunk.
- [x] Add sentence-windowing:
  - [x] window size `W` (e.g. 1–3 sentences)
  - [x] stride `S` (e.g. 1 sentence)
  - [x] each window has absolute doc offsets
- [ ] Add deduplication/normalization for near-identical candidates (optional).

**Acceptance**
- [x] Evidence spanning sentence boundaries can be returned as a single contiguous span.
- [x] Passage offsets always map back to the original source text.

---

## 4) Tokenization + normalization upgrades (matching-quality foundation)

- [x] Replace the current “alnum runs only” tokenizer with a configurable tokenizer family:
  - [x] keep `token_spans` into the original text
  - [x] handle decimals and numbers (`5.2`, `1,200`, `34%`)
  - [x] handle hyphens (`state-of-the-art`) and apostrophes (`company’s`, `don't`)
  - [x] handle currency/units (`$5.2B`, `5.2 billion`)
- [x] Add a configurable normalization pipeline:
  - [x] NFKC + casefold (current behavior)
  - [x] numeric normalization (opt-in): `5.20`→`5.2`, `34%`→`34 percent`
  - [ ] light morphology (opt-in): stemming/lemmatization behind an extra (e.g. spaCy)
- [x] Ensure tokenization is deterministic and stable across OS/Python.

**Acceptance**
- [x] “Same meaning, different formatting” cases match reliably (numbers/units/punctuation).
- [x] Token spans remain character-accurate in the original text.

---

## 5) Candidate generation (scale to many sources)

### 5.1 Lexical prefilter (core, fast)

- [x] Build per-candidate token sets / counters.
- [x] Compute lightweight match signals for each `AnswerSpan`:
  - [x] token overlap
  - [x] weighted overlap via IDF computed from the retrieved set
- [ ] Optional extra: `rapidfuzz` string similarity prefilter.
- [x] Select top-N candidates for expensive alignment.

**Acceptance**
- [x] Runtime scales to 50+ sources with long text without aligning against every candidate.

### 5.2 Embedding retrieval (optional but integrated)

- [x] Provide an `Embedder` interface for sentence/passage embeddings.
- [x] Add `EmbeddingIndex` built per call (cacheable by caller):
  - [x] embed all candidates once
  - [x] embed each answer span
  - [x] cosine similarity top-N candidates
- [ ] Allow passing precomputed embeddings from an upstream RAG retriever.

**Acceptance**
- [x] Paraphrase cases are recovered when embeddings are enabled.
- [x] Without embeddings, the library still works (just less recall on paraphrases).

---

## 6) Matching engine (alignment outputs that support citations)

- [x] Extend alignment outputs beyond `(score, token_start, token_end)`:
  - [x] `matched_token_count`
  - [x] `answer_coverage = matched / answer_tokens`
  - [x] `evidence_coverage = matched / evidence_tokens`
  - [x] `normalized_alignment_score` (length-normalized)
  - [ ] (optional) matched token indices for debugging and future multi-span extraction
- [x] Decide whether citations can be **multi-span** (advanced) or contiguous-only (start contiguous-only).
  - [x] Implement `Citation.evidence_spans` and `CitationConfig(multi_span_evidence=True)` (backward compatible with contiguous `Citation.evidence`).
- [ ] Add guardrails to prevent boilerplate matches:
  - [x] minimum `answer_coverage`
  - [ ] minimum `unique_token_match` (optional)

**Acceptance**
- [x] High-confidence citations require meaningful coverage, not just a few common tokens.

---

## 7) Final scoring + ranking (deterministic, configurable)

- [x] Combine signals into a final citation score (deterministic function):
  - [x] normalized alignment score
  - [x] answer coverage
  - [x] evidence coverage
  - [x] embedding similarity (if enabled)
  - [x] lexical/IDF score
- [x] Define and document deterministic tie-break rules for citations.
- [x] Rank and return top-k citations per `AnswerSpan`.
- [ ] Add optional diversity constraints:
  - [x] cap citations per source doc
  - [x] deduplicate near-identical evidence spans across candidates

**Acceptance**
- [x] Results are stable across runs and across Python/Rust backends.
- [x] One answer sentence can cite multiple sources when appropriate.

---

## 8) Rust acceleration (parity-first)

- [x] Extend Rust APIs to return top-k matches (not only best-1):
  - [x] `align_topk(seq1, seqs, ...) -> Vec[(score, index, token_start, token_end)]`
- [x] Ensure Rayon reductions use a deterministic comparator that matches Python.
- [x] Keep GIL released during compute.
- [x] Add parity tests for top-k and tricky tie cases.

**Acceptance**
- [x] Rust and Python produce identical citation selections given identical candidates and scoring config.

---

## 9) Test plan (RAG-realistic)

- [x] Multi-paragraph answers with per-sentence citations across 5/10/20/40/50 sources.
- [x] Sources containing lots of unrelated text and repeated boilerplate.
- [x] Numeric/unit formatting tests (`$5.2B` vs `5.2 billion`, `34%` vs `34 percent`).
- [x] Evidence spanning sentence boundaries (windowing) tests.
- [x] Determinism tests: repeated runs return identical outputs.
- [x] Embedding paraphrase tests (uses lightweight test embedder).
- [x] Property-style regression tests for offsets: evidence slice must equal `source[char_start:char_end]`.

---

## 10) Packaging + CI

- [x] Keep extras split:
  - [x] `spacy` (segmentation + optional lemmatization)
  - [x] `embeddings` (sentence-transformers)
  - [ ] optional `rapidfuzz` extra if used
- [x] CI jobs:
  - [x] core path (no extras)
  - [x] spacy path (model downloaded)
  - [ ] embeddings path (optional; can be nightly if too heavy)
- [x] Wheels workflow smoke-tests `align_citations`.

---

## Open questions (decide early)

- [ ] Citation granularity: per **sentence**, per **clause**, or both?
- [ ] Input format: full documents, retrieved chunks (with offsets), or mixed?
- [x] Evidence format: contiguous-only (v1) vs multi-span (v2)?
- [ ] Ranking philosophy: prefer earlier `char_start` across documents (current policy) vs prefer `source_rank` from retriever?
