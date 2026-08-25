---
type: testing
title: Contract and parity tests
description: High-signal tests that pin status semantics, char-span invariants, Rust↔Python parity, and edge cases across the cite-right pipeline.
tags: [testing, citation-alignment, rust, python, parity, smith-waterman]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-326c42b0fd50f852cd59c6ad
    resource: repo://tests/test_alignment_py.py
  - id: openwiki-source-811d84c9631d27a47d6421e0
    resource: repo://tests/test_alignment_rust_parity.py
  - id: openwiki-source-cce21eea781802cf8abb7d2e
    resource: repo://tests/test_citations_api.py
  - id: openwiki-source-1413c674e60d538eeaadf96c
    resource: repo://tests/test_citations_embedding_only_edge_cases.py
  - id: openwiki-source-c126bef8ff7e71bc028699de
    resource: repo://tests/test_citations_retrieval_support.py
  - id: openwiki-source-80f294a40e89a55a58070064
    resource: repo://tests/test_contradiction_detection.py
  - id: openwiki-source-f65a7e483f703a2b163781db
    resource: repo://tests/test_data2txt_support.py
  - id: openwiki-source-61cbfe170d8c82a627f10456
    resource: repo://tests/test_inverted_index.py
  - id: openwiki-source-5ddf2e3b4fca9c3c6270fdcf
    resource: repo://tests/test_rust_prepare_with_embeddings.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

Contract tests in cite-right validate the behavioral contracts that the rest of the system depends on: correctness of the Smith-Waterman algorithm, byte-for-byte parity between Rust and Python backends, status determination semantics, contradiction downgrades, retrieval support boundaries, and structured field parsing. These tests are deliberately sensitive — they fail loudly when invariants drift, making them the first line of defense against regressions.

## Smith-Waterman alignment (`tests/test_alignment_py.py`)

Pure-Python implementation tests that pin the contract for local sequence alignment.

### Core correctness

| Test | What it pins |
|------|-------------|
| `test_alignment_basic` | Exact subsequence match returns correct score, token boundaries |
| `test_alignment_prefers_earlier_start` | Equal-score tie-breaking favors earlier `token_start` |
| `test_alignment_no_match` | Zero-score return when no alignment exists |
| `test_alignment_exact_match` | Perfect match on identical sequences |
| `test_alignment_partial_match` | Subspan extraction within a longer candidate |
| `test_alignment_single_element_match` | Single-token alignment boundary handling |

### Edge cases

| Test | What it pins |
|------|-------------|
| `test_alignment_empty_query` | Empty query produces zero score, zero-length span |
| `test_alignment_empty_target` | Empty target produces zero score, zero-length span |
| `test_fill_matrix_tracks_single_best_endpoint` | Fill phase returns one winning endpoint, not all max cells |
| `test_reduced_state_fill_tracks_best_endpoint_for_default_path` | Reduced-state matrix dimensions match full-state expectations |
| `test_align_batch_preserves_single_alignment_results_in_order` | Batch results are ordered identically to sequential `align` calls |

### Match block traceback

| Test | What it pins |
|------|-------------|
| `test_alignment_prefers_more_matches_across_equal_score_endpoints` | Traceback explores all equal-score predecessor cells, selects path with most matches |
| `test_alignment_prefers_more_matches_within_single_optimal_endpoint` | Same-cell equal-score neighbors resolved by match count |
| `test_default_path_matches_detailed_path_without_match_blocks` | Reduced-state and full-state paths agree on span boundaries |

## Rust↔Python parity (`tests/test_alignment_rust_parity.py`)

Parity tests that verify both backends produce identical results for every variant of every operation. These are the tests that would break if the Rust implementation diverged from the Python reference.

### `requires_rust` and `requires_rust_blocks` markers

The `conftest.py` fixtures gate these tests on Rust extension availability:

```python
requires_rust = pytest.mark.skipif(not _rust_available(), reason="Rust extension not built")
requires_rust_blocks = pytest.mark.skipif(
    not _rust_has_blocks_details(),
    reason="Rust extension missing align_pair_blocks_details",
)
```

### Core parity contracts

| Test | What it pins |
|------|-------------|
| `test_rust_parity` | `align_pair_details` output matches Python `align()` on score, token boundaries, query boundaries, match count |
| `test_rust_parity_for_equal_score_more_matches_case` | Equal-score traceback regression fixture: Rust block collector matches Python match-block output |
| `test_rust_wrapper_align_best_details_matches_extension` | `RustSmithWatermanAligner.align_best_details` wrapper returns correct tuple |
| `test_rust_align_pair_blocks_details_matches_python_blocks` | Rust block entrypoint output matches Python `match_blocks` field |

### Best-match selection parity

| Test | What it pins |
|------|-------------|
| `test_rust_align_best_matches_python_selection` | Rust `align_best_details` picks the same candidate as Python's full sort-key comparison across all candidates |
| `test_rust_align_best_empty_returns_none` | Empty candidate list returns `None` in Rust (not a crash or empty tuple) |

### Batch operations

| Test | What it pins |
|------|-------------|
| `test_rust_wrapper_align_batch_matches_python_ordered_results` | Rust batch wrapper preserves input order and returns full `Alignment` objects |
| `test_rust_block_and_non_block_entrypoints_share_alignment` | Block-collection mode does not alter the chosen span vs. non-block mode |

## Citation API (`tests/test_citations_api.py`)

High-level API contract tests for `align_citations` and `PreparedCitationCorpus`.

### Status semantics docstring check

`test_how_it_works_describes_status_using_answer_coverage` is the single test that pins the documentation to the correct behavioral rule:

```python
def test_how_it_works_describes_status_using_answer_coverage() -> None:
    docs_path = Path(__file__).resolve().parents[1] / "docs/concepts/how-it-works.md"
    docs_text = docs_path.read_text(encoding="utf-8")

    assert "best citation score" not in docs_text
    assert "best citation\'s answer coverage" in docs_text
```

This test fails if documentation ever refers to "best citation score" — the correct concept is `answer_coverage` of the best citation. See [`/openwiki/concepts/status-semantics.md`](/openwiki/concepts/status-semantics.md) for the full semantics contract.

### Backend selection

| Test | What it pins |
|------|-------------|
| `test_align_citations_auto_falls_back_when_rust_core_lacks_details` | `backend="auto"` falls back to Python when Rust extension lacks detailed alignment API |
| `test_align_citations_rust_backend_requires_detailed_core` | `backend="rust"` raises `RuntimeError` if detailed API is missing |

### Batch alignment dispatch

| Test | What it pins |
|------|-------------|
| `test_align_citations_uses_batch_alignment_api` | Aligner with `align_batch` method is called once for a batch, not per-candidate |
| `test_align_citations_accepts_legacy_single_alignment_api` | Aligner with only `align` method is accepted and called per-candidate |

### Prepared corpus alignment

| Test | What it pins |
|------|-------------|
| `test_prepared_corpus_align_resolves_default_aligner_once` | Default aligner is resolved once and reused across spans |

### Citation ranking tie-breaking

| Test | What it pins |
|------|-------------|
| `test_rank_and_limit_citations_prefers_source_order_in_equal_score_ties` | `prefer_source_order=True` breaks equal-score ties by source index ascending |
| `test_rank_and_limit_citations_prefers_earlier_position_when_source_order_disabled` | Default: equal-score ties broken by earlier `char_start` |
| `test_rank_and_limit_citations_dedupes_by_source_and_evidence_span_tuple` | Duplicate citations from the same source with identical evidence spans are deduplicated |

## Retrieval support and embedding-only edge cases (`tests/test_citations_retrieval_support.py`, `tests/test_citations_embedding_only_edge_cases.py`)

### Retrieval support does not flip status

These tests confirm the invariant: `retrieval_support` is surfaced for transparency but never upgrades status to `supported`.

| Test | What it pins |
|------|-------------|
| `test_align_citations_embedding_only_returns_retrieval_support_only` | Embedding-only path (zero lexical candidates) returns `status="unsupported"` with `retrieval_support` populated |
| `test_align_citations_embedding_support_does_not_upgrade_exact_status` | Semantic match to a document still marked `unsupported` when no lexical alignment succeeds |
| `test_align_citations_retrieval_support_respects_own_limit` | `max_retrieval_support` caps the list independently of citation count |
| `test_align_citations_lexical_only_returns_retrieval_support_when_alignment_fails` | Failed lexical alignment falls back to retrieval support with `lexical_score > 0` |

### Token guard for strict mode

Strict mode enforces that all answer tokens appear in evidence. The token guard prevents false positives from alignment with a mismatched slot count.

| Test | What it pins |
|------|-------------|
| `test_answer_token_guard_stops_after_all_required_tokens_are_found` | Token iterator is not exhausted after all answer tokens are matched |
| `test_answer_token_guard_exact_sequence_avoids_frequency_map` | Exact token lists bypass `Counter` allocation |
| `test_answer_token_guard_exact_lists_use_native_comparison` | `NoPythonIterationList` subclasses are compared natively, not via Python iteration |
| `test_answer_token_guard_trusts_complete_exact_alignment` | Full exact alignment does not rescan tokens (optimization) |
| `test_answer_token_guard_does_not_trust_custom_match_count` | Custom aligner claiming high match count does not bypass token guard |

### Numeric and negation strictness

| Test | What it pins |
|------|-------------|
| `test_strict_exact_citation_rejects_numeric_token_mismatch` | "125 days" vs. "124 days" produces `unsupported`, not `supported` |
| `test_strict_exact_citation_rejects_negation_token_mismatch` | "shall make every law" vs. "shall make no law" produces `unsupported` |
| `test_strict_exact_citation_does_not_split_u_s_abbreviation_into_supported_stub` | U.S. abbreviation is not split into partial tokens that could incorrectly support unrelated claims |

## Contradiction detection (`tests/test_contradiction_detection.py`)

Tests for the five contradiction checks that downgrade `supported` to `partial` when the cited passage contradicts the answer. See [`/openwiki/concepts/contradiction-detection.md`](/openwiki/concepts/contradiction-detection.md) for the full architecture.

### Contradiction downgrades

| Test | What it pins |
|------|-------------|
| `test_negation_mismatch_marked_unsupported` | Negation mismatch → `status="partial"` (not `supported`) |
| `test_affirmative_match_is_supported` | Matching affirmative → `status="supported"` |
| `test_number_mismatch_not_supported` | Number mismatch → `status="partial"` |
| `test_matching_numbers_are_supported` | Matching numbers → `status="supported"` |
| `test_entity_mismatch_not_supported` | Entity swap → `status="partial"` |

### Issue #48 regression fixtures

Issue #48 discovered that Smith-Waterman truncation can leave behind n-gram "leftovers" that attach to the wrong semantic slot. The contradiction check must operate on the full passage, not the truncated span.

| Test | What it pins |
|------|-------------|
| `test_issue48_number_leftover_rebounds` | "10 of which came in the first half" vs. "10 rebounds": leftover "10" must not bless the mismatch; status remains `partial` |
| `test_issue48_entity_swap_india_france` | Shared "opposed" / "involvement" must not bless India↔France entity swap; status is `partial` |
| `test_issue48_temporal_polarity_bc_vs_ago` | Truncated span "300 years" vs. full passage "300 years BC" must catch "ago" vs. "BC" polarity flip via contradiction check |
| `test_issue48_polarity_flip_oppose_vs_urged` | "oppose laws" vs. "urged laws": leftover "laws" + "prohibit" must not bless the flip |
| `test_issue48_extractive_near_copy_stays_supported` | Extractive near-copy remains `supported` |
| `test_issue48_extractive_subset_stays_supported` | Faithful subset ("18 points" from "18 points, 10 of which...") remains `supported` |

### Passage vs. truncated span contract

| Test | What it pins |
|------|-------------|
| `test_check_contradiction_uses_passage_not_truncated_span` | `check_contradiction` with truncated evidence flags `ago` vs. `BC`; full passage check returns `True`; truncated check on truncated span also returns `True` |

## Data2txt structured field support (`tests/test_data2txt_support.py`)

Tests for paraphrase support of field:value lines in structured sources. The `_looks_like_structured_source` heuristic detects structured input and relaxes tokenization for field-value rewrites.

### Field:value paraphrase contracts

| Test | What it pins |
|------|-------------|
| `test_business_stars_and_wifi_field_rewrite_is_supported` | "business_stars: 4.5" + "attributes.WiFi: free" rewritten as "4.5 stars" + "free WiFi" → `supported` or `partial` |
| `test_hours_field_rewrite_is_supported` | "hours.Monday: 9:0-17:0" rewritten as "Monday–Friday 9:00 AM–5:00 PM" → not `unsupported` |
| `test_null_wifi_with_invented_amenity_stays_unsupported` | `attributes.WiFi: null` + invented "free Wi-Fi" → `unsupported` |
| `test_platform_mismatch_not_fully_supported` | Star value grounded; "on Google" invented → not `supported` |
| `test_mixed_field_source_with_review_text` | Structured fields + prose review → field content cited correctly |
| `test_field_rewrite_still_works_beside_unrelated_prose` | Correct source selected when structured content coexists with unrelated prose |
| `test_structured_leniency_does_not_relax_prose_coverage` | Structured source leniency does not reduce prose coverage requirements |

### Heuristic boundary

| Test | What it pins |
|------|-------------|
| `test_field_value_heuristic_requires_multiple_field_lines` | Two `key: value` lines → structured; "Headline: X\nBody text" → prose |

## Inverted index (`tests/test_inverted_index.py`)

Tests for the Rust-based inverted index used to seed lexical candidate selection efficiently.

### Index construction and persistence

| Test | What it pins |
|------|-------------|
| `test_inverted_index_is_built_with_rust_prepare` | `PreparedCitationCorpus` with `use_rust=True` creates `rust_corpus` with `query_index` method |
| `test_rust_corpus_stays_in_rust` | The same `rust_corpus` Python object is reused across queries (no per-query rebuild) |

### Index behavior

| Test | What it pins |
|------|-------------|
| `test_inverted_index_improves_retrieval` | Index-seeded retrieval finds relevant candidates among 100 similar candidates |
| `test_inverted_index_never_returns_empty_when_tokens_exist` | Query with tokens present in corpus never produces empty seeds |
| `test_inverted_index_uses_intersection` | Rare-token query returns fewer candidates than full corpus; unique-token passage is in results |

### Rust/Python fallback

| Test | What it pins |
|------|-------------|
| `test_python_fallback_without_index` | `use_rust=False` produces `rust_corpus=None`; alignment still works |

### Token lazy-loading

| Test | What it pins |
|------|-------------|
| `test_rust_corpus_with_embedder` | Rust prepare runs even when an embedder is provided; embeddings built afterward |
| `test_prepare_does_not_fetch_all_tokens` | Candidates have empty `token_ids`, `token_spans`, `token_set` at prepare time; tokens fetched on-demand during alignment |

## Rust prepare with embeddings (`tests/test_rust_prepare_with_embeddings.py`)

Tests for the combined Rust-inverted-index + embedding-index prepare path.

### Embedding dimension compatibility

| Test | What it pins |
|------|-------------|
| `test_rust_prepare_with_dummy_embedder_dim8` | Rust prepare succeeds with 8-dimensional embedder; `embedding_index` and `idf` populated |
| `test_rust_prepare_with_dummy_embedder_dim384` | Rust prepare succeeds with 384-dimensional embedder; vector shape matches |

### Rust vs. Python prepare parity

| Test | What it pins |
|------|-------------|
| `test_rust_prepare_candidate_count_close_to_python` | Rust and Python prepare produce candidate counts within 20% (segmentation may differ slightly) |
| `test_rust_prepare_citation_fixture_still_works` | Full fixture: finance source cited as `supported` or `partial`; irrelevant source excluded |

### Tokenizer fallback

| Test | What it pins |
|------|-------------|
| `test_custom_tokenizer_falls_back_to_python` | `TiktokenTokenizer` (non-simple) triggers Python fallback; `rust_corpus=None` |

### Build-time tracking

| Test | What it pins |
|------|-------------|
| `test_rust_prepare_embedding_build_time_tracked` | `embedding_build_time_ms >= 0.0` when Rust prepare runs with embedder |

### Backward compatibility

| Test | What it pins |
|------|-------------|
| `test_rust_prepare_without_embedder_still_works` | Rust prepare without embedder produces `rust_corpus` with candidates and IDF weights |
