---
type: testing-reference
title: Rust/Python Contract Tests
description: Agent-only reference for the Python vs Rust parity contract enforced by tests/test_alignment_rust_parity.py. Compares status, offsets, scores, matches, match_blocks, and best-candidate selection between SmithWatermanAligner and the cite_right._core extension. Points at src/cite_right/core/aligner_py.py and src/cite_right/core/aligner_rust.py.
tags: [contract-tests, rust, python, smith-waterman, parity, alignment, _core, match-blocks, best-match, tie-breaking, skip, fixtures]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-0b1b3279f2fdef17b4081691
    resource: repo://src/cite_right/_core.pyi
  - id: openwiki-source-565dc547e636f5aa89fb94bd
    resource: repo://src/cite_right/core/aligner_py.py
  - id: openwiki-source-edff227b75f84c03f46d0ad0
    resource: repo://src/cite_right/core/aligner_rust.py
  - id: openwiki-source-f0a6e7dc03522b2682f88655
    resource: repo://tests/conftest.py
  - id: openwiki-source-811d84c9631d27a47d6421e0
    resource: repo://tests/test_alignment_rust_parity.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Rust/Python Contract Tests

This page is the agent-only reference for the parity contract that `tests/test_alignment_rust_parity.py` enforces between the pure-Python `SmithWatermanAligner` and the Rust-backed `RustSmithWatermanAligner` (which calls into the `cite_right._core` extension module). Cite-Right ships the Rust extension as an optional acceleration of the same pipeline; the contract is that the two backends return identical alignment tuples, so `align_citations(answer, sources, backend="auto")` can substitute one for the other without behavior change.

If you change either backend and any test on this page fails, the contract has regressed. Fix the change so the Python and Rust outputs match, do not weaken the test.

## What The Contract Requires

The tests assert that for the same query sequence, the same candidate sequence, and the same scoring parameters, the Rust entry point returns exactly the same tuple the Python aligner returns. The two aligners are interchangeable on the same prepared corpus; status, offsets, and `evidence_spans` are the same.

The contract is enforced on three different entry points, in increasing order of detail:

- `align_pair_details` — pair alignment, six integers plus a match count.
- `align_pair_blocks_details` — pair alignment with per-alignment `match_blocks`, the tuple expected by `CitationConfig(multi_span_evidence=True)`.
- `align_best_details` — best-candidate selection across a list, with deterministic tie-breaking.

The contract also requires:

- Empty candidate lists return `None` from `align_best` and `align_best_details`.
- The `RustSmithWatermanAligner` wrapper exposes the same `align_best_details` tuple shape as the raw extension.
- The `RustSmithWatermanAligner` `align_batch` returns alignments in input order.
- Collecting `match_blocks` does not change which alignment is chosen; the first six tuple elements of `align_pair_blocks_details` must equal the `align_pair_details` tuple on the same input.

The scoring parameters used by every parity test are `match_score=2`, `mismatch_score=-1`, `gap_score=-1`. Those are the defaults exposed by both `SmithWatermanAligner` and `RustSmithWatermanAligner`. The `match_blocks` test uses `return_match_blocks=True` on both sides.

## Tuple Shape

The tuple shape is the same on both backends and is what the tests compare element-for-element.

- `align_pair_details` returns `(score, token_start, token_end, query_start, query_end, matches)`. Six integers. `token_start` / `token_end` are the half-open span in `seq2`; `query_start` / `query_end` are the half-open span in `seq1`.
- `align_pair_blocks_details` returns the same six integers followed by `match_blocks: list[tuple[int, int]]` — a list of `(token_start, token_end)` runs in `seq2` for the exact matches that participated in the selected alignment.
- `align_best_details` returns `(score, index, token_start, token_end, query_start, query_end, matches)`, where `index` is the position in the input candidate list, or `None` for an empty candidate list.

The `Alignment` dataclass returned by the Python `SmithWatermanAligner.align` has the same field names: `score`, `token_start`, `token_end`, `query_start`, `query_end`, `matches`, and optionally `match_blocks`. Tests unpack the Rust tuple and the Python `Alignment` and assert element-wise equality.

## Tie-Breaking Contract

When two alignments score the same number of points, the Python and Rust backends must agree on which one is best. The Python tie-breaker is implemented in `_fill_matrix_reduced_state` (and the block-collecting `_fill_matrix`) in `src/cite_right/core/aligner_py.py` and is encoded into a key in `test_alignment_rust_parity.py` for the `align_best_details` parity test.

The key, in priority order, is:

1. Higher `score` wins.
2. Higher `matches` wins.
3. Smaller `token_start` wins (earlier start in `seq2`).
4. Larger span length (`token_end - token_start`) wins — longer coverage first.
5. Smaller `query_start` wins.
6. Original `index` in the candidate list wins.
7. `token_end` and `query_end` are the final tie-breakers for the single-pair case.

The equal-score coverage regression test (`test_rust_parity_for_equal_score_more_matches_case`) targets rule 2 in particular: when two cells have the same score, the one with more matches must win, and the Rust traceback must agree. The sequences `[0, 1, 0]` against `[0, 1, 1, 1, 0]` is the canonical regression case for that rule.

## Skip Decorators And Fixtures

Tests skip cleanly when the Rust extension is missing or out of date. The skip mechanism lives in `tests/conftest.py`.

- `requires_rust` — skips the test if `cite_right._core` cannot be imported. Reason: "Rust extension not built".
- `requires_rust_blocks` — additionally skips if `cite_right._core` does not expose `align_pair_blocks_details`. Reason: "Rust extension missing align_pair_blocks_details".

Two fixtures return the imported module, skipping the same way:

- `rust_core` — returns the `cite_right._core` module, skips on `ImportError`.
- `rust_core_with_blocks` — returns the same module, additionally skips if `hasattr(_core, "align_pair_blocks_details")` is false, with reason "Rust extension is missing align_pair_blocks_details (rebuild required)".

`requires_rust` tests call the raw extension entry points directly. `requires_rust_blocks` tests call `align_pair_blocks_details` and the `RustSmithWatermanAligner(return_match_blocks=True)` wrapper. The wrapper also checks for `align_pair_blocks_details` and `align_batch_blocks_details` at construction time and raises `RuntimeError` with a rebuild message if either is missing.

## Test Map

Each test below lives in `tests/test_alignment_rust_parity.py`. The headline tests are documented in the page brief (`test_rust_parity`, `test_rust_parity_for_equal_score_more_matches_case`, `test_rust_align_best_matches_python_selection`); the remaining tests are part of the same contract and are documented here for completeness.

### `test_rust_parity`

Verifies that the six-element `align_pair_details` tuple from Rust equals the Python `Alignment` tuple element-for-element on a small set of representative cases.

Skips via `requires_rust`. The three cases cover a clean repeat, a match embedded in a longer candidate, and a no-overlap case:

- `([1, 2], [1, 2, 1, 2])` — both tokens of `seq1` appear twice in `seq2`. The best alignment covers all of `seq1` and the first two tokens of `seq2`.
- `([1, 2, 3], [0, 1, 2, 3, 4])` — a contiguous sub-match at positions 1..4 in `seq2`.
- `([1, 2], [3, 4])` — no match, score is 0 and the aligner returns the empty alignment.

The comparison is element-wise on the six-tuple: `score`, `token_start`, `token_end`, `query_start`, `query_end`, `matches`. Any mismatch on any of those six is a contract failure.

### `test_rust_parity_for_equal_score_more_matches_case`

Verifies the equal-score coverage regression. Skips via `requires_rust_blocks`. The Python aligner is constructed with `return_match_blocks=True` so its `Alignment` includes `match_blocks`; the Rust call uses `align_pair_blocks_details`. Both must return the same seven-tuple plus the same `match_blocks` list.

The sequences are `seq1 = [0, 1, 0]` and `seq2 = [0, 1, 1, 1, 0]`. The traceback must follow the rule "more matches wins on equal score". The regression that this test pins down: a previous Rust version picked the lower-coverage endpoint on equal-score cells, producing a different `token_start` / `token_end` / `match_blocks` from the Python side. The test fails the build if that regression returns.

### `test_rust_align_best_matches_python_selection`

Verifies that `align_best_details` on the Rust side picks the same candidate the Python selector would pick, applying the eight-element sort key from `_python_alignment_sort_key`. Skips via `requires_rust`.

The Python side iterates every candidate, calls `SmithWatermanAligner().align`, builds the key, and keeps the minimum key (i.e. the best alignment under that key). The Rust side calls `align_best_details` directly. The test asserts the seven-tuple Rust returns matches the seven-tuple the Python selector built.

The candidate set is `[[3, 4], [1, 2, 1, 2], [1, 2], [0, 1, 2, 3]]` and the claim is `[1, 2]`. The expected winner is index 1 with score 4, span `[0, 2)` in `seq2`, span `[0, 2)` in `seq1`, and 2 matches — the same value the `test_rust_wrapper_align_best_details_matches_extension` test pins with the `RustSmithWatermanAligner` wrapper.

### `test_rust_align_best_empty_returns_none`

Verifies that the Rust entry points return `None` (not an empty tuple, not a crash) when the candidate list is empty. Skips via `requires_rust`. Both `align_best` and `align_best_details` are checked.

### `test_rust_wrapper_align_best_details_matches_extension`

Verifies the `RustSmithWatermanAligner.align_best_details` wrapper exposes the same tuple shape and the same selection as the raw `cite_right._core.align_best_details`. Skips via `requires_rust`. The expected value is `(4, 1, 0, 2, 0, 2, 2)` for the same `claim = [1, 2]` and `candidates = [[3, 4], [1, 2, 1, 2], [1, 2], [0, 1, 2, 3]]` used in the raw-extension test.

### `test_rust_wrapper_align_batch_matches_python_ordered_results`

Verifies the `RustSmithWatermanAligner.align_batch` wrapper preserves input order. Skips via `requires_rust_blocks`. The Python side builds a list of `Alignment` objects in the same order as the candidate list; the Rust side calls `align_batch_blocks_details` and rebuilds `Alignment` objects from the returned tuples. The test asserts the two lists are equal element-for-element.

Both aligners are constructed with `return_match_blocks=True`. The candidate set is `[[0, 1, 2, 3, 4], [1, 2, 9, 3], [8, 9, 10]]` against the claim `[1, 2, 3]`.

### `test_rust_align_pair_blocks_details_matches_python_blocks`

Verifies that `align_pair_blocks_details` matches the Python blocks output on a clean two-block case. Skips via `requires_rust_blocks`. Sequences are `seq1 = [1, 2, 3, 4]` and `seq2 = [1, 2, 9, 9, 3, 4]`. The expected `match_blocks` are two runs: `[0, 2)` and `[4, 6)` in `seq2`. Any divergence on `match_blocks` is a contract failure even when the six-tuple matches.

### `test_rust_block_and_non_block_entrypoints_share_alignment`

Verifies that the choice of alignment does not depend on whether `match_blocks` are being collected. Skips via `requires_rust_blocks`. The test calls `align_pair_details` on the non-blocks entry point and `align_pair_blocks_details` on the blocks entry point on the same input, then asserts `with_blocks[:6] == without_blocks`. The first six elements of the blocks tuple must equal the non-blocks tuple; only the `match_blocks` element is allowed to differ.

This pins the invariant that the traceback choice is independent of the traceback-collection flag. A regression in that invariant would change the chosen alignment when `multi_span_evidence` is enabled, which would silently change citation offsets.

## What This Test File Does Not Cover

The contract here is the Smith-Waterman local aligner contract, not the full pipeline contract. Things that other files cover:

- Status assignment (`"supported"`, `"partial"`, `"unsupported"`) is checked end-to-end in `tests/test_citations_api.py`, `tests/test_hallucination.py`, and `tests/test_dspy_paper_scenarios.py`. They use `align_citations(answer, sources, backend="python")` and `backend="rust"` and compare the resulting `SpanCitations` lists, which is the public-API form of the same parity guarantee.
- Inverted index parity is in `tests/test_inverted_index.py`.
- Embedder interaction with Rust prepare is in `tests/test_rust_prepare_with_embeddings.py`.
- The `match_blocks`-using citation path is in `tests/test_citations_multi_span.py`. That test also uses `requires_rust_blocks` because `multi_span_evidence=True` routes through `align_pair_blocks_details`.

## Pointers

- `tests/test_alignment_rust_parity.py` — the test file.
- `tests/conftest.py` — `requires_rust`, `requires_rust_blocks`, `rust_core`, `rust_core_with_blocks`.
- `src/cite_right/core/aligner_py.py` — `SmithWatermanAligner`, the matrix fill, the tie-breaker, the traceback that produces `match_blocks`.
- `src/cite_right/core/aligner_rust.py` — `RustSmithWatermanAligner`, the wrapper that picks the right `_core` entry point based on `return_match_blocks` and the construction-time check that the `*_details` entry points exist.
- `src/cite_right/_core.pyi` — the type stub for the `cite_right._core` extension: `align_pair`, `align_pair_details`, `align_pair_blocks_details`, `align_best`, `align_best_details`, `align_topk_details`, `align_batch_details`, `align_batch_blocks_details`.
- `rust_core/` — the Rust extension source. `Cargo.toml` and `rust_core/src/`.
- `openwiki/advanced/rust-acceleration.md` — the public-facing Rust extension guide, including the `backend="auto" | "python" | "rust"` switch and the fallback when `_core` is missing.
- `openwiki/concepts/how-it-works.md` — where Smith-Waterman sits in the pipeline.
- `openwiki/testing/pytest-markers.md` — the other testing reference page, covering the optional-dependency markers (`rust`, `spacy`, `embeddings`, `tiktoken`, `huggingface`, `pysbd`, `slow`).
