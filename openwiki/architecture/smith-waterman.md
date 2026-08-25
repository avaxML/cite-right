---
type: architecture
title: Smith-Waterman Aligners
description: Pure-Python and Rust implementations of Smith-Waterman local alignment for token sequences, used to localize answer evidence in candidate documents.
tags: [alignment, citation, smith-waterman, rust, python]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-5e3251d7fd54ced7f7fb97fd
    resource: repo://rust_core/src/lib.rs
  - id: openwiki-source-8c2260658c1a4514202dea35
    resource: repo://rust_core/src/smith_waterman.rs
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-565dc547e636f5aa89fb94bd
    resource: repo://src/cite_right/core/aligner_py.py
  - id: openwiki-source-edff227b75f84c03f46d0ad0
    resource: repo://src/cite_right/core/aligner_rust.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
  - id: openwiki-source-811d84c9631d27a47d6421e0
    resource: repo://tests/test_alignment_rust_parity.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

The cite-right pipeline uses Smith-Waterman local sequence alignment to find the best-scoring token-level match between an answer query and a candidate passage. Two backend implementations share a common algorithm contract: `SmithWatermanAligner` (pure Python) and `RustSmithWatermanAligner` (Rust extension via PyO3). Both backends produce the same alignment results and expose the same public interface.

Alignment output feeds directly into `_compute_alignment_metrics`, which derives coverage ratios and normalized alignment scores that drive final citation ranking. Match blocks are required when `cfg.multi_span_evidence` is enabled, allowing the system to cite non-contiguous evidence spans.

## Core Data Structures

### ScoreParams (Rust)

```rust
pub struct ScoreParams {
    pub match_score: i32,
    pub mismatch_score: i32,
    pub gap_score: i32,
}
```

The default scoring used throughout the pipeline is `match_score=2`, `mismatch_score=-1`, `gap_score=-1`. All scoring parameters are identical across both backends.

### Alignment Result

Both backends return an `Alignment` object (defined in `src/cite_right/core/results.py`) with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `score` | `int` | Raw alignment score (sum of match/mismatch/gap contributions) |
| `token_start` | `int` | Start index (inclusive) of the aligned span in the candidate token sequence |
| `token_end` | `int` | End index (exclusive) of the aligned span in the candidate token sequence |
| `query_start` | `int` | Start index (inclusive) of the aligned span in the query (answer) sequence |
| `query_end` | `int` | End index (exclusive) of the aligned span in the query sequence |
| `matches` | `int` | Count of exact token matches in the selected alignment |
| `match_blocks` | `list[tuple[int, int]]` | Disjoint token-index ranges for contiguous runs of exact matches; populated only when `return_match_blocks=True` |

The `match_blocks` field uses half-open intervals `[start, end)`. For example, `[(0, 2), (4, 6)]` represents two separate runs covering token indices 0–1 and 4–5 in the candidate sequence.

## Algorithm

### Dynamic Programming Fill

Smith-Waterman computes a scoring matrix using three transition choices at each cell `(i, j)`:

1. **Diagonal (match/mismatch)**: Extend the current alignment if the current tokens match. Score contribution is `match_score` on match, `mismatch_score` on mismatch.
2. **Up (gap in seq2)**: Insert a gap in the candidate sequence, penalizing with `gap_score`.
3. **Left (gap in seq1)**: Insert a gap in the query sequence, penalizing with `gap_score`.

Every cell with a score of zero or below terminates any alignment passing through it. The best-scoring cell anywhere in the matrix marks the alignment endpoint. The traceback then follows direction pointers back to the alignment start.

### Tie-Breaking

When multiple cells achieve the same maximum score, the algorithm selects the alignment endpoint according to a deterministic ordering:

1. **Higher match count** — More exact token matches are preferred.
2. **Earlier token_start** — Earlier start position in the candidate is preferred.
3. **Shorter span** — Tighter evidence spans are preferred.
4. **Earlier query_start** — Earlier start in the query is preferred.
5. **Earlier endpoint column** — For equal-quality alignments, earlier column position wins.
6. **Earlier endpoint row** — For still-equal alignments, earlier row position wins.

The tie-breaking logic is implemented identically in both backends (`cmp_candidate` in Rust; inline in Python).

### Memory Optimization: Reduced-State Fill

The default alignment path (without match blocks) uses a memory-optimized fill function. Rather than storing match counts, query starts, and token starts for every cell, the algorithm maintains two rolling rows of metadata. This reduces memory from `O(rows × cols × metadata_fields)` to `O(cols × metadata_fields)`. The full-state fill is used only when `return_match_blocks=True`.

## Backends

### SmithWatermanAligner (`src/cite_right/core/aligner_py.py`)

Pure Python implementation using list-of-lists matrices. Two fill variants are exposed:

- `_fill_matrix`: Full state tracking for match-block traceback.
- `_fill_matrix_reduced_state`: Rolling-row optimization for the default path.

The `align` method dispatches to the appropriate fill based on `self.return_match_blocks`.

### RustSmithWatermanAligner (`src/cite_right/core/aligner_rust.py`)

Wrapper around the compiled `cite_right._core` extension module (built with PyO3 and Maturin). The extension exposes individual PyO3 functions for each operation variant:

| Python method | Rust PyO3 function |
|--------------|-------------------|
| `align` (basic) | `align_pair_details` |
| `align` (with blocks) | `align_pair_blocks_details` |
| `align_batch` (basic) | `align_batch_details` |
| `align_batch` (with blocks) | `align_batch_blocks_details` |
| `align_best` (basic) | `align_best` |
| `align_best` (detailed) | `align_best_details` |
| `align_topk` (detailed) | `align_topk_details` |

The wrapper validates at construction time that required functions are present in the extension. This guards against using a partial or outdated Rust build for citation scoring.

## Aligner Selection Policy

The `_default_aligner` function in `src/cite_right/citations.py` implements the selection policy:

```python
def _default_aligner(cfg: CitationConfig, *, backend: str) -> Aligner:
```

| `backend` argument | Result |
|-------------------|--------|
| `"python"` | Returns `SmithWatermanAligner` |
| `"rust"` | Returns `RustSmithWatermanAligner` |
| `"auto"` | Attempts `RustSmithWatermanAligner`; falls back to `SmithWatermanAligner` if the Rust extension raises `RuntimeError` on import |

The `backend="auto"` path is the default, enabling deployments to run without a compiled Rust extension while gaining its performance benefits when available.

## Match Blocks and Citation Scoring

Detailed alignment outputs (`matches`, `match_blocks`) are required for citation scoring. The `_compute_alignment_metrics` function consumes them:

```python
def _compute_alignment_metrics(
    alignment: Alignment, answer_tokens: list[int], cfg: CitationConfig
) -> dict[str, float]:
    matches = alignment.matches
    if alignment.score > 0 and matches <= 0:
        raise RuntimeError(
            "Alignment metrics require detailed match counts; use a traceback-capable "
            "aligner backend for citation scoring"
        )
    answer_len = len(answer_tokens)
    evidence_len = max(1, alignment.token_end - alignment.token_start)
    return {
        "matches": matches,
        "answer_coverage": matches / max(1, answer_len),
        "evidence_coverage": matches / evidence_len,
        "normalized_alignment": alignment.score / max(1, cfg.match_score * answer_len),
    }
```

These metrics feed directly into `_compute_final_score`, which combines them with lexical overlap and embedding similarity to produce the final citation score. If `alignment.score > 0` but `matches == 0`, the function raises an error—indicating a backend that lacks traceback capability.

The `match_blocks` output is consumed when `cfg.multi_span_evidence=True`. The system can then extract multiple disjoint evidence spans from a single candidate, building citations that reference scattered but relevant content.

## Structured-Field Retry: `gap_score=0`

When an answer references structured data sources (e.g., `business_stars: 4.5`), the standard gap penalty can suppress valid alignments because the source content has been reordered and flattened into field:value lines. The `_retry_structured_field_citations` function handles this case by re-running alignment with `gap_score=0` on candidates that resemble structured sources.

### Detection: `_looks_like_structured_source`

```python
def _looks_like_structured_source(text: str) -> bool:
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return False
    field_value_lines = 0
    non_empty_lines = 0
    for raw_line in lines[:10]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        non_empty_lines += 1
        if _is_field_value_line(stripped):
            field_value_lines += 1
    return (
        non_empty_lines > 0
        and field_value_lines >= 2
        and field_value_lines / non_empty_lines >= 0.5
    )
```

A source is considered structured when at least half of its first ten non-empty lines match the `field:value` pattern checked by `_is_field_value_line`. Field names may contain alphanumerics, dots, underscores, and hyphens. Values are limited to ten space-separated tokens.

### Aligner Reconfiguration: `_field_reorder_aligner`

```python
def _field_reorder_aligner(aligner: Aligner) -> Aligner | None:
    gap_score = getattr(aligner, "gap_score", None)
    if not isinstance(gap_score, int) or gap_score >= 0:
        return None
    # Creates a new aligner of the same type with gap_score=0
```

Only aligners with a negative `gap_score` (the standard case) are reconfigured. The function creates a fresh aligner instance of the same concrete type (either `RustSmithWatermanAligner` or `SmithWatermanAligner`) with all other parameters preserved but `gap_score` set to zero.

### Retokenization: `_python_tokens_for_candidate`

Rust corpus preparation preserves compound identifiers like `business_stars` as single tokens. Python tokenization splits on underscores, which better matches how structured field names appear in rephrased answers. The retry pass re-tokenizes candidates using the answer's tokenizer to ensure consistent token boundaries:

```python
def _python_tokens_for_candidate(
    candidate: Candidate,
    tokenizer: Tokenizer,
) -> Candidate:
    tokenized = tokenizer.tokenize(candidate.source.text)
    sliced = slice_tokenized_text(tokenized, candidate.passage)
    return Candidate(
        global_index=candidate.global_index,
        source=candidate.source,
        passage=candidate.passage,
        token_ids=sliced.token_ids,
        token_spans=sliced.token_spans,
        token_set=frozenset(sliced.token_ids),
    )
```

### Integration Point

`_retry_structured_field_citations` is called from `align_citations` after the primary alignment pass completes (line ~643 in `citations.py`). It processes only uncited candidates whose sources pass the structured-source detection, running one additional alignment per candidate and building citations from any that meet quality thresholds. The `trusted_alignment_match_counts` flag is set based on whether the retry aligner is a known Smith-Waterman type, enabling downstream functions to trust the match count outputs.

## Control Flow Summary

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    subgraph align_citations["align_citations()"]
        A1[align candidates with configured aligner]
        A2["_retry_structured_field_citations()"]
        A3["_rank_and_limit_citations()"]
        A1 --> A2 --> A3
    end

    subgraph retry["_retry_structured_field_citations()"]
        R1["_looks_like_structured_source?"]
        R2["_field_reorder_aligner() → gap_score=0"]
        R3["_python_tokens_for_candidate()"]
        R4[align with reordered tokens]
        R5["_build_exact_citation()"]
        R1 -->|Yes| R2 --> R3 --> R4 --> R5
    end

    subgraph scoring["_compute_alignment_metrics()"]
        S1[assert matches > 0]
        S2[answer_coverage = matches / answer_len]
        S3[evidence_coverage = matches / evidence_len]
        S4[normalized_alignment = score / match_score × answer_len]
        S1 --> S2 --> S3 --> S4
    end

    A1 --> scoring
    R5 --> scoring
```

## Key Invariants

1. **Backend parity**: `SmithWatermanAligner.align` and `RustSmithWatermanAligner.align` produce identical `score`, `token_start`, `token_end`, `query_start`, `query_end`, and `matches` for all inputs.
2. **Traceback requirement**: `_compute_alignment_metrics` raises if `alignment.score > 0` but `alignment.matches == 0`, enforcing that citation scoring requires a traceback-capable backend.
3. **Single best endpoint**: The fill loop selects exactly one winning endpoint per alignment, breaking ties by the deterministic ordering described above.
4. **Match block disjointness**: `consolidate_match_blocks` produces non-overlapping, non-adjacent blocks; runs separated by at least one non-match token become separate blocks.
5. **Auto-fallback safety**: When `backend="auto"` and the Rust extension is unavailable, `_default_aligner` catches the `RuntimeError` and returns the Python aligner transparently.

## Testing

| File | Coverage |
|------|----------|
| `rust_core/src/smith_waterman.rs` | Unit tests within `#[cfg(test)]` module; covers tie-breaking, match blocks, reduced-state fill, top-k, batch, and parity with Python defaults |
| `src/cite_right/core/aligner_py.py` | Unit tests in `tests/test_alignment_py.py`; covers basic alignment, tie-breaking, edge cases, match blocks, and batch ordering |
| `tests/test_alignment_rust_parity.py` | Cross-backend parity tests; verifies Rust and Python produce identical results across all operation variants |

The Rust test suite includes a `default_path_matches_detailed_path_without_match_blocks` test that validates the reduced-state and full-state code paths produce identical alignment results.
