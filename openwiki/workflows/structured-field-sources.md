---
type: workflow
title: Structured-field sources (data2txt)
description: "How _retry_structured_field_citations rescues field:value style sources where the answer reorders values; gap=0 retry on _looks_like_structured_source candidates."
tags: [cite-right, citation-alignment, structured-data, field-sources, workflow]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

## Overview

Data2txt sources are flat `field: value` lines generated from structured records. A faithful LLM rewrite may reorder those lines freely — "4.5 stars, reviews 120" instead of "stars: 4.5, reviews: 120". Standard Smith-Waterman alignment penalizes the gaps introduced by reordering, so it misses these candidates even when every token is present.

`_retry_structured_field_citations` detects this source shape, retokenizes candidates with the answer's tokenizer, and reruns alignment with `gap_score=0`. Caller thresholds (`min_alignment_score`, `min_final_score`, etc.) stay unchanged; only the gap penalty is relaxed for structured-field candidates.

## Detection heuristic

`_looks_like_structured_source` returns `True` when a candidate source looks like a field-value list:

1. Splits the source on newlines and examines only the first 10 non-empty lines.
2. Counts lines that pass `_is_field_value_line`.
3. Returns `True` if at least 2 lines pass **and** they represent ≥50% of non-empty lines.

A line passes `_is_field_value_line` when:

- It contains exactly one `:` separator (split on `:` with `maxsplit=1`).
- The **field** (left side) is non-empty and consists only of alphanumerics, dots, underscores, or hyphens: `all(char.isalnum() or char in '._-' for char in field)`.
- The **value** (right side) is non-empty and contains at most 10 whitespace-separated tokens.

Examples that match:

- `stars: 4.5`
- `hours.Monday: 9:0-17:0`
- `review_count: 120`

Examples that do not match:

- `The restaurant has 4.5 stars and 120 reviews.` (prose, no `:` field)
- `: value` (empty field)
- `very_long_field_name_that_exceeds_the_limit: x` (value >10 tokens)

## Entry point

`_retry_structured_field_citations` is called at the end of `_process_answer_span`, after all standard citations and retrieval-support candidates have been evaluated. It operates only on candidates that were **not** already cited.

```python
# Called from _process_answer_span (citations.py:642-654)
citations, retrieval_support, extra_alignments = (
    _retry_structured_field_citations(
        citations=citations,
        retrieval_support=retrieval_support,
        selected=selected,
        selected_candidates=selected_candidates,
        answer_span_text=answer_span.text,
        tokenizer=tokenizer,
        cfg=cfg,
        aligner=aligner,
        lexical_scores=lexical_scores,
    )
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `citations` | `list[Citation]` | Citations already built for the span (candidates already in this list are skipped). |
| `retrieval_support` | `list[RetrievalSupport]` | Retrieval-support entries for the span. |
| `selected` | `CandidateSelection` | List of `(candidate_index, embedding_score, lexical_score)` tuples from candidate selection. |
| `selected_candidates` | `list[Candidate]` | Parallel list of `Candidate` objects. |
| `answer_span_text` | `str` | Raw answer span text (not tokenized yet). |
| `tokenizer` | `Tokenizer` | Answer tokenizer; used for retokenization and answer tokenization. |
| `cfg` | `CitationConfig` | Caller's thresholds and weights (unchanged for retry). |
| `aligner` | `Aligner` | Primary aligner; its gap score is overridden for structured-field retry. |
| `lexical_scores` | `LexicalScores` | IDF-weighted lexical scores from the primary pass. |

### Returns

A 3-tuple: `(citations, retrieval_support, extra_alignments)`. Newly built citations are appended to `citations`. Retrieval-support entries whose candidates are now cited are removed. `extra_alignments` counts how many candidates were re-evaluated.

## Gap-score-zero aligner factory

`_field_reorder_aligner` clones the given aligner with `gap_score=0` while preserving `match_score`, `mismatch_score`, and `return_match_blocks`:

```python
def _field_reorder_aligner(aligner: Aligner) -> Aligner | None:
    gap_score = getattr(aligner, "gap_score", None)
    if not isinstance(gap_score, int) or gap_score >= 0:
        return None  # Already gap-neutral or unknown; nothing to do
    match_score = int(getattr(aligner, "match_score", 2))
    mismatch_score = int(getattr(aligner, "mismatch_score", -1))
    return_match_blocks = bool(getattr(aligner, "return_match_blocks", False))
    if isinstance(aligner, RustSmithWatermanAligner):
        return RustSmithWatermanAligner(
            match_score=match_score, mismatch_score=mismatch_score,
            gap_score=0, return_match_blocks=return_match_blocks,
        )
    if isinstance(aligner, SmithWatermanAligner):
        return SmithWatermanAligner(
            match_score=match_score, mismatch_score=mismatch_score,
            gap_score=0, return_match_blocks=return_match_blocks,
        )
    return None
```

If the primary aligner has a non-negative gap score (or is an unknown type), the function returns `None` and the retry is skipped entirely. With `gap_score=0`, the algorithm no longer penalizes gaps in the aligned sequence, allowing reordered field-value tokens to align without penalty.

## Retokenization with the answer tokenizer

Rust corpus preparation keeps compound identifiers like `business_stars` intact as single tokens. When aligning against an answer that spells out the fields — "business stars" — the answer tokenizer produces two separate tokens, so no match is found.

`_python_tokens_for_candidate` re-tokenizes the entire source passage using the same tokenizer as the answer:

```python
def _python_tokens_for_candidate(
    candidate: Candidate,
    tokenizer: Tokenizer,
) -> Candidate:
    """Retokenize a field source with the same tokenizer as the answer.

    Rust prepare keeps identifiers like ``business_stars`` intact. Python
    splits on underscores, which is what field rewrites need.
    """
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

This ensures that field names containing underscores align with space-separated equivalents in the answer. The function preserves `global_index`, `source`, and `passage`, but replaces `token_ids`, `token_spans`, and `token_set` with values produced by the answer tokenizer.

## Control flow

```mermaid
flowchart TD
    A[_process_answer_span: standard alignment done] --> B{_field_reorder_aligner\nreturned a gap=0 aligner?}
    B -->|No| Z[return unchanged]
    B -->|Yes| C[Tokenize answer span\nwith range-dash splitting]
    C --> D{Answer has tokens?}
    D -->|No| Z
    D -->|Yes| E[Iterate selected candidates\nnot already cited]
    E --> F{_looks_like_structured_source\n(candidate.source.text)?}
    F -->|No| E
    F -->|Yes| G[_python_tokens_for_candidate\nretokenize with answer tokenizer]
    G --> H{Rust token_ids\nalready populated?}
    H -->|Yes, non-empty| I[Use existing token_ids]
    H -->|No| I
    I --> J[align answer_tokens\nwith gap=0 aligner]
    J --> K{_build_exact_citation\npasses thresholds?}
    K -->|Yes| L[Append citation\ntrack in newly_cited]
    K -->|No| E
    L --> M{Any newly cited\ncandidates?}
    M -->|Yes| N[Remove cited entries\nfrom retrieval_support]
    M -->|No| O[return citations, support, extra_alignments]
    N --> O
```

## Dash normalization

`_split_range_dashes` converts en-dashes, em-dashes, and minus signs to spaces before tokenization so that "Monday–Friday" tokenizes as two separate day tokens matching the source:

```python
def _split_range_dashes(text: str) -> str:
    """Turn en/em range dashes into spaces so Monday–Friday tokenizes as two days.

    ASCII hyphens stay put (Wi-Fi, state-of-the-art).
    """
    return text.replace("\u2013", " ").replace("\u2014", " ").replace("\u2212", " ")
```

## Invariants and failure modes

- **Gap score unchanged for prose**: Only candidates detected as structured-field sources receive the gap=0 aligner. Prose candidates always use the caller's gap penalty.
- **Already-cited candidates skipped**: Candidates that produced a citation in the primary pass are not re-evaluated.
- **Threshold gates preserved**: `_build_exact_citation` enforces `min_alignment_score`, `min_answer_coverage`, `min_final_score`, and `require_all_answer_tokens_in_evidence`. The retry cannot lower these bars.
- **Empty answer span**: If the answer span tokenizes to an empty list (e.g., after dash normalization), the retry returns immediately with no extra work.
- **Unknown aligner type**: `_field_reorder_aligner` returns `None` for non-Smith-Waterman aligners; the retry is skipped.
- **No invented values**: Values that appear in the answer but are not present in the field-value lines still fail alignment, as the gap=0 relaxations do not create false positive matches.

## Configuration

No additional configuration is required. The mechanism activates automatically when:

1. The primary aligner has a negative gap score (the default is `-1`).
2. The source text passes the structured-field heuristic.
3. A candidate was not cited in the primary alignment pass.
