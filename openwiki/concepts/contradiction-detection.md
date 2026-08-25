---
type: concept
title: Contradiction detection
description: Lightweight post-alignment check that detects when a cited answer contradicts the source, downgrading span status to partial rather than unsupported.
tags: [contradiction, citation-alignment, validation]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-ad97420c2a0900f616ed0fef
    resource: repo://src/cite_right/contradiction.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

# Contradiction detection

Contradiction detection is a cheap post-alignment validation step that runs after Smith-Waterman evidence extraction. It checks whether a cited answer claim contradicts the source passage it was aligned to, and if so, downgrades the span status from `supported` to `partial` — preserving the citation but flagging the contradiction.

## Integration point

The check is invoked inside `_span_status()` (repo://src/cite_right/citations.py#L1610-L1633) after citations have been ranked:

```python
def _span_status(
    citations: Sequence[Citation],
    cfg: CitationConfig,
    answer_text: str | None = None,
    candidates: Sequence[Candidate] | None = None,
) -> Literal["supported", "partial", "unsupported"]:
    if not citations:
        return "unsupported"
    best = citations[0]
    coverage = float(best.components.get("answer_coverage", 0.0))

    # Check for contradictions if answer text is provided.
    # Use the candidate passage so leftover tokens beyond truncated evidence
    # (e.g. "BC", "of which came in the first half") are visible.
    if answer_text is not None and check_contradiction(
        answer_text, _contradiction_context(best, candidates)
    ):
        # Downgrade to partial (not unsupported) if contradiction detected
        # because we have evidence, it just contradicts the claim
        return "partial"

    if coverage >= cfg.supported_answer_coverage:
        return "supported"
    return "partial"
```

The function is only called when the answer span text is available. When `answer_text` is `None` (e.g. during batch evaluation without full text), contradiction checks are skipped.

## The `_contradiction_context` helper

`_contradiction_context()` (repo://src/cite_right/citations.py#L1592-L1607) selects which text to pass to `check_contradiction`:

```python
def _contradiction_context(
    citation: Citation,
    candidates: Sequence[Candidate] | None,
) -> str:
    """Prefer the candidate passage over truncated Smith-Waterman evidence.

    Leftover n-grams (issue #48) attach to the wrong slot when alignment
    truncates evidence and hides the contradicting remainder of the passage.
    """
    if candidates:
        for candidate in candidates:
            if candidate.global_index == citation.candidate_index:
                passage = candidate.passage.text
                if passage:
                    return passage
    return citation.evidence
```

Smith-Waterman local alignment returns a *truncated* evidence span — the window of the source passage that best matches the answer. This truncated window can omit contextual words that appear after the aligned region. For example, if the source is `"The vaccine is safe and effective."` and the aligned evidence is `"is safe"`, a negated answer `"The vaccine is not safe"` would not be detected against the truncated `"is safe"` alone. By preferring the full candidate passage, `_contradiction_context` ensures that the negation word `"not"` (which may appear before the aligned window) remains visible to the contradiction checks.

## The five contradiction checks

`check_contradiction()` (repo://src/cite_right/contradiction.py#L67-L87) runs five independent checks in sequence:

```python
def check_contradiction(answer_text: str, evidence_text: str) -> bool:
    if has_negation_mismatch(answer_text, evidence_text):
        return True
    if has_number_mismatch(answer_text, evidence_text):
        return True
    if has_entity_swap(answer_text, evidence_text):
        return True
    if has_temporal_polarity_mismatch(answer_text, evidence_text):
        return True
    if has_number_context_mismatch(answer_text, evidence_text):
        return True
    return False
```

### `has_negation_mismatch`

Detects when one string contains negation and the other does not. Uses `contains_negation()` which searches for patterns including `not`, `never`, `n't` contractions (`isn't`, `doesn't`, `won't`, etc.), `no`, `neither`, `nor`, `nobody`, `nothing`, `cannot`.

```python
def has_negation_mismatch(answer_text: str, evidence_text: str) -> bool:
    return contains_negation(answer_text) != contains_negation(evidence_text)
```

### `has_number_mismatch`

Extracts all numbers from both strings using `extract_numbers()` (normalizing Unicode NFKC and stripping commas). If both strings contain numbers, it checks whether the answer has numbers with no overlap in the evidence set. The presence of *any* answer-unique number triggers a mismatch.

```python
def has_number_mismatch(answer_text: str, evidence_text: str) -> bool:
    answer_numbers = extract_numbers(answer_text)
    evidence_numbers = extract_numbers(evidence_text)
    if answer_numbers and evidence_numbers:
        answer_set = set(answer_numbers)
        evidence_set = set(evidence_numbers)
        unique_in_answer = answer_set - evidence_set
        if unique_in_answer and len(unique_in_answer) == len(answer_set):
            return True
    return False
```

This means `answer: "15%"` vs `evidence: "Revenue grew 15% in Q4"` is *not* a contradiction (15% appears in both). But `answer: "15%"` vs `evidence: "Revenue grew 20% in Q4"` *is* a contradiction.

### `has_entity_swap`

Extracts capitalized words as potential entity names via `extract_entities()`, skipping function words and sentence-initial stopwords. If both strings have entities and each contains unique entities absent from the other, an entity swap is flagged.

```python
def has_entity_swap(answer_text: str, evidence_text: str) -> bool:
    answer_entities = extract_entities(answer_text)
    evidence_entities = extract_entities(evidence_text)
    if not answer_entities or not evidence_entities:
        return False
    answer_set = set(answer_entities)
    evidence_set = set(evidence_entities)
    unique_in_answer = answer_set - evidence_set
    unique_in_evidence = evidence_set - answer_set
    return bool(unique_in_answer and unique_in_evidence)
```

### `has_temporal_polarity_mismatch`

Detects temporal and polarity contradictions using `_paired_marker_mismatch()`, which returns `True` when exactly one side of a contradictory pair matches:

- **Temporal**: `BC` ↔ `ago`, `CE` ↔ `BC`, `AD` ↔ `BC`, `BCE` ↔ `CE`
- **Polarity**: `oppose` ↔ `support/urge/advocate`, `support` ↔ `oppose/reject/resist`, `reject` ↔ `accept/approve/endorse`, `denied` ↔ `confirmed/admitted`, `failed` ↔ `succeeded/achieved`

```python
def _paired_marker_mismatch(
    answer_lower: str,
    evidence_lower: str,
    pairs: list[tuple[str, str]],
) -> bool:
    """Return True if exactly one side of a contradictory marker pair matches."""
    for left, right in pairs:
        answer_left = bool(re.search(left, answer_lower))
        answer_right = bool(re.search(right, answer_lower))
        evidence_left = bool(re.search(left, evidence_lower))
        evidence_right = bool(re.search(right, evidence_lower))
        if (evidence_left and answer_right) or (evidence_right and answer_left):
            return True
    return False
```

### `has_number_context_mismatch`

Handles the "same number, different slot" problem (leftover n-gram issue #48). When the answer and evidence share a number, the function checks whether the content words *immediately surrounding* that number differ:

- `"10 rebounds"` in the answer vs `"10 of which came in the first half"` in the evidence — the word "rebounds" is absent from the evidence context → contradiction.
- `"10 rebounds"` in the answer vs `"He scored 10 rebounds"` in the evidence — "rebounds" appears in both contexts → not a contradiction.

Uses `extract_number_context()` to grab 1–2 words before and after the number, filters out function words via `_content_words()`, and checks whether the answer's content words are absent from the evidence's.

```python
def _number_slot_mismatch(
    answer_context: list[str],
    evidence_context: list[str],
    evidence_text: str,
    number: str,
) -> bool:
    """True when a shared number is attached to different content words."""
    if not answer_context:
        return False
    if not evidence_context:
        return bool(
            evidence_ends_with_number(evidence_text, number)
            and _content_words(answer_context)
        )
    return bool(_content_words(answer_context) - _content_words(evidence_context))
```

## Status downgrade behavior

When contradiction is detected, the span status is downgraded to `partial`, **not** `unsupported`. This distinction matters because:

- `unsupported` means no evidence was found — the citation is discarded.
- `partial` means evidence was found and cited, but the claim contradicts that evidence — the citation is retained for transparency.

For example, given source `"The vaccine is safe and effective."` and answer `"The vaccine is not safe."`:

1. Smith-Waterman aligns `"safe"` → citation created with evidence `"safe"`.
2. `_span_status()` calls `check_contradiction("The vaccine is not safe.", "The vaccine is safe and effective.")`
3. `has_negation_mismatch` detects `"not"` in answer but not in evidence → returns `True`
4. Status is set to `partial` — citation is kept, contradiction is flagged.

## Function word set

`_FUNCTION_WORDS` (repo://src/cite_right/contradiction.py#L16-L37) is a frozenset of 19 words used to filter content words in entity extraction and number-context comparison. It includes standard English stopwords (`the`, `of`, `in`, `and`, etc.) plus normalization variants for percentage (`percent`, `percentage`, `pct`). Words in this set are excluded from entity-name comparison and from content-word comparison in number-context mismatch detection.
