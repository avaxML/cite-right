
# Multi-Span Evidence

A standard `Citation` carries one contiguous evidence view: `char_start`, `char_end`, and `evidence` slice the source as a single region. When the answer is supported by content that lives in two or more separate places in the same source, that single enclosing span includes "bridge" text that the answer never actually matched. Multi-span evidence exposes the individual matched regions so highlighters, fact-checkers, and audit UIs can attribute precisely to what was matched.

The feature is off by default. Turn it on with `CitationConfig(multi_span_evidence=True)`.

## How It Fits The Pipeline

Multi-span evidence is a presentation layer on top of Smith-Waterman. The local aligner still produces a single best local alignment with one enclosing `token_start` / `token_end` window. When the aligner runs with `return_match_blocks=True`, it also returns the list of disjoint exact-match regions in the candidate passage (`Alignment.match_blocks`). `_alignment_to_evidence_spans` in `src/cite_right/citations.py` turns those match blocks into one or more `EvidenceSpan` objects, while the legacy `Citation.evidence` / `char_start` / `char_end` keep the enclosing slice for backward compatibility.

```text
flowchart LR
    SW[Smith-Waterman<br/>return_match_blocks=True] --> MB[alignment.match_blocks]
    MB --> EX[_extract_multi_span_evidence]
    EX --> MG[merge by gap <= multi_span_merge_gap_chars]
    MG --> CHK{spans > multi_span_max_spans?}
    CHK -- no --> OUT[Citation.evidence_spans]
    CHK -- yes --> FB[fall back to single<br/>contiguous span]
    FB --> OUT
    EX --> LEG[Citation.evidence<br/>Citation.char_start/end<br/>stay enclosing]
```

The default / Rust path produces identical multi-span output to the pure-Python fallback. Rust prepare and Rust alignment both populate `match_blocks` when `multi_span_evidence=True`; the result is normalized through the same `_alignment_to_evidence_spans` path before it reaches a `Citation`.

## Enabling Multi-Span Evidence

```python
from cite_right import CitationConfig, SourceDocument, align_citations

answer = "The company increased revenue and reduced costs."
sources = [
    SourceDocument(
        id="report",
        text=(
            "In Q4, the company increased revenue by 15% through new product launches.\n"
            "Various cost reduction initiatives were implemented throughout the year.\n"
            "Operating costs were reduced by 8% compared to the previous quarter."
        ),
    )
]

config = CitationConfig(multi_span_evidence=True)
results = align_citations(answer, sources, config=config)
```

When the option is off, `Citation.evidence_spans` still carries a single-element list whose span matches the enclosing `evidence` field. When the option is on, the list has one entry per distinct matched region.

## Reading The Spans

Each `EvidenceSpan` is a frozen Pydantic model with three fields, all expressed as half-open offsets in the source document (rebased through `SourceChunk` if needed):

- `char_start` and `char_end` — absolute offsets, inclusive start, exclusive end.
- `evidence` — the source substring, equivalent to `source.text[char_start:char_end]`.

```python
for result in results:
    for citation in result.citations:
        print(f"Source: {citation.source_id} ({len(citation.evidence_spans)} spans)")
        for span in citation.evidence_spans:
            print(f"  {span.char_start}:{span.char_end} -> {span.evidence!r}")
```

`Citation.exact_evidence` is a computed string that joins the ordered spans with `" ... "`. Use it for any UI that wants to show the matched regions inline without the bridge text. `Citation.evidence` is the legacy enclosing slice and is what `assert source.text[citation.char_start:citation.char_end] == citation.evidence` is still guaranteed to hold against.

```python
# enclosing slice (legacy)
print(citation.evidence)

# matched regions only, joined with " ... "
print(citation.exact_evidence)

# individual regions
for span in citation.evidence_spans:
    print(span.evidence)
```

## Backward Compatibility

The legacy `evidence` / `char_start` / `char_end` fields stay a contiguous enclosing span. After chunk rebasing, `source.text[citation.char_start:citation.char_end] == citation.evidence` still holds exactly as in single-span mode. The new fields are additive:

- `citation.evidence_spans` — list of `EvidenceSpan`, one per matched region.
- `citation.exact_evidence` — `" ... "`-joined `evidence` from those spans, sorted by `(char_start, char_end)`.

For precise attribution, prefer `evidence_spans` or `exact_evidence`. Tracing, highlighting, and attribution-oriented consumers should not over-attribute bridge text by relying on the legacy `evidence` field when multi-span is on. Applications that do not need that granularity can keep using `evidence` unchanged, including code that does the half-open offset assertion above.

When `multi_span_evidence=False` (the default), `evidence_spans` contains exactly one span equal to the enclosing slice, and `exact_evidence` equals `evidence`.

## Gap Merging

Adjacent or near-adjacent match regions are merged before they leave the pipeline. The `multi_span_merge_gap_chars` knob sets the maximum gap, in source characters, between two regions that should be combined into one. `merge_gap_chars <= 0` disables merging, so every disjoint match becomes its own span.

```python
config = CitationConfig(
    multi_span_evidence=True,
    multi_span_merge_gap_chars=30,
)
```

The default `multi_span_merge_gap_chars` is **16**. With a 16-character gap, two evidence regions separated by short connector text or punctuation combine into one span. Regions separated by more than 16 characters stay distinct. Set this higher if the source is heavy with sentence-joining connectors; set it to `0` if you want every disjoint match to be its own span for audit-style display.

## Max Spans And Fallback

Even after merging, a single citation can carry many spans if the candidate passage is very fragmented. The `multi_span_max_spans` knob caps how many spans a single citation is allowed to expose. When the post-merge count exceeds that cap, the citation falls back to the legacy single contiguous slice: `evidence_spans` becomes a single-element list equal to the enclosing span, and `exact_evidence` equals `evidence`.

```python
config = CitationConfig(
    multi_span_evidence=True,
    multi_span_max_spans=3,  # if more than 3 regions match, collapse to one
)
```

The default is `5`. Setting it lower makes the citation UI simpler at the cost of losing the per-region breakdown. The fallback is automatic; there is no warning or status change. Status (`"supported"`, `"partial"`, `"unsupported"`) is computed from the same best-citation path either way, and the fallback still produces a real `Citation` rather than dropping to `retrieval_support`.

## When It Helps

Multi-span evidence is most useful when the answer stitches together facts from non-adjacent regions of the same source document.

- **Compound claims.** A claim that says "X increased Y and reduced Z" when one paragraph covers Y and a different paragraph covers Z. Multi-span evidence exposes both regions separately.
- **Scattered entity references.** A name, role, or numeric fact that appears in one sentence and is referenced again several sentences later. The spans let you see both the entity's introduction and the matching predicate.
- **Audit and fact-check UIs.** Reviewers want to see exactly which tokens supported the claim, not a slice that includes the connective text in between.
- **Document review tools.** A reviewer clicking a citation expects to land on each region, not on the whole enclosing paragraph.

## When It Is Not Necessary

- **Single contiguous matches.** A claim that quotes a single sentence in the source. The legacy `evidence` field is already a single span and the multi-span machinery adds nothing.
- **High-throughput pipelines that only need a status.** If downstream code only reads `SpanCitations.status` and `citation.evidence` for display, leave `multi_span_evidence=False` to avoid running the extra match-block traceback. The aligner runs `return_match_blocks=True` only when the option is on.
- **Short sources.** A source that is one or two sentences long cannot produce disjoint match regions through the alignment windowing defaults. The single-span view is already precise.

## Components And Debugging

`Citation.components` carries two fields that are only useful when multi-span is on, exposed regardless of whether the flag is set:

- `num_evidence_spans` — count of spans after merging and after any fallback collapse.
- `evidence_chars_total` — sum of `span.char_end - span.char_start` across the spans in `evidence_spans`.

These let you detect, without iterating the spans, whether the citation actually exposed multiple regions or collapsed to a single one. They are also convenient in tests and metrics where you want to assert multi-span behavior without relying on the underlying span objects.

## How Status Interacts

Status is determined by the best exact citation, not by the number of spans. A multi-span citation with the same `answer_coverage` and no contradiction produces the same status as a single-span citation with the same coverage. Cheap contradiction (negation, number, leftover n-gram slot, entity swap) still downgrades to `"partial"` regardless of how many spans the citation carries.

If multi-span is on, the same `Citation` is still what shows up under `SpanCitations.citations` ranked by score; the difference is that the citation now has a richer view of which source regions supported it. `retrieval_support` is unchanged. A passage that did not produce a localized `Citation` is still a `RetrievalSupport`, not a `Citation` with one empty span.

## SourceChunk And Offset Rebasing

`SourceChunk` lets you pass a pre-chunked excerpt with offsets that point back to the original document. Multi-span evidence respects that rebasing: every `EvidenceSpan.char_start` and `EvidenceSpan.char_end` is in the parent document's coordinate system, not the chunk-local one. When `SourceChunk.document_text` is provided, the spans can be re-sliced directly from the parent text; otherwise the chunk-local text is sliced with `base_doc_offset` subtracted.

```python
from cite_right import SourceChunk, CitationConfig, align_citations

chunk = SourceChunk(
    source_id="report",
    text="Revenue grew 15%.\n\nCosts dropped 8%.",
    doc_char_start=100,
    doc_char_end=140,
    document_text=full_report_text,  # optional, enables direct reslicing
)

config = CitationConfig(multi_span_evidence=True, multi_span_merge_gap_chars=0)
results = align_citations(answer, [chunk], config=config)

for citation in results[0].citations:
    for span in citation.evidence_spans:
        # span offsets are into full_report_text
        assert full_report_text[span.char_start:span.char_end] == span.evidence
```

## Backend Parity

`CitationConfig(multi_span_evidence=True)` produces identical `Citation.evidence_spans` and `Citation.exact_evidence` on the pure-Python and Rust backends. The Python `SmithWatermanAligner` populates `match_blocks` through a detailed traceback in `src/cite_right/core/aligner_py.py`; the Rust `RustSmithWatermanAligner` populates them through the `align_pair_blocks_details` / `align_batch_blocks_details` paths. The build raises if the Rust extension is too old to expose those entry points, so a mismatched extension fails fast at construction time rather than producing silently empty `evidence_spans`.

Rust prepare, including the index, still runs with multi-span on when `SimpleTokenizer` and `SimpleSegmenter` are in use. The same `PreparedCitationCorpus` is reused across many answers; the multi-span extraction is a per-alignment step, not a per-corpus step.

## Configuration Reference

The three `CitationConfig` fields that control this feature live next to the rest of the citation tuning knobs. All three are off by default in the sense that the feature is opt-in, and the two numeric knobs take their documented defaults when the feature is on:

| Field | Default | Effect |
|-------|---------|--------|
| `multi_span_evidence` | `False` | Master switch. When off, every `Citation.evidence_spans` is a one-element list equal to the enclosing slice. |
| `multi_span_merge_gap_chars` | `16` | Maximum source-character gap between two match regions before they are merged into a single span. `<= 0` disables merging. |
| `multi_span_max_spans` | `5` | Maximum number of spans per citation after merging. If exceeded, the citation falls back to a single contiguous span. |

For the wider context on how multi-span evidence slots into scoring, candidate selection, and the rest of the configuration, see [Citation Config](../configuration/citation-config.md) and [Citation Alignment](../concepts/citation-alignment.md). For the alignment stage that produces the match blocks, see [How It Works](../concepts/how-it-works.md).
