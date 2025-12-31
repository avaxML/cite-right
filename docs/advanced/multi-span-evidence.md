# Multi-Span Evidence

Standard citations point to a single contiguous region of text. Multi-span evidence extends this by allowing citations to reference multiple separate regions that together support a claim. This feature is particularly valuable when the source information is spread across non-adjacent sentences.

## Enabling Multi-Span Evidence

Multi-span evidence is disabled by default to maintain backward compatibility and simplicity. Enable it through the configuration.

```python
from cite_right import CitationConfig, align_citations

config = CitationConfig(multi_span_evidence=True)
results = align_citations(answer, sources, config=config)
```

With this configuration, each citation may contain multiple evidence spans rather than just one.

## The Evidence Spans Field

When multi-span evidence is enabled, the `evidence_spans` field on each citation contains a list of span objects.

```python
for result in results:
    for citation in result.citations:
        print(f"Citation from {citation.source_id}")

        for span in citation.evidence_spans:
            print(f"  Evidence: {span.evidence}")
            print(f"  Position: {span.char_start} to {span.char_end}")
```

Each span object has its own `char_start`, `char_end`, and `evidence` fields pointing to a distinct region in the source.

## Backward Compatibility

The single `evidence` field on citations remains available and always contains a contiguous span. When multi-span evidence is enabled, this field contains the enclosing span from the first matched token to the last matched token, including any gaps between matched regions.

```python
# evidence contains the full enclosing span
full_evidence = citation.evidence

# evidence_spans contains the individual regions
individual_spans = citation.evidence_spans

# The full evidence encompasses all individual spans
assert full_evidence.startswith(individual_spans[0].evidence)
```

Applications that do not need multi-span granularity can continue using the `evidence` field unchanged.

## Gap Merging

Adjacent or nearby evidence spans are merged to avoid fragmentation. The `multi_span_merge_gap_chars` parameter controls the maximum gap between spans before they remain separate.

```python
config = CitationConfig(
    multi_span_evidence=True,
    multi_span_merge_gap_chars=30  # Spans within 30 chars are merged
)
```

With a merge gap of 30 characters, two evidence regions separated by 25 characters of punctuation or connector words will be combined into a single span. Regions separated by more than 30 characters remain distinct.

The default merge gap is 50 characters, which handles typical cases where related information appears in adjacent sentences.

## Use Cases

### Compound Claims

Consider an answer span that makes a compound statement: "The company increased revenue and reduced costs." The source document might discuss revenue in one paragraph and costs in another.

```python
answer = "The company increased revenue and reduced costs."
sources = [SourceDocument(
    id="report",
    text="""
    In Q4, the company increased revenue by 15% through new product launches.

    Various cost reduction initiatives were implemented throughout the year.
    Operating costs were reduced by 8% compared to the previous quarter.
    """
)]

config = CitationConfig(multi_span_evidence=True)
results = align_citations(answer, sources, config=config)

for citation in results[0].citations:
    print(f"Number of evidence spans: {len(citation.evidence_spans)}")
    for span in citation.evidence_spans:
        print(f"  {span.evidence!r}")
```

This produces two evidence spans: one from the revenue paragraph and one from the cost paragraph.

### Scattered Facts

When source documents present information in a non-linear order, multi-span evidence captures all relevant regions.

```python
answer = "The CEO, John Smith, announced the acquisition."
sources = [SourceDocument(
    id="news",
    text="""
    A major acquisition was announced today at the annual shareholder meeting.
    The deal is valued at 2.5 billion dollars.
    John Smith, the company's CEO since 2018, made the announcement personally.
    """
)]
```

The relevant information about "John Smith" and "CEO" appears in the third sentence, while "announced the acquisition" aligns with the first sentence. Multi-span evidence captures both regions.

## Frontend Display

Displaying multi-span evidence requires handling non-contiguous highlighting. Here is a pattern for rendering evidence with multiple highlighted regions.

```python
def render_evidence_with_highlights(source_text, evidence_spans):
    """
    Render source text with multiple highlighted regions.
    Returns HTML with <mark> tags around each evidence span.
    """
    # Sort spans by position
    sorted_spans = sorted(evidence_spans, key=lambda s: s.char_start)

    parts = []
    position = 0

    for span in sorted_spans:
        # Add text before this span
        if span.char_start > position:
            parts.append(html_escape(source_text[position:span.char_start]))

        # Add highlighted span
        parts.append(f"<mark>{html_escape(span.evidence)}</mark>")
        position = span.char_end

    # Add remaining text
    if position < len(source_text):
        parts.append(html_escape(source_text[position:]))

    return "".join(parts)
```

## Performance Considerations

Multi-span evidence adds minimal overhead to the alignment process. The additional computation occurs during evidence extraction rather than alignment, which is already the most expensive step.

The main consideration is evidence size. With multi-span evidence enabled, the combined evidence text may be larger than a single contiguous span, potentially increasing response payload size when evidence is included in API responses.

## When to Use Multi-Span Evidence

Enable multi-span evidence when your application needs to display precise supporting text for complex claims. Document review interfaces, fact-checking applications, and detailed citation displays benefit from this granularity.

Keep multi-span evidence disabled when you only need to identify which source supports a claim without fine-grained highlighting. API responses that do not display evidence text, simple supported/unsupported classifications, and bulk processing pipelines can use the simpler single-span model.

## Limitations

Multi-span evidence identifies non-contiguous matching regions but does not understand semantic relationships between them. The spans are determined by alignment matching, not by comprehension of what information is needed to support the claim.

In some cases, the identified spans may be fragments that individually seem incomplete. The value comes from presenting all matched regions together, allowing users to understand the full evidence picture.
