# Quickstart

This guide walks through building a complete citation pipeline from scratch. By the end, you will understand how to align generated text to source documents and extract actionable citation information.

## The Basic Pattern

Working with Cite-Right follows a straightforward pattern. You provide an answer (the generated text you want to cite) and a collection of source documents (the reference material that may contain supporting evidence). The library analyzes the answer, segments it into individual claims, and finds the best matching evidence in your sources.

```python
from cite_right import SourceDocument, align_citations

answer = "Acme Corporation reported revenue of 5.2 billion dollars in 2024."

sources = [
    SourceDocument(
        id="annual_report",
        text="Acme Corporation reported revenue of 5.2 billion dollars in 2024, representing a 12% increase over the previous year."
    )
]

results = align_citations(answer, sources)
```

The `results` variable now contains a list of `SpanCitations` objects, one for each sentence or clause in the answer. Each object includes the answer text, its position within the full answer, and any citations found.

## Understanding the Results

Let us examine what the alignment returns in detail.

```python
for result in results:
    span = result.answer_span
    print(f"Answer text: {span.text!r}")
    print(f"Position: characters {span.char_start} to {span.char_end}")
    print(f"Status: {result.status}")

    for citation in result.citations:
        print(f"  Source: {citation.source_id}")
        print(f"  Evidence: {citation.evidence!r}")
        print(f"  Score: {citation.score:.3f}")
```

The `status` field indicates how well the answer span is supported. A status of "supported" means the alignment found strong evidence in the sources. "Partial" indicates some support was found but it may not cover the entire claim. "Unsupported" means no matching evidence was found.

Each citation provides the `source_id` identifying which document contains the evidence, the `evidence` text itself, and character offsets (`char_start` and `char_end`) pointing to the exact location in that source document.

## Working with Multiple Sources

Real applications often retrieve several documents that might contain relevant information. Cite-Right handles multiple sources naturally, finding the best evidence across all of them.

```python
sources = [
    SourceDocument(
        id="earnings_call",
        text="During the Q4 earnings call, CEO Jane Smith noted that revenue reached 5.2 billion dollars, exceeding analyst expectations."
    ),
    SourceDocument(
        id="press_release",
        text="Acme Corporation today announced fourth quarter revenue of 5.2 billion dollars, a new company record."
    ),
    SourceDocument(
        id="market_analysis",
        text="Industry analysts had predicted Acme would report between 4.8 and 5.0 billion in revenue for the quarter."
    )
]

answer = "Revenue reached 5.2 billion dollars, exceeding expectations."

results = align_citations(answer, sources)

for result in results:
    print(f"'{result.answer_span.text}' -> {result.status}")
    for citation in result.citations:
        print(f"  From {citation.source_id}: {citation.evidence!r}")
```

The library automatically ranks citations by quality. By default, it returns the single best match, but you can request multiple citations using the configuration object.

## Handling Multi-Sentence Answers

Generated answers typically contain multiple sentences, each potentially drawing from different sources. The library segments the answer and processes each span independently.

```python
answer = """Acme Corporation reported record revenue in Q4.
The company attributed growth to its new product line.
European sales exceeded expectations."""

sources = [
    SourceDocument(id="financial", text="Q4 revenue hit a record high at 5.2 billion dollars."),
    SourceDocument(id="products", text="The new product line launched in March drove significant growth."),
    SourceDocument(id="regional", text="Sales in Europe surpassed all projections by 15%."),
]

results = align_citations(answer, sources)

for result in results:
    print(f"\n{result.answer_span.text}")
    print(f"  Status: {result.status}")
    if result.citations:
        best = result.citations[0]
        print(f"  Best match from '{best.source_id}': {best.evidence!r}")
```

Each sentence receives its own status and citation list. This granular approach allows your application to display per-sentence confidence indicators.

## Checking for Hallucinations

When you need to assess the overall quality of a generated response, the hallucination detection functions provide aggregate metrics.

```python
from cite_right import align_citations, compute_hallucination_metrics

answer = "Revenue grew 15% year-over-year. The company will acquire its main competitor next month."

sources = [
    SourceDocument(id="report", text="Annual revenue growth was 15% compared to the previous fiscal year.")
]

results = align_citations(answer, sources)
metrics = compute_hallucination_metrics(results)

print(f"Groundedness score: {metrics.groundedness_score:.1%}")
print(f"Hallucination rate: {metrics.hallucination_rate:.1%}")
print(f"Supported spans: {metrics.num_supported}")
print(f"Unsupported spans: {metrics.num_unsupported}")
```

In this example, the first sentence about revenue growth will be marked as supported because it aligns with the source. The second sentence about acquiring a competitor has no source support and will be flagged as unsupported, contributing to the hallucination rate.

## Convenience Functions

For common patterns, Cite-Right provides high-level convenience functions that combine multiple steps.

```python
from cite_right import is_grounded, check_groundedness, annotate_answer

answer = "The project was completed on time and under budget."
sources = [SourceDocument(id="status", text="Project completion occurred ahead of schedule and below the allocated budget.")]

# Simple boolean check
if is_grounded(answer, sources, threshold=0.5):
    print("Answer is well-grounded in sources")

# Get detailed metrics in one call
metrics = check_groundedness(answer, sources)
print(f"Score: {metrics.groundedness_score:.1%}")

# Add inline citations to the text
annotated = annotate_answer(answer, sources)
print(annotated)  # Adds [1] markers to supported text
```

The `annotate_answer` function inserts citation markers directly into the text, producing output suitable for display in applications that use footnote-style citations.

## Next Steps

This quickstart covered the fundamental operations. The [Building a Check Sources UI](check-sources-ui.md) guide shows how to structure citation data for frontend applications. The [Citation Alignment](../concepts/citation-alignment.md) page explains the algorithm and scoring in depth. The [Configuration](../configuration/citation-config.md) section describes how to tune the alignment behavior for your specific use case.
