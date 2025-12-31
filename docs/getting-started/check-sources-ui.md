# Building a Check Sources UI

This guide demonstrates how to build an interactive "check sources" feature like the one found in Perplexity. Users click on any part of the generated answer and see the exact source text that supports it.

## The User Experience

In a check sources interface, the generated answer is displayed with visual indicators showing which parts are well-supported, partially supported, or potentially hallucinated. When users click on a sentence, the interface highlights the corresponding passage in the source document and scrolls it into view.

The key technical requirement is having precise character offsets that map answer text to source text. Cite-Right provides exactly this through the `char_start` and `char_end` fields in its citation objects.

## Structuring Data for the Frontend

The first step is formatting the alignment results into a structure suitable for your frontend framework. Here is a function that transforms Cite-Right output into JSON-serializable data.

```python
from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig

def prepare_check_sources_data(answer: str, sources: list[SourceDocument]) -> dict:
    """
    Prepare citation data structured for frontend rendering.

    Returns a dictionary containing the original answer, annotated spans
    with their citation information, and source documents indexed by ID.
    """
    config = CitationConfig(top_k=3)  # Get up to 3 citations per span
    results = align_citations(answer, sources, config=config)

    spans = []
    for result in results:
        span_data = {
            "text": result.answer_span.text,
            "start": result.answer_span.char_start,
            "end": result.answer_span.char_end,
            "status": result.status,
            "citations": []
        }

        for citation in result.citations:
            citation_data = {
                "source_id": citation.source_id,
                "evidence_text": citation.evidence,
                "evidence_start": citation.char_start,
                "evidence_end": citation.char_end,
                "confidence": citation.components.get("answer_coverage", 0)
            }
            span_data["citations"].append(citation_data)

        spans.append(span_data)

    source_map = {source.id: source.text for source in sources}

    return {
        "answer": answer,
        "spans": spans,
        "sources": source_map
    }
```

This function produces output that your frontend can use to render interactive answer text with clickable citations.

## Frontend Implementation Pattern

While the specific implementation depends on your frontend framework, the general pattern remains consistent. Here is a conceptual example using HTML and JavaScript.

```javascript
// Render the answer with clickable spans
function renderAnswer(data) {
    const container = document.getElementById('answer-container');

    data.spans.forEach((span, index) => {
        const element = document.createElement('span');
        element.textContent = span.text;
        element.className = `answer-span status-${span.status}`;
        element.dataset.spanIndex = index;

        element.addEventListener('click', () => showCitations(span, data.sources));

        container.appendChild(element);
    });
}

// Show citations when a span is clicked
function showCitations(span, sources) {
    const panel = document.getElementById('citation-panel');
    panel.innerHTML = '';

    span.citations.forEach(citation => {
        const sourceText = sources[citation.source_id];
        const evidenceElement = document.createElement('div');
        evidenceElement.className = 'citation';

        // Render source with highlighted evidence
        const before = sourceText.substring(0, citation.evidence_start);
        const evidence = sourceText.substring(citation.evidence_start, citation.evidence_end);
        const after = sourceText.substring(citation.evidence_end);

        evidenceElement.innerHTML = `
            <div class="source-id">${citation.source_id}</div>
            <div class="source-text">
                ${before}<mark>${evidence}</mark>${after}
            </div>
            <div class="confidence">Confidence: ${(citation.confidence * 100).toFixed(0)}%</div>
        `;

        panel.appendChild(evidenceElement);
    });
}
```

The corresponding CSS provides visual feedback for different support levels.

```css
.answer-span {
    cursor: pointer;
    padding: 2px 4px;
    border-radius: 3px;
    transition: background-color 0.2s;
}

.status-supported {
    background-color: #e8f5e9;  /* Light green */
}

.status-partial {
    background-color: #fff8e1;  /* Light yellow */
}

.status-unsupported {
    background-color: #ffebee;  /* Light red */
}

.answer-span:hover {
    opacity: 0.8;
}

.citation mark {
    background-color: #a5d6a7;  /* Green highlight */
    padding: 2px;
}
```

## React Implementation Example

For React applications, the pattern translates into component state and event handlers.

```jsx
import { useState } from 'react';

function CheckSourcesView({ data }) {
    const [selectedSpan, setSelectedSpan] = useState(null);

    const statusColors = {
        supported: '#e8f5e9',
        partial: '#fff8e1',
        unsupported: '#ffebee'
    };

    return (
        <div className="check-sources">
            <div className="answer-panel">
                {data.spans.map((span, index) => (
                    <span
                        key={index}
                        onClick={() => setSelectedSpan(span)}
                        style={{
                            backgroundColor: statusColors[span.status],
                            cursor: 'pointer',
                            padding: '2px 4px',
                            borderRadius: '3px'
                        }}
                    >
                        {span.text}
                    </span>
                ))}
            </div>

            <div className="citation-panel">
                {selectedSpan && selectedSpan.citations.map((citation, idx) => (
                    <CitationCard
                        key={idx}
                        citation={citation}
                        sourceText={data.sources[citation.source_id]}
                    />
                ))}
            </div>
        </div>
    );
}

function CitationCard({ citation, sourceText }) {
    const before = sourceText.substring(0, citation.evidence_start);
    const evidence = sourceText.substring(citation.evidence_start, citation.evidence_end);
    const after = sourceText.substring(citation.evidence_end);

    return (
        <div className="citation-card">
            <div className="source-label">{citation.source_id}</div>
            <div className="source-content">
                <span>{before}</span>
                <mark style={{ backgroundColor: '#a5d6a7' }}>{evidence}</mark>
                <span>{after}</span>
            </div>
        </div>
    );
}
```

## Handling Long Source Documents

When source documents are lengthy, scrolling to the evidence location improves the user experience. The character offsets make this straightforward.

```javascript
function scrollToEvidence(citation, sourceText) {
    // Calculate approximate scroll position based on character offset
    const totalLength = sourceText.length;
    const scrollRatio = citation.evidence_start / totalLength;

    const sourcePanel = document.getElementById('source-panel');
    const scrollTarget = sourcePanel.scrollHeight * scrollRatio;

    sourcePanel.scrollTo({
        top: scrollTarget - 100,  // Offset to show context above
        behavior: 'smooth'
    });
}
```

A more precise approach renders the source text with span elements for each segment and uses DOM methods to scroll the highlighted element into view.

## Multi-Source Citations

Some answer spans may have citations from multiple sources. The data structure supports this naturally through the citations array.

```python
for result in results:
    if len(result.citations) > 1:
        print(f"'{result.answer_span.text}' has multiple sources:")
        for i, citation in enumerate(result.citations):
            print(f"  {i+1}. {citation.source_id}: {citation.evidence!r}")
```

Your frontend can display these as numbered tabs or a list of source cards, letting users explore all the supporting evidence for a given claim.

## Confidence Indicators

The citation components provide detailed scoring information that can inform visual indicators.

```python
for result in results:
    for citation in result.citations:
        coverage = citation.components.get("answer_coverage", 0)

        if coverage > 0.8:
            indicator = "strong"
        elif coverage > 0.5:
            indicator = "moderate"
        else:
            indicator = "weak"

        print(f"{result.answer_span.text}: {indicator} support")
```

Displaying these indicators helps users understand not just that a source exists, but how well it supports the specific claim.

## API Endpoint Example

For server-rendered applications or API-driven frontends, here is a Flask endpoint that provides the check sources data.

```python
from flask import Flask, request, jsonify
from cite_right import SourceDocument, align_citations

app = Flask(__name__)

@app.route('/api/check-sources', methods=['POST'])
def check_sources():
    data = request.json
    answer = data['answer']
    sources = [
        SourceDocument(id=s['id'], text=s['text'])
        for s in data['sources']
    ]

    result = prepare_check_sources_data(answer, sources)
    return jsonify(result)
```

This endpoint accepts an answer and sources array, runs the alignment, and returns the structured data ready for frontend rendering.
