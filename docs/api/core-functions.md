# Core Functions

This page documents the primary functions in Cite-Right. These functions are imported directly from the `cite_right` module and form the main interface for citation alignment and analysis.

## align_citations

The primary function for computing citations from answer text against source documents.

**Location:** `src/cite_right/citations.py`

```python
def align_citations(
    answer: str,
    sources: Sequence[SourceDocument | SourceChunk],
    config: CitationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> list[SpanCitations]
```

### Parameters

**answer** (`str`): The generated text to cite. This text will be segmented into spans, each receiving its own citations.

**sources** (`Sequence[SourceDocument | SourceChunk]`): The reference documents to search for supporting evidence. Can be complete documents or pre-chunked excerpts with position information.

**config** (`CitationConfig | None`): Configuration controlling alignment behavior including thresholds, window sizes, and scoring weights. Defaults to `CitationConfig()` with balanced settings.

**tokenizer** (`Tokenizer | None`): The tokenizer for converting text to token sequences. Defaults to `SimpleTokenizer()`.

**answer_segmenter** (`AnswerSegmenter | None`): The segmenter for splitting the answer into citeable spans. Defaults to `SimpleAnswerSegmenter()`.

**source_segmenter** (`Segmenter | None`): The segmenter for splitting sources into sentences for passage windowing. Defaults to `SimpleSegmenter()`.

**embedder** (`Embedder | None`): Optional embedder for semantic retrieval of candidates. When provided, enables embedding-based candidate selection.

**backend** (`Literal["auto", "python", "rust"]`): The alignment implementation to use. "auto" uses Rust if available, "python" forces pure Python, "rust" requires Rust.

### Returns

`list[SpanCitations]`: A list of citation results, one per answer span. Each result contains the span text, its status, and ranked citations.

### Example

```python
from cite_right import SourceDocument, align_citations, CitationConfig

answer = "The company grew revenue by 20% in 2024."
sources = [
    SourceDocument(id="report", text="Annual revenue increased by 20% during fiscal year 2024.")
]

config = CitationConfig(top_k=3)
results = align_citations(answer, sources, config=config)

for result in results:
    print(f"Span: {result.answer_span.text}")
    print(f"Status: {result.status}")
    for cite in result.citations:
        print(f"  Evidence: {cite.evidence}")
```

## compute_hallucination_metrics

Computes aggregate metrics measuring how well an answer is grounded in source documents.

**Location:** `src/cite_right/hallucination.py`

```python
def compute_hallucination_metrics(
    results: list[SpanCitations],
    config: HallucinationConfig | None = None,
) -> HallucinationMetrics
```

### Parameters

**results** (`list[SpanCitations]`): Citation alignment results from `align_citations`.

**config** (`HallucinationConfig | None`): Configuration controlling metric computation including thresholds for weak citations and partial support handling.

### Returns

`HallucinationMetrics`: Aggregate metrics including groundedness score, hallucination rate, span counts, and detailed per-span analysis.

### Example

```python
from cite_right import align_citations, compute_hallucination_metrics

results = align_citations(answer, sources)
metrics = compute_hallucination_metrics(results)

print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Unsupported spans: {metrics.num_unsupported}")
```

## verify_facts

Performs claim-level verification by decomposing sentences into atomic claims.

**Location:** `src/cite_right/fact_verification.py`

```python
def verify_facts(
    answer: str,
    sources: Sequence[SourceDocument | SourceChunk],
    config: CitationConfig | None = None,
    claim_decomposer: ClaimDecomposer | None = None,
    **kwargs,
) -> FactVerificationResult
```

### Parameters

**answer** (`str`): The generated text to verify.

**sources** (`Sequence[SourceDocument | SourceChunk]`): The reference documents.

**config** (`CitationConfig | None`): Configuration for alignment behavior.

**claim_decomposer** (`ClaimDecomposer | None`): Strategy for splitting sentences into claims. Defaults to `SimpleClaimDecomposer()`.

**kwargs**: Additional arguments passed to `align_citations`.

### Returns

`FactVerificationResult`: Verification results including claim-level status, aggregate counts, and verification rate.

### Example

```python
from cite_right import verify_facts
from cite_right.claims import SpacyClaimDecomposer

result = verify_facts(
    answer,
    sources,
    claim_decomposer=SpacyClaimDecomposer()
)

print(f"Verified claims: {result.num_verified}/{result.total_claims}")
```

## Convenience Functions

### is_grounded

Quick boolean check for whether an answer meets a groundedness threshold.

```python
def is_grounded(
    answer: str,
    sources: Sequence[SourceDocument | SourceChunk],
    threshold: float = 0.5,
    **kwargs,
) -> bool
```

Returns `True` if the groundedness score meets or exceeds the threshold.

### is_hallucinated

Quick boolean check for whether an answer exceeds a hallucination rate threshold.

```python
def is_hallucinated(
    answer: str,
    sources: Sequence[SourceDocument | SourceChunk],
    threshold: float = 0.3,
    **kwargs,
) -> bool
```

Returns `True` if the hallucination rate exceeds the threshold.

### check_groundedness

One-step function combining alignment and metric computation.

```python
def check_groundedness(
    answer: str,
    sources: Sequence[SourceDocument | SourceChunk],
    config: CitationConfig | None = None,
    hallucination_config: HallucinationConfig | None = None,
    **kwargs,
) -> HallucinationMetrics
```

Returns `HallucinationMetrics` computed from fresh alignment results.

### annotate_answer

Adds citation markers to answer text.

```python
def annotate_answer(
    answer: str,
    sources: Sequence[SourceDocument | SourceChunk],
    format: Literal["bracket", "superscript", "footnote"] = "bracket",
    unsupported_marker: str = "?",
    **kwargs,
) -> str
```

Returns the answer text with inline citation markers like `[1]` or `[?]` for unsupported content.

### Example

```python
from cite_right import annotate_answer

annotated = annotate_answer(answer, sources)
print(annotated)
# "Revenue grew 20%.[1] Mars mission planned.[?]"
```

## Integration Functions

### from_langchain_documents

Converts LangChain Document objects to Cite-Right sources.

**Location:** `src/cite_right/integrations.py`

```python
def from_langchain_documents(
    documents: Sequence[Any],
) -> list[SourceDocument]
```

### from_langchain_chunks

Converts LangChain chunks with offset metadata to Cite-Right sources.

```python
def from_langchain_chunks(
    chunks: Sequence[Any],
) -> list[SourceChunk]
```

### from_llamaindex_nodes

Converts LlamaIndex nodes to Cite-Right sources.

```python
def from_llamaindex_nodes(
    nodes: Sequence[Any],
) -> list[SourceDocument]
```

### from_dicts

Converts dictionary objects to Cite-Right sources.

```python
def from_dicts(
    dicts: Sequence[dict],
) -> list[SourceDocument]
```

Looks for text in fields named "text", "content", "page_content", or "body".
Looks for IDs in fields named "id", "doc_id", "document_id", or "source".

### Example

```python
from cite_right.integrations import from_dicts, from_langchain_documents

# From dictionaries
docs = [{"id": "doc1", "content": "Document text..."}]
sources = from_dicts(docs)

# From LangChain
from langchain_core.documents import Document
lc_docs = [Document(page_content="...", metadata={"source": "file.pdf"})]
sources = from_langchain_documents(lc_docs)
```
