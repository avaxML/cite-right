# Data Models

This page documents the data structures used throughout Cite-Right. All models are Pydantic classes with validation and serialization support.

## Input Types

### SourceDocument

Represents a complete source document.

**Location:** `src/cite_right/core/results.py`

```python
class SourceDocument(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = {}
```

**Fields:**

**id** (`str`): Unique identifier for the document. Used in citation results to reference the source.

**text** (`str`): The complete document text. Citation character offsets refer to positions within this text.

**metadata** (`dict[str, Any]`): Optional additional information. Preserved through alignment and accessible in results.

**Example:**

```python
from cite_right import SourceDocument

doc = SourceDocument(
    id="annual_report_2024",
    text="The full document text goes here...",
    metadata={"author": "Finance Team", "date": "2024-03-15"}
)
```

### SourceChunk

Represents a pre-chunked excerpt with position information.

**Location:** `src/cite_right/core/results.py`

```python
class SourceChunk(BaseModel):
    source_id: str
    text: str
    doc_char_start: int
    doc_char_end: int
    metadata: dict[str, Any] = {}
```

**Fields:**

**source_id** (`str`): Identifier of the parent document this chunk came from.

**text** (`str`): The chunk text.

**doc_char_start** (`int`): Starting character position in the original document.

**doc_char_end** (`int`): Ending character position in the original document.

**metadata** (`dict[str, Any]`): Optional additional information.

**Example:**

```python
from cite_right import SourceChunk

chunk = SourceChunk(
    source_id="annual_report_2024",
    text="Revenue increased by 20% year-over-year.",
    doc_char_start=1500,
    doc_char_end=1540
)
```

When citations are computed against chunks, the `doc_char_start` is added to citation offsets, producing positions in the original document.

## Output Types

### SpanCitations

Contains citation results for a single answer span.

**Location:** `src/cite_right/core/results.py`

```python
class SpanCitations(BaseModel):
    answer_span: AnswerSpan
    citations: list[Citation]
    status: Literal["supported", "partial", "unsupported"]
```

**Fields:**

**answer_span** (`AnswerSpan`): The answer text segment being cited.

**citations** (`list[Citation]`): Ranked list of citations, best match first.

**status** (`Literal["supported", "partial", "unsupported"]`): Overall support level based on best citation quality.

### AnswerSpan

Represents a segment of the answer text.

**Location:** `src/cite_right/core/results.py`

```python
class AnswerSpan(BaseModel):
    text: str
    char_start: int
    char_end: int
    kind: Literal["sentence", "clause", "paragraph"] = "sentence"
```

**Fields:**

**text** (`str`): The span text.

**char_start** (`int`): Starting position in the original answer.

**char_end** (`int`): Ending position in the original answer.

**kind** (`Literal["sentence", "clause", "paragraph"]`): The type of segment, determined by the answer segmenter.

### Citation

Contains details about a source match.

**Location:** `src/cite_right/core/results.py`

```python
class Citation(BaseModel):
    score: float
    source_id: str
    source_index: int
    char_start: int
    char_end: int
    evidence: str
    evidence_spans: list[EvidenceSpan] = []
    components: dict[str, float] = {}
```

**Fields:**

**score** (`float`): Overall match quality score.

**source_id** (`str`): Identifier of the source document.

**source_index** (`int`): Index of the source in the original sources list.

**char_start** (`int`): Starting character position of evidence in the source.

**char_end** (`int`): Ending character position of evidence in the source.

**evidence** (`str`): The matched text from the source document.

**evidence_spans** (`list[EvidenceSpan]`): When multi-span evidence is enabled, contains individual evidence regions.

**components** (`dict[str, float]`): Breakdown of score into components like alignment_score, answer_coverage, evidence_coverage.

**Verification:**

The evidence text always equals the source document slice:
```python
assert source.text[citation.char_start:citation.char_end] == citation.evidence
```

### EvidenceSpan

Represents a single region of evidence text within a multi-span citation.

**Location:** `src/cite_right/core/results.py`

```python
class EvidenceSpan(BaseModel):
    char_start: int
    char_end: int
    evidence: str
```

### Alignment

Internal result from the Smith-Waterman alignment operation.

**Location:** `src/cite_right/core/results.py`

```python
class Alignment(BaseModel):
    score: float
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    matches: int
```

**Fields:**

**score** (`float`): Raw alignment score.

**query_start** (`int`): Start token position in the query (answer span).

**query_end** (`int`): End token position in the query.

**target_start** (`int`): Start token position in the target (source passage).

**target_end** (`int`): End token position in the target.

**matches** (`int`): Number of matching tokens in the alignment.

## Metric Types

### HallucinationMetrics

Aggregate metrics from hallucination analysis.

**Location:** `src/cite_right/hallucination.py`

```python
class HallucinationMetrics(BaseModel):
    groundedness_score: float
    hallucination_rate: float
    supported_ratio: float
    partial_ratio: float
    unsupported_ratio: float
    avg_confidence: float
    min_confidence: float
    num_supported: int
    num_partial: int
    num_unsupported: int
    num_weak_citations: int
    unsupported_spans: list[AnswerSpan]
    weakly_supported_spans: list[AnswerSpan]
    span_confidences: list[SpanConfidence]
```

All ratio and score fields range from 0 to 1. Count fields are non-negative integers.

### SpanConfidence

Per-span confidence information.

**Location:** `src/cite_right/hallucination.py`

```python
class SpanConfidence(BaseModel):
    span: AnswerSpan
    confidence: float
    status: Literal["supported", "partial", "unsupported"]
    top_source_id: str | None
```

### FactVerificationResult

Results from fact-level verification.

**Location:** `src/cite_right/fact_verification.py`

```python
class FactVerificationResult(BaseModel):
    claims: list[ClaimVerification]
    total_claims: int
    num_verified: int
    num_partial: int
    num_unverified: int
    verification_rate: float
```

### ClaimVerification

Verification status for a single claim.

**Location:** `src/cite_right/fact_verification.py`

```python
class ClaimVerification(BaseModel):
    text: str
    status: Literal["verified", "partial", "unverified"]
    citations: list[Citation]
```

## Serialization

All models support Pydantic serialization methods.

```python
# To dictionary
data = result.model_dump()

# To JSON string
json_str = result.model_dump_json()

# From dictionary
result = SpanCitations.model_validate(data)

# From JSON string
result = SpanCitations.model_validate_json(json_str)
```

This enables easy integration with APIs, databases, and logging systems.
