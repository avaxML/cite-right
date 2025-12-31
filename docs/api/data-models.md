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
    metadata: Mapping[str, Any] = {}
```

**Fields:**

**id** (`str`): Unique identifier for the document. Used in citation results to reference the source.

**text** (`str`): The complete document text. Citation character offsets refer to positions within this text.

**metadata** (`Mapping[str, Any]`): Optional additional information. Preserved through alignment and accessible in results.

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
    metadata: Mapping[str, Any] = {}
    document_text: str | None = None
    source_index: int | None = None
```

**Fields:**

**source_id** (`str`): Identifier of the parent document this chunk came from.

**text** (`str`): The chunk text.

**doc_char_start** (`int`): Starting character position in the original document.

**doc_char_end** (`int`): Ending character position in the original document.

**metadata** (`Mapping[str, Any]`): Optional additional information.

**document_text** (`str | None`): Full original document text. If provided, citation offsets are computed against the original document text.

**source_index** (`int | None`): Index of this source in the sources list. If None, the position in the sources list is used.

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

**status** (`Literal["supported", "partial", "unsupported"]`): Overall support level based on answer coverage and optional embedding thresholds.

### AnswerSpan

Represents a segment of the answer text.

**Location:** `src/cite_right/core/results.py`

```python
class AnswerSpan(BaseModel):
    text: str
    char_start: int
    char_end: int
    kind: Literal["sentence", "clause", "paragraph"] = "sentence"
    paragraph_index: int | None = None
    sentence_index: int | None = None
```

**Fields:**

**text** (`str`): The span text.

**char_start** (`int`): Starting position in the original answer.

**char_end** (`int`): Ending position in the original answer.

**kind** (`Literal["sentence", "clause", "paragraph"]`): The type of segment, determined by the answer segmenter.

**paragraph_index** (`int | None`): Paragraph index containing this span.

**sentence_index** (`int | None`): Sentence index within the answer.

### Citation

Contains details about a source match.

**Location:** `src/cite_right/core/results.py`

```python
class Citation(BaseModel):
    score: float
    source_id: str
    source_index: int
    candidate_index: int
    char_start: int
    char_end: int
    evidence: str
    evidence_spans: list[EvidenceSpan] = []
    components: Mapping[str, float] = {}
```

**Fields:**

**score** (`float`): Overall match quality score.

**source_id** (`str`): Identifier of the source document.

**source_index** (`int`): Index of the source in the original sources list.

**candidate_index** (`int`): Internal index of the passage candidate.

**char_start** (`int`): Starting character position of evidence in the source.

**char_end** (`int`): Ending character position of evidence in the source.

**evidence** (`str`): The matched text from the source document.

**evidence_spans** (`list[EvidenceSpan]`): When multi-span evidence is enabled, contains individual evidence regions.

**components** (`Mapping[str, float]`): Breakdown of score components. Typical keys include alignment_score, normalized_alignment, matches, answer_coverage, evidence_coverage, lexical_score, embedding_score, embedding_only, num_evidence_spans, evidence_chars_total, passage_char_start, and passage_char_end.

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
    score: int
    token_start: int
    token_end: int
    query_start: int = 0
    query_end: int = 0
    matches: int = 0
    match_blocks: list[tuple[int, int]] = []
```

**Fields:**

**score** (`int`): Raw alignment score.

**token_start** (`int`): Start token position in the candidate passage.

**token_end** (`int`): End token position in the candidate passage.

**query_start** (`int`): Start token position in the query (answer span).

**query_end** (`int`): End token position in the query.

**matches** (`int`): Number of matching tokens in the alignment.

**match_blocks** (`list[tuple[int, int]]`): Non-contiguous match blocks used for multi-span evidence.

## Metric Types

### HallucinationMetrics

Aggregate metrics from hallucination analysis.
