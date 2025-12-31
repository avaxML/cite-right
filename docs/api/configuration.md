# Configuration Classes

This page documents the configuration classes used to customize Cite-Right behavior.

## CitationConfig

Controls citation alignment behavior.

**Location:** `src/cite_right/core/citation_config.py`

```python
class CitationConfig(BaseModel):
    # Result parameters
    top_k: int = 1
    min_score_threshold: float = 0.2
    supported_threshold: float = 0.5

    # Passage windowing
    window_size_sentences: int = 3
    window_stride_sentences: int = 1

    # Candidate selection
    max_candidates: int = 50
    lexical_weight: float = 0.5
    embedding_weight: float = 0.5

    # Alignment scoring
    match_score: int = 2
    mismatch_penalty: int = -1
    gap_penalty: int = -1

    # Multi-span evidence
    multi_span_evidence: bool = False
    multi_span_merge_gap_chars: int = 50

    # Embedding behavior
    allow_embedding_only: bool = False

    # Scoring weights
    weights: CitationWeights = CitationWeights()
```

### Result Parameters

**top_k** (`int`): Maximum citations to return per answer span. Default is 1, returning only the best match.

**min_score_threshold** (`float`): Minimum score for a citation to be included. Citations below this are discarded.

**supported_threshold** (`float`): Minimum score for "supported" status. Spans with best citation above this are marked supported.

### Passage Windowing

**window_size_sentences** (`int`): Number of sentences in each passage window.

**window_stride_sentences** (`int`): Step between consecutive windows. Stride of 1 means maximum overlap.

### Candidate Selection

**max_candidates** (`int`): Maximum passages to consider for full alignment per answer span.

**lexical_weight** (`float`): Weight of lexical overlap in candidate scoring.

**embedding_weight** (`float`): Weight of embedding similarity in candidate scoring.

### Alignment Scoring

**match_score** (`int`): Score added when tokens match.

**mismatch_penalty** (`int`): Penalty when tokens do not match.

**gap_penalty** (`int`): Penalty for gaps in alignment.

### Multi-Span Evidence

**multi_span_evidence** (`bool`): Enable extraction of non-contiguous evidence spans.

**multi_span_merge_gap_chars** (`int`): Maximum gap between spans before they are kept separate.

### Embedding Behavior

**allow_embedding_only** (`bool`): Allow citations based solely on embedding similarity when alignment fails.

### Class Methods

```python
@classmethod
def balanced(cls) -> "CitationConfig":
    """Default balanced configuration."""

@classmethod
def strict(cls) -> "CitationConfig":
    """High-precision configuration for fact-checking."""

@classmethod
def permissive(cls) -> "CitationConfig":
    """Lenient configuration for paraphrased content."""

@classmethod
def fast(cls) -> "CitationConfig":
    """Speed-optimized configuration."""
```

## CitationWeights

Controls how score components combine.

**Location:** `src/cite_right/core/citation_config.py`

```python
class CitationWeights(BaseModel):
    alignment_score: float = 0.4
    answer_coverage: float = 0.3
    evidence_coverage: float = 0.2
    embedding_similarity: float = 0.1
```

**alignment_score** (`float`): Weight of raw Smith-Waterman score.

**answer_coverage** (`float`): Weight of matched answer tokens fraction.

**evidence_coverage** (`float`): Weight of matched evidence tokens fraction.

**embedding_similarity** (`float`): Weight of embedding cosine similarity.

Weights are normalized during scoring, so relative values matter.

## HallucinationConfig

Controls hallucination metric computation.

**Location:** `src/cite_right/hallucination.py`

```python
class HallucinationConfig(BaseModel):
    weak_citation_threshold: float = 0.4
    include_partial_in_grounded: bool = True
```

**weak_citation_threshold** (`float`): Minimum answer coverage for a citation to be considered adequate. Below this is "weak".

**include_partial_in_grounded** (`bool`): Whether partial matches contribute to groundedness score. False for stricter metrics.

## TokenizerConfig

Controls tokenizer behavior.

**Location:** `src/cite_right/text/tokenizer.py`

```python
class TokenizerConfig(BaseModel):
    normalize_numbers: bool = True
    normalize_percent: bool = True
    normalize_currency: bool = True
```

**normalize_numbers** (`bool`): Convert numeric separators like "1,200" to "1200".

**normalize_percent** (`bool`): Convert "%" to "percent".

**normalize_currency** (`bool`): Convert "$" to "dollar", "€" to "euro", etc.

## Usage Examples

### Custom Configuration

```python
from cite_right import CitationConfig, align_citations
from cite_right.core.citation_config import CitationWeights

config = CitationConfig(
    top_k=5,
    min_score_threshold=0.3,
    supported_threshold=0.6,
    window_size_sentences=5,
    max_candidates=100,
    weights=CitationWeights(
        alignment_score=0.5,
        answer_coverage=0.3,
        evidence_coverage=0.2,
        embedding_similarity=0.0
    )
)

results = align_citations(answer, sources, config=config)
```

### Preset with Modifications

```python
# Start from a preset
base = CitationConfig.strict()

# Create modified version
config = CitationConfig(
    top_k=3,
    min_score_threshold=base.min_score_threshold,
    supported_threshold=base.supported_threshold,
    window_size_sentences=base.window_size_sentences,
    max_candidates=base.max_candidates
)
```

### Hallucination Configuration

```python
from cite_right import HallucinationConfig, compute_hallucination_metrics

config = HallucinationConfig(
    weak_citation_threshold=0.3,
    include_partial_in_grounded=False
)

metrics = compute_hallucination_metrics(results, config=config)
```

### Tokenizer Configuration

```python
from cite_right import SimpleTokenizer
from cite_right.text.tokenizer import TokenizerConfig

config = TokenizerConfig(
    normalize_numbers=True,
    normalize_percent=True,
    normalize_currency=False  # Keep $ and € as-is
)

tokenizer = SimpleTokenizer(config=config)
```

## Validation

Configuration values are validated by Pydantic. Invalid values raise `ValidationError`.

```python
try:
    config = CitationConfig(top_k=-1)
except ValidationError as e:
    print(e)  # top_k must be positive
```

Range constraints are enforced for thresholds (0-1), counts (non-negative), and other parameters.
