# Configuration Classes

This page documents the configuration classes used to customize Cite-Right behavior.

## CitationConfig

Controls citation alignment behavior.

**Location:** `src/cite_right/core/citation_config.py`

```python
class CitationConfig(BaseModel):
    # Result filtering and status
    top_k: int = 3
    min_final_score: float = 0.0
    min_alignment_score: int = 0
    min_answer_coverage: float = 0.2
    supported_answer_coverage: float = 0.6
    allow_embedding_only: bool = False
    min_embedding_similarity: float = 0.3
    supported_embedding_similarity: float = 0.6

    # Passage windowing
    window_size_sentences: int = 1
    window_stride_sentences: int = 1

    # Candidate selection
    max_candidates_lexical: int = 200
    max_candidates_embedding: int = 200
    max_candidates_total: int = 400

    # Ranking
    max_citations_per_source: int = 2
    prefer_source_order: bool = True

    # Alignment scoring
    match_score: int = 2
    mismatch_score: int = -1
    gap_score: int = -1

    # Multi-span evidence
    multi_span_evidence: bool = False
    multi_span_merge_gap_chars: int = 16
    multi_span_max_spans: int = 5

    # Scoring weights
    weights: CitationWeights = CitationWeights()
```

### Result Filtering and Status

**top_k** (`int`): Maximum citations to return per answer span. Default is 3.

**min_final_score** (`float`): Minimum final citation score required for inclusion. This score is a weighted sum of alignment, coverage, lexical, and embedding components.

**min_alignment_score** (`int`): Minimum raw Smith-Waterman alignment score required to use alignment evidence.

**min_answer_coverage** (`float`): Minimum fraction of answer tokens that must match to use alignment evidence.

**supported_answer_coverage** (`float`): Answer coverage threshold for `supported` status.

**allow_embedding_only** (`bool`): Allow citations based solely on embedding similarity when alignment evidence fails.

**min_embedding_similarity** (`float`): Minimum embedding similarity for embedding-only citations.

**supported_embedding_similarity** (`float`): Embedding similarity threshold for `supported` status when `allow_embedding_only=True`.

### Passage Windowing

**window_size_sentences** (`int`): Number of sentences per source passage window.

**window_stride_sentences** (`int`): Step between consecutive passage windows.

### Candidate Selection

**max_candidates_lexical** (`int`): Maximum lexical candidates to consider per answer span.

**max_candidates_embedding** (`int`): Maximum embedding candidates to consider per answer span (requires an embedder).

**max_candidates_total** (`int`): Maximum candidates after combining lexical and embedding candidates.

### Ranking

**max_citations_per_source** (`int`): Cap on citations returned from a single source per answer span.

**prefer_source_order** (`bool`): When scores tie, prefer earlier sources (True) or earlier character positions (False).

### Alignment Scoring

**match_score** (`int`): Score added when tokens match.

**mismatch_score** (`int`): Penalty when tokens do not match.

**gap_score** (`int`): Penalty for gaps in alignment.

### Multi-Span Evidence

**multi_span_evidence** (`bool`): Enable extraction of non-contiguous evidence spans.

**multi_span_merge_gap_chars** (`int`): Maximum gap between spans before they are merged.

**multi_span_max_spans** (`int`): Maximum number of spans returned per citation after merging.

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
    alignment: float = 1.0
    answer_coverage: float = 1.0
    evidence_coverage: float = 0.0
    lexical: float = 0.5
    embedding: float = 0.5
```

**alignment** (`float`): Weight of normalized Smith-Waterman alignment score.

**answer_coverage** (`float`): Weight of matched answer token fraction.

**evidence_coverage** (`float`): Weight of matched evidence token fraction.

**lexical** (`float`): Weight of IDF-weighted lexical overlap score.

**embedding** (`float`): Weight of embedding similarity when embeddings are enabled.

Weights are summed directly (not normalized), so absolute values matter.

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

Controls tokenizer normalization behavior.

**Location:** `src/cite_right/text/tokenizer.py`

```python
class TokenizerConfig:
    def __init__(
        self,
        *,
        normalize_numbers: bool = True,
        normalize_percent: bool = True,
        normalize_currency: bool = True,
    ) -> None:
        ...
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
    min_final_score=0.3,
    supported_answer_coverage=0.7,
    window_size_sentences=2,
    max_candidates_lexical=150,
    weights=CitationWeights(
        alignment=1.0,
        answer_coverage=1.0,
        evidence_coverage=0.0,
        lexical=0.3,
        embedding=0.0
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
    min_answer_coverage=base.min_answer_coverage,
    supported_answer_coverage=base.supported_answer_coverage,
    window_size_sentences=base.window_size_sentences,
    max_candidates_lexical=base.max_candidates_lexical
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

`CitationConfig` is a Pydantic model but does not declare explicit range constraints. Supplying extreme or inconsistent values can lead to degraded results or runtime errors. `align_citations` returns an empty list when `top_k <= 0`.
