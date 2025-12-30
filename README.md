# Cite-Right

[![CI](https://github.com/avaxML/cite-right/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/avaxML/cite-right/actions/workflows/ci.yml)
![Coverage](./coverage.svg)

**Link every piece of your AI-generated response back to its source text** - like Perplexity's "check sources", but for your own applications.

Cite-Right lets you **select any part of a generated answer and see the exact source text** it came from, with character-accurate offsets you can use for highlighting, extraction, and verification. Built on Smith–Waterman local sequence alignment for precision.

## Core Features

- **Document-Source Linking**: Select any sentence in a generated answer and get the exact source text with character offsets - perfect for building "check sources" UI experiences
- **Character-Accurate Citations**: Every citation includes `char_start` and `char_end` offsets that point to the exact location in the original source document
- **Multi-Paragraph Support**: Handles RAG-style responses with multiple paragraphs and sentences, returning citations per answer span
- **Hallucination Detection**: Identify which parts of your answer are well-grounded vs. potentially hallucinated

The public API is Python-first for correctness and determinism, with an optional Rust extension (`cite_right._core`) for speed:

- `cite_right.align_citations`: Align multi-paragraph answers to source documents, returning citations per answer sentence/clause
- `cite_right.compute_hallucination_metrics`: Aggregate metrics measuring how well an answer is grounded in sources

## Install

Requirements:

- Python 3.11+ (tested on 3.11–3.13)
- Rust (only needed when building from source / when no wheel exists for your platform)

Core (lightweight):

```bash
pip install cite-right
```

Optional extras:

```bash
pip install "cite-right[spacy]"          # spaCy segmentation
pip install "cite-right[embeddings]"     # Sentence embeddings for semantic retrieval
pip install "cite-right[huggingface]"    # HuggingFace tokenizers (BERT, RoBERTa, etc.)
pip install "cite-right[tiktoken]"       # OpenAI tiktoken (GPT-3.5, GPT-4)
```

## Quickstart

Build a Perplexity-style "check sources" feature in minutes:

```python
from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig

# Your AI-generated answer and source documents
answer = "Acme reported revenue of 5.2 billion dollars in 2020.\n\nHeat pumps cut household emissions."
sources = [
    SourceDocument(id="finance_report", text="... Acme reported revenue of 5.2 billion dollars in 2020. ..."),
    SourceDocument(id="energy_study", text="... Heat pumps cut household emissions. ..."),
]

# Get citations for each answer sentence
results = align_citations(answer, sources, config=CitationConfig(top_k=1))

# Link each answer span to its source
for result in results:
    answer_text = result.answer_span.text
    print(f"Answer: {answer_text!r}")
    print(f"Status: {result.status}")  # "supported", "partial", or "unsupported"

    for citation in result.citations:
        # Get the exact source text using character offsets
        source_doc = sources[citation.source_index]
        source_text = source_doc.text[citation.char_start:citation.char_end]

        # Verify character-accuracy (always passes!)
        assert source_text == citation.evidence

        print(f"  → Source: {citation.source_id}")
        print(f"  → Evidence: {citation.evidence!r}")
        print(f"  → Location: chars {citation.char_start}-{citation.char_end}")
```

Output:
```
Answer: 'Acme reported revenue of 5.2 billion dollars in 2020.'
Status: supported
  → Source: finance_report
  → Evidence: 'Acme reported revenue of 5.2 billion dollars in 2020'
  → Location: chars 4-56

Answer: 'Heat pumps cut household emissions.'
Status: supported
  → Source: energy_study
  → Evidence: 'Heat pumps cut household emissions'
  → Location: chars 4-38
```

## Building a "Check Sources" UI

The character offsets make it easy to build interactive UIs where users can click on answer text and see the source:

```python
from cite_right import align_citations

# Structure your response for the frontend
def format_answer_with_sources(answer: str, sources: list[SourceDocument]) -> dict:
    results = align_citations(answer, sources)

    # Build clickable answer spans
    answer_spans = []
    for result in results:
        span_data = {
            "text": result.answer_span.text,
            "start": result.answer_span.char_start,
            "end": result.answer_span.char_end,
            "status": result.status,  # Visual indicator: green/yellow/red
            "citations": [
                {
                    "source_id": c.source_id,
                    "evidence": c.evidence,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "confidence": c.components.get("answer_coverage", 0),
                }
                for c in result.citations
            ],
        }
        answer_spans.append(span_data)

    return {
        "answer": answer,
        "spans": answer_spans,
        "sources": {s.id: s.text for s in sources},
    }

# In your frontend: users click on answer text → see highlighted source text
# Use char_start/char_end to scroll to and highlight the exact source location
```

**Key capabilities:**
- ✅ Click any answer sentence → jump to exact source location
- ✅ Highlight source text that supports each claim
- ✅ Show confidence indicators (supported/partial/unsupported)
- ✅ Handle multi-source citations (one sentence cites multiple documents)
- ✅ Support non-contiguous quotes with `multi_span_evidence=True`

## Configuration

Common tuning knobs live in `CitationConfig` (see `src/cite_right/core/citation_config.py`), and you can force the alignment backend:

```python
spans = align_citations(answer, sources, backend="auto")  # or: "python", "rust"
```

To return non-contiguous quotes, enable multi-span evidence and use `Citation.evidence_spans`:

```python
from cite_right.core.citation_config import CitationConfig

config = CitationConfig(multi_span_evidence=True)
spans = align_citations(answer, sources, config=config)
for span in spans:
    for citation in span.citations:
        for ev in citation.evidence_spans:
            print(ev.char_start, ev.char_end, ev.evidence)
```

## How it works

At a high level, Cite-Right turns text into token IDs, runs Smith–Waterman alignment, then converts the best token span back into **absolute character offsets** in the original source document.

### Concepts

- **Answer**: the generated text you want to cite (a `str`).
- **Source**: a candidate document (a `str`) that may contain evidence.
- **Passage**: a span of a source document (usually 1–N sentences), with absolute document offsets.
- **Token**: a normalized unit used for alignment; each token also carries a `(start_char, end_char)` span into the original text.

All character offsets are **0-based half-open**: `[char_start, char_end)`.

### Pipeline (align_citations)

`cite_right.align_citations` is designed for RAG/summarization outputs (multiple sentences/paragraphs) and returns citations **per answer sentence**:

1. **Split the answer** into `AnswerSpan`s (default: paragraph-aware sentence segmentation).
2. **Split each source** into sentences and build **passage windows** (configurable `window_size_sentences` / `window_stride_sentences`).
3. **Tokenize** each answer span and each passage using the same tokenizer instance (stable token IDs within the call).
4. **Select candidates** for expensive alignment:
   - lexical prefilter with IDF-weighted overlap (core)
   - optional embedding retrieval when an `Embedder` is provided (`cite-right[embeddings]`)
5. **Align** each answer span against selected passages using Smith–Waterman.
6. Convert token spans → **absolute char spans** in the original source document, and return `Citation` objects.
7. Rank citations deterministically per span and return `SpanCitations(status="supported|partial|unsupported")`.

### Determinism + tie-breaking

For `align_citations`, ranking is configurable via `CitationConfig`. By default citations are ordered by:

1. Higher final `score`
2. Lower `source_index` (to preserve retrieved source ordering)
3. Earlier `char_start`
4. Longer evidence span

## Tokenization

`align_citations` accepts an optional `tokenizer=` argument. If not specified, it defaults to `SimpleTokenizer()`.

### SimpleTokenizer (default)

Location: `src/cite_right/text/tokenizer.py`

Default lightweight tokenizer with:
- Alphanumeric tokens (with internal hyphens/apostrophes: `state-of-the-art`, `company's`)
- Numeric tokens with separators (`5.2`, `1,200`)
- `%` and currency symbols (`$`, `€`, `£`) with optional normalization (`%`→`percent`, `$`→`dollar`, etc.)
- Unicode **NFKC** normalization and casefolding for matching
- Character spans always point to the **original (unnormalized) text**, preserving casing/punctuation in evidence

Configuration via `TokenizerConfig`:
```python
from cite_right import SimpleTokenizer, align_citations
from cite_right.text.tokenizer import TokenizerConfig

config = TokenizerConfig(
    normalize_numbers=True,    # Convert "1,200" → "1200"
    normalize_percent=True,    # Convert "%" → "percent"
    normalize_currency=True,   # Convert "$" → "dollar", "€" → "euro", etc.
)
tokenizer = SimpleTokenizer(config=config)

spans = align_citations(answer, sources, tokenizer=tokenizer)
```

### HuggingFaceTokenizer (transformer-based)

Location: `src/cite_right/text/tokenizer_huggingface.py`

Wraps HuggingFace tokenizers (BERT, RoBERTa, GPT-2, etc.):

```python
from cite_right import HuggingFaceTokenizer, align_citations

# Using a pre-downloaded model
tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")

# Or with a custom tokenizer instance
from transformers import AutoTokenizer
hf_tok = AutoTokenizer.from_pretrained("roberta-base")
tokenizer = HuggingFaceTokenizer(hf_tok)

spans = align_citations(answer, sources, tokenizer=tokenizer)
```

Options:
- `add_special_tokens=False` (default) to exclude `[CLS]`, `[SEP]`, etc.
- Supports WordPiece (BERT), BPE, and SentencePiece encodings
- Requires `pip install "cite-right[huggingface]"`

### TiktokenTokenizer (OpenAI BPE)

Location: `src/cite_right/text/tokenizer_tiktoken.py`

Uses OpenAI's tiktoken for GPT models:

```python
from cite_right import TiktokenTokenizer, align_citations

# Default encoding (cl100k_base, used by GPT-4 and GPT-3.5-turbo)
tokenizer = TiktokenTokenizer()

# Or specify an encoding
tokenizer = TiktokenTokenizer("p50k_base")  # for Codex models
tokenizer = TiktokenTokenizer("r50k_base")  # for GPT-3 models

spans = align_citations(answer, sources, tokenizer=tokenizer)
```

Available encodings:
- `cl100k_base` (default): GPT-4, GPT-3.5-turbo, text-embedding-ada-002
- `p50k_base`: Codex models
- `r50k_base`: GPT-3 models

Requires `pip install "cite-right[tiktoken]"`

## Segmentation

Default segmenter: `SimpleSegmenter` (`src/cite_right/text/segmenter_simple.py`)

- Splits on sentence-ish punctuation (`.?!`), semicolons, and (optionally) newlines.
- Trims surrounding whitespace but preserves absolute `doc_char_start/doc_char_end`.

Optional spaCy segmenter: `SpacySegmenter` (`src/cite_right/text/segmenter_spacy.py`)

- Uses spaCy sentence boundaries and a conservative conjunction-based clause split.
- Requires `cite-right[spacy]` and a spaCy model (e.g. `en_core_web_sm`).
- Use it with citations:

```python
from cite_right import SpacyAnswerSegmenter, SpacySegmenter, align_citations

spans = align_citations(
    answer,
    sources,
    answer_segmenter=SpacyAnswerSegmenter(split_clauses=True),
    source_segmenter=SpacySegmenter(),
)
```

## Alignment (Smith–Waterman)

Reference aligner: `SmithWatermanAligner` (`src/cite_right/core/aligner_py.py`)

- Local alignment over integer token IDs.
- Default scoring: `match=2`, `mismatch=-1`, `gap=-1`.
- Returns the best matching **contiguous span in the candidate segment token list** (`token_start/token_end`) and the alignment `score` (plus `query_start/query_end/matches` for citation scoring).

The evidence text is extracted as the substring spanning from the first matched token’s start char to the last matched token’s end char. This yields a **contiguous** evidence slice (it may include intervening punctuation/whitespace between tokens).

## Embeddings (optional)

`align_citations` can take an `embedder=` argument to improve recall on paraphrases by retrieving candidates via cosine similarity before alignment. This is opt-in and lives behind `cite-right[embeddings]` so core installs stay lightweight.

```python
from cite_right import SentenceTransformerEmbedder, align_citations
from cite_right.core.citation_config import CitationConfig

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
spans = align_citations(answer, sources, embedder=embedder, config=CitationConfig(top_k=2))
```

If your answer is heavily paraphrased (low lexical overlap), set `CitationConfig(allow_embedding_only=True)` to allow returning the whole passage window as evidence based on embedding similarity.

## Hallucination Detection

Cite-Right provides aggregate metrics to measure how well a generated answer is grounded in source documents. Use `compute_hallucination_metrics()` to analyze citation alignment results:

```python
from cite_right import align_citations, compute_hallucination_metrics, HallucinationConfig

answer = "Acme reported revenue of 5.2 billion dollars. They also announced plans to colonize Mars."
sources = ["In the annual report, Acme reported revenue of 5.2 billion dollars for fiscal year 2023."]

# Get citation alignments
results = align_citations(answer, sources)

# Compute hallucination metrics
metrics = compute_hallucination_metrics(results)

print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Hallucination rate: {metrics.hallucination_rate:.1%}")
print(f"Supported: {metrics.num_supported}, Partial: {metrics.num_partial}, Unsupported: {metrics.num_unsupported}")

# Identify problematic spans
for span in metrics.unsupported_spans:
    print(f"  Unsupported: {span.text!r}")
```

### Metrics Returned

`HallucinationMetrics` provides:

| Metric | Description |
|--------|-------------|
| `groundedness_score` | Weighted confidence score (0-1), higher = better grounded |
| `hallucination_rate` | Proportion of ungrounded content (0-1), lower = better |
| `supported_ratio` | Proportion of spans (by char count) that are fully supported |
| `partial_ratio` | Proportion of spans with partial support |
| `unsupported_ratio` | Proportion of spans with no source support |
| `avg_confidence` / `min_confidence` | Confidence statistics across all spans |
| `num_supported` / `num_partial` / `num_unsupported` | Span counts by status |
| `num_weak_citations` | Spans with low-quality citations |
| `unsupported_spans` | List of `AnswerSpan` objects with no source support |
| `weakly_supported_spans` | List of spans with weak evidence |
| `span_confidences` | Per-span `SpanConfidence` details with source attribution |

### Configuration

Customize thresholds with `HallucinationConfig`:

```python
from cite_right import HallucinationConfig, compute_hallucination_metrics

config = HallucinationConfig(
    weak_citation_threshold=0.4,      # Citations below this answer_coverage are "weak"
    include_partial_in_grounded=True, # Count partial matches toward groundedness
)

metrics = compute_hallucination_metrics(results, config=config)
```

Setting `include_partial_in_grounded=False` gives a stricter groundedness score that only counts fully "supported" spans.

## Convenience Functions

For common RAG post-processing workflows, cite-right provides high-level convenience functions:

### Quick Groundedness Checks

```python
from cite_right import is_grounded, is_hallucinated

answer = "Revenue grew 15% in Q4."
sources = ["Annual report: Revenue grew 15% in Q4 2024."]

# Simple boolean check for quality gates
if is_grounded(answer, sources, threshold=0.6):
    print("Answer is well-grounded!")

# Or check for hallucinations
if is_hallucinated(answer, sources, threshold=0.3):
    print("Warning: Answer may contain hallucinations!")
```

### One-Step Groundedness Metrics

```python
from cite_right import check_groundedness

metrics = check_groundedness(answer, sources)
print(f"Groundedness: {metrics.groundedness_score:.1%}")
print(f"Unsupported: {[s.text for s in metrics.unsupported_spans]}")
```

### Inline Citation Annotation

Add citation markers directly to your answer text:

```python
from cite_right import SourceDocument, annotate_answer

answer = "Revenue grew 15%. Profits doubled."
sources = [SourceDocument(id="report", text="Revenue grew 15% in Q4.")]

annotated = annotate_answer(answer, sources)
# Output: "Revenue grew 15%.[1] Profits doubled.[?]"

# Different formats available
annotated = annotate_answer(answer, sources, format="superscript")  # ^1
annotated = annotate_answer(answer, sources, format="footnote")     # [^1]
```

## Configuration Presets

For common use cases, use pre-configured settings:

```python
from cite_right import CitationConfig, align_citations

# High-precision mode for fact-checking
config = CitationConfig.strict()

# Lenient mode for paraphrased content
config = CitationConfig.permissive()

# Speed-optimized for high-volume processing
config = CitationConfig.fast()

# Default balanced configuration
config = CitationConfig.balanced()

results = align_citations(answer, sources, config=config)
```

| Preset | Use Case |
|--------|----------|
| `strict()` | Fact-checking, high-stakes applications |
| `permissive()` | Summarization, paraphrased content |
| `fast()` | High-volume processing, latency-sensitive |
| `balanced()` | General purpose (same as default) |

## Framework Integrations

Cite-right provides helpers for popular RAG frameworks:

### LangChain

```python
from cite_right import align_citations
from cite_right.integrations import from_langchain_documents

# After retrieval
lc_docs = retriever.invoke(query)

# Convert to cite-right format
sources = from_langchain_documents(lc_docs)

# Use with align_citations
results = align_citations(answer, sources)
```

For pre-chunked documents with offsets:

```python
from cite_right.integrations import from_langchain_chunks

# If your chunks have start_index/end_index metadata
sources = from_langchain_chunks(lc_chunks)
```

### LlamaIndex

```python
from cite_right.integrations import from_llamaindex_nodes

# After retrieval
nodes = retriever.retrieve(query)

# Convert to cite-right format
sources = from_llamaindex_nodes(nodes)
```

### Plain Dictionaries

```python
from cite_right.integrations import from_dicts

# From API responses or custom pipelines
docs = [
    {"id": "doc1", "text": "Document content...", "score": 0.9},
    {"id": "doc2", "text": "Another document...", "score": 0.8},
]
sources = from_dicts(docs)
```

## Rust acceleration

If built, the Rust extension is importable as `cite_right._core` and is used by default by `align_citations` (falls back to pure Python if unavailable).

- `align_pair(seq1, seq2, ...) -> (score, token_start, token_end)` accelerates the DP/traceback for one pair.
- `align_best(seq1, seqs, ...) -> (score, index, token_start, token_end) | None` runs one-to-many alignment with Rayon and a deterministic reduction.
- The GIL is released during compute.

Build locally:

```bash
uv run maturin develop
```

## API details

### `align_citations`

See `src/cite_right/citations.py` for the full signature and `src/cite_right/core/results.py` for the result types (`SpanCitations`, `Citation`, `AnswerSpan`, `SourceDocument`, `SourceChunk`).

### `compute_hallucination_metrics`

See `src/cite_right/hallucination.py` for the full signature and result types (`HallucinationMetrics`, `HallucinationConfig`, `SpanConfidence`).

## Developer setup

```bash
uv sync --frozen
uv run maturin develop
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Repository layout

- `src/cite_right/`: Python reference implementation
- `rust_core/`: Rust extension (`cite_right._core`)
- `tests/`: test suite
- `.github/workflows/`: CI and wheel builds
- `CITATION_ALIGNMENT_PLAN.md`: design notes / roadmap

## Limitations (current)

- `Citation.evidence` is a single **contiguous** (enclosing) span; use `Citation.evidence_spans` for multi-span highlighting.
- Without embeddings enabled, paraphrases may not match (lexical overlap + alignment is strongest on near-verbatim support).
- Default segmentation is heuristic and may miss clause boundaries; enable spaCy for richer clause splitting.

## Notes

- All alignment results include absolute character offsets into the original source text.
- Python is the reference implementation; Rust accelerates alignment while matching Python outputs.

## License

Apache-2.0 (see `LICENSE`).
