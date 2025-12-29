# Cite-Right v3

[![CI](https://github.com/avaxML/cite-right/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/avaxML/cite-right/actions/workflows/ci.yml)
![Coverage](./coverage.svg)

Cite-Right aligns a generated answer to source documents using Smith–Waterman **local sequence alignment** and returns **character-accurate citations** suitable for highlighting and extraction.

The public API is Python-first for correctness and determinism, with an optional Rust extension (`cite_right._core`) for speed:

- `cite_right.align_citations`: RAG-style citation alignment for **multi-paragraph answers**, returning citations per answer sentence/clause

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
pip install "cite-right[spacy]"
pip install "cite-right[embeddings]"
```

## Quickstart

```python
from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig

answer = "Acme reported revenue of 5.2 billion dollars in 2020.\n\nHeat pumps cut household emissions."
sources = [
    SourceDocument(id="finance", text="... Acme reported revenue of 5.2 billion dollars in 2020. ..."),
    SourceDocument(id="energy", text="... Heat pumps cut household emissions. ..."),
]

spans = align_citations(answer, sources, config=CitationConfig(top_k=1))
for span in spans:
    print(span.answer_span.text, span.status)
    for citation in span.citations:
        doc_text = sources[citation.source_index].text
        assert doc_text[citation.char_start : citation.char_end] == citation.evidence
        print(citation.source_id, citation.char_start, citation.char_end, citation.evidence)
```

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

Default tokenizer: `SimpleTokenizer` (`src/cite_right/text/tokenizer.py`)

- Supports:
  - alphanumeric tokens (with internal hyphens/apostrophes: `state-of-the-art`, `company’s`)
  - numeric tokens with separators (`5.2`, `1,200`)
  - `%` and currency symbols (`$`, `€`, `£`) with lightweight normalization (`%`→`percent`, `$`→`dollar`, etc.)
- Each token is normalized with Unicode **NFKC** and **casefolding** for matching (and small numeric/currency normalization by default).
- `TokenizedText.token_spans` always point into the **original (unnormalized) text**, so evidence slices preserve original casing/punctuation.

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
