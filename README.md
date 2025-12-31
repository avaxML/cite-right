# Cite-Right

[![CI](https://github.com/avaxML/cite-right/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/avaxML/cite-right/actions/workflows/ci.yml)
![Coverage](./coverage.svg)

**Character-accurate citations for AI outputs.** Cite-Right aligns generated answers to source text and returns exact character offsets for highlighting, extraction, and verification. The Python API is the reference implementation, with an optional Rust extension for speed.

## Core features

- **Document-source linking**: Map each answer span to the exact source substring.
- **Character-accurate offsets**: `char_start` / `char_end` are ready for UI highlights.
- **Multi-paragraph support**: Works on RAG-style answers with multiple sentences.
- **Grounding metrics**: Compute hallucination and groundedness stats.

## How it works (high level)

1. Segment the answer into spans (sentences/clauses).
2. Find candidate passages in each source and align with Smith-Waterman.
3. Return citations with absolute character offsets into the original source text.

## Docs

- Site: https://avaxml.github.io/cite-right/
- Start here: `docs/index.md`
- MkDocs config: `mkdocs.yml`


## Install

Requirements: Python 3.11+ (Rust is only needed when building from source or if no wheel exists for your platform).

```bash
pip install cite-right
```

For the embedding-backed quickstart below, install extras:

```bash
pip install "cite-right[embeddings,tiktoken]"
```

See `docs/getting-started/` for optional extras (spaCy, embeddings, HuggingFace, tiktoken) and deeper examples.

## Quickstart

```python
from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig
from cite_right.models.sbert_embedder import SentenceTransformerEmbedder
from cite_right.text.tokenizer_tiktoken import TiktokenTokenizer

question = (
    "What method is introduced to improve sample efficiency, and what gains does it "
    "report over GRPO and MIPROv2?"
)
answer = (
    "GEPA (Genetic-Pareto) is introduced as a reflective prompt optimizer for compound AI systems. "
    "On Qwen3 8B, GEPA outperforms GRPO by up to 19% while requiring up to 35x fewer rollouts. "
    "It surpasses MIPROv2 with aggregate optimization gains of +14%, more than doubling MIPROv2's +7%."
)
sources = [
    SourceDocument(
        id="gepa_intro",
        text=(
            "To operationalize this, we introduce GEPA (Genetic-Pareto), a reflective prompt "
            "optimizer for compound AI systems that merges textual reflection with multi-objective "
            "evolutionary search."
        ),
    ),
    SourceDocument(
        id="grpo_results",
        text=(
            "Our results show that GEPA demonstrates robust generalization and is highly sample efficient: "
            "on Qwen3 8B, GEPA outperforms GRPO (24,000 rollouts with LoRA) by up to 19% while requiring up to "
            "35x fewer rollouts."
        ),
    ),
    SourceDocument(
        id="mipro_results",
        text=(
            "GEPA surpasses the previous state-of-the-art prompt optimizer, MIPROv2, on every benchmark and model, "
            "obtaining aggregate optimization gains of +14%, more than doubling the gains achieved by MIPROv2 (+7%)."
        ),
    ),
]

results = align_citations(
    answer,
    sources,
    config=CitationConfig(top_k=1),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    tokenizer=TiktokenTokenizer(),
)
for result in results:
    print(result.answer_span.text, result.status)
    for citation in result.citations:
        source_doc = sources[citation.source_index]
        evidence = source_doc.text[citation.char_start:citation.char_end]
        print(" ", citation.source_id, evidence)
```

Why embeddings help here:

- The last sentence paraphrases the source, so token overlap alone can fall below the supported threshold.
- The embedder pulls semantically similar passages into the candidate set; alignment then confirms the exact span and returns precise offsets.
- If you want embeddings to directly mark a span as supported with low lexical overlap, tune `supported_embedding_similarity` or use `allow_embedding_only=True` in `CitationConfig`.



## Development

```bash
uv sync --frozen
uv run maturin develop
uv run pytest
```

Optional checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## License

Apache-2.0 (see `LICENSE`).
