---
type: concept
title: Text pipeline pluggability
description: Pluggable Tokenizer, Segmenter, AnswerSegmenter, and Embedder Protocols and their bundled implementations for the citation alignment pipeline.
tags: [tokenization, segmentation, embedding, pluggability, cite-right]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-70a6feac670e6bc0185a21c7
    resource: repo://src/cite_right/core/interfaces.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-323579fe89d07517b6f31615
    resource: repo://src/cite_right/models/base.py
  - id: openwiki-source-b90e114ae4f90cba0402e394
    resource: repo://src/cite_right/models/sbert_embedder.py
  - id: openwiki-source-dac6ef0fde0d1e9a4af0de06
    resource: repo://src/cite_right/text/answer_segmenter.py
  - id: openwiki-source-550db9930b2cb7cc86a30bd9
    resource: repo://src/cite_right/text/content_coverage.py
  - id: openwiki-source-0fe20dd2f783e447ead4ce9f
    resource: repo://src/cite_right/text/segmenter_pysbd.py
  - id: openwiki-source-476081e91c3d48f452514c18
    resource: repo://src/cite_right/text/segmenter_simple.py
  - id: openwiki-source-5aae6732a9c8e118a74dd279
    resource: repo://src/cite_right/text/tokenizer_huggingface.py
  - id: openwiki-source-280e6689245ed27fbc16e8ee
    resource: repo://src/cite_right/text/tokenizer_tiktoken.py
  - id: openwiki-source-ccf29287cebbf95d80aebc2f
    resource: repo://src/cite_right/text/tokenizer.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

# Text Pipeline Pluggability

The cite-right library uses a layered text processing pipeline with pluggable components for tokenization, sentence/segment splitting, answer span detection, and semantic embedding. This design lets callers swap in custom implementations—for example, to match the tokenization scheme of a specific LLM or use a domain-specific sentence splitter—without changing the alignment logic.

## Protocol Interfaces

All pipeline components are defined as `Protocol` interfaces in `src/cite_right/core/interfaces.py` using `typing.Protocol`. The `@runtime_checkable` decorator allows `isinstance()` checks at runtime.

### Tokenizer Protocol

```python
@runtime_checkable
class Tokenizer(Protocol):
    def tokenize(self, text: str) -> TokenizedText: ...
```

`Tokenizer` splits text into integer token IDs and character-accurate `(start, end)` spans. The returned `TokenizedText` (defined in `src/cite_right/core/results.py`) is a frozen Pydantic model containing the original text, a list of token IDs, and a parallel list of half-open character offsets. The offset contract is strict: `len(token_ids) == len(token_spans)` and spans must be monotonic, non-overlapping, and within text bounds.

### Segmenter Protocol

```python
@runtime_checkable
class Segmenter(Protocol):
    def segment(self, text: str) -> list[Segment]: ...
```

`Segmenter` splits a document into cohesive units—typically sentences—returning `Segment` objects with `text`, `doc_char_start`, and `doc_char_end` fields. Offsets are absolute to the original document and are used by `passage.py`'s `generate_passages()` to build sliding windows over consecutive segments.

### AnswerSegmenter Protocol

```python
@runtime_checkable
class AnswerSegmenter(Protocol):
    def segment(self, text: str) -> list[AnswerSpan]: ...
```

`AnswerSegmenter` identifies spans in a generated answer that correspond to individual claims. Each `AnswerSpan` carries `char_start`, `char_end`, a `kind` discriminator (`"sentence"`, `"clause"`, or `"paragraph"`), and `paragraph_index` / `sentence_index` for positional context. The answer segmenter is called by `citations.py`'s `align_citations()` to split the answer before candidate retrieval.

### Embedder Protocol

```python
@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...
```

`Embedder` encodes text strings as dense float vectors for semantic similarity scoring. The protocol is defined in `src/cite_right/models/base.py`. Implementations handle batching, caching, and optional GPU acceleration.

## Tokenizer Implementations

### SimpleTokenizer

Located in `src/cite_right/text/tokenizer.py`, `SimpleTokenizer` is a pure-Python rule-based tokenizer with an internal vocabulary (`_vocab: dict[str, int]`). It performs NFKC Unicode normalization, casefolding, and punctuation standardization before assigning integer IDs.

**Normalization toggles** are controlled by `TokenizerConfig`:

| Option | Effect |
|--------|--------|
| `normalize_numbers` | Strips commas from digit sequences (e.g., `"1,200"` → `"1200"`) |
| `normalize_percent` | Converts `"%"` → `"percent"` |
| `normalize_currency` | Converts `"$"` → `"dollar"`, `"€"` → `"euro"`, `"£"` → `"pound"` |

The `_vocab` maps normalized token strings to integer IDs and is built incrementally on first encounter. This is essential for **content coverage**—the pipeline relies on consistent token IDs between answer and source to detect lexical overlap. If two different tokenizers produce different IDs for the same normalized form, alignment quality degrades.

Internal helper functions:
- `_iter_token_spans()` — regex-free token boundary detection via character classification, with LRU caching
- `_normalize_token_cached()` — applies the config normalization rules with memoization
- `_consume_word()` / `_consume_number()` — handle internal apostrophes/hyphens within words and decimal/comma-separated numbers

### TiktokenTokenizer

`src/cite_right/text/tokenizer_tiktoken.py` wraps OpenAI's `tiktoken` library for byte-pair encoding (BPE) tokenization. This is useful when you need citation alignment to use the same tokenization as GPT-4 or GPT-3.5-turbo.

Key behavior:
- Defaults to `cl100k_base` (GPT-4 encoding); supports `p50k_base` (Codex) and `r50k_base` (GPT-3)
- Computes character-accurate spans by building a UTF-8 byte-to-character mapping, then decoding each BPE token back to its original byte range
- Handles multi-byte Unicode characters by tracing byte boundaries back to character offsets
- Raises `ValueError` if tiktoken produces a zero-width span (inconsistent encoding)

Install: `pip install cite-right[tiktoken]`

### HuggingFaceTokenizer

`src/cite_right/text/tokenizer_huggingface.py` wraps HuggingFace tokenizers for subword tokenization (BERT, RoBERTa, etc.). It accepts either `tokenizers.Tokenizer` (Rust-based, fast) or `transformers.PreTrainedTokenizer` instances.

Key behavior:
- `from_pretrained()` convenience classmethod loads models from the HuggingFace Hub
- `add_special_tokens=False` by default (alignment tasks need consistent spans)
- Filters out tokens with empty `(0, 0)` offsets (typically special tokens like `[CLS]`, `[SEP]`)
- For transformers tokenizers, uses `return_offsets_mapping=True` to obtain character spans

Install: `pip install cite-right[huggingface]`

## Segmenter Implementations

### SimpleSegmenter

`src/cite_right/text/segmenter_simple.py` splits text on `.`, `?`, `!` (followed by whitespace), and `;`. A hardcoded abbreviation set (`dr`, `mr`, `mrs`, `ms`, `prof`, `sr`, `jr`, `st`, `vs`, `etc`, `e.g`, `i.e`) prevents false splits on common titles. Optionally splits on `\n` characters via the `split_on_newlines` constructor flag.

### SpacySegmenter

`src/cite_right/text/segmenter_spacy.py` uses spaCy for sentence detection, then further splits at clause-level coordinating conjunctions (`"and"`, `"or"`, `"but"`) when the conjunction's head word is a verb, auxiliary, or adjective. This produces finer-grained segments suitable for granular citation targets.

Dependency parsing determines whether a conjunction links clauses (e.g., "Apples and oranges are fruit" → one segment) versus items in a list (e.g., "I like apples, oranges, and pears" → one segment).

Install: `pip install cite-right[spacy]`

### PySBDSegmenter

`src/cite_right/text/segmenter_pysbd.py` uses pySBD (Python Sentence Boundary Disambiguation), a rule-based library that handles abbreviations, URLs, emails, and ellipses without requiring a full NLP pipeline. It is significantly faster than spaCy.

The `clean=False` default preserves original text offsets for accurate character mapping. Setting `clean=True` makes pySBD preprocess the text but may shift offsets.

Install: `pip install cite-right[pysbd]`

## AnswerSegmenter Implementations

### SimpleAnswerSegmenter

`src/cite_right/text/answer_segmenter.py` first splits the answer into paragraphs (delimited by two or more consecutive newlines), then applies `SimpleSegmenter` to each paragraph. Each sentence becomes an `AnswerSpan` with `kind="sentence"` and paragraph/sentence indices.

### SpacyAnswerSegmenter

`src/cite_right/text/answer_segmenter_spacy.py` uses spaCy for sentence boundary detection, with an optional `split_clauses=True` mode that further divides sentences at coordinating conjunctions (delegating to `segmenter_spacy._split_sentence()`). When enabled, spans receive `kind="clause"`.

## Embedder Implementation

### SentenceTransformerEmbedder

`src/cite_right/models/sbert_embedder.py` wraps `sentence-transformers` for dense semantic encoding. It includes:
- **LRU caching** of encoded vectors (global `OrderedDict` with a 10,000-entry limit)
- Batched encoding via `model.encode()` for efficiency
- Configurable model name (defaults to `"all-MiniLM-L6-v2"`)

The `EmbeddingIndex` class in `src/cite_right/models/embedding_index.py` stores precomputed L2 norms alongside vectors and exposes `top_k()` for cosine similarity search.

Install: `pip install cite-right[embeddings]`

## Content Coverage

`src/cite_right/text/content_coverage.py` implements **content-word overlap** scoring, which prevents paraphrases from falling out of citation scope. A hardcoded stopword list (function words only; polarity markers like "not" are retained as content) distinguishes content tokens from functional ones.

`content_token_coverage()` computes the fraction of non-stopword answer tokens that appear in the source passage, with multiplicity counting to handle reordered content words. This complements Smith-Waterman's sequential matching by rewarding scattered but semantically aligned content.

## Rust Fast Path and Pluggability Trade-offs

`PreparedCitationCorpus.from_sources()` in `src/cite_right/core/prepared_corpus.py` attempts a **Rust fast path** when all three conditions hold:
1. The Rust `_core` extension is installed (`RUST_PREPARE_AVAILABLE = True`)
2. The tokenizer is an instance of `SimpleTokenizer`
3. The source segmenter is an instance of `SimpleSegmenter`

When these conditions are met, `rust_tokenize_and_prepare()` tokenizes, segments, and builds candidates entirely in Rust, marshaling only minimal metadata back to Python. The `SimpleTokenizer`'s `_vocab` is populated from Rust's vocabulary so subsequent answer tokenization remains consistent.

**Using any custom tokenizer or segmenter disables the Rust fast path**, falling back to Python-only processing. This has measurable performance implications for large corpora: the Python path marshals token data across the FFI boundary per candidate, whereas the Rust path keeps data in-process.

## Integration Points

| Function | Component | Default |
|----------|-----------|---------|
| `align_citations()` | `tokenizer` | `SimpleTokenizer()` |
| `align_citations()` | `source_segmenter` | `SimpleSegmenter()` |
| `align_citations()` | `answer_segmenter` | `SimpleAnswerSegmenter()` |
| `PreparedCitationCorpus.from_sources()` | `embedder` | `None` (lexical-only) |

All components are keyword arguments; pass your own implementation to override defaults. Verify custom implementations satisfy the Protocol contract and produce monotonic, non-overlapping character offsets for correct citation localization.
