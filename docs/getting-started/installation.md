
# Installation

This page covers installing Cite-Right, the requirements you need to meet, and the optional extras you can pull in for tokenization and segmentation backends.

## Requirements

Cite-Right targets Python 3.11 and newer. The 0.4.0 release ships abi3 wheels for the Rust extension, which is built with `abi3-py311`, so a single wheel per platform covers every supported CPython 3.11+ interpreter. 0.4.0 also publishes linux/aarch64 wheels and an sdist, so arm64 Docker installs can use a published wheel or fall back to the sdist without a missing-platform-tag error. The core install pulls in `numpy>=1.24` and `pydantic>=2.0`. No other dependencies are required for the default citation pipeline.

## Basic Installation

Install Cite-Right from PyPI with `pip`.

```bash
pip install cite-right==0.4.0
```

If you manage environments with uv, the same install works through the pip-compatible interface.

```bash
uv pip install cite-right==0.4.0
```

This base install is enough to call `align_citations` and `PreparedCitationCorpus` against `SourceDocument` and `SourceChunk` inputs with the default `SimpleTokenizer` and `SimpleSegmenter`. The optional Rust extension is bundled in the wheel and is used automatically when importable; if a wheel is not available for your platform, the sdist is installed and the library runs on the pure-Python fallback.

## Optional Extras

Cite-Right defines several optional extras in `pyproject.toml`. Each one pulls in a focused dependency group. Extras can be combined by listing them with commas, so a single install can opt into more than one backend at a time.

The extras that affect the citation pipeline itself are `embeddings`, `spacy`, `huggingface`, and `tiktoken`. The package also publishes `pysbd`, `langchain`, and `llamaindex` extras for the corresponding segmenter and integration modules, but those are not part of the citation pipeline itself.

### Embeddings

The `embeddings` extra installs `sentence-transformers>=2.2` and its dependencies. It enables semantic candidate expansion on top of the lexical inverted index. Use it when answer text paraphrases the source instead of quoting it directly.

```bash
pip install "cite-right[embeddings]==0.4.0"
```

With an embedder set, the pipeline still runs Rust prepare when `SimpleTokenizer` and `SimpleSegmenter` are in use; the embedding index is built on the Rust-prepared candidates. Lexical scores are filled only for inverted-index seeds, and embedding-only entries on `retrieval_support` still respect the configured `min_embedding_similarity` (default 0.3). The skip of Rust prepare on the embedder path that existed in 0.3.x is gone.

### SpaCy

The `spacy` extra installs `spacy>=3.7` and `click>=8.0`. It unlocks `SpacyAnswerSegmenter` and `SpacySegmenter` for higher-quality sentence boundary detection and clause-level splitting.

```bash
pip install "cite-right[spacy]==0.4.0"
```

After installing, download a spaCy language model. The small English model is enough for most use cases.

```bash
python -m spacy download en_core_web_sm
```

A custom tokenizer or segmenter takes the lexical fallback path: `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and `_select_candidates` uses lexical prefilter on each span. spaCy segmentation is therefore also the switch to enter that fallback path intentionally.

### HuggingFace

The `huggingface` extra installs `transformers>=4.30` and `tokenizers>=0.15`. It enables `HuggingFaceTokenizer`, which is useful when the tokenization scheme should match a transformer model such as BERT or RoBERTa.

```bash
pip install "cite-right[huggingface]==0.4.0"
```

### Tiktoken

The `tiktoken` extra installs `tiktoken>=0.5` and enables `TiktokenTokenizer`, which uses OpenAI's GPT tokenizer. This is the right choice when answers are produced by a GPT-4 or GPT-3.5-turbo model and you want cite-side tokens to match the model's own tokenization.

```bash
pip install "cite-right[tiktoken]==0.4.0"
```

### Combining Extras

Extras compose. List them with commas in a single `pip install` invocation.

```bash
pip install "cite-right[embeddings,spacy]==0.4.0"
pip install "cite-right[embeddings,huggingface,tiktoken]==0.4.0"
```

You can stack as many as you need; the constraints in `pyproject.toml` resolve them together.

## Verifying The Install

A quick smoke test confirms the public surface imports and the default pipeline runs.

```python
from cite_right import SourceDocument, align_citations

answer = "Hello world."
sources = [SourceDocument(id="test", text="Hello world, this is a test.")]

results = align_citations(answer, sources)
print(f"Found {len(results)} span(s) with status: {results[0].status}")
```

To check whether the Rust extension is active, import `cite_right._core` directly. A missing extension raises `ImportError`, and the library still runs on the pure-Python fallback.

```python
try:
    from cite_right._core import align_pair
    print("Rust extension is available")
except ImportError:
    print("Rust extension is not available, using pure Python")
```

When the extension is active, the default `PreparedCitationCorpus` builds an inverted index and uses rare-token intersect to choose which source windows get Smith-Waterman. When the extension is missing, the same corpus falls back to lexical prefilter and pure-Python Smith-Waterman. Either way, the public API and the resulting `status` values are the same.

## Wheels And Platform Notes

0.4.0 publishes abi3 wheels (`abi3-py311`) and an sdist. The abi3 build means one wheel per platform covers every supported CPython 3.11+ interpreter, and a single linux/aarch64 wheel covers arm64 Linux installs without needing to compile. The sdist is the fallback when a wheel is not available; building from the sdist needs a Rust toolchain.

If you only need the Python pipeline and a wheel exists for your platform, no Rust toolchain is required at install time. A Rust toolchain is only required when building the extension yourself, for example to run an unreleased change or to target a platform without a published wheel.

For background on what the Rust extension actually accelerates, see [Rust Acceleration](../advanced/rust-acceleration.md). For the embedder path and the role of `retrieval_support`, see [Embedding Retrieval](../advanced/embedding-retrieval.md).
