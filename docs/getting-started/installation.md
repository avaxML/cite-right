# Installation

This page covers how to install Cite-Right and configure optional features for your specific use case.

## Requirements

Cite-Right requires Python 3.11 or later. The library has been tested on Python versions 3.11 through 3.13. While the core functionality is implemented in pure Python, an optional Rust extension provides significant performance improvements for alignment operations.

## Basic Installation

The simplest way to install Cite-Right is through pip. This installs the core library with its minimal dependencies: NumPy for numerical operations and Pydantic for data validation.

```bash
pip install cite-right
```

If you use uv for package management, the same command works with uv's pip interface.

```bash
uv pip install cite-right
```

The core installation is lightweight and suitable for most use cases where you need basic citation alignment without semantic retrieval capabilities.

## Optional Dependencies

Cite-Right provides several optional extras that add specialized functionality. Each extra can be installed independently, and you can combine multiple extras by listing them together.

### Sentence Embeddings

The embeddings extra enables semantic retrieval of candidate passages before alignment. This feature significantly improves recall when your generated text paraphrases the source material rather than quoting it directly.

```bash
pip install "cite-right[embeddings]"
```

This extra installs the sentence-transformers library and its dependencies. The default embedding model is `all-MiniLM-L6-v2`, which provides a good balance between quality and speed.

### SpaCy Segmentation

The spacy extra provides improved sentence boundary detection and optional clause-level splitting. SpaCy's statistical models produce more accurate segmentation than the default rule-based approach, particularly for complex sentences with nested clauses.

```bash
pip install "cite-right[spacy]"
```

After installing, you must download a spaCy language model. The small English model is sufficient for most use cases.

```bash
python -m spacy download en_core_web_sm
```

### HuggingFace Tokenizers

The huggingface extra enables tokenization using transformer models like BERT and RoBERTa. This option is valuable when you want the tokenization scheme used during alignment to match your language model's tokenization.

```bash
pip install "cite-right[huggingface]"
```

This extra installs the transformers and tokenizers libraries from Hugging Face.

### OpenAI Tokenizers

The tiktoken extra provides tokenization compatible with OpenAI's GPT models. If your application uses GPT-4 or GPT-3.5-turbo, aligning text using the same tokenization scheme can improve citation accuracy.

```bash
pip install "cite-right[tiktoken]"
```

### PySBD Segmentation

The pysbd extra offers fast sentence boundary detection using the pysbd library. This option provides better accuracy than the simple rule-based segmenter while being faster than the full spaCy pipeline.

```bash
pip install "cite-right[pysbd]"
```

### Combining Extras

You can install multiple extras at once by listing them with commas.

```bash
pip install "cite-right[embeddings,spacy]"
```

For a full-featured installation with all optional capabilities, you can install all extras.

```bash
pip install "cite-right[embeddings,spacy,huggingface,tiktoken,pysbd]"
```

## Building from Source

If you need to build Cite-Right from source, perhaps to modify the code or use unreleased features, you will need a Rust toolchain to compile the performance extension.

First, ensure you have Rust installed. The recommended approach is through rustup.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/avaxML/cite-right.git
cd cite-right
```

Install the development dependencies using uv.

```bash
uv sync --frozen
```

Build the Rust extension in development mode.

```bash
uv run maturin develop
```

The maturin tool compiles the Rust code and links it with the Python package. After this step, the high-performance alignment functions will be available automatically.

## Verifying the Installation

You can verify that Cite-Right is installed correctly by running a simple test.

```python
from cite_right import SourceDocument, align_citations

answer = "Hello world."
sources = [SourceDocument(id="test", text="Hello world, this is a test.")]

results = align_citations(answer, sources)
print(f"Found {len(results)} span(s) with status: {results[0].status}")
```

If the Rust extension is available, you can verify its presence by checking the backend.

```python
from cite_right._core import align_pair

print("Rust extension is available!")
```

If the Rust extension is not installed, this import will raise an ImportError, but the library will still function correctly using the pure Python implementation.

## Platform-Specific Notes

On Apple Silicon Macs (M1/M2/M3), the Rust extension compiles natively for ARM64. No special configuration is required.

On Windows, you may need Visual Studio Build Tools with the C++ workload installed to compile the Rust extension. The pure Python implementation works without any additional requirements.

On Linux, the manylinux wheels are available for most common configurations. Building from source requires a C compiler and the Python development headers.
