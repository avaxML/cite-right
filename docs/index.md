# Cite-Right Documentation

**Link every piece of your AI-generated response back to its source text.**

Cite-Right is a Python library that enables you to build citation-backed AI applications, similar to the "check sources" feature found in Perplexity. When your language model generates an answer from retrieved documents, Cite-Right identifies exactly which parts of the source text support each claim in the response.

## What Makes Cite-Right Different

Traditional citation systems link entire paragraphs or documents to generated text. Cite-Right takes a more precise approach by providing **character-accurate offsets** that point to the exact location in your source documents. This precision enables features like highlighting the specific supporting text when a user clicks on any sentence in your AI's response.

The library is built on the Smith-Waterman local sequence alignment algorithm, the same algorithm used in bioinformatics to find similar regions between DNA sequences. This approach excels at finding matching text even when there are minor differences in phrasing, punctuation, or formatting.

## Core Capabilities

Cite-Right addresses three fundamental challenges in RAG (Retrieval-Augmented Generation) applications.

### Document-Source Linking

Every sentence in your generated answer can be traced back to its origin. When you call `align_citations`, the library analyzes each sentence and returns a `Citation` object containing the source document identifier along with the precise character positions where the supporting text begins and ends. Your frontend can use these offsets to scroll to and highlight the exact source passage.

### Hallucination Detection

Not every claim in a generated answer will have source support. Cite-Right categorizes each answer span as "supported", "partial", or "unsupported" based on how well it aligns with the provided sources. The `compute_hallucination_metrics` function aggregates these results into a groundedness score that quantifies the overall reliability of the response.

### Fact-Level Verification

For applications requiring fine-grained analysis, the `verify_facts` function decomposes sentences into individual claims and verifies each one independently. This approach catches situations where a sentence combines a factual statement with an unsupported assertion.

## The Citation Object

At the heart of Cite-Right is the `Citation` data structure. Each citation contains several key pieces of information.

The `source_id` field identifies which document contains the supporting evidence. The `char_start` and `char_end` fields specify the exact byte positions within that document, using Python's standard half-open interval convention where the start is inclusive and the end is exclusive.

The `evidence` field contains the actual text extracted from the source document. You can verify that this matches the document slice by checking that `source.text[citation.char_start:citation.char_end] == citation.evidence`, a property that always holds true.

The `score` field indicates the alignment quality, with higher values representing stronger matches. The `components` dictionary breaks down this score into its constituent parts, including answer coverage, evidence coverage, and optional embedding similarity.

## Installation

The core library has minimal dependencies and installs quickly.

```bash
pip install cite-right
```

Several optional features require additional packages. Semantic retrieval using sentence embeddings needs the embeddings extra. SpaCy-based sentence segmentation requires the spacy extra. Support for transformer tokenizers from HuggingFace or OpenAI's tiktoken are available through their respective extras.

```bash
pip install "cite-right[embeddings]"  # For semantic retrieval
pip install "cite-right[spacy]"       # For improved segmentation
pip install "cite-right[huggingface]" # For BERT/RoBERTa tokenizers
pip install "cite-right[tiktoken]"    # For GPT tokenizers
```

## A Quick Example

Here is a minimal example demonstrating the core functionality.

```python
from cite_right import SourceDocument, align_citations

answer = "The company reported record revenue in Q4."
sources = [
    SourceDocument(
        id="earnings_call",
        text="During the earnings call, the CEO announced that the company reported record revenue in Q4 of 2024."
    )
]

results = align_citations(answer, sources)

for result in results:
    print(f"Text: {result.answer_span.text}")
    print(f"Status: {result.status}")
    for citation in result.citations:
        print(f"Evidence: {citation.evidence}")
        print(f"Location: {citation.char_start}:{citation.char_end}")
```

This code produces output showing that the answer sentence is "supported" with evidence extracted from character positions 45 through 90 in the source document.

## Where to Go Next

The [Installation](getting-started/installation.md) page covers detailed setup instructions including optional dependencies and platform-specific considerations.

The [Quickstart](getting-started/quickstart.md) guide walks through building a complete citation pipeline from scratch.

The [How It Works](concepts/how-it-works.md) section explains the alignment algorithm and scoring mechanisms in depth.

The [API Reference](api/core-functions.md) provides comprehensive documentation for all public functions and classes.

## Project Information

Cite-Right is released under the Apache 2.0 license. The source code is available on [GitHub](https://github.com/avaxML/cite-right). The library requires Python 3.11 or later and includes an optional Rust extension for improved performance.

The design draws inspiration from academic work on text alignment and citation extraction. The Smith-Waterman algorithm was originally described in "Identification of Common Molecular Subsequences" by Smith and Waterman (1981). The application of sequence alignment to citation tasks builds on research in document similarity and plagiarism detection.
