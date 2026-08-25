# Cite-Right Documentation

**Link every piece of your AI-generated response back to its source text.**

Cite-Right is a Python library for citation-backed AI applications, similar to a "check sources" feature. When a language model generates an answer from retrieved documents, Cite-Right identifies which parts of the source text support each claim and returns character offsets for highlighting.

## What Cite-Right Does

Traditional citation systems link entire paragraphs or documents to generated text. Cite-Right returns **character-accurate offsets** that point to the exact location in your source documents. A frontend can highlight the supporting text when a user clicks a sentence in the answer.

0.4.0 is index-first. An inverted index and rare-token intersect choose which source windows are worth aligning. Smith-Waterman still localizes the citation. The index does not replace alignment, and it does not run Smith-Waterman over every window.

The public API is unchanged: `align_citations` and `PreparedCitationCorpus`. Span status is `"supported"`, `"partial"`, or `"unsupported"`. There is no `"partially_supported"` status.

## Core Capabilities

Cite-Right covers three parts of a RAG citation pipeline.

### Document-Source Linking

Every sentence in your generated answer can be traced back to its origin. When you call `align_citations`, the library analyzes each sentence and returns a `Citation` object containing the source document identifier along with the precise character positions where the supporting text begins and ends. Your frontend can use these offsets to scroll to and highlight the exact source passage.

### Groundedness Tagging

Not every claim in a generated answer will have source support. Cite-Right categorizes each answer span as `"supported"`, `"partial"`, or `"unsupported"` based on how well it aligns with the provided sources. Cheap contradiction (negation, number, leftover n-gram slot, entity swap) downgrades to `"partial"`, not `"unsupported"`.

The `compute_hallucination_metrics` function aggregates these results into a groundedness score. Cite-Right is a groundedness and citation tagger, not a clean hallucination detector. On RAGTruth test (2,675 answers), quality matched 0.3.1: false-supported on gold hallucinations is about 1.6%, and unsupported precision is about 14%. If `partial` counts, gold hallucinations are rarely blessed as `supported`.

### Fact-Level Verification

For applications requiring fine-grained analysis, the `verify_facts` function decomposes sentences into individual claims and verifies each one independently. This approach catches situations where a sentence combines a factual statement with an unsupported assertion.

## The Citation Object

At the heart of Cite-Right is the `Citation` data structure. Each citation contains several key pieces of information.

The `source_id` field identifies which document contains the supporting evidence. The `char_start` and `char_end` fields specify the exact byte positions within that document, using Python's standard half-open interval convention where the start is inclusive and the end is exclusive.

The `evidence` field contains the actual text extracted from the source document. You can verify that this matches the document slice by checking that `source.text[citation.char_start:citation.char_end] == citation.evidence`, a property that always holds true.

The `score` field indicates the alignment quality, with higher values representing stronger matches. The `components` dictionary breaks down this score into its constituent parts, including normalized alignment, answer coverage, evidence coverage, lexical overlap, and optional embedding similarity.

## Installation

```bash
pip install cite-right==0.4.0
```

0.4.0 ships abi3 wheels (`abi3-py311`), linux/aarch64 wheels, and an sdist.

Several optional features require additional packages. Semantic retrieval using sentence embeddings needs the embeddings extra. SpaCy-based sentence segmentation requires the spacy extra. Support for transformer tokenizers from HuggingFace or OpenAI's tiktoken are available through their respective extras.

```bash
pip install "cite-right[embeddings]==0.4.0"  # For semantic retrieval
pip install "cite-right[spacy]==0.4.0"       # For improved segmentation
pip install "cite-right[huggingface]==0.4.0" # For BERT/RoBERTa tokenizers
pip install "cite-right[tiktoken]==0.4.0"    # For GPT tokenizers
```

Rust prepare still runs when an embedder is set. Embedding-only `retrieval_support` still respects `min_embedding_similarity`.

## A Quick Example

Here is a minimal example demonstrating the core functionality.

```python
from cite_right import SourceDocument, align_citations

answer = "The company reported record revenue in Q4."
sources = [
    SourceDocument(
        id="earnings_call",
        text="During the earnings call, the CEO announced that the company reported record revenue in Q4 of 2024.",
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

This code produces output showing that the answer sentence is `"supported"` with evidence extracted from character positions 45 through 90 in the source document.

## Where to Go Next

The [Installation](getting-started/installation.md) page covers setup, wheels, and optional extras.

The [Quickstart](getting-started/quickstart.md) guide walks through building a complete citation pipeline from scratch.

The [How It Works](concepts/how-it-works.md) section explains index-first retrieval, alignment, and scoring.

The [API Reference](api/core-functions.md) documents the public functions and classes.

## Project Information

Cite-Right is released under the Apache 2.0 license. The source code is available on [GitHub](https://github.com/avaxML/cite-right). The library requires Python 3.11 or later. On the 50-case pack with no embedder, 0.4.0 p50 wall time is about 12.4ms versus about 175.8ms in 0.3.1 (roughly 14×). spp is 81.3% versus 83.4%.

The design draws inspiration from academic work on text alignment and citation extraction. The Smith-Waterman algorithm was originally described in "Identification of Common Molecular Subsequences" by Smith and Waterman (1981). The application of sequence alignment to citation tasks builds on research in document similarity and plagiarism detection.
