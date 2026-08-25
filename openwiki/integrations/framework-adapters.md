---
type: Concept
title: Framework adapters
description: Utility functions that convert LangChain Documents, LlamaIndex nodes, and plain dictionaries into cite-right SourceDocument and SourceChunk objects for citation alignment.
tags: [integrations, langchain, llamaindex, adapters, citations]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-25T18:44:12.628Z
sources:
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-6bdbae01887b73d94b8375db
    resource: repo://src/cite_right/integrations.py
  - id: openwiki-source-fa870c43cb320c9eee84e785
    resource: repo://tests/test_integrations.py
generated: {by: "openwiki/0.4.0", at: "2026-08-25T18:44:12.628Z"}
---

# Framework adapters

The `cite_right.integrations` module bridges cite-right's citation alignment to popular RAG frameworks. It provides functions that convert external framework types into cite-right's internal source representations, plus runtime flags and type guards for mixed-pipeline use.

## Optional dependencies

Framework integrations require optional extras:

```bash
pip install cite-right[langchain]   # For LangChain support
pip install cite-right[llamaindex]  # For LlamaIndex support
```

The `from_dicts` helper has no external dependencies.

## Runtime availability flags

At import time, `cite_right.integrations` attempts to import each framework and sets a boolean flag:

```python
from cite_right.integrations import (
    LANGCHAIN_AVAILABLE,
    LLAMAINDEX_AVAILABLE,
    is_langchain_available,
    is_llamaindex_available,
)

assert is_langchain_available() == LANGCHAIN_AVAILABLE
assert is_llamaindex_available() == LLAMAINDEX_AVAILABLE
```

Both flags are `False` if the respective package is not installed. Functions that require a framework raise `ImportError` with an installation message when called without the dependency.

## Type guards for mixed pipelines

When a pipeline may return mixed types (for example, a retriever that sometimes yields LangChain documents and sometimes other objects), use the type guards to dispatch:

```python
from cite_right.integrations import (
    is_langchain_document,
    is_llamaindex_node,
)

for doc in retrieved_documents:
    if is_langchain_document(doc):
        sources.append(from_langchain_documents([doc])[0])
    elif is_llamaindex_node(doc):
        sources.append(from_llamaindex_nodes([doc])[0])
    else:
        # Handle other types
        pass
```

Both guards return `False` if their respective framework is not installed, so you do not need to check availability separately.

## Adapter functions

### `from_langchain_documents`

Converts LangChain `Document` objects into cite-right `SourceDocument` instances.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `documents` | — | Sequence of LangChain `Document` objects |
| `id_key` | `"source"` | Metadata key used as the document identifier |

Reads `page_content` for text and `metadata[id_key]` for the document ID. Falls back to the zero-based index as a string if the key is absent.

```python
from langchain_core.documents import Document
from cite_right.integrations import from_langchain_documents

docs = [
    Document(page_content="Revenue grew 15% in Q4.", metadata={"source": "report.pdf"}),
]
sources = from_langchain_documents(docs)
# sources[0] is a SourceDocument with id="report.pdf" and text="Revenue grew 15% in Q4."
```

### `from_langchain_chunks`

Converts pre-chunked LangChain `Document` objects into cite-right `SourceChunk` instances. Use this when the documents are already split from a larger source and you have character offsets.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `documents` | — | Sequence of LangChain `Document` chunks |
| `id_key` | `"source"` | Metadata key for the source document ID |
| `start_key` | `"start_index"` | Metadata key for the chunk's start offset |
| `end_key` | `"end_index"` | Metadata key for the chunk's end offset |
| `full_text_key` | `None` | Optional metadata key containing the full document text |

```python
from cite_right.integrations import from_langchain_chunks

chunks = [
    Document(
        page_content="Revenue grew 15% in Q4.",
        metadata={
            "source": "report.pdf",
            "start_index": 100,
            "end_index": 130,
        },
    )
]
sources = from_langchain_chunks(chunks)
# sources[0] is a SourceChunk with doc_char_start=100 and doc_char_end=130
```

When `full_text_key` is provided, the full document text is stored in `SourceChunk.document_text`, enabling absolute offset computation.

### `from_llamaindex_nodes`

Converts LlamaIndex `TextNode` or `NodeWithScore` objects into cite-right `SourceDocument` instances.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nodes` | — | Sequence of LlamaIndex nodes |
| `id_key` | `"file_name"` | Metadata key used as the document identifier |

Extracts content via `node.get_content()` and handles both bare `TextNode` and wrapped `NodeWithScore` objects (unwraps via `getattr(node, "node", node)`).

```python
from llama_index.core.schema import TextNode
from cite_right.integrations import from_llamaindex_nodes

nodes = [
    TextNode(text="Revenue grew 15% in Q4.", metadata={"file_name": "report.pdf"}),
]
sources = from_llamaindex_nodes(nodes)
# sources[0] is a SourceDocument with id="report.pdf"
```

### `from_llamaindex_chunks`

Converts LlamaIndex nodes with offset metadata into cite-right `SourceChunk` instances.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nodes` | — | Sequence of LlamaIndex nodes with offset metadata |
| `id_key` | `"file_name"` | Metadata key for the source document ID |
| `start_key` | `"start_char_idx"` | Metadata key for the chunk's start offset |
| `end_key` | `"end_char_idx"` | Metadata key for the chunk's end offset |

```python
from llama_index.core.schema import TextNode
from cite_right.integrations import from_llamaindex_chunks

nodes = [
    TextNode(
        text="Revenue grew 15% in Q4.",
        metadata={
            "file_name": "report.pdf",
            "start_char_idx": 100,
            "end_char_idx": 130,
        },
    )
]
sources = from_llamaindex_chunks(nodes)
# sources[0] is a SourceChunk with doc_char_start=100 and doc_char_end=130
```

### `from_dicts`

Converts plain Python dictionaries into cite-right `SourceDocument` instances. This is useful for custom RAG pipelines, API responses, or any data that is already in dict form.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `documents` | — | Sequence of dictionaries |
| `text_key` | `"text"` | Key containing the document text |
| `id_key` | `"id"` | Key containing the document identifier |

Any dict fields not used for `text_key` or `id_key` are preserved as metadata.

```python
from cite_right.integrations import from_dicts

docs = [
    {"id": "report", "text": "Revenue grew 15% in Q4.", "score": 0.95},
]
sources = from_dicts(docs)
# sources[0].id == "report"
# sources[0].metadata["score"] == 0.95
```

## Integration with `align_citations`

All adapter functions return `SourceDocument` or `SourceChunk` objects that are directly usable with `align_citations`:

```python
from cite_right import align_citations

answer = "Revenue grew 15%."
results = align_citations(answer, sources)
```

## Metadata key mapping summary

| Adapter | ID key | Start key | End key | Full text key | Text extraction |
|---------|--------|-----------|---------|---------------|-----------------|
| `from_langchain_documents` | `source` | — | — | — | `page_content` |
| `from_langchain_chunks` | `source` | `start_index` | `end_index` | optional | `page_content` |
| `from_llamaindex_nodes` | `file_name` | — | — | — | `node.get_content()` |
| `from_llamaindex_chunks` | `file_name` | `start_char_idx` | `end_char_idx` | — | `node.get_content()` |
| `from_dicts` | `id` | — | — | — | dict key |

## Exported symbols

All integration symbols are re-exported from the top-level `cite_right` package:

```python
from cite_right import (
    # Flags
    LANGCHAIN_AVAILABLE,
    LLAMAINDEX_AVAILABLE,
    # Type aliases (None when library not installed)
    LangChainDocument,
    LlamaIndexNode,
    LlamaIndexNodeWithScore,
    LlamaIndexTextNode,
    # Adapters
    from_dicts,
    from_langchain_chunks,
    from_langchain_documents,
    from_llamaindex_chunks,
    from_llamaindex_nodes,
    # Guards
    is_langchain_available,
    is_langchain_document,
    is_llamaindex_available,
    is_llamaindex_node,
)
```
