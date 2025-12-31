# Custom Sources

Not all applications use LangChain or LlamaIndex. Cite-Right provides flexible input options for integrating with any retrieval system or custom data pipeline.

## Using SourceDocument Directly

The most straightforward approach is creating `SourceDocument` objects directly. This class is the primary input format for citation alignment.

```python
from cite_right import SourceDocument, align_citations

sources = [
    SourceDocument(
        id="doc_1",
        text="The full text of the first document goes here.",
        metadata={"author": "Smith", "year": 2024}
    ),
    SourceDocument(
        id="doc_2",
        text="The full text of the second document goes here."
    )
]

results = align_citations(answer, sources)
```

The `id` field is a unique identifier used to reference the source in citation results. Choose IDs that are meaningful in your application, such as database keys, file paths, or URLs.

The `text` field contains the complete document text. Citation alignment will find matching regions within this text and return character offsets pointing to specific locations.

The `metadata` field is optional and can contain any additional information you want to associate with the document. This metadata is preserved through the alignment process and accessible in results.

## Using from_dicts

For data coming from APIs, databases, or JSON files, the `from_dicts` function provides convenient conversion.

```python
from cite_right.integrations import from_dicts
from cite_right import align_citations

# Data from an API response
api_response = [
    {"id": "result_1", "content": "First document text...", "score": 0.95},
    {"id": "result_2", "content": "Second document text...", "score": 0.87},
]

sources = from_dicts(api_response)
results = align_citations(answer, sources)
```

### Field Mapping

The function looks for content in several common field names.

For the text content, it checks "text", "content", "page_content", and "body" in that order. The first non-empty field found is used.

For the identifier, it checks "id", "doc_id", "document_id", and "source". If none are found, it generates IDs like "doc_0", "doc_1", etc.

All other fields become metadata.

```python
docs = [
    {"body": "Document content", "url": "https://example.com/doc1"}
]

sources = from_dicts(docs)
# sources[0].text == "Document content"
# sources[0].id == "doc_0"
# sources[0].metadata == {"url": "https://example.com/doc1"}
```

### Custom Field Names

If your data uses different field names, you can map them explicitly.

```python
docs = [
    {"document_text": "Content here", "document_uuid": "abc123"}
]

# Rename fields before conversion
standardized = [
    {"text": d["document_text"], "id": d["document_uuid"]}
    for d in docs
]

sources = from_dicts(standardized)
```

## Using SourceChunk for Pre-Chunked Content

When your retrieval system already divides documents into chunks and tracks their positions, use `SourceChunk` to preserve offset information.

```python
from cite_right import SourceChunk, align_citations

# Chunks from a retrieval system that tracks positions
chunks = [
    SourceChunk(
        source_id="report_2024",
        text="This is the text of chunk 1.",
        doc_char_start=0,
        doc_char_end=28
    ),
    SourceChunk(
        source_id="report_2024",
        text="This is the text of chunk 2.",
        doc_char_start=29,
        doc_char_end=57
    ),
]

results = align_citations(answer, chunks)
```

The `source_id` identifies the parent document that the chunk came from. Multiple chunks can share the same `source_id`.

The `doc_char_start` and `doc_char_end` specify where this chunk appears in the original document. These offsets are added to citation character positions, so the final offsets refer to the complete document rather than the chunk.

### When to Use Chunks

Use `SourceChunk` when your retrieval produces excerpts with known positions and you want citations to reference the original document. This is common with vector databases that store chunked content alongside offset metadata.

Use `SourceDocument` when you have complete documents or when chunk positions are unknown. The library will segment and window the documents internally.

## Database Integration Example

Here is an example of integrating with a PostgreSQL database containing documents.

```python
import psycopg2
from cite_right import SourceDocument, align_citations

def get_relevant_docs(query, connection, limit=10):
    cursor = connection.cursor()
    cursor.execute("""
        SELECT id, content, title, created_at
        FROM documents
        WHERE to_tsvector(content) @@ plainto_tsquery(%s)
        ORDER BY ts_rank(to_tsvector(content), plainto_tsquery(%s)) DESC
        LIMIT %s
    """, (query, query, limit))

    sources = []
    for row in cursor.fetchall():
        doc_id, content, title, created_at = row
        sources.append(SourceDocument(
            id=str(doc_id),
            text=content,
            metadata={"title": title, "created_at": str(created_at)}
        ))

    return sources

# Usage
connection = psycopg2.connect(...)
sources = get_relevant_docs("quarterly results", connection)
results = align_citations(answer, sources)
```

## Elasticsearch Integration Example

For Elasticsearch-based retrieval systems:

```python
from elasticsearch import Elasticsearch
from cite_right import SourceDocument, align_citations

es = Elasticsearch()

def search_and_cite(query, answer):
    # Retrieve from Elasticsearch
    response = es.search(
        index="documents",
        body={
            "query": {"match": {"content": query}},
            "size": 10
        }
    )

    # Convert to SourceDocument
    sources = []
    for hit in response["hits"]["hits"]:
        sources.append(SourceDocument(
            id=hit["_id"],
            text=hit["_source"]["content"],
            metadata={
                "score": hit["_score"],
                "index": hit["_index"],
                **hit["_source"]
            }
        ))

    # Compute citations
    return align_citations(answer, sources)
```

## REST API Integration

For custom retrieval APIs:

```python
import requests
from cite_right.integrations import from_dicts
from cite_right import align_citations

def fetch_and_cite(query, answer):
    # Call your retrieval API
    response = requests.post(
        "https://api.example.com/search",
        json={"query": query, "limit": 10}
    )
    response.raise_for_status()

    # Convert the response
    docs = response.json()["results"]
    sources = from_dicts(docs)

    # Compute citations
    return align_citations(answer, sources)
```

## Mixing Source Types

You can mix `SourceDocument` and `SourceChunk` in the same call. The library handles each according to its type.

```python
sources = [
    SourceDocument(id="full_doc", text="Complete document text..."),
    SourceChunk(source_id="chunked_doc", text="Chunk text...", doc_char_start=100, doc_char_end=200),
]

results = align_citations(answer, sources)
```

This flexibility allows integration with hybrid retrieval systems that return both complete documents and pre-chunked excerpts.

## Validation

The library validates input types and raises clear errors for invalid data.

```python
# Empty text raises no error but produces no citations
sources = [SourceDocument(id="empty", text="")]
results = align_citations(answer, sources)  # Works, but finds no matches

# Invalid types raise errors
sources = [{"text": "..."}]  # Not a SourceDocument
results = align_citations(answer, sources)  # TypeError
```

For production systems, validate your data before calling the alignment functions.

```python
def validate_sources(sources):
    if not sources:
        raise ValueError("At least one source document is required")
    for i, source in enumerate(sources):
        if not source.text.strip():
            raise ValueError(f"Source at index {i} has empty text")
    return sources
```
