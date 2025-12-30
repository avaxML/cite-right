"""Integration helpers for popular RAG frameworks.

This module provides utility functions to convert between cite-right's
source document types and those used by popular RAG frameworks like
LangChain and LlamaIndex.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from cite_right.core.results import SourceChunk, SourceDocument


@runtime_checkable
class LangChainDocument(Protocol):
    """Protocol for LangChain Document objects."""

    page_content: str
    metadata: dict[str, Any]


@runtime_checkable
class LlamaIndexNode(Protocol):
    """Protocol for LlamaIndex NodeWithScore or TextNode objects."""

    def get_content(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


def from_langchain_documents(
    documents: Sequence[LangChainDocument],
    *,
    id_key: str = "source",
) -> list[SourceDocument]:
    """Convert LangChain Document objects to cite-right SourceDocuments.

    Args:
        documents: Sequence of LangChain Document objects with
            ``page_content`` and ``metadata`` attributes.
        id_key: Metadata key to use as the document ID. If the key is not
            present, falls back to the document's index.

    Returns:
        List of SourceDocument objects.

    Example:
        >>> from langchain.schema import Document
        >>> from cite_right import align_citations
        >>> from cite_right.integrations import from_langchain_documents
        >>>
        >>> lc_docs = retriever.invoke(query)
        >>> sources = from_langchain_documents(lc_docs)
        >>> results = align_citations(answer, sources)
    """
    result: list[SourceDocument] = []
    for idx, doc in enumerate(documents):
        doc_id = doc.metadata.get(id_key, str(idx))
        result.append(
            SourceDocument(
                id=str(doc_id),
                text=doc.page_content,
                metadata=doc.metadata,
            )
        )
    return result


def from_langchain_chunks(
    documents: Sequence[LangChainDocument],
    *,
    id_key: str = "source",
    start_key: str = "start_index",
    end_key: str = "end_index",
    full_text_key: str | None = None,
) -> list[SourceChunk]:
    """Convert LangChain Document chunks to cite-right SourceChunks.

    Use this when your LangChain documents are pre-chunked from larger
    documents and you want citation offsets relative to the original.

    Args:
        documents: Sequence of LangChain Document chunks.
        id_key: Metadata key for the source document ID.
        start_key: Metadata key for the chunk's start offset in the original.
        end_key: Metadata key for the chunk's end offset in the original.
        full_text_key: Optional metadata key containing the full document text.
            If provided, enables absolute offset computation.

    Returns:
        List of SourceChunk objects.

    Example:
        >>> from cite_right.integrations import from_langchain_chunks
        >>>
        >>> # Assuming chunks have start_index/end_index in metadata
        >>> sources = from_langchain_chunks(lc_chunks)
        >>> results = align_citations(answer, sources)
    """
    result: list[SourceChunk] = []
    for idx, doc in enumerate(documents):
        doc_id = doc.metadata.get(id_key, str(idx))
        start = doc.metadata.get(start_key, 0)
        end = doc.metadata.get(end_key, start + len(doc.page_content))
        full_text = doc.metadata.get(full_text_key) if full_text_key else None

        result.append(
            SourceChunk(
                source_id=str(doc_id),
                text=doc.page_content,
                doc_char_start=start,
                doc_char_end=end,
                metadata=doc.metadata,
                document_text=full_text,
                source_index=idx,
            )
        )
    return result


def from_llamaindex_nodes(
    nodes: Sequence[LlamaIndexNode],
    *,
    id_key: str = "file_name",
) -> list[SourceDocument]:
    """Convert LlamaIndex nodes to cite-right SourceDocuments.

    Args:
        nodes: Sequence of LlamaIndex TextNode or NodeWithScore objects.
        id_key: Metadata key to use as the document ID.

    Returns:
        List of SourceDocument objects.

    Example:
        >>> from cite_right.integrations import from_llamaindex_nodes
        >>>
        >>> nodes = retriever.retrieve(query)
        >>> sources = from_llamaindex_nodes(nodes)
        >>> results = align_citations(answer, sources)
    """
    result: list[SourceDocument] = []
    for idx, node in enumerate(nodes):
        # Handle NodeWithScore wrapper
        actual_node = getattr(node, "node", node)
        content = (
            actual_node.get_content()
            if hasattr(actual_node, "get_content")
            else str(actual_node)
        )
        metadata = getattr(actual_node, "metadata", {})
        doc_id = metadata.get(id_key, str(idx))

        result.append(
            SourceDocument(
                id=str(doc_id),
                text=content,
                metadata=metadata,
            )
        )
    return result


def from_llamaindex_chunks(
    nodes: Sequence[LlamaIndexNode],
    *,
    id_key: str = "file_name",
    start_key: str = "start_char_idx",
    end_key: str = "end_char_idx",
) -> list[SourceChunk]:
    """Convert LlamaIndex nodes (with offsets) to cite-right SourceChunks.

    Use this when your LlamaIndex nodes contain character offset metadata
    from the original documents.

    Args:
        nodes: Sequence of LlamaIndex nodes with offset metadata.
        id_key: Metadata key for the source document ID.
        start_key: Metadata key for the chunk's start offset.
        end_key: Metadata key for the chunk's end offset.

    Returns:
        List of SourceChunk objects.

    Example:
        >>> from cite_right.integrations import from_llamaindex_chunks
        >>>
        >>> nodes = retriever.retrieve(query)
        >>> sources = from_llamaindex_chunks(nodes)
        >>> results = align_citations(answer, sources)
    """
    result: list[SourceChunk] = []
    for idx, node in enumerate(nodes):
        actual_node = getattr(node, "node", node)
        content = (
            actual_node.get_content()
            if hasattr(actual_node, "get_content")
            else str(actual_node)
        )
        metadata = getattr(actual_node, "metadata", {})

        doc_id = metadata.get(id_key, str(idx))
        start = metadata.get(start_key, 0)
        end = metadata.get(end_key, start + len(content))

        result.append(
            SourceChunk(
                source_id=str(doc_id),
                text=content,
                doc_char_start=start if start is not None else 0,
                doc_char_end=end if end is not None else len(content),
                metadata=metadata,
                source_index=idx,
            )
        )
    return result


def from_dicts(
    documents: Sequence[dict[str, Any]],
    *,
    text_key: str = "text",
    id_key: str = "id",
) -> list[SourceDocument]:
    """Convert plain dictionaries to cite-right SourceDocuments.

    This is useful for custom RAG pipelines or API responses.

    Args:
        documents: Sequence of dictionaries with text content.
        text_key: Key containing the document text.
        id_key: Key containing the document ID.

    Returns:
        List of SourceDocument objects.

    Example:
        >>> docs = [{"id": "doc1", "text": "...", "score": 0.9}]
        >>> sources = from_dicts(docs)
        >>> results = align_citations(answer, sources)
    """
    result: list[SourceDocument] = []
    for idx, doc in enumerate(documents):
        text = doc.get(text_key, "")
        doc_id = doc.get(id_key, str(idx))
        # Store all other keys as metadata
        metadata = {k: v for k, v in doc.items() if k not in (text_key, id_key)}

        result.append(
            SourceDocument(
                id=str(doc_id),
                text=str(text),
                metadata=metadata,
            )
        )
    return result
