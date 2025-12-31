"""Integration helpers for popular RAG frameworks.

This module provides utility functions to convert between cite-right's
source document types and those used by popular RAG frameworks like
LangChain and LlamaIndex.

To use these integrations, install the optional dependencies:
    pip install cite-right[langchain]    # For LangChain support
    pip install cite-right[llamaindex]   # For LlamaIndex support
"""

from __future__ import annotations

from typing import Any, Sequence

from cite_right.core.results import SourceChunk, SourceDocument

# Flags to track which libraries are available
LANGCHAIN_AVAILABLE: bool = False
LLAMAINDEX_AVAILABLE: bool = False

# Type aliases - these will be set based on library availability
LangChainDocument: type | None = None
LlamaIndexTextNode: type | None = None
LlamaIndexNodeWithScore: type | None = None
LlamaIndexNode: tuple[type, ...] | None = None

# Try to import LangChain types
try:
    from langchain_core.documents import Document as _LangChainDocument

    LangChainDocument = _LangChainDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

# Try to import LlamaIndex types
try:
    from llama_index.core.schema import NodeWithScore as _LlamaIndexNodeWithScore
    from llama_index.core.schema import TextNode as _LlamaIndexTextNode

    LlamaIndexTextNode = _LlamaIndexTextNode
    LlamaIndexNodeWithScore = _LlamaIndexNodeWithScore
    LlamaIndexNode = (_LlamaIndexTextNode, _LlamaIndexNodeWithScore)
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    pass


def _require_langchain() -> None:
    """Raise ImportError if langchain-core is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for LangChain integration. "
            "Install it with: pip install cite-right[langchain]"
        )


def _require_llamaindex() -> None:
    """Raise ImportError if llama-index-core is not installed."""
    if not LLAMAINDEX_AVAILABLE:
        raise ImportError(
            "llama-index-core is required for LlamaIndex integration. "
            "Install it with: pip install cite-right[llamaindex]"
        )


def is_langchain_available() -> bool:
    """Check if LangChain is installed and available.

    Returns:
        True if langchain-core is installed and can be imported.

    Example:
        >>> from cite_right.integrations import is_langchain_available
        >>> if is_langchain_available():
        ...     from langchain_core.documents import Document
        ...     # Use LangChain features
    """
    return LANGCHAIN_AVAILABLE


def is_llamaindex_available() -> bool:
    """Check if LlamaIndex is installed and available.

    Returns:
        True if llama-index-core is installed and can be imported.

    Example:
        >>> from cite_right.integrations import is_llamaindex_available
        >>> if is_llamaindex_available():
        ...     from llama_index.core.schema import TextNode
        ...     # Use LlamaIndex features
    """
    return LLAMAINDEX_AVAILABLE


def is_langchain_document(obj: Any) -> bool:
    """Check if an object is a LangChain Document.

    Args:
        obj: Object to check.

    Returns:
        True if the object is a LangChain Document instance.
        Returns False if langchain-core is not installed.

    Example:
        >>> from cite_right.integrations import is_langchain_document
        >>> doc = retriever.invoke(query)[0]
        >>> if is_langchain_document(doc):
        ...     print(f"Document content: {doc.page_content}")
    """
    if not LANGCHAIN_AVAILABLE or LangChainDocument is None:
        return False
    return isinstance(obj, LangChainDocument)


def is_llamaindex_node(obj: Any) -> bool:
    """Check if an object is a LlamaIndex TextNode or NodeWithScore.

    Args:
        obj: Object to check.

    Returns:
        True if the object is a LlamaIndex TextNode or NodeWithScore instance.
        Returns False if llama-index-core is not installed.

    Example:
        >>> from cite_right.integrations import is_llamaindex_node
        >>> node = retriever.retrieve(query)[0]
        >>> if is_llamaindex_node(node):
        ...     print(f"Node content: {node.get_content()}")
    """
    if not LLAMAINDEX_AVAILABLE:
        return False
    if LlamaIndexTextNode is None or LlamaIndexNodeWithScore is None:
        return False
    return isinstance(obj, (LlamaIndexTextNode, LlamaIndexNodeWithScore))


def from_langchain_documents(
    documents: Sequence[Any],
    *,
    id_key: str = "source",
) -> list[SourceDocument]:
    """Convert LangChain Document objects to cite-right SourceDocuments.

    Requires langchain-core to be installed.

    Args:
        documents: Sequence of LangChain Document objects with
            ``page_content`` and ``metadata`` attributes.
        id_key: Metadata key to use as the document ID. If the key is not
            present, falls back to the document's index.

    Returns:
        List of SourceDocument objects.

    Raises:
        ImportError: If langchain-core is not installed.

    Example:
        >>> from langchain_core.documents import Document
        >>> from cite_right import align_citations
        >>> from cite_right.integrations import from_langchain_documents
        >>>
        >>> lc_docs = retriever.invoke(query)
        >>> sources = from_langchain_documents(lc_docs)
        >>> results = align_citations(answer, sources)
    """
    _require_langchain()

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
    documents: Sequence[Any],
    *,
    id_key: str = "source",
    start_key: str = "start_index",
    end_key: str = "end_index",
    full_text_key: str | None = None,
) -> list[SourceChunk]:
    """Convert LangChain Document chunks to cite-right SourceChunks.

    Use this when your LangChain documents are pre-chunked from larger
    documents and you want citation offsets relative to the original.

    Requires langchain-core to be installed.

    Args:
        documents: Sequence of LangChain Document chunks.
        id_key: Metadata key for the source document ID.
        start_key: Metadata key for the chunk's start offset in the original.
        end_key: Metadata key for the chunk's end offset in the original.
        full_text_key: Optional metadata key containing the full document text.
            If provided, enables absolute offset computation.

    Returns:
        List of SourceChunk objects.

    Raises:
        ImportError: If langchain-core is not installed.

    Example:
        >>> from cite_right.integrations import from_langchain_chunks
        >>>
        >>> # Assuming chunks have start_index/end_index in metadata
        >>> sources = from_langchain_chunks(lc_chunks)
        >>> results = align_citations(answer, sources)
    """
    _require_langchain()

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
    nodes: Sequence[Any],
    *,
    id_key: str = "file_name",
) -> list[SourceDocument]:
    """Convert LlamaIndex nodes to cite-right SourceDocuments.

    Requires llama-index-core to be installed.

    Args:
        nodes: Sequence of LlamaIndex TextNode or NodeWithScore objects.
        id_key: Metadata key to use as the document ID.

    Returns:
        List of SourceDocument objects.

    Raises:
        ImportError: If llama-index-core is not installed.

    Example:
        >>> from cite_right.integrations import from_llamaindex_nodes
        >>>
        >>> nodes = retriever.retrieve(query)
        >>> sources = from_llamaindex_nodes(nodes)
        >>> results = align_citations(answer, sources)
    """
    _require_llamaindex()

    result: list[SourceDocument] = []
    for idx, node in enumerate(nodes):
        # Handle NodeWithScore wrapper - unwrap to get the actual TextNode
        actual_node = getattr(node, "node", node)
        content = actual_node.get_content()
        metadata = actual_node.metadata
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
    nodes: Sequence[Any],
    *,
    id_key: str = "file_name",
    start_key: str = "start_char_idx",
    end_key: str = "end_char_idx",
) -> list[SourceChunk]:
    """Convert LlamaIndex nodes (with offsets) to cite-right SourceChunks.

    Use this when your LlamaIndex nodes contain character offset metadata
    from the original documents.

    Requires llama-index-core to be installed.

    Args:
        nodes: Sequence of LlamaIndex nodes with offset metadata.
        id_key: Metadata key for the source document ID.
        start_key: Metadata key for the chunk's start offset.
        end_key: Metadata key for the chunk's end offset.

    Returns:
        List of SourceChunk objects.

    Raises:
        ImportError: If llama-index-core is not installed.

    Example:
        >>> from cite_right.integrations import from_llamaindex_chunks
        >>>
        >>> nodes = retriever.retrieve(query)
        >>> sources = from_llamaindex_chunks(nodes)
        >>> results = align_citations(answer, sources)
    """
    _require_llamaindex()

    result: list[SourceChunk] = []
    for idx, node in enumerate(nodes):
        actual_node = getattr(node, "node", node)
        content = actual_node.get_content()
        metadata = actual_node.metadata

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
