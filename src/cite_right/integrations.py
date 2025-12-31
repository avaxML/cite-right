"""Integration helpers for popular RAG frameworks.

This module provides utility functions to convert between cite-right's
source document types and those used by popular RAG frameworks like
LangChain and LlamaIndex.

When the optional dependencies are installed (cite-right[langchain] or
cite-right[llamaindex]), this module uses the actual library types for
better type checking and IDE support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence, Union, runtime_checkable

from cite_right.core.results import SourceChunk, SourceDocument

# Flags to track which libraries are available
LANGCHAIN_AVAILABLE = False
LLAMAINDEX_AVAILABLE = False

# Try to import LangChain types
try:
    from langchain_core.documents import Document as LangChainDocumentClass

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LangChainDocumentClass = None  # type: ignore[misc, assignment]

# Try to import LlamaIndex types
try:
    from llama_index.core.schema import NodeWithScore as LlamaIndexNodeWithScoreClass
    from llama_index.core.schema import TextNode as LlamaIndexTextNodeClass

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LlamaIndexTextNodeClass = None  # type: ignore[misc, assignment]
    LlamaIndexNodeWithScoreClass = None  # type: ignore[misc, assignment]


@runtime_checkable
class LangChainDocumentProtocol(Protocol):
    """Protocol for LangChain Document objects.

    This protocol is used when langchain-core is not installed to allow
    duck-typed objects that match the Document interface.
    """

    page_content: str
    metadata: dict[str, Any]


@runtime_checkable
class LlamaIndexNodeProtocol(Protocol):
    """Protocol for LlamaIndex NodeWithScore or TextNode objects.

    This protocol is used when llama-index-core is not installed to allow
    duck-typed objects that match the TextNode interface.
    """

    def get_content(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


# Type aliases that use real types when available
if TYPE_CHECKING:
    # For static type checking, prefer the real types if they could be available
    if LANGCHAIN_AVAILABLE:
        LangChainDocument = LangChainDocumentClass
    else:
        LangChainDocument = LangChainDocumentProtocol  # type: ignore[misc]

    if LLAMAINDEX_AVAILABLE:
        LlamaIndexNode = Union[LlamaIndexTextNodeClass, LlamaIndexNodeWithScoreClass]
        LlamaIndexTextNode = LlamaIndexTextNodeClass
        LlamaIndexNodeWithScore = LlamaIndexNodeWithScoreClass
    else:
        LlamaIndexNode = LlamaIndexNodeProtocol  # type: ignore[misc]
        LlamaIndexTextNode = LlamaIndexNodeProtocol  # type: ignore[misc]
        LlamaIndexNodeWithScore = LlamaIndexNodeProtocol  # type: ignore[misc]
else:
    # At runtime, use the real types if available, otherwise the protocols
    if LANGCHAIN_AVAILABLE:
        LangChainDocument = LangChainDocumentClass
    else:
        LangChainDocument = LangChainDocumentProtocol

    if LLAMAINDEX_AVAILABLE:
        LlamaIndexNode = (LlamaIndexTextNodeClass, LlamaIndexNodeWithScoreClass)
        LlamaIndexTextNode = LlamaIndexTextNodeClass
        LlamaIndexNodeWithScore = LlamaIndexNodeWithScoreClass
    else:
        LlamaIndexNode = LlamaIndexNodeProtocol
        LlamaIndexTextNode = LlamaIndexNodeProtocol
        LlamaIndexNodeWithScore = LlamaIndexNodeProtocol


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

    This function checks against the real LangChain Document class if
    langchain-core is installed, and also accepts objects matching the
    Document protocol (page_content and metadata attributes).

    Args:
        obj: Object to check.

    Returns:
        True if the object is a LangChain Document or matches the protocol.

    Example:
        >>> from cite_right.integrations import is_langchain_document
        >>> doc = retriever.invoke(query)[0]
        >>> if is_langchain_document(doc):
        ...     print(f"Document content: {doc.page_content}")
    """
    if LANGCHAIN_AVAILABLE and LangChainDocumentClass is not None:
        if isinstance(obj, LangChainDocumentClass):
            return True
    # Fall back to protocol check for duck-typed objects
    return isinstance(obj, LangChainDocumentProtocol)


def is_llamaindex_node(obj: Any) -> bool:
    """Check if an object is a LlamaIndex TextNode or NodeWithScore.

    This function checks against the real LlamaIndex classes if
    llama-index-core is installed, and also accepts objects matching the
    node protocol (get_content method and metadata property).

    Args:
        obj: Object to check.

    Returns:
        True if the object is a LlamaIndex node or matches the protocol.

    Example:
        >>> from cite_right.integrations import is_llamaindex_node
        >>> node = retriever.retrieve(query)[0]
        >>> if is_llamaindex_node(node):
        ...     print(f"Node content: {node.get_content()}")
    """
    if LLAMAINDEX_AVAILABLE:
        if LlamaIndexTextNodeClass is not None and LlamaIndexNodeWithScoreClass is not None:
            if isinstance(obj, (LlamaIndexTextNodeClass, LlamaIndexNodeWithScoreClass)):
                return True
    # Fall back to protocol check for duck-typed objects
    return isinstance(obj, LlamaIndexNodeProtocol)


def from_langchain_documents(
    documents: Sequence[LangChainDocumentProtocol],
    *,
    id_key: str = "source",
) -> list[SourceDocument]:
    """Convert LangChain Document objects to cite-right SourceDocuments.

    This function accepts both real LangChain Document objects (when
    langchain-core is installed) and any object matching the Document
    protocol (page_content and metadata attributes).

    Args:
        documents: Sequence of LangChain Document objects with
            ``page_content`` and ``metadata`` attributes.
        id_key: Metadata key to use as the document ID. If the key is not
            present, falls back to the document's index.

    Returns:
        List of SourceDocument objects.

    Example:
        >>> from langchain_core.documents import Document
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
    documents: Sequence[LangChainDocumentProtocol],
    *,
    id_key: str = "source",
    start_key: str = "start_index",
    end_key: str = "end_index",
    full_text_key: str | None = None,
) -> list[SourceChunk]:
    """Convert LangChain Document chunks to cite-right SourceChunks.

    Use this when your LangChain documents are pre-chunked from larger
    documents and you want citation offsets relative to the original.

    This function accepts both real LangChain Document objects (when
    langchain-core is installed) and any object matching the Document
    protocol.

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
    nodes: Sequence[LlamaIndexNodeProtocol],
    *,
    id_key: str = "file_name",
) -> list[SourceDocument]:
    """Convert LlamaIndex nodes to cite-right SourceDocuments.

    This function accepts both real LlamaIndex TextNode/NodeWithScore objects
    (when llama-index-core is installed) and any object matching the node
    protocol (get_content method and metadata property).

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
        # Handle NodeWithScore wrapper - unwrap to get the actual TextNode
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
    nodes: Sequence[LlamaIndexNodeProtocol],
    *,
    id_key: str = "file_name",
    start_key: str = "start_char_idx",
    end_key: str = "end_char_idx",
) -> list[SourceChunk]:
    """Convert LlamaIndex nodes (with offsets) to cite-right SourceChunks.

    Use this when your LlamaIndex nodes contain character offset metadata
    from the original documents.

    This function accepts both real LlamaIndex TextNode/NodeWithScore objects
    (when llama-index-core is installed) and any object matching the node
    protocol.

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
