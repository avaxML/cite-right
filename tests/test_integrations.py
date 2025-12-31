"""Tests for framework integration helpers."""

import pytest

from cite_right import (
    LANGCHAIN_AVAILABLE,
    LLAMAINDEX_AVAILABLE,
    SourceChunk,
    SourceDocument,
    align_citations,
    from_dicts,
    from_langchain_chunks,
    from_langchain_documents,
    from_llamaindex_chunks,
    from_llamaindex_nodes,
    is_langchain_available,
    is_langchain_document,
    is_llamaindex_available,
    is_llamaindex_node,
)


class TestAvailabilityChecks:
    """Tests for library availability check functions."""

    def test_is_langchain_available_returns_bool(self):
        """is_langchain_available() should return a boolean."""
        result = is_langchain_available()
        assert isinstance(result, bool)
        assert result == LANGCHAIN_AVAILABLE

    def test_is_llamaindex_available_returns_bool(self):
        """is_llamaindex_available() should return a boolean."""
        result = is_llamaindex_available()
        assert isinstance(result, bool)
        assert result == LLAMAINDEX_AVAILABLE


class TestTypeChecksWithoutLibraries:
    """Tests for type checking functions when libraries are not installed."""

    @pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="Test for when langchain is NOT installed")
    def test_is_langchain_document_returns_false_when_not_installed(self):
        """is_langchain_document() should return False when library not installed."""
        assert is_langchain_document("anything") is False
        assert is_langchain_document({"page_content": "text"}) is False

    @pytest.mark.skipif(LLAMAINDEX_AVAILABLE, reason="Test for when llamaindex is NOT installed")
    def test_is_llamaindex_node_returns_false_when_not_installed(self):
        """is_llamaindex_node() should return False when library not installed."""
        assert is_llamaindex_node("anything") is False
        assert is_llamaindex_node({"text": "content"}) is False


class TestLangChainFunctionsRequireLibrary:
    """Tests that LangChain functions require the library to be installed."""

    @pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="Test for when langchain is NOT installed")
    def test_from_langchain_documents_raises_without_library(self):
        """from_langchain_documents() should raise ImportError without langchain."""
        with pytest.raises(ImportError, match="langchain-core is required"):
            from_langchain_documents([])

    @pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="Test for when langchain is NOT installed")
    def test_from_langchain_chunks_raises_without_library(self):
        """from_langchain_chunks() should raise ImportError without langchain."""
        with pytest.raises(ImportError, match="langchain-core is required"):
            from_langchain_chunks([])


class TestLlamaIndexFunctionsRequireLibrary:
    """Tests that LlamaIndex functions require the library to be installed."""

    @pytest.mark.skipif(LLAMAINDEX_AVAILABLE, reason="Test for when llamaindex is NOT installed")
    def test_from_llamaindex_nodes_raises_without_library(self):
        """from_llamaindex_nodes() should raise ImportError without llamaindex."""
        with pytest.raises(ImportError, match="llama-index-core is required"):
            from_llamaindex_nodes([])

    @pytest.mark.skipif(LLAMAINDEX_AVAILABLE, reason="Test for when llamaindex is NOT installed")
    def test_from_llamaindex_chunks_raises_without_library(self):
        """from_llamaindex_chunks() should raise ImportError without llamaindex."""
        with pytest.raises(ImportError, match="llama-index-core is required"):
            from_llamaindex_chunks([])


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-core not installed")
class TestLangChainIntegration:
    """Tests that use actual LangChain types."""

    def test_from_langchain_documents_basic(self):
        """Should convert LangChain Document objects to SourceDocuments."""
        from langchain_core.documents import Document

        docs = [
            Document(page_content="First document.", metadata={"source": "doc1.pdf"}),
            Document(page_content="Second document.", metadata={"source": "doc2.pdf"}),
        ]

        sources = from_langchain_documents(docs)

        assert len(sources) == 2
        assert all(isinstance(s, SourceDocument) for s in sources)
        assert sources[0].text == "First document."
        assert sources[0].id == "doc1.pdf"
        assert sources[1].id == "doc2.pdf"

    def test_from_langchain_documents_custom_id_key(self):
        """Should support custom ID key."""
        from langchain_core.documents import Document

        docs = [Document(page_content="Content", metadata={"doc_id": "custom-id"})]
        sources = from_langchain_documents(docs, id_key="doc_id")
        assert sources[0].id == "custom-id"

    def test_from_langchain_documents_fallback_to_index(self):
        """Should use index when ID key is missing."""
        from langchain_core.documents import Document

        docs = [Document(page_content="Content", metadata={})]
        sources = from_langchain_documents(docs, id_key="source")
        assert sources[0].id == "0"

    def test_from_langchain_documents_preserves_metadata(self):
        """Should preserve metadata from original documents."""
        from langchain_core.documents import Document

        docs = [Document(page_content="Content", metadata={"source": "doc.pdf", "page": 5})]
        sources = from_langchain_documents(docs)
        assert sources[0].metadata.get("page") == 5

    def test_is_langchain_document_with_real_type(self):
        """is_langchain_document() should detect real Document objects."""
        from langchain_core.documents import Document

        doc = Document(page_content="test", metadata={})
        assert is_langchain_document(doc) is True

    def test_is_langchain_document_with_non_document(self):
        """is_langchain_document() should return False for non-documents."""
        assert is_langchain_document("not a document") is False
        assert is_langchain_document({"page_content": "text"}) is False
        assert is_langchain_document(42) is False

    def test_from_langchain_chunks_with_offsets(self):
        """Should convert Document chunks with character offsets."""
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="chunk text here",
                metadata={"source": "doc.pdf", "start_index": 100, "end_index": 115},
            )
        ]

        chunks = from_langchain_chunks(docs)

        assert len(chunks) == 1
        assert isinstance(chunks[0], SourceChunk)
        assert chunks[0].doc_char_start == 100
        assert chunks[0].doc_char_end == 115

    def test_from_langchain_chunks_custom_offset_keys(self):
        """Should support custom offset key names."""
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="chunk text", metadata={"source": "doc", "begin": 50, "finish": 60}
            )
        ]
        chunks = from_langchain_chunks(docs, start_key="begin", end_key="finish")
        assert chunks[0].doc_char_start == 50
        assert chunks[0].doc_char_end == 60

    def test_langchain_with_align_citations(self):
        """LangChain Documents should work with align_citations."""
        from langchain_core.documents import Document

        docs = [Document(page_content="Revenue grew 15% in Q4.", metadata={"source": "report"})]
        sources = from_langchain_documents(docs)

        answer = "Revenue grew 15%."
        results = align_citations(answer, sources)

        assert len(results) > 0
        assert results[0].citations[0].source_id == "report"


@pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="llama-index-core not installed")
class TestLlamaIndexIntegration:
    """Tests that use actual LlamaIndex types."""

    def test_from_llamaindex_nodes_basic(self):
        """Should convert LlamaIndex TextNode objects to SourceDocuments."""
        from llama_index.core.schema import TextNode

        nodes = [
            TextNode(text="First node content.", metadata={"file_name": "doc1.pdf"}),
            TextNode(text="Second node content.", metadata={"file_name": "doc2.pdf"}),
        ]

        sources = from_llamaindex_nodes(nodes)

        assert len(sources) == 2
        assert all(isinstance(s, SourceDocument) for s in sources)
        assert sources[0].text == "First node content."
        assert sources[0].id == "doc1.pdf"

    def test_from_llamaindex_nodes_with_node_with_score(self):
        """Should work with NodeWithScore objects."""
        from llama_index.core.schema import NodeWithScore, TextNode

        inner_node = TextNode(text="Node content.", metadata={"file_name": "doc.pdf"})
        nodes = [NodeWithScore(node=inner_node, score=0.95)]

        sources = from_llamaindex_nodes(nodes)

        assert len(sources) == 1
        assert sources[0].text == "Node content."
        assert sources[0].id == "doc.pdf"

    def test_from_llamaindex_nodes_custom_id_key(self):
        """Should support custom ID key."""
        from llama_index.core.schema import TextNode

        nodes = [TextNode(text="Content", metadata={"doc_id": "custom"})]
        sources = from_llamaindex_nodes(nodes, id_key="doc_id")
        assert sources[0].id == "custom"

    def test_is_llamaindex_node_with_text_node(self):
        """is_llamaindex_node() should detect real TextNode objects."""
        from llama_index.core.schema import TextNode

        node = TextNode(text="test", metadata={})
        assert is_llamaindex_node(node) is True

    def test_is_llamaindex_node_with_node_with_score(self):
        """is_llamaindex_node() should detect real NodeWithScore objects."""
        from llama_index.core.schema import NodeWithScore, TextNode

        inner = TextNode(text="test", metadata={})
        wrapped = NodeWithScore(node=inner, score=0.9)
        assert is_llamaindex_node(wrapped) is True

    def test_is_llamaindex_node_with_non_node(self):
        """is_llamaindex_node() should return False for non-nodes."""
        assert is_llamaindex_node("not a node") is False
        assert is_llamaindex_node({"content": "text"}) is False
        assert is_llamaindex_node(42) is False

    def test_from_llamaindex_chunks_with_offsets(self):
        """Should convert TextNodes with character offset metadata."""
        from llama_index.core.schema import TextNode

        nodes = [
            TextNode(
                text="chunk text",
                metadata={"file_name": "doc.pdf", "start_char_idx": 100, "end_char_idx": 110},
            )
        ]

        chunks = from_llamaindex_chunks(nodes)

        assert len(chunks) == 1
        assert isinstance(chunks[0], SourceChunk)
        assert chunks[0].doc_char_start == 100
        assert chunks[0].doc_char_end == 110

    def test_llamaindex_with_align_citations(self):
        """LlamaIndex nodes should work with align_citations."""
        from llama_index.core.schema import TextNode

        nodes = [TextNode(text="Revenue grew 15% in Q4.", metadata={"file_name": "report.pdf"})]
        sources = from_llamaindex_nodes(nodes)

        answer = "Revenue grew 15%."
        results = align_citations(answer, sources)

        assert len(results) > 0
        assert sources[0].id == "report.pdf"


class TestFromDicts:
    """Tests for from_dicts() - works without external dependencies."""

    def test_converts_basic_dicts(self):
        """Should convert plain dictionaries to SourceDocuments."""
        docs = [
            {"id": "doc1", "text": "First document."},
            {"id": "doc2", "text": "Second document."},
        ]

        sources = from_dicts(docs)

        assert len(sources) == 2
        assert sources[0].id == "doc1"
        assert sources[0].text == "First document."

    def test_custom_keys(self):
        """Should support custom key names."""
        docs = [{"doc_id": "myid", "content": "My content."}]
        sources = from_dicts(docs, id_key="doc_id", text_key="content")

        assert sources[0].id == "myid"
        assert sources[0].text == "My content."

    def test_stores_extra_fields_as_metadata(self):
        """Should store extra dict fields as metadata."""
        docs = [{"id": "doc", "text": "Content", "score": 0.9, "page": 5}]
        sources = from_dicts(docs)

        assert sources[0].metadata.get("score") == 0.9
        assert sources[0].metadata.get("page") == 5

    def test_fallback_to_index_for_missing_id(self):
        """Should use index when ID is missing."""
        docs = [{"text": "Content without ID."}]
        sources = from_dicts(docs)
        assert sources[0].id == "0"

    def test_works_with_align_citations(self):
        """Converted dicts should work with align_citations."""
        docs = [{"id": "report", "text": "Revenue grew 15% in Q4.", "relevance": 0.95}]
        sources = from_dicts(docs)

        answer = "Revenue grew 15%."
        results = align_citations(answer, sources)

        assert len(results) > 0
