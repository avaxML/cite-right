"""Tests for framework integration helpers."""

from cite_right import (
    SourceChunk,
    SourceDocument,
    align_citations,
    from_dicts,
    from_langchain_chunks,
    from_langchain_documents,
    from_llamaindex_chunks,
    from_llamaindex_nodes,
)


class MockLangChainDocument:
    """Mock LangChain Document for testing."""

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockLlamaIndexNode:
    """Mock LlamaIndex TextNode for testing."""

    def __init__(self, content: str, metadata: dict | None = None):
        self._content = content
        self._metadata = metadata or {}

    def get_content(self) -> str:
        return self._content

    @property
    def metadata(self) -> dict:
        return self._metadata


class MockNodeWithScore:
    """Mock LlamaIndex NodeWithScore wrapper."""

    def __init__(self, node: MockLlamaIndexNode, score: float = 0.9):
        self.node = node
        self.score = score

    def get_content(self) -> str:
        """Forward to inner node for protocol compliance."""
        return self.node.get_content()

    @property
    def metadata(self) -> dict:
        """Forward to inner node for protocol compliance."""
        return self.node.metadata


class TestFromLangChainDocuments:
    """Tests for from_langchain_documents()."""

    def test_converts_basic_documents(self):
        """Should convert LangChain documents to SourceDocuments."""
        docs = [
            MockLangChainDocument("First document content.", {"source": "doc1.pdf"}),
            MockLangChainDocument("Second document content.", {"source": "doc2.pdf"}),
        ]

        sources = from_langchain_documents(docs)

        assert len(sources) == 2
        assert all(isinstance(s, SourceDocument) for s in sources)
        assert sources[0].text == "First document content."
        assert sources[0].id == "doc1.pdf"
        assert sources[1].id == "doc2.pdf"

    def test_custom_id_key(self):
        """Should support custom ID key."""
        docs = [MockLangChainDocument("Content", {"doc_id": "custom-id"})]
        sources = from_langchain_documents(docs, id_key="doc_id")
        assert sources[0].id == "custom-id"

    def test_fallback_to_index_for_missing_id(self):
        """Should use index when ID key is missing."""
        docs = [MockLangChainDocument("Content", {})]
        sources = from_langchain_documents(docs, id_key="source")
        assert sources[0].id == "0"

    def test_preserves_metadata(self):
        """Should preserve metadata from original documents."""
        docs = [MockLangChainDocument("Content", {"source": "doc.pdf", "page": 5})]
        sources = from_langchain_documents(docs)
        assert sources[0].metadata.get("page") == 5

    def test_works_with_align_citations(self):
        """Converted documents should work with align_citations."""
        docs = [MockLangChainDocument("Revenue grew 15% in Q4.", {"source": "report"})]
        sources = from_langchain_documents(docs)

        answer = "Revenue grew 15%."
        results = align_citations(answer, sources)

        assert len(results) > 0
        assert results[0].citations[0].source_id == "report"


class TestFromLangChainChunks:
    """Tests for from_langchain_chunks()."""

    def test_converts_chunks_with_offsets(self):
        """Should convert chunks with character offsets."""
        docs = [
            MockLangChainDocument(
                "chunk text here",
                {"source": "doc.pdf", "start_index": 100, "end_index": 115},
            )
        ]

        chunks = from_langchain_chunks(docs)

        assert len(chunks) == 1
        assert isinstance(chunks[0], SourceChunk)
        assert chunks[0].doc_char_start == 100
        assert chunks[0].doc_char_end == 115

    def test_custom_offset_keys(self):
        """Should support custom offset key names."""
        docs = [
            MockLangChainDocument(
                "chunk text", {"source": "doc", "begin": 50, "finish": 60}
            )
        ]
        chunks = from_langchain_chunks(docs, start_key="begin", end_key="finish")
        assert chunks[0].doc_char_start == 50
        assert chunks[0].doc_char_end == 60


class TestFromLlamaIndexNodes:
    """Tests for from_llamaindex_nodes()."""

    def test_converts_text_nodes(self):
        """Should convert LlamaIndex TextNodes to SourceDocuments."""
        nodes = [
            MockLlamaIndexNode("First node content.", {"file_name": "doc1.pdf"}),
            MockLlamaIndexNode("Second node content.", {"file_name": "doc2.pdf"}),
        ]

        sources = from_llamaindex_nodes(nodes)

        assert len(sources) == 2
        assert all(isinstance(s, SourceDocument) for s in sources)
        assert sources[0].text == "First node content."
        assert sources[0].id == "doc1.pdf"

    def test_handles_node_with_score_wrapper(self):
        """Should unwrap NodeWithScore objects."""
        inner_node = MockLlamaIndexNode("Content", {"file_name": "doc.pdf"})
        nodes = [MockNodeWithScore(inner_node, 0.95)]

        sources = from_llamaindex_nodes(nodes)

        assert len(sources) == 1
        assert sources[0].text == "Content"
        assert sources[0].id == "doc.pdf"

    def test_custom_id_key(self):
        """Should support custom ID key."""
        nodes = [MockLlamaIndexNode("Content", {"doc_id": "custom"})]
        sources = from_llamaindex_nodes(nodes, id_key="doc_id")
        assert sources[0].id == "custom"


class TestFromLlamaIndexChunks:
    """Tests for from_llamaindex_chunks()."""

    def test_converts_nodes_with_offsets(self):
        """Should convert nodes with character offset metadata."""
        nodes = [
            MockLlamaIndexNode(
                "chunk text",
                {"file_name": "doc.pdf", "start_char_idx": 100, "end_char_idx": 110},
            )
        ]

        chunks = from_llamaindex_chunks(nodes)

        assert len(chunks) == 1
        assert isinstance(chunks[0], SourceChunk)
        assert chunks[0].doc_char_start == 100
        assert chunks[0].doc_char_end == 110


class TestFromDicts:
    """Tests for from_dicts()."""

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
