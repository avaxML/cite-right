# LlamaIndex Integration

LlamaIndex is a data framework for connecting custom data sources to large language models. Cite-Right provides helper functions that convert LlamaIndex node types to the formats expected by the citation alignment functions.

## Node Conversion

LlamaIndex represents text chunks as nodes with rich metadata including relationships to parent documents and retrieval scores. The `from_llamaindex_nodes` function converts retrieved nodes to Cite-Right source objects.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from cite_right import align_citations
from cite_right.integrations import from_llamaindex_nodes

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create retriever and retrieve nodes
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("What were the quarterly results?")

# Convert to cite-right format
sources = from_llamaindex_nodes(nodes)

# Generate answer and get citations
answer = generate_answer(query, nodes)
results = align_citations(answer, sources)
```

### ID Extraction

LlamaIndex nodes have multiple identifiers. The conversion function uses the following priority order.

The `node_id` field is preferred as it uniquely identifies the node within the index.

If the node has a `ref_doc_id` (reference to the parent document), that may be used depending on the conversion mode.

For display purposes, the original document's file name or source is often available in metadata.

```python
sources = from_llamaindex_nodes(nodes)
for source in sources:
    print(f"Source ID: {source.id}")
    print(f"Original file: {source.metadata.get('file_name', 'unknown')}")
```

### Score Preservation

LlamaIndex nodes include retrieval scores when returned from a retriever. These scores are preserved in the source metadata.

```python
sources = from_llamaindex_nodes(nodes)
for source in sources:
    retrieval_score = source.metadata.get("score")
    print(f"{source.id}: score = {retrieval_score}")
```

## Working with NodeWithScore

When you retrieve from a LlamaIndex index, you receive `NodeWithScore` objects that wrap the underlying node with its retrieval score. The integration function handles both raw nodes and scored nodes.

```python
# NodeWithScore objects (from retriever.retrieve())
nodes_with_scores = retriever.retrieve(query)
sources = from_llamaindex_nodes(nodes_with_scores)

# Raw TextNode objects also work
raw_nodes = [nws.node for nws in nodes_with_scores]
sources = from_llamaindex_nodes(raw_nodes)
```

## Document Offset Tracking

LlamaIndex nodes created by text splitters maintain relationships to their parent documents. When nodes have start and end character positions, you can use `from_llamaindex_chunks` to preserve these offsets.

```python
from cite_right.integrations import from_llamaindex_chunks

# Nodes with position metadata
sources = from_llamaindex_chunks(nodes)

for source in sources:
    if hasattr(source, 'doc_char_start'):
        print(f"Position in original: {source.doc_char_start} to {source.doc_char_end}")
```

The function checks for metadata fields like "start_char_idx" and "end_char_idx" that LlamaIndex splitters may populate.

## Complete RAG Pipeline Example

Here is a complete example showing citation integration in a LlamaIndex RAG pipeline.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from cite_right import align_citations, check_groundedness
from cite_right.integrations import from_llamaindex_nodes

# Configure LlamaIndex settings
Settings.llm = Ollama(model="llama2")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents and create index
documents = SimpleDirectoryReader("./knowledge_base").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=5)

def query_with_citations(question):
    # Execute query
    response = query_engine.query(question)

    # Get the source nodes
    source_nodes = response.source_nodes

    # Convert to cite-right format
    sources = from_llamaindex_nodes(source_nodes)

    # Compute citations
    citations = align_citations(str(response), sources)

    # Check groundedness
    metrics = check_groundedness(str(response), sources)

    return {
        "answer": str(response),
        "citations": citations,
        "groundedness": metrics.groundedness_score,
        "source_nodes": source_nodes
    }

# Use the pipeline
result = query_with_citations("What is the company's mission statement?")
print(f"Answer: {result['answer']}")
print(f"Groundedness: {result['groundedness']:.1%}")

for citation_result in result['citations']:
    print(f"\n{citation_result.answer_span.text}")
    for cite in citation_result.citations:
        print(f"  From {cite.source_id}: {cite.evidence[:50]}...")
```

## Custom Query Engines

For custom query engines that need citation support, you can wrap the retrieval and generation steps.

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever

class CitationQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    llm: Ollama

    def custom_query(self, query_str: str):
        # Retrieve nodes
        nodes = self.retriever.retrieve(query_str)

        # Generate response
        context = "\n".join(n.get_content() for n in nodes)
        prompt = f"Context: {context}\n\nQuestion: {query_str}\nAnswer:"
        response = self.llm.complete(prompt)

        # Compute citations
        sources = from_llamaindex_nodes(nodes)
        citations = align_citations(str(response), sources)

        return {
            "response": str(response),
            "citations": citations,
            "nodes": nodes
        }
```

## Response Synthesis

LlamaIndex provides various response synthesis strategies. Citations work with all of them since they operate on the final response text and retrieved nodes.

```python
from llama_index.core.response_synthesizers import ResponseMode

# Tree summarize mode
query_engine = index.as_query_engine(
    response_mode=ResponseMode.TREE_SUMMARIZE
)

response = query_engine.query(question)
sources = from_llamaindex_nodes(response.source_nodes)
citations = align_citations(str(response), sources)
```

## Metadata Filtering

LlamaIndex supports metadata filtering during retrieval. The filtered results work seamlessly with the citation integration.

```python
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value="2024"),
    ]
)

retriever = index.as_retriever(
    similarity_top_k=5,
    filters=filters
)

nodes = retriever.retrieve(query)
sources = from_llamaindex_nodes(nodes)  # Only 2024 documents
```

The metadata including filter criteria is preserved in the converted sources.

## Handling Empty Results

When retrieval returns no nodes, the integration function returns an empty list. Your application should handle this case.

```python
nodes = retriever.retrieve(query)
sources = from_llamaindex_nodes(nodes)

if not sources:
    # No relevant documents found
    return {"answer": None, "reason": "No relevant sources found"}

citations = align_citations(answer, sources)
```
