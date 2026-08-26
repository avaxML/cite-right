---
type: integration-guide
title: LlamaIndex Integration
description: How to feed LlamaIndex retrievers into Cite-Right — convert TextNode and NodeWithScore lists to SourceDocument with from_llamaindex_nodes, preserve chunk offsets with from_llamaindex_chunks, and call align_citations. Covers the id_key fallback, the file_name default, start_char_idx handling, and LLAMAINDEX_AVAILABLE.
tags: [llamaindex, integration, from-llamaindex-nodes, from-llamaindex-chunks, source-document, source-chunk, text-node, node-with-score, file-name, start-char-idx, align-citations, rag, retriever, query-engine, is-llamaindex-available]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
  - id: openwiki-source-6bdbae01887b73d94b8375db
    resource: repo://src/cite_right/integrations.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# LlamaIndex Integration

LlamaIndex retrievers hand back `TextNode` and `NodeWithScore` objects from `llama_index.core.schema`, each with a `get_content()` method and a `metadata` dict. Cite-Right ships two adapter functions that turn those into `SourceDocument` and `SourceChunk` so you can call `align_citations` on the result without rewriting your pipeline. The adapters live in `src/cite_right/integrations.py` and are also exported from the top-level `cite_right` package.

This page walks the two conversions, the `id_key` and `start_key` / `end_key` knobs, what happens when offsets are missing, and the full pattern for a LlamaIndex query engine plus a `CustomQueryEngine` subclass. The underlying I/O contract is in [Citation Alignment](../concepts/citation-alignment.md); the chunk-rebase offset convention is in [Custom Sources](custom-sources.md). The parallel LangChain adapters are in [LangChain Integration](langchain.md).

## Install The Optional Extra

The LlamaIndex adapters import `llama_index.core.schema` at module load. If that package is not present, the import paths are still exported but resolve to `None`, the `LLAMAINDEX_AVAILABLE` flag is `False`, and the adapter functions raise `ImportError` with a hint.

```bash
pip install "cite-right[llamaindex]==0.4.0"
```

The extra is `llama-index-core>=0.11.0` (see `pyproject.toml`). You do not need the full LlamaIndex distribution; `llama-index-core` is enough to use the adapters with any LlamaIndex retriever or query engine (`VectorStoreIndex`, `SimpleDirectoryReader`, the `llama-index-integrations` readers, etc.).

You can check whether the import succeeded without trying an adapter call:

```python
from cite_right import is_llamaindex_available, LLAMAINDEX_AVAILABLE

if is_llamaindex_available():
    # llama_index.core is importable; the adapters will work
    ...
```

`LLAMAINDEX_AVAILABLE` is also exposed as a module-level boolean. `is_llamaindex_node(obj)` is the type guard for retriever results: it returns `True` for real `llama_index.core.schema.TextNode` and `NodeWithScore` instances, and `False` for anything else.

The two type handles are also re-exported from the top-level package so you can use them in `isinstance` checks without an extra import:

```python
from cite_right import LlamaIndexTextNode, LlamaIndexNodeWithScore, LlamaIndexNode

if isinstance(obj, LlamaIndexNode):
    # TextNode or NodeWithScore
    ...
```

`LlamaIndexNode` is the 2-tuple `(LlamaIndexTextNode, LlamaIndexNodeWithScore)`, intended for use with `isinstance(obj, LlamaIndexNode)`.

## Converting Nodes

`from_llamaindex_nodes(nodes, *, id_key="file_name")` takes a sequence of LlamaIndex `TextNode` or `NodeWithScore` objects and returns a list of `SourceDocument`. The text comes from `node.get_content()`, the metadata is passed through unchanged, and the `id` is read from `node.metadata[id_key]` — falling back to the node's integer index stringified when that key is missing.

The function unwraps `NodeWithScore` transparently: when the element has a `.node` attribute (the `NodeWithScore` shape that `Retriever.retrieve` returns), the adapter reads `get_content()` and `metadata` off the inner node and discards the score wrapper.

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

The function calls `_require_llamaindex()` first, so a missing `llama-index-core` install fails fast with a clear message rather than producing a half-built list.

### The id_key Fallback And Why The Default Is `file_name`

LlamaIndex text splitters and `SimpleDirectoryReader` write the source file path into `node.metadata["file_name"]` by default. That is the convention the adapter's `id_key="file_name"` default matches, so for a typical `SimpleDirectoryReader` retriever you get the original filename on every `Citation.source_id` without any extra configuration.

```python
from llama_index.core.schema import TextNode
from cite_right.integrations import from_llamaindex_nodes

# With "file_name" in metadata: that value becomes the SourceDocument id
nodes = [TextNode(text="...", metadata={"file_name": "annual_report.pdf"})]
sources = from_llamaindex_nodes(nodes)
assert sources[0].id == "annual_report.pdf"

# Without the key: the index stringifies to "0", "1", ...
nodes = [TextNode(text="...")]
sources = from_llamaindex_nodes(nodes)
assert sources[0].id == "0"
```

If your loader writes a different metadata key, pass it explicitly:

```python
# A loader that puts the document id under "doc_id"
sources = from_llamaindex_nodes(nodes, id_key="doc_id")

# A loader that uses the node id
sources = from_llamaindex_nodes(nodes, id_key="id_")
```

The `id` is what every `Citation.source_id` will carry, so a stable identifier here is what makes it possible to map a citation back to the source row in your retrieval layer.

### Score Preservation On NodeWithScore

`NodeWithScore` wraps a `BaseNode` with a retrieval score, and the wrapper has its own `.score` attribute. The adapter unwraps it to read content and metadata but does not pull the score into the resulting `SourceDocument`. Retrieval scores stay in your application code; `metadata` is the bridge.

```python
nodes = retriever.retrieve(query)  # list[NodeWithScore]
sources = from_llamaindex_nodes(nodes)
for source, node in zip(sources, nodes):
    retrieval_score = getattr(node, "score", None)
    print(f"{source.id}: score = {retrieval_score}")
```

If you need the score inside `SourceDocument.metadata` for downstream logging, copy it in yourself before calling `align_citations`:

```python
sources = from_llamaindex_nodes(nodes)
for source, node in zip(sources, nodes):
    if hasattr(node, "score"):
        source.metadata["retrieval_score"] = node.score
```

### Metadata Preservation

The full `node.metadata` dict is copied onto the resulting `SourceDocument.metadata`. Cite-Right does not read it during alignment — it is opaque to the pipeline — but your application code can read it back to recover anything the retriever or splitter attached: file names, page numbers, retrieval scores, custom tags.

```python
sources = from_llamaindex_nodes(nodes)
for source in sources:
    print(f"id={source.id}")
    print(f"file_name={source.metadata.get('file_name')}")
    print(f"ref_doc_id={source.metadata.get('ref_doc_id')}")
```

Metadata round-trips through `align_citations` unchanged, so it is the right place to keep any retriever-specific extras you want to expose in your final answer payload.

## Working With NodeWithScore And Raw TextNode

`Retriever.retrieve` returns `list[NodeWithScore]`; `index.docstore.get_node(node_id)` and most other docstore reads return raw `TextNode`. The adapter accepts either shape, and unwraps `NodeWithScore` automatically.

```python
# NodeWithScore objects (from retriever.retrieve())
nodes_with_scores = retriever.retrieve(query)
sources = from_llamaindex_nodes(nodes_with_scores)

# Raw TextNode objects also work
raw_nodes = [nws.node for nws in nodes_with_scores]
sources = from_llamaindex_nodes(raw_nodes)
```

The unwrap is one line: `actual_node = getattr(node, "node", node)`. The same unwrap is used by `from_llamaindex_chunks`, so a list with mixed `TextNode` and `NodeWithScore` elements is handled the same way.

## Converting Chunks With Offsets

LlamaIndex text splitters (`SentenceSplitter`, `TokenTextSplitter`, `SemanticSplitterNodeParser`, and others) split a `Document` into smaller nodes and write character offsets into `node.metadata["start_char_idx"]` and `node.metadata["end_char_idx"]` by default. `from_llamaindex_chunks` reads those offsets and produces `SourceChunk` objects whose `doc_char_start` / `doc_char_end` point at the parent. Citation offsets then come out absolute in the original document, not in the chunk.

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from cite_right.integrations import from_llamaindex_chunks

documents = SimpleDirectoryReader("./data").load_data()
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# Nodes now have start_char_idx and end_char_idx in metadata
sources = from_llamaindex_chunks(nodes)
results = align_citations(answer, sources)

for chunk in sources:
    print(f"Chunk from {chunk.source_id}")
    print(f"Position in original: {chunk.doc_char_start} to {chunk.doc_char_end}")
```

The function signature is:

```python
from_llamaindex_chunks(
    nodes,
    *,
    id_key="file_name",
    start_key="start_char_idx",
    end_key="end_char_idx",
) -> list[SourceChunk]
```

`start_key` and `end_key` default to `"start_char_idx"` and `"end_char_idx"`, matching the LlamaIndex splitter convention. If the node has `start_char_idx` but no `end_char_idx`, the adapter computes the end as `start + len(content)`, so a single-offset splitter still produces a valid `SourceChunk` range.

### Why from_llamaindex_chunks Omits document_text

The chunk adapter does not set `SourceChunk.document_text`, even when the parent text is reachable. The reason is that the typical LlamaIndex node carries the chunk text and the offsets, but not the full parent — the parent lives in the docstore, not on the node itself. Passing `document_text` would require an extra docstore lookup that the adapter does not perform.

The rebase still produces correct absolute offsets without `document_text`: `_slice_source_text` in the alignment pipeline falls back to chunk-local slicing when `full_text` is not supplied. The rebase contract is in [Custom Sources](custom-sources.md#chunk-rebase-and-the-evidence-equality-invariant).

If you have the parent text in memory and want the slice-equality check, build the `SourceChunk` directly with `document_text=parent_text` rather than going through the adapter.

### Missing Offsets

Not every LlamaIndex node carries offsets. Nodes that came from a manual `TextNode(text=...)` construction, a custom node parser, or a reader that does not write `start_char_idx` will have neither metadata field. The adapter falls back to `0` and `start + len(content)`, producing a well-formed `SourceChunk` whose offsets no longer point at the parent — they point at the chunk. If you need parent-absolute offsets, configure the splitter to emit them; there is no post-hoc recovery.

```python
# Splitter that does not write offsets
from llama_index.core.node_parser import SimpleNodeParser

parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)
# nodes have no start_char_idx / end_char_idx in metadata
sources = from_llamaindex_chunks(nodes)
# sources[0].doc_char_start == 0, doc_char_end == len(chunk.text)
```

## End-To-End RAG Pipeline

The typical flow: load documents into a `VectorStoreIndex`, attach an LLM and embedder, retrieve nodes from a query, generate the answer, then convert the retrieved nodes and align citations. Cite-Right fits in after the LLM step; the retrieval and generation layers are unchanged.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from cite_right import align_citations, check_groundedness
from cite_right.integrations import from_llamaindex_nodes

# Configure LlamaIndex settings
Settings.llm = Ollama(model="llama2")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load documents and create index
documents = SimpleDirectoryReader("./knowledge_base").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=5)


def query_with_citations(question: str) -> dict:
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
        "source_nodes": source_nodes,
    }


result = query_with_citations("What is the company's mission statement?")
print(f"Answer: {result['answer']}")
print(f"Groundedness: {result['groundedness']:.1%}")

for citation_result in result["citations"]:
    print(f"\n{citation_result.answer_span.text}")
    for cite in citation_result.citations:
        print(f"  From {cite.source_id}: {cite.evidence[:50]}...")
```

The same pattern works with any LlamaIndex retriever, response synthesizer, or chat model. The integration boundary is the `from_llamaindex_nodes(source_nodes)` call — everything before it is plain LlamaIndex; everything after is plain Cite-Right.

## Custom Query Engines

For custom query engines that need citation support, the adapter fits between retrieval and the final response. The pattern is the same as the standard `query_engine.query` flow: retrieve, generate, then convert and align.

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.llms.types import CompletionResponse
from cite_right import align_citations
from cite_right.integrations import from_llamaindex_nodes


class CitationQueryEngine(CustomQueryEngine):
    """Custom query engine that returns citations alongside the answer."""

    retriever: BaseRetriever

    def custom_query(self, query_str: str):
        # Retrieve nodes
        nodes = self.retriever.retrieve(query_str)

        # Build context from retrieved nodes
        context = "\n".join(n.get_content() for n in nodes)

        # Generate response via the configured LLM
        response = self.llm.complete(
            f"Context: {context}\n\nQuestion: {query_str}\nAnswer:"
        )
        answer_text = response.text

        # Convert and align
        sources = from_llamaindex_nodes(nodes)
        citations = align_citations(answer_text, sources)

        return {
            "response": answer_text,
            "citations": citations,
            "nodes": nodes,
        }
```

The adapter does not care whether the LLM is an Ollama, OpenAI, Anthropic, or a local model. It operates on the final answer text and the list of retrieved nodes, both of which the custom engine already has in scope.

## Response Synthesis Modes

LlamaIndex's response synthesizers (`ResponseMode.TREE_SUMMARIZE`, `ResponseMode.REFINE`, `ResponseMode.COMPACT`, `ResponseMode.SIMPLE_SUMMARIZE`, and others) all produce a final response with `source_nodes` attached. The adapter is mode-agnostic because it operates on the post-synthesis result.

```python
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

synth = get_response_synthesizer(response_mode="tree_summarize")
query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)

response = query_engine.query(question)
sources = from_llamaindex_nodes(response.source_nodes)
citations = align_citations(str(response), sources)
```

The same pattern holds for `as_chat_engine`, sub-question query engines, and any other engine that exposes `response.source_nodes`.

## Metadata Filtering

LlamaIndex retrievers support pre-filtering on metadata. The filter does not affect the adapter — the nodes that come back are still `TextNode` or `NodeWithScore`, and the conversion is unchanged. The full metadata, including any filter criteria and the values they matched on, is preserved on the resulting `SourceDocument.metadata`.

```python
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="year", value="2024"),
    ]
)

retriever = index.as_retriever(similarity_top_k=5, filters=filters)
nodes = retriever.retrieve(query)
sources = from_llamaindex_nodes(nodes)  # only 2024 documents
```

## Error Handling

The adapters are permissive on the input side and strict on the integration boundary. The behavior, in order of failure detection:

1. **Missing `llama-index-core`.** `from_llamaindex_nodes` and `from_llamaindex_chunks` call `_require_llamaindex()`, which raises `ImportError` with the install hint. The function never returns a half-built list.
2. **Missing `id_key` metadata.** Falls back to `str(idx)`, where `idx` is the node's position in the input sequence. You get a stable id per call but no connection to your real source row. Pass `id_key` explicitly when your loader uses a non-default name.
3. **Missing offsets in chunk conversion.** Falls back to `0` and `start + len(content)`. The resulting `SourceChunk` is valid but its offsets do not point at the parent. There is no post-hoc recovery.
4. **Empty nodes.** A `TextNode(text="")` produces a `SourceDocument` with empty text. `align_citations` will not produce a citation against an empty source, but it will not crash either. Filter empty nodes at the retrieval layer if you want to skip them entirely.
5. **Empty result list.** When retrieval returns no nodes, the adapter returns an empty list. Your application should treat that as "no relevant sources found" rather than passing an empty list to `align_citations`.

For production pipelines, validate the conversion result before calling `align_citations`:

```python
sources = from_llamaindex_nodes(nodes)
if not sources:
    raise ValueError("No relevant nodes retrieved.")
if not any(s.text.strip() for s in sources):
    raise ValueError("All retrieved nodes are empty.")
```

The `is_llamaindex_node(obj)` guard is also useful at the boundary between your retriever and the adapter, so you can detect retriever outputs that are not `TextNode` or `NodeWithScore` and route them to `from_dicts` instead.

## What Cite-Right Does Not Touch

`metadata` is opaque to the alignment pipeline. Cite-Right does not index it, does not score on it, and does not emit it on the resulting `Citation` or `SpanCitations`. Round-trip retrieval scores, file paths, and page numbers back to the user through your own application code. The same is true of the `id` you used to build the `SourceDocument` — that is your handle for joining citations back to your retrieval layer.

The adapter does not segment, tokenize, or index anything by itself. It is a one-line mapping from `TextNode` to `SourceDocument` (or `SourceChunk`); all of the alignment work happens inside `align_citations` on the same path described in [Citation Alignment](../concepts/citation-alignment.md).

The `NodeWithScore` score attribute is read by the unwrap but is not copied into the resulting `SourceDocument`. If you need it on the citation side, copy it into `metadata` yourself before the alignment call.

## Common Pitfalls

- **Do not pass a node whose `id_key` is not set without an explicit `id_key` argument.** The default `id_key="file_name"` is right for `SimpleDirectoryReader` but not every reader. If you see `"0"`, `"1"`, ... as the `source_id` on your citations, the metadata key is wrong.
- **Do not assume `NodeWithScore` scores land on the `SourceDocument`.** The adapter unwraps `NodeWithScore` to read content and metadata but does not copy the score. If you need it, copy it yourself.
- **Do not forget `start_char_idx` / `end_char_idx` on the node metadata.** Without them, the chunk adapter falls back to chunk-local offsets and citations will not point at the parent document. Configure your splitter to write them, or build `SourceChunk` directly.
- **Do not expect `document_text` validation to run through the chunk adapter.** The adapter omits `document_text`; if you have the parent in memory, build `SourceChunk` directly so the slice-equality check fires.
- **Do not pass nodes with empty `text` and expect citations to come back.** They will not, and the resulting `SourceDocument` will be useless to the alignment pipeline. Filter or skip them at the retrieval layer.
- **Do not rely on `metadata` for citation output.** The pipeline does not surface it on `Citation` or `SpanCitations`. If you need to expose the source path or page number in your final answer, carry it through your own code.
- **Do not pass the same node twice.** The adapter does not deduplicate. Two copies of the same source produce independent `SourceDocument` objects, and `align_citations` will not realize they point at the same parent.

## See Also

- [Citation Alignment](../concepts/citation-alignment.md) — `SourceDocument` and `SourceChunk` I/O, `SpanCitations` and `Citation` results, the half-open offset convention, and the rules that drive `status`.
- [Custom Sources](custom-sources.md) — building `SourceDocument` and `SourceChunk` directly, the chunk-rebase offset contract, the evidence equality invariant, and the `from_dicts` adapter for plain dictionaries.
- [LangChain Integration](langchain.md) — `from_langchain_documents` and `from_langchain_chunks` for the LangChain equivalent of the same workflow.
- [Quickstart](../getting-started/quickstart.md) — the basic `align_citations` call, reading the result, multiple sources, and `PreparedCitationCorpus` reuse.
