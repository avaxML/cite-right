
# LangChain Integration

LangChain retrievers hand back `Document` objects from `langchain_core.documents` with a `page_content` string and a `metadata` dict. Cite-Right ships two adapter functions that turn those into `SourceDocument` and `SourceChunk` so you can call `align_citations` on the result without rewriting your pipeline. The adapters live in `src/cite_right/integrations.py` and are also exported from the top-level `cite_right` package.

This page walks the two conversions, the `id_key` and `start_key` / `end_key` knobs, what happens when offsets are missing, and the full pattern for a LangChain RAG pipeline plus an LCEL post-processing step. The underlying I/O contract is in [Citation Alignment](../concepts/citation-alignment.md); the chunk-rebase offset convention is in [Custom Sources](custom-sources.md).

## Install The Optional Extra

The LangChain adapters import `langchain_core.documents` at module load. If that package is not present, the import paths are still exported but resolve to `None`, the `LANGCHAIN_AVAILABLE` flag is `False`, and the adapter functions raise `ImportError` with a hint.

```bash
pip install "cite-right[langchain]==0.4.0"
```

The extra is `langchain-core>=0.3.0` (see `pyproject.toml`). You do not need the full `langchain` distribution; `langchain-core` is enough to use the adapters with any LangChain-compatible retriever (FAISS, Chroma, Pinecone, the `langchain_community` integrations, etc.).

You can check whether the import succeeded without trying an adapter call:

```python
from cite_right import is_langchain_available, LANGCHAIN_AVAILABLE

if is_langchain_available():
    # langchain_core is importable; the adapters will work
    ...
```

`LANGCHAIN_AVAILABLE` is also exposed as a module-level boolean. `is_langchain_document(obj)` is the type guard for retriever results: it returns `True` for real `langchain_core.documents.Document` instances and `False` for anything else, including plain dicts and strings that happen to have a `page_content` attribute.

## Converting Documents

`from_langchain_documents(documents, *, id_key="source")` takes a sequence of LangChain `Document` objects and returns a list of `SourceDocument`. The text comes from `doc.page_content`, the metadata is passed through unchanged, and the `id` is read from `doc.metadata[id_key]` — falling back to the document's integer index stringified when that key is missing.

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from cite_right import align_citations
from cite_right.integrations import from_langchain_documents

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("my_index", embeddings)
retriever = vectorstore.as_retriever()

query = "What were the Q4 results?"
lc_docs = retriever.invoke(query)

sources = from_langchain_documents(lc_docs)
answer = generate_answer(query, lc_docs)  # your generation step
results = align_citations(answer, sources)
```

The function calls `_require_langchain()` first, so a missing `langchain-core` install fails fast with a clear message rather than producing a half-built list.

### The id_key Fallback

LangChain `Document` does not have a required `id` field, and most loaders do not set one — the metadata usually has `"source"`, `"source_path"`, or a loader-specific name. The adapter reads `doc.metadata[id_key]` and only falls back to the index when that key is missing. The default `id_key="source"` matches the convention used by LangChain's `PyPDFLoader`, `TextLoader`, `UnstructuredFileLoader`, and most file-based loaders, so for those retrievers you get the original filename or path as the `source_id` on every `Citation`.

```python
from langchain_core.documents import Document
from cite_right.integrations import from_langchain_documents

# With "source" in metadata: that value becomes the SourceDocument id
docs = [Document(page_content="...", metadata={"source": "annual_report.pdf"})]
sources = from_langchain_documents(docs)
assert sources[0].id == "annual_report.pdf"

# Without the key: the index stringifies to "0", "1", ...
docs = [Document(page_content="...")]
sources = from_langchain_documents(docs)
assert sources[0].id == "0"
```

If your loader uses a different metadata key, pass it explicitly:

```python
# A retriever that puts the document id under "doc_id"
sources = from_langchain_documents(docs, id_key="doc_id")

# A retriever that uses LangChain's "id" key
sources = from_langchain_documents(docs, id_key="id")
```

The resulting `id` is what every `Citation.source_id` will carry, so a stable identifier here is what makes it possible to map a citation back to the source row in your retrieval layer.

### Metadata Preservation

The full `Document.metadata` dict is copied onto the resulting `SourceDocument.metadata`. Cite-Right does not read it during alignment — it is opaque to the pipeline — but your application code can read it back to recover anything the retriever attached: page numbers, file paths, retrieval scores, timestamps, custom tags.

```python
sources = from_langchain_documents(lc_docs)
for source in sources:
    page = source.metadata.get("page")
    score = source.metadata.get("relevance_score")
    path = source.metadata.get("source")
```

Metadata round-trips through `align_citations` unchanged, so it is the right place to keep any retriever-specific extras you want to expose in your final answer payload.

## Converting Chunks With Offsets

LangChain text splitters (`RecursiveCharacterTextSplitter`, `CharacterTextSplitter`, `TokenTextSplitter`, and others) split a `Document` into smaller chunks. When you pass `add_start_index=True` to the splitter, each resulting chunk's metadata has a `"start_index"` field with the character offset of the chunk in the parent document. `from_langchain_chunks` reads that offset and produces `SourceChunk` objects whose `doc_char_start` / `doc_char_end` point at the parent. Citation offsets then come out absolute in the original document, not in the chunk.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from cite_right.integrations import from_langchain_chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,  # writes "start_index" into each chunk's metadata
)

full_doc = Document(page_content=long_text, metadata={"source": "report.pdf"})
chunks = splitter.split_documents([full_doc])

sources = from_langchain_chunks(chunks)
results = align_citations(answer, sources)

for chunk in sources:
    print(f"Chunk from {chunk.source_id}")
    print(f"Position in original: {chunk.doc_char_start} to {chunk.doc_char_end}")
```

The function signature is:

```python
from_langchain_chunks(
    documents,
    *,
    id_key="source",
    start_key="start_index",
    end_key="end_index",
    full_text_key=None,
) -> list[SourceChunk]
```

`start_key` and `end_key` default to `"start_index"` and `"end_index"`, matching the LangChain splitter convention. If the chunk has `start_index` but no `end_index`, the adapter computes the end as `start + len(doc.page_content)`, so a single-key splitter still produces a valid `SourceChunk` range.

### Passing the Full Document For Slice Validation

If you have the original document text in memory, pass its metadata key as `full_text_key`. The adapter then sets `SourceChunk.document_text` to that value, which enables the slice-equality check inside `SourceChunk` (`_validate_document_text_alignment` in `src/cite_right/core/results.py`) and lets `_slice_source_text` re-slice every citation's `evidence` directly from the full parent rather than from the chunk-local coordinates.

```python
chunks = splitter.split_documents([full_doc])
sources = from_langchain_chunks(chunks, full_text_key="full_text")
```

Without `full_text_key`, the rebase still produces correct absolute offsets — `_slice_source_text` falls back to chunk-local slicing (`source.text[abs - base_doc_offset]`) when `full_text` is not supplied. The rebase contract is in [Custom Sources](custom-sources.md#chunk-rebase-and-the-evidence-equality-invariant).

### Missing Offsets

Not every splitter sets `start_index`. If you forget `add_start_index=True`, or use a loader that does not track offsets, `doc.metadata.get("start_index", 0)` returns `0` and `end` is computed as `start + len(doc.page_content)`. The resulting `SourceChunk` is well-formed, but the offsets no longer point at the original document — they point at the chunk. If you need parent-absolute offsets, configure the splitter to emit them. There is no post-hoc recovery.

## End-To-End RAG Pipeline

The typical flow: retrieve from a vector store, build a context string, generate the answer with your LLM, then convert the retrieved documents and align citations. Cite-Right fits in after the LLM step; the retrieval and generation layers are unchanged.

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from cite_right import align_citations, check_groundedness
from cite_right.integrations import from_langchain_documents

embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = Ollama(model="llama2")
prompt = PromptTemplate.from_template("""
Answer the question based on the following context:

{context}

Question: {question}
Answer:""")


def rag_with_citations(question: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = llm.invoke(prompt.format(context=context, question=question))

    sources = from_langchain_documents(docs)
    citations = align_citations(answer, sources)
    metrics = check_groundedness(answer, sources)

    return {
        "answer": answer,
        "citations": citations,
        "groundedness": metrics.groundedness_score,
        "sources": docs,
    }


result = rag_with_citations("What were the company's key achievements?")
print(f"Answer: {result['answer']}")
print(f"Groundedness: {result['groundedness']:.1%}")
```

The same pattern works with any LangChain retriever, prompt template, or chat model. The integration boundary is the `from_langchain_documents(docs)` call — everything before it is plain LangChain; everything after is plain Cite-Right.

## LangChain Expression Language (LCEL)

For LCEL chains, citation computation is a post-processing step on the chain's output. Wrap the conversion in a `RunnableLambda` and add the result to the chain's output dict.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def add_citations(data: dict) -> dict:
    sources = from_langchain_documents(data["documents"])
    citations = align_citations(data["answer"], sources)
    return {**data, "citations": citations}


chain = (
    {"documents": retriever, "question": RunnablePassthrough()}
    | RunnablePassthrough.assign(answer=generation_chain)
    | RunnableLambda(add_citations)
)

result = chain.invoke("What is the return policy?")
```

The `add_citations` step keeps the functional style of LCEL: it takes the previous step's dict, computes a new field, and returns the same dict plus that field. Downstream steps see `data["citations"]` and never need to know the documents are LangChain objects.

## Error Handling

The adapters are permissive on the input side and strict on the integration boundary. The behavior, in order of failure detection:

1. **Missing `langchain-core`.** `from_langchain_documents` and `from_langchain_chunks` call `_require_langchain()`, which raises `ImportError` with the install hint. The function never returns a half-built list.
2. **Missing `id_key` metadata.** Falls back to `str(idx)`, where `idx` is the document's position in the input sequence. You get a stable id per call but no connection to your real source row. Pass `id_key` explicitly when the loader uses a non-default name.
3. **Missing offsets in chunk conversion.** Falls back to `0` and `start + len(page_content)`. The resulting `SourceChunk` is valid but its offsets do not point at the parent. There is no post-hoc recovery.
4. **Empty documents.** A `Document(page_content="")` produces a `SourceDocument` with empty text. `align_citations` will not produce a citation against an empty source, but it will not crash either. Filter empty documents at the retrieval layer if you want to skip them entirely.

For production pipelines, validate the conversion result before calling `align_citations`:

```python
sources = from_langchain_documents(docs)
if not sources:
    raise ValueError("No valid source documents retrieved.")
if not any(s.text.strip() for s in sources):
    raise ValueError("All retrieved documents are empty.")
```

The `is_langchain_document(obj)` guard is also useful at the boundary between your retriever and the adapter, so you can detect retriever outputs that are not LangChain `Document` instances (some retrievers return dicts or other types) and route them to `from_dicts` instead.

## What Cite-Right Does Not Touch

`metadata` is opaque to the alignment pipeline. Cite-Right does not index it, does not score on it, and does not emit it on the resulting `Citation` or `SpanCitations`. Round-trip retrieval scores, file paths, and page numbers back to the user through your own application code. The same is true of the `id` you used to build the `SourceDocument` — that is your handle for joining citations back to your retrieval layer.

The adapter does not segment, tokenize, or index anything by itself. It is a one-line mapping from `Document` to `SourceDocument` (or `SourceChunk`); all of the alignment work happens inside `align_citations` on the same path described in [Citation Alignment](../concepts/citation-alignment.md).

## Common Pitfalls

- **Do not pass a `Document` whose `id_key` is not set without an explicit `id_key` argument.** The default `id_key="source"` is right for the file-based loaders but not for every retriever. If you see `"0"`, `"1"`, ... as the `source_id` on your citations, the metadata key is wrong.
- **Do not forget `add_start_index=True` on the splitter.** Without it, the chunk adapter falls back to chunk-local offsets and citations will not point at the parent document.
- **Do not pass `Document` objects with empty `page_content` and expect citations to come back.** They will not, and the resulting `SourceDocument` will be useless to the alignment pipeline. Filter or skip them at the retrieval layer.
- **Do not rely on `metadata` for citation output.** The pipeline does not surface it on `Citation` or `SpanCitations`. If you need to expose the source path or page number in your final answer, carry it through your own code.
- **Do not pass `from_langchain_documents` the same `Document` twice.** The adapter does not deduplicate. Two copies of the same source produce independent `SourceDocument` objects, and `align_citations` will not realize they point at the same parent.

## See Also

- [Citation Alignment](../concepts/citation-alignment.md) — `SourceDocument` and `SourceChunk` I/O, `SpanCitations` and `Citation` results, the half-open offset convention, and the rules that drive `status`.
- [Custom Sources](custom-sources.md) — building `SourceDocument` and `SourceChunk` directly, the chunk-rebase offset contract, the evidence equality invariant, and the `from_dicts` adapter for plain dictionaries.
- [LlamaIndex Integration](llamaindex.md) — `from_llamaindex_nodes` and `from_llamaindex_chunks` for the LlamaIndex equivalent of the same workflow.
- [Quickstart](../getting-started/quickstart.md) — the basic `align_citations` call, reading the result, multiple sources, and `PreparedCitationCorpus` reuse.
