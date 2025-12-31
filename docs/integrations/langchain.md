# LangChain Integration

LangChain is a popular framework for building applications with large language models. Cite-Right provides helper functions that convert LangChain document types to the formats expected by the citation alignment functions.

## Document Conversion

LangChain represents retrieved documents using the `Document` class from `langchain_core.documents`. This class has `page_content` for the text and `metadata` for additional information.

The `from_langchain_documents` function converts a list of LangChain documents to Cite-Right `SourceDocument` objects.

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from cite_right import align_citations
from cite_right.integrations import from_langchain_documents

# Create a retriever
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("my_index", embeddings)
retriever = vectorstore.as_retriever()

# Retrieve documents
query = "What were the Q4 results?"
lc_docs = retriever.invoke(query)

# Convert to cite-right format
sources = from_langchain_documents(lc_docs)

# Generate an answer (using your LLM of choice)
answer = generate_answer(query, lc_docs)

# Get citations
results = align_citations(answer, sources)
```

### ID Generation

LangChain documents do not have a required ID field. The conversion function generates IDs using several strategies.

If the document metadata contains a "source" field (common when loading from files), that value is used as the ID.

If the metadata contains an "id" field, that value is used.

Otherwise, the function generates an ID from the document index: "doc_0", "doc_1", and so on.

```python
# Documents with source metadata
docs = [Document(page_content="...", metadata={"source": "annual_report.pdf"})]
sources = from_langchain_documents(docs)
# sources[0].id == "annual_report.pdf"

# Documents without source metadata
docs = [Document(page_content="...")]
sources = from_langchain_documents(docs)
# sources[0].id == "doc_0"
```

### Metadata Preservation

The original LangChain metadata is preserved in the Cite-Right document.

```python
sources = from_langchain_documents(lc_docs)
for source in sources:
    original_metadata = source.metadata
```

This allows you to access any additional information attached to the documents, such as page numbers, file paths, or retrieval scores.

## Working with Chunks

LangChain applications often use text splitters to break documents into smaller chunks for retrieval. When these chunks maintain offsets into the original document, you can use `from_langchain_chunks` to preserve positional information.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cite_right.integrations import from_langchain_chunks

# Split a document while tracking offsets
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True  # Important: enables offset tracking
)

full_doc = Document(page_content=long_text, metadata={"source": "report.pdf"})
chunks = splitter.split_documents([full_doc])

# Convert to cite-right chunks
sources = from_langchain_chunks(chunks)
```

### Offset Handling

When `add_start_index=True` is set on the splitter, each chunk's metadata contains a "start_index" field indicating where in the original document the chunk begins.

The `from_langchain_chunks` function uses this information to create `SourceChunk` objects with proper document offsets.

```python
for chunk in sources:
    print(f"Chunk from {chunk.source_id}")
    print(f"Position in original: {chunk.doc_char_start} to {chunk.doc_char_end}")
```

When citations are computed against these chunks, the resulting character offsets refer to positions in the original document, not the chunk. This enables linking back to the complete source for display purposes.

### Missing Offsets

If chunks do not have start_index metadata (because the splitter was not configured with `add_start_index=True`), the function falls back to treating each chunk as an independent document.

## Complete RAG Pipeline Example

Here is a complete example showing citation integration in a LangChain RAG pipeline.

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from cite_right import align_citations, check_groundedness
from cite_right.integrations import from_langchain_documents

# Set up the retrieval components
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Set up the generation component
llm = Ollama(model="llama2")
prompt = PromptTemplate.from_template("""
Answer the question based on the following context:

{context}

Question: {question}
Answer:""")

def rag_with_citations(question):
    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Format context for the prompt
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate answer
    formatted_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(formatted_prompt)

    # Convert documents and compute citations
    sources = from_langchain_documents(docs)
    citations = align_citations(answer, sources)

    # Check groundedness
    metrics = check_groundedness(answer, sources)

    return {
        "answer": answer,
        "citations": citations,
        "groundedness": metrics.groundedness_score,
        "sources": docs
    }

# Use the pipeline
result = rag_with_citations("What were the company's key achievements?")
print(f"Answer: {result['answer']}")
print(f"Groundedness: {result['groundedness']:.1%}")
```

## LangChain Expression Language (LCEL)

For applications using LCEL chains, citations can be computed as a post-processing step.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def add_citations(data):
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

This approach integrates citation computation into the chain while maintaining the functional style of LCEL.

## Error Handling

The integration functions are designed to be permissive. If document metadata is missing expected fields, the functions use fallbacks rather than raising errors.

```python
# Even empty documents are handled
docs = [Document(page_content="")]
sources = from_langchain_documents(docs)  # Returns empty source list
```

For production applications, you may want to add validation.

```python
sources = from_langchain_documents(docs)
if not sources:
    raise ValueError("No valid source documents found")
if not any(s.text.strip() for s in sources):
    raise ValueError("All source documents are empty")
```
