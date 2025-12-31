# Performance Tuning

As citation workloads scale, performance optimization becomes important. This page covers strategies for tuning Cite-Right to handle high-volume processing and latency-sensitive applications.

## Understanding the Cost Model

Citation alignment has several computational components, each with different scaling characteristics.

Tokenization is linear in text length. Processing a 10,000 word document takes roughly 10x longer than a 1,000 word document, but the absolute time is typically small.

Passage windowing is linear in document length and window configuration. More windows mean more candidates to consider.

Candidate selection computes lexical overlap or embedding similarity for each answer-passage pair. This scales with the product of answer spans and passages.

Smith-Waterman alignment is quadratic in sequence length for each pair. This is typically the dominant cost for workloads with many or long passages.

The Rust extension dramatically reduces alignment cost, making candidate selection and passage creation relatively more significant.

## Configuration Strategies

### Reducing Candidates

The `max_candidates` parameter limits how many passages undergo full alignment.

```python
from cite_right import CitationConfig, align_citations

config = CitationConfig(max_candidates=20)  # Default is 50
results = align_citations(answer, sources, config=config)
```

Reducing candidates improves speed but may miss some matches. Monitor citation quality when adjusting this parameter.

### Smaller Windows

Reducing passage window size decreases both the number of passages and the length of each alignment operation.

```python
config = CitationConfig(
    window_size_sentences=2,  # Default is 3
    window_stride_sentences=2  # Default is 1
)
```

Smaller, non-overlapping windows process faster but may miss matches that span window boundaries.

### Fast Preset

The fast configuration preset combines several speed-oriented settings.

```python
config = CitationConfig.fast()
```

This preset reduces candidates, shrinks windows, and adjusts thresholds for throughput rather than precision.

## Batching Strategies

### Reusing Components

Tokenizers, segmenters, and embedders have initialization costs. Reuse instances across calls.

```python
from cite_right import (
    SimpleTokenizer,
    SentenceTransformerEmbedder,
    SpacyAnswerSegmenter,
    align_citations,
)

# Initialize once
tokenizer = SimpleTokenizer()
embedder = SentenceTransformerEmbedder()
segmenter = SpacyAnswerSegmenter()

# Reuse across many calls
for answer, sources in workload:
    results = align_citations(
        answer,
        sources,
        tokenizer=tokenizer,
        embedder=embedder,
        answer_segmenter=segmenter,
    )
```

This eliminates repeated model loading and configuration parsing.

### Parallel Processing

For batch workloads, process multiple answers concurrently.

```python
from concurrent.futures import ThreadPoolExecutor

def process_batch(items, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(align_citations, answer, sources)
            for answer, sources in items
        ]
        return [f.result() for f in futures]
```

The Rust extension releases the GIL, enabling true parallelism. With pure Python, threading provides limited benefit due to GIL contention.

### Multiprocessing for Large Batches

For very large batches, multiprocessing avoids GIL limitations entirely.

```python
from multiprocessing import Pool

def align_worker(args):
    answer, sources = args
    return align_citations(answer, sources)

with Pool(processes=8) as pool:
    results = pool.map(align_worker, workload)
```

Each worker process has its own Python interpreter and GIL, enabling full CPU utilization.

## Memory Optimization

### Document Chunking

Very long documents consume memory for tokenization and passage creation. Pre-chunk documents to limit per-call memory.

```python
def chunk_document(text, max_length=10000):
    """Split document into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_length:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"

    if current:
        chunks.append(current.strip())

    return chunks
```

### Streaming Results

For real-time applications, stream results as they complete rather than waiting for all alignments.

```python
def stream_citations(answer, sources):
    """Yield citation results as they are computed."""
    # This is conceptual; actual implementation would require
    # modifications to the core alignment function
    for span in segment_answer(answer):
        result = align_single_span(span, sources)
        yield result
```

This approach reduces time-to-first-result for user-facing applications.

## Monitoring and Profiling

### Timing Breakdown

Measure where time is spent in your pipeline.

```python
import time

start = time.perf_counter()
results = align_citations(answer, sources)
elapsed = time.perf_counter() - start

print(f"Total time: {elapsed:.3f}s")
print(f"Answer length: {len(answer)} chars")
print(f"Sources: {len(sources)}, total {sum(len(s.text) for s in sources)} chars")
print(f"Result spans: {len(results)}")
```

### Component Profiling

For detailed analysis, profile individual components.

```python
from cite_right import SimpleTokenizer
from cite_right.text.passage import generate_passages

tokenizer = SimpleTokenizer()

# Profile tokenization
start = time.perf_counter()
tokens = tokenizer.tokenize(long_text)
print(f"Tokenization: {time.perf_counter() - start:.3f}s")

# Profile passage generation
start = time.perf_counter()
passages = list(generate_passages(long_text, window_size=3, stride=1))
print(f"Passage generation: {time.perf_counter() - start:.3f}s")
```

### Memory Profiling

Track memory usage for large workloads.

```python
import tracemalloc

tracemalloc.start()
results = align_citations(answer, sources)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
```

## Architecture Patterns

### Caching

Cache alignment results for repeated queries.

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_align(answer, source_key):
    sources = load_sources(source_key)
    return align_citations(answer, sources)
```

This helps when the same answer-source combinations are requested multiple times.

### Tiered Processing

Use fast configuration for initial filtering, detailed configuration for final results.

```python
def tiered_citation(answer, sources):
    # Quick check with fast config
    fast_config = CitationConfig.fast()
    quick_results = align_citations(answer, sources, config=fast_config)

    # If well-grounded, return fast results
    if all(r.status == "supported" for r in quick_results):
        return quick_results

    # Otherwise, do detailed analysis
    strict_config = CitationConfig.strict()
    return align_citations(answer, sources, config=strict_config)
```

### Async Processing

For web applications, run citation in background tasks.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

async def async_align(answer, sources):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        align_citations,
        answer,
        sources
    )
```

This keeps the event loop responsive while citations compute.

## Benchmarking Your Workload

Create representative benchmarks for your specific use case.

```python
import statistics

def benchmark(workload, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for answer, sources in workload:
            align_citations(answer, sources)
        times.append(time.perf_counter() - start)

    print(f"Mean: {statistics.mean(times):.3f}s")
    print(f"Std:  {statistics.stdev(times):.3f}s")
    print(f"Min:  {min(times):.3f}s")
    print(f"Max:  {max(times):.3f}s")
```

Run benchmarks before and after configuration changes to quantify impact.
