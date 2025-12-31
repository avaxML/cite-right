# Embedding Retrieval

By default, Cite-Right uses lexical matching to identify candidate passages for alignment. When source content is heavily paraphrased, semantic similarity provides a complementary signal that improves recall. The embedding retrieval feature enables this capability.

## How It Works

The citation pipeline has two stages. First, candidate selection identifies passages worth aligning. Second, Smith-Waterman alignment computes precise matches.

Without embeddings, candidate selection relies entirely on lexical overlap. Passages with more shared words rank higher. This approach works well when the answer closely mirrors the source text but struggles with paraphrased content.

With embeddings enabled, the pipeline encodes answer spans and source passages as dense vectors. Cosine similarity between these vectors identifies semantically similar passages regardless of exact word overlap. High-similarity passages join the candidate set even if they share few words with the answer.

## Enabling Embeddings

Embedding retrieval requires the embeddings optional dependency.

```bash
pip install "cite-right[embeddings]"
```

Then provide an embedder to the alignment function.

```python
from cite_right import SentenceTransformerEmbedder, align_citations

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
results = align_citations(answer, sources, embedder=embedder)
```

The embedder encodes all answer spans and passages on first use, caching the results for subsequent comparisons.

## SentenceTransformerEmbedder

The included embedder wraps the sentence-transformers library, which provides access to many pre-trained models.

### Model Selection

Different models offer tradeoffs between quality and speed.

```python
# Fast, good for most English content
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

# Higher quality, slower
embedder = SentenceTransformerEmbedder("all-mpnet-base-v2")

# Multilingual support
embedder = SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
```

The default model `all-MiniLM-L6-v2` provides a good balance for general English content. For specialized domains, fine-tuned models may perform better.

### Embedding Dimensions

Model output dimensions affect memory usage and similarity computation speed. Smaller dimensions are faster; larger dimensions capture more semantic nuance.

| Model | Dimensions |
|-------|------------|
| all-MiniLM-L6-v2 | 384 |
| all-mpnet-base-v2 | 768 |
| all-distilroberta-v1 | 768 |

For high-volume applications, smaller models reduce memory and latency with modest quality impact.

## Configuration Interaction

Several configuration parameters affect how embeddings influence candidate selection.

### Candidate limits

Candidate selection combines lexical overlap and (optionally) embedding similarity. You can control how many of each enter the full alignment stage.

```python
from cite_right import CitationConfig, align_citations

config = CitationConfig(
    max_candidates_lexical=200,
    max_candidates_embedding=100,
    max_candidates_total=250
)
results = align_citations(answer, sources, embedder=embedder, config=config)
```

### weights.embedding and weights.lexical

These weights affect the final citation score (after alignment), not candidate selection.

```python
from cite_right import CitationConfig, align_citations
from cite_right.core.citation_config import CitationWeights

config = CitationConfig(
    weights=CitationWeights(
        alignment=1.0,
        answer_coverage=1.0,
        evidence_coverage=0.0,
        lexical=0.3,
        embedding=0.7
    )
)
results = align_citations(answer, sources, embedder=embedder, config=config)
```

### allow_embedding_only

When alignment fails to find a good match but embedding similarity is high, this setting allows returning the passage as evidence.

```python
config = CitationConfig(allow_embedding_only=True)
```

With this enabled, heavily paraphrased content receives citations based on semantic similarity even without token-level matching. The evidence will be the entire passage window rather than a precise span.

Use this setting when recall is more important than precision and you accept coarser evidence regions.

## Custom Embedders

You can implement custom embedders by following the `Embedder` protocol.

```python
from typing import Sequence

from cite_right.models.base import Embedder

class OpenAIEmbedder:
    def __init__(self, client, model="text-embedding-ada-002"):
        self.client = client
        self.model = model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
```

Custom embedders allow integration with any embedding service or locally-hosted model.

## Performance Considerations

### Startup Cost

Sentence transformer models load on first use, adding several seconds to initial latency. For server applications, initialize the embedder at startup.

```python
# Initialize at server startup
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

# Reuse across requests
def handle_request(answer, sources):
    return align_citations(answer, sources, embedder=embedder)
```

### Batch Encoding

The embedder encodes all passages in a single batch operation. This is efficient for typical workloads with 5-50 passages. For very large source sets, consider pre-computing and caching embeddings.

### GPU Acceleration

Sentence transformers automatically use GPU when available. For high-throughput applications, GPU acceleration significantly reduces embedding latency.

```python
# Check if GPU is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## When Embeddings Help

Embeddings improve recall in several scenarios.

Paraphrased content where the answer expresses source facts using different words benefits significantly. "The project was completed successfully" may match "Successfully finishing the initiative" even without word overlap.

Synonyms and related terms are captured. "Increase" may match "growth" or "rise" through semantic similarity.

Domain-specific vocabulary with consistent meaning across variations benefits from the contextual understanding embeddings provide.

## When Embeddings May Not Help

Near-verbatim content where lexical matching already works well gains little from embeddings. The additional computation adds latency without improving results.

Highly technical content with specialized terminology may not be well-represented by general-purpose embedding models. Domain-specific fine-tuning may be necessary.

Very short passages may not contain enough context for meaningful embeddings. Single-word or very brief excerpts produce less reliable similarity scores.

## Observability

When debugging citation quality, examine both lexical and embedding scores.

```python
for result in results:
    for citation in result.citations:
        components = citation.components
        print(f"Lexical score: {components.get('lexical_score', 0):.3f}")
        print(f"Embedding score: {components.get('embedding_score', 0):.3f}")
        print(f"Alignment score: {components.get('alignment_score', 0):.3f}")
```

This breakdown reveals whether citations are driven primarily by word overlap or semantic similarity, informing configuration tuning.
