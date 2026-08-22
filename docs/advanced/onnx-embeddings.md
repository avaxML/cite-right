# ONNX Embeddings

cite-right supports fast, quantized embeddings via ONNX Runtime, providing significant performance improvements without requiring PyTorch or sentence-transformers at inference time.

## Installation

Install cite-right with ONNX support:

```bash
pip install cite-right[onnx]
```

This installs:
- `onnxruntime` - Fast inference engine
- `tokenizers` - HuggingFace tokenizers library

## Usage

The `OnnxMiniLmEmbedder` provides a drop-in replacement for `SentenceTransformerEmbedder`:

```python
from cite_right import align_citations, OnnxMiniLmEmbedder

# Initialize the ONNX embedder (downloads model on first use)
embedder = OnnxMiniLmEmbedder()

# Use it just like SentenceTransformerEmbedder
results = align_citations(
    answer="Climate policy reduces emissions.",
    sources=["Climate policy reduces emissions quickly."],
    embedder=embedder
)
```

## Performance

The ONNX embedder offers several advantages:

- **Faster inference**: Quantized int8 model runs faster than full-precision PyTorch
- **Smaller memory footprint**: No PyTorch dependency needed at inference
- **Same quality**: Uses the same all-MiniLM-L6-v2 model architecture (384-dimensional)

### Benchmark Comparison

On a CPU with a 50-case test pack:

| Configuration | p50 Latency | Speedup |
|--------------|-------------|---------|
| Default (no embeddings) | ~177ms | 1.0x |
| MiniLM-L6 + PyTorch | ~1757ms | 1.0x |
| MiniLM-L6 + ONNX (quantized) | ~1100-1300ms | **1.3-1.6x** |

*Note: Actual speedup depends on hardware and batch size.*

## Model Details

The ONNX embedder uses:
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Quantization**: int8 (via ONNX Runtime)
- **Embedding dimension**: 384
- **Max sequence length**: 512 tokens

On first use, the model and tokenizer are automatically downloaded to your cache directory (`~/.cache/cite-right/onnx-models/`).

## Rust Prepare Path

The ONNX embedder is fully compatible with cite-right's Rust acceleration:

```python
from cite_right import PreparedCitationCorpus, OnnxMiniLmEmbedder

embedder = OnnxMiniLmEmbedder()

# Rust prepare + ONNX embeddings = maximum performance
corpus = PreparedCitationCorpus.from_sources(
    sources=["Your source documents here"],
    embedder=embedder,
    use_rust=True  # Default, uses Rust for tokenize/passages/candidates
)

# Repeated alignments reuse the prepared corpus
results1 = corpus.align("First answer...")
results2 = corpus.align("Second answer...")
```

This combines the benefits of:
- Rust tokenization and passage generation (~40x faster than Python)
- ONNX quantized inference (~1.3-1.6x faster than PyTorch)
- **No spaCy required** for the fast embedded path

## Custom ONNX Models

You can use custom ONNX models by providing paths:

```python
embedder = OnnxMiniLmEmbedder(
    model_path="/path/to/your/model.onnx",
    tokenizer_path="/path/to/your/tokenizer.json",
    model_name="custom-model-name"  # Used for caching
)
```

## Exporting Custom Models

To export a sentence-transformers model to ONNX:

```python
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import QuantizationConfig

# Load the model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Export to ONNX with quantization
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    model_name,
    export=True,
)

# Quantize to int8
qconfig = QuantizationConfig(is_static=False, format="onnx")
ort_model.quantize(save_dir="./onnx-model", quantization_config=qconfig)
```

Then use the exported model:

```python
embedder = OnnxMiniLmEmbedder(
    model_path="./onnx-model/model.onnx",
    tokenizer_path="./onnx-model/tokenizer.json"
)
```

## API Reference

### OnnxMiniLmEmbedder

```python
class OnnxMiniLmEmbedder:
    def __init__(
        self,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...
```

**Parameters:**
- `model_path`: Path to ONNX model file. If None, downloads default model.
- `tokenizer_path`: Path to tokenizer.json. If None, downloads default tokenizer.
- `model_name`: Model name for caching embeddings.

**Methods:**
- `encode(texts)`: Encode texts into 384-dimensional embeddings.

## See Also

- [Embedding Retrieval Guide](embedding-retrieval.md) - General embedding usage
- [Rust Acceleration](rust-acceleration.md) - Rust prepare path details
- [Performance Tuning](performance-tuning.md) - Optimization strategies
