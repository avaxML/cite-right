# Tokenizers

Tokenization is the process of splitting text into units for alignment comparison. The choice of tokenizer affects how text is matched and can significantly impact citation quality. Cite-Right provides several tokenizer options to suit different requirements.

## SimpleTokenizer

The default tokenizer handles common cases without external dependencies. It is defined in `src/cite_right/text/tokenizer.py`.

```python
from cite_right import SimpleTokenizer, align_citations

tokenizer = SimpleTokenizer()
results = align_citations(answer, sources, tokenizer=tokenizer)
```

### Token Handling

SimpleTokenizer recognizes alphanumeric tokens as contiguous sequences of letters and digits. Internal hyphens and apostrophes are preserved as part of tokens, allowing "state-of-the-art" to match as a single unit rather than four separate words. Similarly, "company's" remains intact.

Numeric values with separators are handled intelligently. The tokenizer recognizes patterns like "5.2" and "1,200" as single tokens, improving alignment of financial and scientific content.

### Normalization

By default, the tokenizer applies Unicode NFKC normalization and case-folding. This ensures that typographic variations and capitalization differences do not prevent matching. The text "REVENUE" will match "revenue" and "Revenue" equally well.

Importantly, the tokenizer preserves the original character positions. While matching uses normalized forms, the evidence extracted from sources retains original casing and formatting.

### Configuration

The `TokenizerConfig` class provides control over normalization behavior.

```python
from cite_right import SimpleTokenizer
from cite_right.text.tokenizer import TokenizerConfig

config = TokenizerConfig(
    normalize_numbers=True,
    normalize_percent=True,
    normalize_currency=True
)

tokenizer = SimpleTokenizer(config=config)
```

The `normalize_numbers` option converts numeric separators, so "1,200" becomes "1200" for matching purposes.

The `normalize_percent` option converts the percent symbol to the word "percent", allowing "15%" to match "15 percent".

The `normalize_currency` option converts currency symbols to words: "$" becomes "dollar", "€" becomes "euro", and "£" becomes "pound".

These normalizations improve matching of financial content where authors may express the same value in different formats.

## HuggingFaceTokenizer

For applications using transformer models, the HuggingFace tokenizer wrapper provides compatibility with model-specific tokenization schemes. This tokenizer is defined in `src/cite_right/text/tokenizer_huggingface.py` and requires the huggingface optional dependency.

```python
pip install "cite-right[huggingface]"
```

### From Pretrained

The most common usage loads a tokenizer from a model identifier.

```python
from cite_right import HuggingFaceTokenizer, align_citations

tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
results = align_citations(answer, sources, tokenizer=tokenizer)
```

This approach downloads and caches the tokenizer configuration automatically.

### Custom Instance

If you already have a HuggingFace tokenizer instance, you can wrap it directly.

```python
from transformers import AutoTokenizer
from cite_right import HuggingFaceTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer = HuggingFaceTokenizer(hf_tokenizer)
```

### Subword Handling

Transformer tokenizers use subword units like WordPiece (BERT), BPE (GPT-2), or SentencePiece (many multilingual models). The HuggingFaceTokenizer handles these appropriately, maintaining character spans that account for subword boundaries.

### Special Tokens

By default, the wrapper excludes special tokens like `[CLS]` and `[SEP]` from the token sequence. These tokens are added by models for sequence classification but should not participate in alignment.

```python
tokenizer = HuggingFaceTokenizer.from_pretrained(
    "bert-base-uncased",
    add_special_tokens=False  # Default
)
```

## TiktokenTokenizer

For applications using OpenAI models, the tiktoken tokenizer provides compatible tokenization. This tokenizer is defined in `src/cite_right/text/tokenizer_tiktoken.py` and requires the tiktoken optional dependency.

```python
pip install "cite-right[tiktoken]"
```

### Default Encoding

Without arguments, the tokenizer uses `cl100k_base`, the encoding used by GPT-4 and GPT-3.5-turbo.

```python
from cite_right import TiktokenTokenizer, align_citations

tokenizer = TiktokenTokenizer()
results = align_citations(answer, sources, tokenizer=tokenizer)
```

### Specific Encodings

Different OpenAI model families use different encodings. Specify the encoding explicitly when needed.

```python
# For GPT-4 and GPT-3.5-turbo
tokenizer = TiktokenTokenizer("cl100k_base")

# For Codex models
tokenizer = TiktokenTokenizer("p50k_base")

# For older GPT-3 models
tokenizer = TiktokenTokenizer("r50k_base")
```

Using the same encoding as your generation model ensures that token boundaries in the answer match what the model produced.

## Choosing a Tokenizer

The default SimpleTokenizer works well for most applications. It handles common text patterns without external dependencies and produces intuitive token boundaries.

Consider HuggingFaceTokenizer when your application uses a specific transformer model and you want tokenization to match that model's behavior. This is particularly relevant when the answer comes from a model that uses unusual tokenization, such as multilingual models with specialized vocabularies.

Consider TiktokenTokenizer when working with OpenAI models. Matching the tokenization scheme can improve alignment quality, especially for content that the model generates in specific token patterns.

## Performance Considerations

SimpleTokenizer is the fastest option as it uses a simple regular expression-based approach with minimal overhead.

HuggingFaceTokenizer and TiktokenTokenizer incur the overhead of loading their respective vocabularies. For batch processing, reuse the tokenizer instance across calls to avoid repeated initialization.

```python
tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")

for answer, sources in batch:
    results = align_citations(answer, sources, tokenizer=tokenizer)
```

## Custom Tokenizers

You can implement custom tokenizers by following the `Tokenizer` protocol defined in `src/cite_right/core/interfaces.py`.

```python
from cite_right.core.interfaces import Tokenizer
from typing import Sequence

class MyTokenizer:
    def tokenize(self, text: str) -> tuple[Sequence[int], Sequence[tuple[int, int]]]:
        """
        Returns a tuple of (token_ids, spans).
        token_ids: Sequence of integer token identifiers
        spans: Sequence of (start_char, end_char) tuples
        """
        # Your implementation here
        pass
```

The protocol requires returning both token IDs and character spans. Token IDs enable efficient comparison during alignment, while character spans enable accurate evidence extraction.
