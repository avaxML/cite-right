# Configuration Presets

Cite-Right includes several pre-configured settings optimized for common use cases. These presets provide sensible defaults that you can use directly or as starting points for customization.

## Available Presets

### Balanced (Default)

The balanced preset provides general-purpose settings suitable for most applications. This is the default behavior when no configuration is specified.

```python
from cite_right import CitationConfig, align_citations

config = CitationConfig.balanced()
results = align_citations(answer, sources, config=config)
```

Balanced uses the default thresholds and candidate limits. It works well for typical RAG applications where sources and answers have reasonable overlap.

### Strict

The strict preset is designed for high-stakes applications like fact-checking, legal document review, or medical information verification. It requires strong evidence before marking content as supported.

```python
config = CitationConfig.strict()
```

This preset increases answer-coverage requirements and sets a minimum final score, filtering out marginal citations. The result is conservative citation behavior that minimizes false positives at the cost of more content being marked as unsupported.

### Permissive

The permissive preset accommodates heavily paraphrased content where answers express source information using substantially different wording.

```python
config = CitationConfig.permissive()
```

This preset lowers answer-coverage thresholds, increases `top_k`, and enables embedding-only citations. It is appropriate for summarization tasks or applications where recall matters more than precision.

### Fast

The fast preset prioritizes processing speed over alignment quality by reducing the number of candidates evaluated.

```python
config = CitationConfig.fast()
```

This preset reduces candidate limits and returns only the single best citation per span. It still produces useful citations but may miss matches that would be found with more thorough search.

## Choosing a Preset

The right preset depends on your application requirements and the nature of your content.

Consider strict when incorrect citations could cause harm, such as in medical, legal, or financial contexts. The higher bar for support protects users from acting on weakly-grounded information.

Consider permissive when sources and answers are expected to differ substantially in wording. Summarization outputs, translated content, and creative paraphrasing all benefit from the relaxed thresholds.

Consider fast when processing large volumes of content and some precision loss is acceptable. Batch processing jobs, preview functionality, and interactive applications with tight latency requirements are good candidates.

Consider balanced for everything else. The default settings handle most scenarios reasonably well without requiring tuning.

## Customizing Presets

Presets can serve as starting points for further customization. Create a preset and then modify specific parameters.

```python
config = CitationConfig.strict()
```

Since `CitationConfig` is a Pydantic model, you can create a modified version by constructing a new instance with updated values.

```python
from cite_right import CitationConfig

base = CitationConfig.strict()
config = CitationConfig(
    top_k=base.top_k,
    min_answer_coverage=base.min_answer_coverage,
    supported_answer_coverage=0.6,  # Slightly lower than strict default
    window_size_sentences=base.window_size_sentences
)
```

This approach lets you start with a well-tuned baseline and adjust only the parameters relevant to your specific needs.

## Preset Comparison

The following table summarizes key parameter differences between presets.

| Parameter | Balanced | Strict | Permissive | Fast |
|-----------|----------|--------|------------|------|
| top_k | 3 | 2 | 5 | 1 |
| min_answer_coverage | 0.2 | 0.4 | 0.15 | 0.2 |
| supported_answer_coverage | 0.6 | 0.7 | 0.4 | 0.6 |
| min_final_score | 0.0 | 0.3 | 0.0 | 0.0 |
| allow_embedding_only | False | False | True | False |
| min_embedding_similarity | 0.3 | 0.3 | 0.25 | 0.3 |
| supported_embedding_similarity | 0.6 | 0.6 | 0.5 | 0.6 |
| max_candidates_lexical | 200 | 200 | 200 | 50 |
| max_candidates_embedding | 200 | 200 | 200 | 50 |
| max_candidates_total | 400 | 400 | 400 | 100 |
| max_citations_per_source | 2 | 1 | 3 | 1 |

Other parameters (window sizes, alignment scoring, weights) remain at their default values across presets.

## Runtime Preset Selection

For applications that need different behavior based on context, presets can be selected dynamically.

```python
def get_config_for_context(context):
    if context.requires_high_precision:
        return CitationConfig.strict()
    elif context.is_summarization_task:
        return CitationConfig.permissive()
    elif context.has_latency_constraint:
        return CitationConfig.fast()
    else:
        return CitationConfig.balanced()

config = get_config_for_context(current_context)
results = align_citations(answer, sources, config=config)
```

This pattern enables a single application to serve diverse use cases with appropriate citation behavior for each.
