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

The balanced configuration uses moderate thresholds that avoid being too strict or too lenient. It works well for typical RAG applications where sources and answers have reasonable overlap.

### Strict

The strict preset is designed for high-stakes applications like fact-checking, legal document review, or medical information verification. It requires strong evidence before marking content as supported.

```python
config = CitationConfig.strict()
```

This preset increases the supported threshold significantly, requiring near-verbatim matches for full support. It also raises the minimum score threshold, filtering out marginal citations entirely. The result is conservative citation behavior that minimizes false positives at the cost of more content being marked as unsupported.

Applications using the strict preset should expect many partial or unsupported spans even when the answer genuinely derives from the sources. This is by design: when consequences of incorrect citation are severe, it is better to under-claim support than over-claim it.

### Permissive

The permissive preset accommodates heavily paraphrased content where answers express source information using substantially different wording.

```python
config = CitationConfig.permissive()
```

This preset lowers all thresholds, allowing weaker matches to qualify as citations. It is appropriate for summarization tasks, creative writing assistance, or applications where the goal is identifying source relevance rather than verifying exact claims.

When using the permissive preset, more content will be marked as supported, but the evidence quality may be lower. Users should understand that "supported" in this context means a plausible connection exists rather than a verified match.

### Fast

The fast preset prioritizes processing speed over alignment quality. It reduces the number of candidates considered and uses smaller passage windows.

```python
config = CitationConfig.fast()
```

This preset is suitable for high-volume processing, real-time applications, or initial filtering where speed matters more than precision. It still produces useful citations but may miss some matches that would be found with more thorough search.

For latency-sensitive applications, combining the fast preset with the Rust backend provides the best performance.

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
    min_score_threshold=base.min_score_threshold,
    supported_threshold=0.6,  # Slightly lower than strict default
    window_size_sentences=base.window_size_sentences
)
```

This approach lets you start with a well-tuned baseline and adjust only the parameters relevant to your specific needs.

## Preset Comparison

The following table summarizes key parameter differences between presets.

| Parameter | Balanced | Strict | Permissive | Fast |
|-----------|----------|--------|------------|------|
| min_score_threshold | 0.2 | 0.4 | 0.1 | 0.2 |
| supported_threshold | 0.5 | 0.7 | 0.3 | 0.5 |
| max_candidates | 50 | 100 | 50 | 20 |
| window_size_sentences | 3 | 3 | 5 | 2 |

These values are illustrative and may differ in the actual implementation. Refer to the source code in `src/cite_right/core/citation_config.py` for exact values.

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
