# Segmenters

Segmentation divides text into units for citation. Answer segmentation determines the granularity of citation results, while source segmentation affects passage window construction. Cite-Right provides several segmenter options with different accuracy and performance characteristics.

## Answer vs Source Segmentation

Two separate segmentation processes occur during citation alignment.

Answer segmentation splits the generated text into spans that receive individual citations. Each span becomes a `SpanCitations` result with its own status and citation list.

Source segmentation splits documents into sentences used to construct passage windows. The segmenter identifies sentence boundaries that define window positions.

These can use different segmenters depending on requirements.

```python
from cite_right import align_citations, SpacyAnswerSegmenter, SimpleSegmenter

results = align_citations(
    answer,
    sources,
    answer_segmenter=SpacyAnswerSegmenter(),
    source_segmenter=SimpleSegmenter()
)
```

## SimpleSegmenter

The default source segmenter uses rule-based pattern matching without external dependencies. It is defined in `src/cite_right/text/segmenter_simple.py`.

```python
from cite_right.text.segmenter_simple import SimpleSegmenter

segmenter = SimpleSegmenter()
```

### Boundary Detection

SimpleSegmenter splits text on common sentence-ending patterns. It recognizes periods, question marks, and exclamation points as sentence terminators when followed by whitespace. Semicolons also trigger splits as they often separate independent clauses.

Paragraph boundaries (double newlines) always create splits regardless of punctuation.

### Limitations

The rule-based approach has known limitations. Abbreviations like "Dr." and "U.S." may trigger incorrect splits. Sentences ending with quoted material or parentheticals may be handled imprecisely. Complex academic or legal prose with nested clauses can confuse the pattern matching.

For content where accurate segmentation is critical, consider the spaCy-based alternatives.

## SimpleAnswerSegmenter

The default answer segmenter extends SimpleSegmenter with paragraph-aware processing. It is defined in `src/cite_right/text/answer_segmenter.py`.

```python
from cite_right.text.answer_segmenter import SimpleAnswerSegmenter

segmenter = SimpleAnswerSegmenter()
```

This segmenter produces `AnswerSpan` objects with character offsets and a `kind` field indicating whether the span represents a sentence, clause, or paragraph.

## SpacySegmenter

The spaCy-based source segmenter provides linguistically-informed sentence boundary detection. It requires the spacy optional dependency and a downloaded language model.

```python
pip install "cite-right[spacy]"
python -m spacy download en_core_web_sm
```

```python
from cite_right import SpacySegmenter

segmenter = SpacySegmenter()
```

### Statistical Boundaries

SpaCy uses a statistical model trained on large text corpora to identify sentence boundaries. This approach handles abbreviations, numbers, and unusual punctuation patterns more accurately than rule-based methods.

The model considers context when making boundary decisions. "Dr. Smith arrived." is correctly identified as a single sentence, while "The meeting ended. Dr. Smith arrived." is correctly split into two.

### Clause Splitting

SpacySegmenter supports optional clause-level splitting for more granular citations.

```python
segmenter = SpacySegmenter(split_clauses=True)
```

When enabled, the segmenter identifies coordinating conjunctions and splits on them. "Revenue increased and profits doubled" becomes two separate segments when split_clauses is True.

This feature uses conservative heuristics to avoid over-splitting. Only top-level coordinations are split; nested or ambiguous structures remain intact.

### Model Selection

By default, SpacySegmenter uses the first available spaCy model. You can specify a particular model.

```python
import spacy

nlp = spacy.load("en_core_web_md")  # Medium model for better accuracy
segmenter = SpacySegmenter(nlp=nlp)
```

Larger models provide slightly better accuracy at the cost of memory and processing time.

## SpacyAnswerSegmenter

The spaCy-based answer segmenter combines statistical boundary detection with clause splitting for fine-grained citation. It is defined in `src/cite_right/text/answer_segmenter_spacy.py`.

```python
from cite_right import SpacyAnswerSegmenter

segmenter = SpacyAnswerSegmenter(split_clauses=True)
```

### Paragraph Awareness

This segmenter processes each paragraph separately, maintaining paragraph boundaries while splitting sentences within each paragraph. Double newlines always create segment breaks.

### Output

Each segment is an `AnswerSpan` with appropriate `kind` labeling.

```python
for result in results:
    span = result.answer_span
    print(f"{span.kind}: {span.text}")
```

## PySBDSegmenter

For applications needing fast sentence boundary detection without the full spaCy pipeline, pysbd provides an efficient alternative. This segmenter requires the pysbd optional dependency.

```python
pip install "cite-right[pysbd]"
```

```python
from cite_right.text.segmenter_pysbd import PySBDSegmenter

segmenter = PySBDSegmenter()
```

### Performance

PySBD is significantly faster than spaCy as it uses optimized rules rather than neural network inference. For high-throughput applications where spaCy's accuracy is not required, pysbd offers a good middle ground between SimpleSegmenter and SpacySegmenter.

### Language Support

PySBD includes rules for multiple languages. Specify the language code for non-English text.

```python
segmenter = PySBDSegmenter(language="de")  # German
```

## Choosing a Segmenter

The right segmenter depends on your accuracy requirements and performance constraints.

SimpleSegmenter and SimpleAnswerSegmenter work well for typical content where sentences follow standard punctuation patterns. They add no dependencies and process quickly.

SpacySegmenter and SpacyAnswerSegmenter provide the highest accuracy for English text. They handle edge cases that confuse rule-based approaches. Use them when citation granularity matters and processing time is not critical.

PySBDSegmenter offers a compromise: better accuracy than simple rules, faster than spaCy, with multilingual support.

## Granularity Considerations

Finer segmentation produces more spans, each receiving its own citation. This enables precise attribution but may split related content. A sentence like "Revenue grew 15% because sales increased" might become two spans, with separate citations for the growth rate and the cause.

Coarser segmentation groups related content but may hide unsupported portions. A paragraph with one hallucinated sentence among five grounded sentences will receive a "partial" status without identifying the specific problem.

The clause splitting options in spaCy segmenters allow adjustment between these extremes.

## Performance Comparison

Approximate relative performance for typical text processing.

SimpleSegmenter and SimpleAnswerSegmenter are the fastest, suitable for any volume.

PySBDSegmenter adds modest overhead, roughly 2-3x slower than simple rules.

SpacySegmenter and SpacyAnswerSegmenter are slowest, roughly 10-50x slower than simple rules depending on model size. The overhead comes primarily from model loading, so reusing the segmenter instance amortizes this cost across many documents.

For latency-sensitive applications, consider using simple segmenters with the fast configuration preset.
