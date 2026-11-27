
# Segmenters

Segmenters split text into the units the alignment pipeline operates on. There are two interfaces, both defined in `src/cite_right/core/interfaces.py`:

- `Segmenter` for source text. `segment(text) -> list[Segment]` returns raw sentence-shaped segments with `text`, `doc_char_start`, and `doc_char_end` (absolute 0-based half-open offsets in the document).
- `AnswerSegmenter` for the answer. `segment(text) -> list[AnswerSpan]` returns sentence or clause spans with `text`, `char_start`, `char_end`, `kind` (`"sentence"`, `"clause"`, or `"paragraph"`), `paragraph_index`, and `sentence_index` (offsets in the answer string).

The default / Rust path runs `SimpleAnswerSegmenter` on the answer and `SimpleSegmenter` on sources. The optional spaCy and pySBD segmenters swap into either role. Any class that exposes the right `segment` method conforms to the corresponding `Protocol`, so a custom segmenter is also a legal value for the `answer_segmenter` and `source_segmenter` arguments.

This page covers the four built-in segmenters and what changes when you swap one in. For the passage-window knobs that control how segments group into candidates, see [Citation Config](./citation-config.md). For the index-first path that depends on the segmenter choice, see [How It Works](../concepts/how-it-works.md). For the embedder interaction, see [Embedding Retrieval](../advanced/embedding-retrieval.md).

## A Small Run

You rarely construct a segmenter by hand; `align_citations` and `PreparedCitationCorpus.from_sources` default to the simple ones. Pass an explicit segmenter only when you need spaCy's clause splitting or pySBD's edge-case handling.

```python
from cite_right import (
    PreparedCitationCorpus,
    PySBDSegmenter,
    SourceDocument,
    SpacyAnswerSegmenter,
    SpacySegmenter,
    align_citations,
)

answer = "Apple revenue is up and stocks are down."
sources = [
    SourceDocument(
        id="earnings",
        text="Intro filler. Apple revenue is up. Outro filler.",
    ),
    "stocks are down. Extra filler follows.",
]

# spaCy with clause splitting on the answer, sentence-only on sources
results = align_citations(
    answer,
    sources,
    answer_segmenter=SpacyAnswerSegmenter(split_clauses=True),
    source_segmenter=SpacySegmenter(),
)
for result in results:
    print(result.answer_span.text, result.status, result.answer_span.kind)
```

For long-lived corpora, build the prepared corpus once with a non-default source segmenter and reuse it for many answers:

```python
corpus = PreparedCitationCorpus.from_sources(
    sources,
    source_segmenter=PySBDSegmenter(language="en"),
)
for later_answer in [answer, "Stocks fell after the report."]:
    print(corpus.align(later_answer)[0].status)
```

## SimpleSegmenter And SimpleAnswerSegmenter

`SimpleSegmenter` and `SimpleAnswerSegmenter` are the defaults. They live in `src/cite_right/text/segmenter_simple.py` and `src/cite_right/text/answer_segmenter.py` and have no extra dependencies.

`SimpleSegmenter` is a rule-based sentence splitter. It walks the text and cuts on `.`, `?`, or `!` when followed by whitespace, on `;`, and on `\n` when `split_on_newlines=True` (default). The abbreviation list is small (`dr`, `mr`, `mrs`, `ms`, `prof`, `sr`, `jr`, `st`, `vs`, `etc`, `e.g`, `i.e`) plus any single-letter initial pattern (e.g. `U.S.`). It does not require a model and runs in a single pass. Construct one with `SimpleSegmenter()` for the default; pass `split_on_newlines=False` if you want sentences only.

`SimpleAnswerSegmenter` is the answer-side wrapper. It first splits the answer into paragraphs on two-or-more consecutive line breaks, then runs `SimpleSegmenter(split_on_newlines=False)` on each paragraph so the line break is treated as a paragraph boundary rather than a sentence cut. The output is `AnswerSpan` objects with `kind="sentence"`, the paragraph's `paragraph_index`, and a global `sentence_index` across paragraphs.

```python
from cite_right import SimpleAnswerSegmenter, SimpleSegmenter

source_segmenter = SimpleSegmenter()            # default; cuts on \n
source_segmenter = SimpleSegmenter(split_on_newlines=False)  # sentence-only

answer_segmenter = SimpleAnswerSegmenter()      # default
```

Because both are the defaults, calling `align_citations(answer, sources)` with no segmenter arguments uses them.

## SpacySegmenter And SpacyAnswerSegmenter

`SpacySegmenter` and `SpacyAnswerSegmenter` wrap a spaCy language model (default `en_core_web_sm`). They require the `spacy` extra and the model itself. Install both before importing:

```bash
pip install "cite-right[spacy]==0.4.0"
python -m spacy download en_core_web_sm
```

The constructor raises `RuntimeError` with a clear message if `spacy` is not importable or if the named model is not installed. Pass `model="..."` to use a different spaCy pipeline.

`SpacySegmenter` runs `self._nlp(text)` and returns sentence spans. After spaCy's sentence boundary detection, `_split_sentence` walks the dependency parse to find coordinating conjunction tokens (`and`, `or`, `but`) that connect clauses. The clause splitter only cuts at a conjunction whose head is a `VERB`, `AUX`, or `ADJ` and where the conjunction sits strictly inside the sentence, so lists like `"Apples, oranges, and pears are tasty."` stay as one segment. The result is a list of `Segment` objects that mix sentence-level and clause-level boundaries, suitable for source-side passage generation.

`SpacyAnswerSegmenter` wraps the same logic for the answer. It splits paragraphs on two-or-more line breaks, runs spaCy per paragraph, and emits `AnswerSpan` objects with `kind="sentence"` by default. Pass `split_clauses=True` to emit clause-level `AnswerSpan` objects with `kind="clause"` and per-clause `sentence_index`. When no clauses are detected, it stays at one `AnswerSpan` per sentence, the same shape as `SimpleAnswerSegmenter`.

```python
from cite_right import SpacyAnswerSegmenter, SpacySegmenter

source_segmenter = SpacySegmenter()                       # sentence + clause
source_segmenter = SpacySegmenter(model="en_core_web_md") # different model

answer_segmenter = SpacyAnswerSegmenter()                         # sentence
answer_segmenter = SpacyAnswerSegmenter(split_clauses=True)       # clause
answer_segmenter = SpacyAnswerSegmenter(model="en_core_web_md")
```

Reuse one segmenter instance across calls; each constructor loads the spaCy pipeline.

## PySBDSegmenter

`PySBDSegmenter` is a rule-based sentence segmenter built on pySBD (Python Sentence Boundary Disambiguation). It handles abbreviations, URLs, emails, decimal numbers, and other edge cases that a simple rule-based splitter gets wrong. It is significantly faster than spaCy while staying rule-based, and it does not need a language model download.

Install the extra before importing:

```bash
pip install "cite-right[pysbd]==0.4.0"
```

`PySBDSegmenter` only implements the `Segmenter` protocol (source side). It is not used as an answer segmenter; `SimpleAnswerSegmenter` or `SpacyAnswerSegmenter` cover that role.

The constructor takes `language` (default `"en"`) and `clean` (default `False`). `clean=False` preserves the original text offsets, so each returned `Segment` is re-locatable in the input string; `clean=True` asks pySBD to pre-normalize the text first, which can break offset integrity. Leave `clean=False` unless you have a specific reason to enable it.

Segmentation is wrapped in an `lru_cache(maxsize=2000)` keyed on `(text, language, clean)`, so repeated calls with the same text and configuration are effectively free.

```python
from cite_right import PySBDSegmenter

source_segmenter = PySBDSegmenter()                  # English
source_segmenter = PySBDSegmenter(language="de")     # German rules
```

Use `PySBDSegmenter` when you want better sentence boundaries than `SimpleSegmenter` without paying the spaCy load cost. Pair it with `SimpleAnswerSegmenter` on the answer side.

## The Lexical Fallback

The default / Rust path runs an inverted index over source windows with a rare-token intersect; Smith-Waterman still localizes `char_start` / `char_end` on the hits. That path is gated on both the tokenizer and the segmenter being the simple defaults. If you supply a custom tokenizer **or** a custom segmenter, `PreparedCitationCorpus.from_sources` skips the Rust prepare path and leaves `inverted_index=None` and `rust_corpus=None`. `_select_candidates` then uses the lexical prefilter over `Candidate.token_set` and IDF, plus optional embedding extras. Smith-Waterman still runs on the chosen candidates; the index only chooses the windows.

A practical consequence: a non-default source segmenter forces the lexical fallback on the prepared corpus. The citation pipeline still works, and offsets are still correct, but the per-candidate selection step is no longer index-accelerated. For long-lived corpora with a custom segmenter, this is usually fine; for one-shot align calls, prefer the defaults unless you specifically need spaCy or pySBD boundaries.

```python
from cite_right import PreparedCitationCorpus, PySBDSegmenter

# Forces the lexical fallback path: inverted_index=None, rust_corpus=None
corpus = PreparedCitationCorpus.from_sources(
    sources,
    source_segmenter=PySBDSegmenter(),
)
```

`align_citations` and `PreparedCitationCorpus.from_sources` accept the segmenter arguments in the same place: as keyword-only `answer_segmenter` and `source_segmenter` parameters. When `align_citations` is called with non-default `source_segmenter`, it forwards the segmenter to `PreparedCitationCorpus.from_sources` so the same fallback rule applies.

## Passage Windows

Segmenter output feeds `generate_passages` in `src/cite_right/text/passage.py`. A `Passage` is a window of `window_size_sentences` consecutive segments sliding by `window_stride_sentences` segments. Default `window_size_sentences=1` and `window_stride_sentences=1` mean each sentence is its own window. Larger windows group sentences for cross-sentence alignment, at the cost of more candidates; stride larger than `1` skips windows and trades recall for speed.

A finer-grained segmenter (clauses from spaCy, sentence-level from pySBD) produces more passages per source. With the default `window_size_sentences=1`, more segments means more candidates; Smith-Waterman runs on each one. If your non-default segmenter produces many more segments than `SimpleSegmenter` would, consider `CitationConfig(window_size_sentences=2, window_stride_sentences=1)` to group them back into sentence-pair windows, or `CitationConfig.fast()` to cap the candidate pool.

## Custom Segmenters

Any class that implements `segment(text) -> list[Segment]` conforms to the `Segmenter` protocol, and any class that implements `segment(text) -> list[AnswerSpan]` conforms to the `AnswerSegmenter` protocol. Both are `runtime_checkable` `Protocol`s, so `isinstance(obj, Segmenter)` works as a sanity check. A custom segmenter takes the lexical fallback path described above; `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and `rust_corpus=None`, and `_select_candidates` uses the lexical prefilter. The public API is unchanged: pass the segmenter to `align_citations` or `PreparedCitationCorpus.from_sources`.

```python
from cite_right import Segmenter, Segment, align_citations


class RegexSegmenter:
    def segment(self, text: str) -> list[Segment]:
        # Return Segment objects with absolute 0-based half-open offsets.
        ...


results = align_citations(answer, sources, source_segmenter=RegexSegmenter())
```

Keep offsets half-open and re-slicable from the original text (`text[seg.doc_char_start:seg.doc_char_end] == seg.text`); the citation pipeline relies on that invariant to keep `Citation.char_start` / `Citation.char_end` rebasable onto the source after chunk rebasing.

## Choosing A Segmenter

Reach for the simple defaults unless you have a concrete reason to change. Both `SimpleSegmenter` and `SimpleAnswerSegmenter` are dependency-free, fast, and keep the inverted-index path on.

Reach for `SpacySegmenter` or `SpacyAnswerSegmenter` when you want finer-grained boundaries. The main use is `SpacyAnswerSegmenter(split_clauses=True)`, which lets a compound sentence produce one citation per clause. Source-side spaCy is useful when source sentences are long and you want clause-level passages; the cost is the spaCy model load on first call and a forced lexical fallback on the prepared corpus.

Reach for `PySBDSegmenter` when you want better sentence boundaries than the simple rule-based splitter without the spaCy load cost. It is well suited to source text that mixes URLs, decimals, and abbreviations. It is source-side only; pair it with `SimpleAnswerSegmenter` on the answer.

Reach for a custom segmenter when the built-in rules do not fit your domain (legal, scientific, multilingual) and you need full control. Remember the lexical fallback: a custom segmenter turns off the inverted-index path on the prepared corpus, so for very large corpora you may want to keep the simple defaults and post-process the answer instead.
