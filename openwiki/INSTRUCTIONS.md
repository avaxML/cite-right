# Cite-Right documentation brief

This file is the user-authored brief for OpenWiki. Generate and update pages
that coding agents read under `openwiki/` and that GitHub Pages publishes from
`docs/`. OpenWiki is the generator only. Readers of the public site must never
be told they are looking at a wiki, at MkDocs, or at OpenWiki. There is one
documentation site. OpenWiki must not rewrite this file on `--init` or
`--update`.

Match the voice of the current public pages. Start from `docs/index.md`,
`docs/getting-started/`, `docs/concepts/how-it-works.md`,
`docs/concepts/hallucination-detection.md`, and
`docs/advanced/rust-acceleration.md`.

## Public Paths

Write public pages at these paths under `openwiki/`, matching `mkdocs.yml`
exactly. Do not invent a `/wiki/` URL, a Wiki tab, a bridge page, or a second
nav.

- `index.md` (Home). This is a real Home page with one H1 and sections, not an
  OKF `# Files` / `# Directories` listing.
- `getting-started/installation.md`
- `getting-started/quickstart.md`
- `concepts/how-it-works.md`
- `concepts/citation-alignment.md`
- `concepts/hallucination-detection.md`
- `concepts/fact-verification.md`
- `configuration/citation-config.md`
- `configuration/presets.md`
- `configuration/tokenizers.md`
- `configuration/segmenters.md`
- `integrations/langchain.md`
- `integrations/llamaindex.md`
- `integrations/custom-sources.md`
- `advanced/multi-span-evidence.md`
- `advanced/embedding-retrieval.md`
- `advanced/rust-acceleration.md`
- `advanced/performance-tuning.md`

API Reference on the site is `docs/api/` via mkdocstrings (`api/core-functions.md`,
`api/data-models.md`, `api/configuration.md`). Do not generate those files under
`openwiki/`. On public pages, point at the public types in prose. Do not send
readers to a wiki.

Public-to-public links must use those paths (relative is fine, for example
`../concepts/how-it-works.md`). Do not use `/openwiki/` or `openwiki/` in
reader-facing hrefs. Do not link public pages at `testing/`.

## Agent-Only Pages

`openwiki/testing/` is for coding agents and is not published.

- `testing/pytest-markers.md`
- `testing/contract-tests.md`

Those pages may be denser. They may mention `openwiki/` paths. They still must
not tell anyone to look for a Wiki tab.

Do not generate a second public tree (`architecture/`, `operations/`,
`workflows/`). Extra depth for agents belongs under `testing/` or not at all.

## Voice And Page Shape

Address the reader as you. In prose the product is Cite-Right. The install name
is `cite-right`. The import package is `cite_right`.

Each generated public page gets one H1, then a short opening paragraph that
states what the page is for, then `##` Title Case sections. Put a small
runnable example before a long theory dump. Show `align_citations` /
`PreparedCitationCorpus` with complete but small Python snippets
(`SourceDocument`, print status, print evidence). Use bash for install.

Be honest, not marketing. Cite-Right is a groundedness and citation tagger, not
a clean hallucination detector. Do not invent benches. The only allowed
measured numbers are: 50-case pack with no embedder, p50 wall about 12.4ms
versus about 175.8ms in 0.3.1 (roughly 14×), spp 81.3% versus 83.4%; RAGTruth
test 2,675 answers, false-supported on gold hallucinations about 1.6%,
unsupported precision about 14%. If a number is not in that list and not in
current source, omit it.

Write status in quotes in prose: `"supported"`, `"partial"`, `"unsupported"`.
`"partial"` means low answer coverage **or** contradiction. The literal is
`"partial"`, never `"partially_supported"`.

Teach the public API. File paths such as `src/cite_right/citations.py` are fine
as orientation. Do not paste long function bodies on getting-started,
how-it-works, or installation pages. Testing pages may go deeper.

Do not write that the pipeline skips Smith-Waterman for speed. Do not write
that Rust prepare is skipped when an embedder is set.

Never use the words wiki, MkDocs, or OpenWiki on a public page. Never contrast
two documentation sites. Never add a "see the wiki" or "see MkDocs" link.

## What Cite-Right Does

Cite-Right aligns generated answer text to source documents and returns
character-accurate citations. When a language model answers from retrieved
documents, you call `align_citations` (in `src/cite_right/citations.py`) and
get per-span status plus `char_start` / `char_end` for highlighting.

0.4.0 is index-first on the default Rust path. An inverted index and rare-token
intersect choose which source windows are worth aligning. Smith-Waterman still
localizes the citation. The public API is unchanged: `align_citations` and
`PreparedCitationCorpus`.

```python
from cite_right import SourceDocument, align_citations

answer = "The company reported record revenue in Q4."
sources = [
    SourceDocument(
        id="earnings_call",
        text="During the earnings call, the CEO announced that the company reported record revenue in Q4 of 2024.",
    )
]

results = align_citations(answer, sources)
for result in results:
    print(result.answer_span.text, result.status)
    for citation in result.citations:
        print(citation.evidence, citation.char_start, citation.char_end)
```

Offsets are a Python half-open interval. After chunk rebasing,
`source.text[citation.char_start:citation.char_end] == citation.evidence`.

## Installation

Maintain `getting-started/installation.md`. Python 3.11+. 0.4.0 ships abi3
wheels (`abi3-py311`), linux/aarch64 wheels, and an sdist.

```bash
pip install cite-right==0.4.0
```

```bash
pip install "cite-right[embeddings]==0.4.0"
pip install "cite-right[spacy]==0.4.0"
pip install "cite-right[huggingface]==0.4.0"
pip install "cite-right[tiktoken]==0.4.0"
```

You can combine extras. SpaCy still needs `python -m spacy download en_core_web_sm`.
Rust prepare still runs when an embedder is set if `SimpleTokenizer` and
`SimpleSegmenter` are in use. Embedding-only `retrieval_support` still respects
`min_embedding_similarity`.

## Quickstart

Maintain `getting-started/quickstart.md`. Cover the basic pattern, reading
results, multiple sources, and `PreparedCitationCorpus`.

```python
from cite_right import CitationConfig, PreparedCitationCorpus, SourceDocument, align_citations

answer = "Acme Corporation reported revenue of 5.2 billion dollars in 2024."
sources = [
    SourceDocument(
        id="annual_report",
        text="Acme Corporation reported revenue of 5.2 billion dollars in 2024, representing a 12% increase over the previous year.",
    )
]
results = align_citations(answer, sources)
print(results[0].status)
print(results[0].citations[0].evidence)

corpus = PreparedCitationCorpus.from_sources(
    sources, config=CitationConfig(top_k=3)
)
for later_answer in [answer, "Revenue increased during fiscal year 2024."]:
    print(corpus.align(later_answer)[0].status)
```

`"supported"` means the best citation's `answer_coverage` is at least
`supported_answer_coverage` (default 0.6) and contradiction did not fire.
`"partial"` means citations exist but coverage is below that threshold, or
contradiction fired. `"unsupported"` means no citation survived filtering.

## How It Works

Maintain `concepts/how-it-works.md`. Pipeline orientation lives in
`src/cite_right/citations.py`. Do not paste long function bodies there.

1. Segment the answer (default `SimpleAnswerSegmenter`).
2. Prepare source passage windows.
3. Tokenize with one tokenizer instance (default `SimpleTokenizer`, Unicode NFKC
   and case-fold, original offsets kept).
4. Select candidates, then Smith-Waterman localizes.

Default / Rust path (`SimpleTokenizer` + `SimpleSegmenter` and
`cite_right._core` present): inverted index over source windows, rare-token
intersect. Smith-Waterman localizes on those hits. The index chooses windows.
Smith-Waterman still localizes `char_start` / `char_end`.

Fallback: if the optional Rust extension is missing, or a custom tokenizer or
segmenter is supplied, `PreparedCitationCorpus.from_sources` leaves
`inverted_index=None` and `_select_candidates` uses lexical selection.

Embedder path: `_add_embedding_candidates` can add windows that were not
inverted-index hits before alignment. Those extras still need Smith-Waterman.
`retrieval_support` is not a `Citation` and does not flip status.

Rust prepare still runs with an embedder when `SimpleTokenizer` and
`SimpleSegmenter` are in use. The embedding index is built on those prepared
candidates. Lexical scores are filled only for inverted-index seeds. 0.3.x
skipped Rust prepare on the embedder path. That skip is gone.

Content-word overlap on the candidate passage can emit a citation when
sequential Smith-Waterman coverage is low. That keeps grounded how-to and news
paraphrases from being tagged `"unsupported"` just because shared content words
are reordered.

Structured field:value sources (Data2txt hours, amenities, and similar) get a
second Smith-Waterman pass per matching candidate with `gap_score=0`. Faithful
rewrites can be `"supported"` or `"partial"`. Invented fields stay
`"unsupported"`.

The how-it-works status section must include the exact words
best citation's answer coverage
as a contiguous phrase with no backticks inside it. Do not write
best citation score. Status still comes from the top citation's
`answer_coverage` versus `supported_answer_coverage` (default 0.6).
Contradiction stays `"partial"`. Never `"partially_supported"`.

## Citation Alignment

Maintain `concepts/citation-alignment.md` for inputs, outputs, and offsets.

`SourceDocument(id, text)` is a full document.
`SourceChunk(source_id, text, doc_char_start, doc_char_end)` is a pre-chunked
excerpt. Chunk offsets are rebased onto the original document.

Each `SpanCitations` has `answer_span`, ranked `citations`,
`retrieval_support`, and `status`. Status comes from the best exact citation,
not from embedding similarity. A high embedding score that never localizes is
`retrieval_support`, not a `Citation`.

## Hallucination Detection

Cite-Right is a groundedness and citation tagger, not a clean hallucination
detector. Maintain `concepts/hallucination-detection.md` in
`src/cite_right/hallucination.py` terms. It marks whether each answer span has
localized source support. Treat `"unsupported"` as "no localized citation
survived," not as a high-precision hallucination label.

On RAGTruth test (2,675 answers), 0.4.0 quality matched 0.3.1. False-supported
on gold hallucinations is about 1.6%. Unsupported precision is about 14%. The
tagger overflags: many spans tagged `"unsupported"` are not gold hallucinations.
If `"partial"` counts as not fully supported, gold hallucinations are rarely
blessed as `"supported"`.

```python
from cite_right import SourceDocument, align_citations, compute_hallucination_metrics

answer = """The company reported record profits in Q4.
They announced plans to expand into Asia."""
sources = [
    SourceDocument(
        id="earnings",
        text="Fourth quarter profits reached an all-time high, beating analyst expectations.",
    )
]
results = align_citations(answer, sources)
metrics = compute_hallucination_metrics(results)
print(metrics.groundedness_score, metrics.hallucination_rate)
```

`HallucinationConfig.include_partial_in_grounded` (default True) controls
whether `"partial"` contributes to groundedness. Convenience helpers
`is_grounded`, `is_hallucinated`, and `check_groundedness` inherit the same
overflag. Do not present their thresholds as calibrated hallucination cutoffs.

## Fact Verification

Maintain `concepts/fact-verification.md` for `verify_facts` in
`src/cite_right/fact_verification.py`. Sentence-level tagging can hide a mixed
sentence. Claim decomposition splits it. `SimpleClaimDecomposer` keeps one
claim per sentence. `SpacyClaimDecomposer` splits coordinated clauses and needs
the spacy extra.

```python
from cite_right import SourceDocument, verify_facts

answer = "The product launched in March and sales exceeded 10 million units."
sources = [
    SourceDocument(
        id="press_release",
        text="The new product line was introduced to the market in March 2024.",
    )
]
result = verify_facts(answer, sources)
print(result.total_claims, result.num_verified, result.num_unverified)
```

## Configuration

Maintain configuration pages for `CitationConfig` / `CitationWeights`
(`src/cite_right/core/citation_config.py`), presets, tokenizers, and
segmenters.

Default `supported_answer_coverage` is 0.6. Other defaults you may document
from source: `top_k=3`, `min_final_score=0.0`, `min_answer_coverage=0.2`,
`min_embedding_similarity=0.3`, `max_candidates_lexical=200`. Presets:
`CitationConfig.balanced()`, `.strict()`, `.permissive()`, `.fast()`.
Permissive still requires localized Smith-Waterman evidence. It does not emit
embedding-only citations.

Tokenizers: `SimpleTokenizer` (default), `HuggingFaceTokenizer`,
`TiktokenTokenizer`. Segmenters: `SimpleSegmenter` / `SimpleAnswerSegmenter`
(default), `SpacySegmenter` / `SpacyAnswerSegmenter`, `PySBDSegmenter`. A
custom tokenizer or segmenter takes the lexical fallback path above.

## Integrations

Maintain LangChain, LlamaIndex, and custom-sources pages.

LangChain: `from_langchain_documents` / `from_langchain_chunks` in
`src/cite_right/integrations.py`. LlamaIndex: `from_llamaindex_nodes` /
`from_llamaindex_chunks`. Custom: `SourceDocument` directly, or `from_dicts`.
Then call `align_citations` as usual.

## Advanced Topics

**Multi-Span Evidence.** Off by default. `CitationConfig(multi_span_evidence=True)`.
Prefer `evidence_spans` or `exact_evidence` for precise attribution. Legacy
`evidence` / `char_start` / `char_end` stay a contiguous enclosing span.

**Embedding Retrieval.** `pip install "cite-right[embeddings]==0.4.0"`, then
`SentenceTransformerEmbedder("all-MiniLM-L6-v2")`. Index-first still chooses
lexical seeds. `_add_embedding_candidates` may add non-index windows. Those
still need Smith-Waterman. `retrieval_support` is not a `Citation` and does
not flip status. Rust prepare still runs with the embedder on the simple
tokenizer/segmenter path.

**Rust Acceleration.** Optional `cite_right._core`.
`backend="auto"|"python"|"rust"`. Prepare, inverted index, and alignment stay
on the hot path. Rust must match Python outputs. If `_core` is missing,
candidate selection falls back to lexical prefilter. Do not claim
Smith-Waterman is skipped for speed.

**Performance.** Index-first means Smith-Waterman runs on index hits plus
optional embedding extras, not on every window. On the 50-case pack with no
embedder, 0.4.0 p50 wall time is about 12.4ms versus about 175.8ms in 0.3.1,
roughly 14×. spp is 81.3% versus 83.4%. RAGTruth test quality on 2,675 answers
matched 0.3.1. Embedder encoding is extra cost on top of the no-embedder
numbers. Do not add other latency or quality figures.

## Invariants

These must stay true on every generated page that touches them.

- Public API: `align_citations`, `PreparedCitationCorpus`. Default
  `supported_answer_coverage` is 0.6.
- Half-open `char_start` / `char_end`. Evidence equals the sliced source after
  chunk rebasing.
- Status is exactly `"supported"`, `"partial"`, or `"unsupported"`. Never
  `"partially_supported"`. `"partial"` is low coverage or contradiction.
- Default / Rust path: inverted index, rare-token intersect, Smith-Waterman
  localizes on hits. The index chooses windows. Smith-Waterman still localizes.
- Fallback: no Rust, or a custom tokenizer/segmenter, then
  `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and
  `_select_candidates` uses lexical selection.
- Embedder: `_add_embedding_candidates` may add non-index windows. Those still
  need Smith-Waterman. `retrieval_support` is not a `Citation` and does not
  flip status.
- Rust prepare still runs with an embedder when `SimpleTokenizer` and
  `SimpleSegmenter` are in use.
- Contradiction (negation, number, leftover n-gram slot, entity swap)
  downgrades to `"partial"` with citations, never `"unsupported"`. The check
  uses the full candidate passage, not only truncated Smith-Waterman evidence.
  Example: source "The vaccine is safe and effective." and answer "The vaccine
  is not safe." is `"partial"` with citations.
- Content-word overlap can emit a citation when sequential Smith-Waterman
  coverage is low.
- Data2txt field:value gets a second Smith-Waterman pass with `gap_score=0`.
  Invented fields stay `"unsupported"`.
- Do not document `evaluation/`, hill-climb, or RAGTruth tables beyond the two
  allowed measured lines.

## Testing Pages

Cover pytest markers from `tests/conftest.py`: `rust`, `spacy`, `embeddings`,
`tiktoken`, `huggingface`, `pysbd`, `slow`. Contract tests compare Python and
Rust backends for status and offsets. Point at `src/cite_right/__init__.py` for
the public surface, `src/cite_right/citations.py` for the pipeline,
`src/cite_right/core/prepared_corpus.py` for prepare,
`src/cite_right/contradiction.py` for the cheap check, and `rust_core/` for the
extension. Do not dump private helper bodies onto getting-started pages.

## CI Model Rotation

OpenWiki in GitHub Actions uses OpenRouter `:free` models.
`scripts/openwiki_pick_model.py` ranks them (prefer `tools`, then Artificial
Analysis `coding_index` if present, else `context_length`, then `created`).
`scripts/openwiki_update.sh` retries the next model on 429/402/rate-limit,
403/404, agentic-harness blocks, model-unavailable errors, and empty/malformed
completions that crash OpenWiki with `Cannot read properties of undefined (reading '0')`.
A 429 sleeps `retry_after_seconds` or until `X-RateLimit-Reset` (cap 90s). After a
successful update, `scripts/publish_openwiki_to_docs.py` copies the public
paths above into `docs/`. That rotation and copy step are CI machinery, not
page content. Do not document a paid model id as required.
