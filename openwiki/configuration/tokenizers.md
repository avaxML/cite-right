---
type: configuration
title: Tokenizers
description: Tokenizer options for the citation alignment pipeline — SimpleTokenizer (default, Unicode NFKC and case-fold with original character offsets), HuggingFaceTokenizer, and TiktokenTokenizer. The Tokenizer protocol, the TokenizedText offset contract, the optional TokenizerConfig normalization knobs, and the rule that a custom tokenizer forces the lexical fallback path with no inverted index.
tags: [configuration, tokenizer, simple-tokenizer, huggingface-tokenizer, tiktoken-tokenizer, tokenizer-config, tokenized-text, unicode-nfkc, case-fold, char-offsets, lexical-fallback, inverted-index, rust-prepare, subword, bpe, token-protocol]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T03:32:47.432Z
sources:
  - id: openwiki-source-0d7249770abeac51acffd6d9
    resource: repo://src/cite_right/__init__.py
  - id: openwiki-source-9420301e9a6eeb80c89f2f99
    resource: repo://src/cite_right/citations.py
  - id: openwiki-source-70a6feac670e6bc0185a21c7
    resource: repo://src/cite_right/core/interfaces.py
  - id: openwiki-source-b3431af41a97a9253d6038b0
    resource: repo://src/cite_right/core/prepared_corpus.py
  - id: openwiki-source-32f69dac67cf7ab0d63041c2
    resource: repo://src/cite_right/core/results.py
  - id: openwiki-source-5aae6732a9c8e118a74dd279
    resource: repo://src/cite_right/text/tokenizer_huggingface.py
  - id: openwiki-source-280e6689245ed27fbc16e8ee
    resource: repo://src/cite_right/text/tokenizer_tiktoken.py
  - id: openwiki-source-ccf29287cebbf95d80aebc2f
    resource: repo://src/cite_right/text/tokenizer.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T03:32:47.432Z"}
---

# Tokenizers

A tokenizer turns the answer text and the source passages into the integer token IDs the rest of the pipeline operates on. The same tokenizer instance is used on both sides, so a token in the answer has the same integer ID as the same token in a source passage. That shared vocabulary is what makes Smith-Waterman meaningful across the two sides.

The public surface is the `Tokenizer` protocol in `src/cite_right/core/interfaces.py`: a single `tokenize(text) -> TokenizedText` method. The three built-in implementations are `SimpleTokenizer` (default, dependency-free, rule-based), `HuggingFaceTokenizer` (subword via the `tokenizers` and `transformers` libraries), and `TiktokenTokenizer` (BPE via OpenAI's `tiktoken` library, used by GPT-4 / GPT-3.5). All three produce the same `TokenizedText` shape with character-accurate offsets into the original text, so the citation rebase works regardless of which one you pick.

A custom tokenizer is allowed, but it takes the lexical fallback path: `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and `rust_corpus=None`, and `_select_candidates` uses the lexical prefilter over `Candidate.token_set` plus optional embedding extras. Smith-Waterman still runs on the chosen candidates.

For the index-first pipeline that the default tokenizer enables, see [How It Works](../concepts/how-it-works.md). For the embedder path and where tokenizers interact with the candidate pool, see [Embedding Retrieval](../advanced/embedding-retrieval.md). For the high-level `CitationConfig` knobs, see [Citation Config](./citation-config.md).

## A Small Run

The default call uses `SimpleTokenizer`; you rarely need to pass one. Reach for an explicit tokenizer when the default's rule-based tokenization is too coarse (BPE / subword) or too narrow (you want the same tokenization as a specific model).

```python
from cite_right import (
    PreparedCitationCorpus,
    SimpleTokenizer,
    SourceDocument,
    TiktokenTokenizer,
    align_citations,
)

answer = "Revenue grew 15% in Q4."
sources = [
    SourceDocument(
        id="earnings",
        text="Annual report: Revenue grew 15% in Q4 2024.",
    ),
]

# Default: SimpleTokenizer is constructed for you
results = align_citations(answer, sources)
print(results[0].status, results[0].citations[0].evidence)

# Explicit tiktoken BPE for the same alignment
tiktoken = TiktokenTokenizer()  # cl100k_base
results = align_citations(answer, sources, tokenizer=tiktoken)

# Long-lived corpus, tokenizer reused across many answers
corpus = PreparedCitationCorpus.from_sources(sources, tokenizer=tiktoken)
for later_answer in [answer, "Q4 revenue was up 15%."]:
    print(corpus.align(later_answer, tokenizer=tiktoken)[0].status)
```

Pass `tokenizer` to both `align_citations` and `PreparedCitationCorpus.from_sources`; `align_citations` forwards it to the prepared corpus under the hood.

## The Tokenizer Contract

The `Tokenizer` protocol in `src/cite_right/core/interfaces.py` is small on purpose:

```python
class Tokenizer(Protocol):
    def tokenize(self, text: str) -> TokenizedText: ...
```

It is `@runtime_checkable`, so `isinstance(obj, Tokenizer)` works as a sanity check. The result is a `TokenizedText` Pydantic model from `src/cite_right/core/results.py` with three fields:

- `text`: the original input text.
- `token_ids`: a list of integer token IDs. Repeated tokens share the same ID within one tokenizer instance.
- `token_spans`: a list of `(start, end)` character offsets into `text`. Offsets are 0-based and half-open (`text[span[0]:span[1]] == token`).

`TokenizedText` validates the result on construction. The invariants are enforced by `_validate_token_spans` in `src/cite_right/core/results.py`:

- `len(token_ids) == len(token_spans)`.
- Every span stays within text bounds: `0 <= start < end <= len(text)`.
- Spans are monotonic and non-overlapping, so the citation pipeline can rebuild a contiguous character range by `start[k]:end[k]` for any token `k`.

If those invariants are violated, the construction raises `ValueError`. The citation pipeline relies on them to rebase `Citation.char_start` / `Citation.char_end` onto the source after chunk rebasing, so any custom tokenizer must keep them. The Smith-Waterman step needs half-open character spans to localize evidence; subword outputs that do not expose offsets (for example, some tokenizers that only return IDs) cannot be used as a `Tokenizer` here.

## SimpleTokenizer

`SimpleTokenizer` is the default. It is dependency-free, rule-based, and built for citation alignment. It lives in `src/cite_right/text/tokenizer.py`.

The tokenization is a single pass over the text. Spans are produced by `_iter_token_spans` (an `lru_cache(maxsize=10000)` helper), then each span is normalized by `_normalize_token_cached` (also cached). The two caches mean repeated calls on the same text are cheap.

Token boundaries come from three rules:

- A run of digits, optionally with internal `.` or `,` between two digit characters, is a number token.
- A single `%`, `$`, `€`, or `£` is its own token.
- A run of word characters (Unicode alphanumerics), optionally containing apostrophes and hyphens that are bracketed by word characters on both sides, is a word token.
- Anything else is skipped (whitespace, standalone punctuation, combining marks attached to a word ride along on the word token).

Token normalization applies, in order, Unicode NFKC, case-folding, and a punctuation map that folds curly quotes and a range of dash variants (U+2010 through U+2013, U+2212) to their ASCII counterparts. So `"company’s"`, `"state–of–the–art"`, and `"STRASSE"` tokenize to the same IDs as their ASCII or lower-cased forms. The `token_spans` always point into the original text, not the normalized one, so the citation rebase stays correct.

Three optional knobs on `TokenizerConfig` control domain-specific normalization on top of that:

| Field | Default | Effect |
|-------|---------|--------|
| `normalize_numbers` | `True` | Strips `,` from number tokens, so `"1,200"` and `"1200"` share an ID. |
| `normalize_percent` | `True` | Maps the standalone `%` token to the word `"percent"`. With it off, `%` and the word `"percent"` stay distinct IDs. |
| `normalize_currency` | `True` | Maps the standalone `$`, `€`, `£` tokens to `"dollar"`, `"euro"`, `"pound"`. |

```python
from cite_right import SimpleTokenizer, TokenizerConfig

tokenizer = SimpleTokenizer()                    # all three defaults on
tokenizer = SimpleTokenizer(TokenizerConfig(
    normalize_numbers=True,
    normalize_percent=True,
    normalize_currency=False,                     # keep $ and "dollar" distinct
))
```

`SimpleTokenizer` builds its vocabulary lazily: `_vocab` maps each normalized token to a stable integer ID, and `_next_id` advances on first sight. Two `SimpleTokenizer` instances started from the same `TokenizerConfig` will assign the same IDs to the same normalized tokens only if they have seen them in the same order. The Rust prepare path overwrites `_vocab` from the Rust `PreparedCorpus.get_vocab()` so the Python tokenizer's vocabulary matches the index that was actually used to prepare the corpus.

## HuggingFaceTokenizer

`HuggingFaceTokenizer` wraps a HuggingFace tokenizer to produce the same `TokenizedText` shape, with subword tokens and character offsets. It supports both the fast `tokenizers.Tokenizer` library and `transformers.PreTrainedTokenizerBase`. It lives in `src/cite_right/text/tokenizer_huggingface.py`.

Install the extra before importing:

```bash
pip install "cite-right[huggingface]==0.4.0"
```

The constructor accepts either a `tokenizers.Tokenizer` or a `transformers.PreTrainedTokenizerBase`. `_check_tokenizer_type` discriminates the two paths. Anything else raises `TypeError` with a message that names both expected types.

```python
from cite_right import HuggingFaceTokenizer

# From a transformers AutoTokenizer (most common)
tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.tokenize("Hello, world!").token_ids)
```

`from_pretrained` is a convenience class method that loads via `transformers.AutoTokenizer.from_pretrained`. It takes `model_name_or_path`, `add_special_tokens` (default `False`), and `use_fast` (default `True`, the Rust-based fast tokenizer). When `transformers` is not installed, `from_pretrained` raises `ImportError` pointing at `cite-right[huggingface]`.

`tokenize(text)` dispatches to one of two private paths:

- For a `transformers` tokenizer, it calls the tokenizer with `return_offsets_mapping=True` and reads the `(start, end)` spans from the result. Special tokens (those with `(0, 0)` offsets) are filtered out.
- For a `tokenizers.Tokenizer` instance, it calls `tokenizer.encode(text, add_special_tokens=...)` and reads `.ids` and `.offsets` from the encoding. Tokens with empty spans (`start == end`) are filtered out.

The `add_special_tokens=False` default is intentional: `[CLS]`, `[SEP]`, and similar markers have empty character spans in the source text, so they would always fail the `TokenizedText` invariant that `start < end`. Set `add_special_tokens=True` only if you want those markers in the token stream and have a downstream reason to keep them.

`HuggingFaceTokenizer` is the right choice when you want alignment to use the same tokenization as a model you control (BERT, RoBERTa, a domain-specific encoder). The shared vocabulary means an answer span and a source span tokenize to overlapping ID sequences, which is what Smith-Waterman matches on.

## TiktokenTokenizer

`TiktokenTokenizer` wraps an OpenAI `tiktoken` BPE encoding and produces the same `TokenizedText` shape. It lives in `src/cite_right/text/tokenizer_tiktoken.py`.

Install the extra before importing:

```bash
pip install "cite-right[tiktoken]==0.4.0"
```

The constructor takes `encoding_name` (default `"cl100k_base"`, used by GPT-4, GPT-3.5-turbo, and `text-embedding-ada-002`) or a pre-initialized `tiktoken.Encoding`. With no `tiktoken` installed, the constructor raises `ImportError` pointing at `cite-right[tiktoken]`. The `encoding_name` property returns the resolved encoding name.

```python
from cite_right import TiktokenTokenizer

tokenizer = TiktokenTokenizer()                      # cl100k_base
tokenizer = TiktokenTokenizer("p50k_base")           # Codex
tokenizer = TiktokenTokenizer("r50k_base")           # GPT-3 davinci/curie
```

The interesting part is the byte-to-character mapping. `tiktoken` operates on UTF-8 bytes, but `TokenizedText` needs Python character offsets. The tokenizer encodes the text, builds a `byte_to_char` lookup by walking the UTF-8 byte stream once, then for each BPE token it decodes the single token to bytes and maps the byte range to a character range through that table. The result is a list of `(char_start, char_end)` pairs that always sum to a non-empty character slice in the original text.

If a BPE token somehow maps to a zero-width character span (a degenerate text / encoding combination), the tokenizer raises `ValueError` explaining that exact span tracing is not reliable for that combination. In practice this only shows up with synthetic inputs; ordinary UTF-8 text is fine.

`TiktokenTokenizer` is the right choice when you want alignment to use the same tokenization as a GPT-family model. The vocabulary is large, so spans are short and matches on a paraphrase are more likely to localize.

## The Lexical Fallback

The default / Rust path runs an inverted index over source windows with a rare-token intersect; Smith-Waterman still localizes `char_start` / `char_end` on the hits. That path is gated on both the tokenizer and the segmenter being the simple defaults: in `PreparedCitationCorpus.from_sources`, `_from_sources_rust` only runs when `isinstance(tokenizer, SimpleTokenizer)` and `isinstance(source_segmenter, SimpleSegmenter)`. The same condition determines whether `rust_corpus` is populated.

If you pass any other tokenizer (`HuggingFaceTokenizer`, `TiktokenTokenizer`, or a custom class), `PreparedCitationCorpus.from_sources` takes the Python prepare path. It builds `source_passages`, `candidates`, and `idf` from the supplied tokenizer, leaves `inverted_index=None` and `rust_corpus=None`, and `_select_candidates` uses the lexical prefilter over `Candidate.token_set` (an IDF-weighted overlap) plus optional embedding extras. Smith-Waterman still runs on the chosen candidates; the index only chooses the windows.

A practical consequence: a non-default tokenizer disables the index-accelerated path on the prepared corpus. The citation pipeline still works, and offsets are still correct, but the per-candidate selection step is no longer index-accelerated. For long-lived corpora, this is usually fine because `PreparedCitationCorpus.from_sources` is called once and the result is reused; for one-shot align calls, prefer the defaults unless you specifically need subword tokenization.

```python
from cite_right import PreparedCitationCorpus, TiktokenTokenizer

# Forces the lexical fallback path: inverted_index=None, rust_corpus=None
corpus = PreparedCitationCorpus.from_sources(
    sources,
    tokenizer=TiktokenTokenizer(),
)
```

`align_citations` and `PreparedCitationCorpus.from_sources` accept the tokenizer in the same place: as a keyword-only `tokenizer` parameter. When `align_citations` is called with a non-default `tokenizer`, it forwards the tokenizer to `PreparedCitationCorpus.from_sources` so the same fallback rule applies.

## Custom Tokenizers

Any class that implements `tokenize(text) -> TokenizedText` conforms to the `Tokenizer` protocol, so a custom tokenizer is a legal value for the `tokenizer` argument. The only hard requirements come from the `TokenizedText` validation:

- `len(token_ids) == len(token_spans)`.
- Every `(start, end)` is inside the text and `start < end`.
- Spans are monotonic and non-overlapping.
- The same token text in two runs gets the same integer ID (so a token in the answer has the same ID as the same token in a source passage).

Anything more elaborate (a vocabulary with a custom mapping, subword IDs, a learned tokenizer) is fine as long as the offset contract holds. The downstream code only touches `token_ids` (for Smith-Waterman matching) and `token_spans` (for offset rebase and `evidence` slicing).

A custom tokenizer takes the lexical fallback path described above: `PreparedCitationCorpus.from_sources` leaves `inverted_index=None` and `rust_corpus=None`, and `_select_candidates` uses the lexical prefilter. The public API is unchanged: pass the tokenizer to `align_citations` or `PreparedCitationCorpus.from_sources`.

```python
from cite_right import Segment, Segmenter, TokenizedText, align_citations


class LowercaseWordTokenizer:
    """A minimal custom tokenizer: lowercase word spans with shared IDs."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def tokenize(self, text: str) -> TokenizedText:
        ids: list[int] = []
        spans: list[tuple[int, int]] = []
        idx = 0
        while idx < len(text):
            ch = text[idx]
            if ch.isalnum():
                end = idx + 1
                while end < len(text) and text[end].isalnum():
                    end += 1
                token = text[idx:end].lower()
                token_id = self._vocab.get(token)
                if token_id is None:
                    token_id = self._next_id
                    self._vocab[token] = token_id
                    self._next_id += 1
                ids.append(token_id)
                spans.append((idx, end))
                idx = end
            else:
                idx += 1
        return TokenizedText(text=text, token_ids=ids, token_spans=spans)


results = align_citations(answer, sources, tokenizer=LowercaseWordTokenizer())
```

Two non-obvious traps when writing a custom tokenizer:

- A tokenizer that returns token IDs without exposing character spans is not usable here. The citation rebase needs `text[span[0]:span[1]]` to equal the original slice of source text.
- A tokenizer whose `token_ids` collide on different strings will break Smith-Waterman. The shared vocabulary between the answer and the source is the only signal the aligner has for "these are the same token."

## Choosing A Tokenizer

Reach for the default `SimpleTokenizer` unless you have a concrete reason to change. It is dependency-free, fast, keeps the inverted-index path on, and handles the Unicode and number normalization that the citation pipeline relies on (so `"café"`, `"CAFÉ"`, and `"cafe\u0301"` tokenize to the same ID, and `"$5"` and `"5 dollars"` share the number side).

Reach for `HuggingFaceTokenizer` when you want alignment to use the same tokenization as a specific encoder model. The most common reason is that you are scoring the answer with a model that has its own vocabulary, and you want the citation's tokens to line up with the model's tokens.

Reach for `TiktokenTokenizer` when you want BPE tokenization that matches a GPT-family model. The cl100k base is the right default; switch to `p50k_base` or `r50k_base` only when matching a specific older model.

Reach for a custom tokenizer when the built-in tokenization is wrong for your domain (highly multilingual text, code with custom identifiers, scientific notation that does not match the simple number rules). Remember the lexical fallback: a custom tokenizer turns off the inverted-index path on the prepared corpus, so for very large corpora you may want to keep the simple defaults and post-process the text instead.
