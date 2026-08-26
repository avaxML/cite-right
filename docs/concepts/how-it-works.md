# How It Works

Understanding the internal mechanics of Cite-Right helps you use the library more effectively and tune it for your specific requirements. This page explains the pipeline from raw text input to citation output as of 0.4.0.

## The Alignment Problem

When a language model generates text based on retrieved documents, we face a fundamental challenge: determining which parts of the source material support which parts of the generated response. This is complicated by several factors.

The generated text rarely quotes sources verbatim. Language models paraphrase, condense, and restructure information. A sentence in the answer might combine facts from multiple source paragraphs or express a source fact using completely different words.

Source documents vary widely in length and structure. Some are brief snippets while others are lengthy articles. The relevant evidence might appear anywhere within a document.

Multiple sources might support the same claim with different wording. We need to find the best match while acknowledging that alternatives exist.

Cite-Right addresses these challenges through text segmentation, index-first retrieval, and local sequence alignment. The public API is unchanged: `align_citations` and `PreparedCitationCorpus`. Span status is `"supported"`, `"partial"`, or `"unsupported"`. There is no `"partially_supported"` status.

## Index-First Retrieval

0.4.0 is index-first. During prepare, Cite-Right builds an inverted index over source passage windows. For each answer span it runs a rare-token intersect against that index and only then runs Smith-Waterman on the hits.

Smith-Waterman still localizes citations. The index only chooses which windows get Smith-Waterman. It does not skip alignment, and it does not scan every window with Smith-Waterman.

If the index returns nothing, the older lexical prefilter is the fallback. When an embedder is set, embedding similarity can add extra windows to that candidate set. Rust prepare still runs in that case. The embedding index is built on the prepared candidates.

## The Smith-Waterman Algorithm

Localization still uses Smith-Waterman, a dynamic programming approach originally developed for biological sequence alignment in 1981 by Temple Smith and Michael Waterman.

The algorithm finds the optimal local alignment between two sequences. Unlike global alignment which tries to align entire sequences end-to-end, local alignment identifies the best matching subsequences, ignoring regions that do not match well. That property fits citation extraction, where the relevant evidence may be a small portion of a larger document.

Given two sequences of tokens, the algorithm constructs a scoring matrix where each cell represents the best alignment score achievable ending at that position. Matches increase the score while mismatches and gaps decrease it. The highest score in the matrix indicates the best local alignment, and a traceback procedure recovers the aligned subsequences.

## From Text to Tokens

Before alignment can occur, both the answer and source texts must be converted to sequences of tokens. This tokenization process serves two purposes.

First, it normalizes the text to handle superficial variations. Different quote characters, Unicode forms, and capitalization should not prevent matching. The tokenizer applies Unicode NFKC normalization and case-folding to ensure consistent comparison.

Second, it maintains a mapping between tokens and their original character positions. Each token carries a `start_char` and `end_char` that point to its location in the original text. After alignment finds matching tokens, these offsets are used to extract the character-accurate evidence spans.

The default `SimpleTokenizer` handles common cases like hyphenated words, apostrophes, and numerical values. It treats "state-of-the-art" as a single token and correctly handles currency symbols and percentages. For specialized needs, alternative tokenizers based on HuggingFace transformers or OpenAI's tiktoken are available.

## The Citation Pipeline

The `align_citations` function orchestrates a multi-step pipeline that transforms raw input into structured citation results. This pipeline is defined in `src/cite_right/citations.py`.

### Step 1: Answer Segmentation

The answer text is split into individual spans, typically sentences. Each span becomes a separate unit for citation, receiving its own alignment score and status.

The default segmenter splits on sentence-ending punctuation and handles paragraph boundaries. The SpaCy-based segmenter provides more sophisticated boundary detection for complex text. The choice of segmenter affects granularity: finer segmentation produces more spans but may split related claims.

### Step 2: Source Passage Creation

Each source document is divided into passage windows. A passage is a contiguous section of the document, typically spanning several sentences. The windowing approach ensures that the alignment algorithm considers context around each sentence rather than matching sentences in isolation.

The window size and stride are configurable. A window of 3 sentences with a stride of 1 means each sentence appears in multiple overlapping windows, improving the chance of finding a good alignment.

This step runs during prepare. With the default tokenizer and segmenter, Rust prepare still runs when an embedder is set. The optional embedding index is built on those prepared windows.

### Step 3: Tokenization

Both answer spans and source passages are tokenized using the same tokenizer instance. Using a consistent tokenizer ensures that the same word receives the same token ID throughout, enabling accurate comparison.

### Step 4: Index-First Candidate Selection

Candidate selection reduces the search space before Smith-Waterman. 0.4.0 does this with an inverted index rather than pairing every answer span with every passage.

The inverted index maps tokens to the windows that contain them. Querying starts from rare tokens and intersects posting lists so Smith-Waterman only runs on hits. `max_candidates_lexical` caps how many of those index seeds are kept.

When an embedder is provided, semantic similarity is a complementary signal. The answer span and passages are encoded as dense vectors, and high-cosine-similarity passages can join the candidate set even if they were not index seeds. Embedding-only `retrieval_support` still respects `min_embedding_similarity`. Lexical scores are filled only for index seeds.

### Step 5: Smith-Waterman Alignment

The selected candidates undergo Smith-Waterman alignment against the answer span. The algorithm finds the best matching region within each passage.

The alignment returns a score indicating match quality along with the token positions of the matching region. Higher scores indicate better matches with more consecutive matching tokens and fewer gaps.

Sequential Smith-Waterman coverage is enough to emit a citation. Content-word overlap on the same candidate passage can also emit a citation when sequential coverage is low. That keeps grounded how-to and news paraphrases from being tagged `unsupported` just because shared content words are reordered.

Structured field:value sources get a second Smith-Waterman pass per matching candidate with `gap_score=0`. Faithful rewrites of hours, amenities, and similar fields can then be `supported` or `partial`. Invented fields stay `unsupported`.

### Step 6: Character Offset Calculation

The token positions are converted back to character offsets in the original source document. This step is critical for accuracy: the passage window introduces its own offset within the document, and the token alignment introduces an offset within the passage.

The final character offsets account for both layers, pointing to the exact location in the original document text. These offsets are absolute within the logical source document, so evidence extraction must use the same rebasing logic as `_slice_source_text()`: slice `source.full_text[char_start:char_end]` when full document text is available, otherwise subtract `base_doc_offset` before slicing `source.text`.

### Step 7: Ranking, Contradiction, and Status Assignment

Citations are ranked deterministically by final citation score. When scores tie, the exact tie-break order depends on `prefer_source_order`: by default ties prefer earlier sources, then earlier character positions, then longer evidence spans; when disabled, earlier character positions are preferred before source order. After sorting, duplicate citations from the same source with the same evidence span tuple are removed, and the ranked list is then trimmed by `max_citations_per_source` and `top_k`.

A cheap contradiction check then runs against the full candidate passage, not only the truncated Smith-Waterman evidence span. Negation, number, leftover n-gram slot, and entity-swap mismatches downgrade the span to `"partial"`. They do not make it `"unsupported"`. Shared tokens that would otherwise bless a contradictory statement as `"supported"` become `"partial"` for the same reason.

Each answer span receives a status from the top-ranked citation's `answer_coverage`, not its overall score, after that contradiction check. If the best citation's answer coverage meets `supported_answer_coverage` and no contradiction fired, the span is `"supported"`. If citations exist but stay below that threshold, or a contradiction fired, the span is `"partial"`. If no citations survive filtering, the span is `"unsupported"`.

## Scoring Components

The final citation score combines several signals, each measuring a different aspect of match quality. The `components` dictionary in each citation breaks down these contributions.

The normalized alignment score from Smith-Waterman forms the base. Answer coverage measures what fraction of the answer tokens appear in the alignment. Evidence coverage measures what fraction of the evidence tokens are matched, penalizing overly long evidence spans that happen to contain the answer.

When embeddings are enabled, cosine similarity between the answer span and evidence provides additional signal. This component helps identify paraphrased content where lexical matching underperforms.

The citation weights configuration controls how these components combine. Applications requiring high precision should emphasize answer coverage, while those tolerating paraphrase should give more weight to embedding similarity.

## Determinism and Reproducibility

Cite-Right prioritizes deterministic behavior. Given the same inputs and configuration, the library produces identical outputs across runs. This property is essential for debugging, testing, and compliance requirements.

The pure Python implementation serves as the reference for correctness. The optional Rust extension reproduces Python's behavior exactly, including tie-breaking order and floating-point rounding. Tests verify this equivalence across a comprehensive suite of inputs.

## Performance Characteristics

Tokenization is linear in text length. The SimpleTokenizer processes text with a single pass through the input.

Index-first retrieval makes candidate selection proportional to posting-list intersection on rare tokens rather than pairing every answer span with every window. Smith-Waterman remains quadratic in the length of the sequences being aligned, but it only runs on index hits (plus optional embedding extras).

On the 50-case pack with no embedder, 0.4.0 p50 wall time is about 12.4ms versus about 175.8ms in 0.3.1, roughly 14×. spp is 81.3% versus 83.4%. RAGTruth test quality on 2,675 answers matched 0.3.1.

The Rust extension keeps prepare, the inverted index, and alignment on the hot path. Released wheels are abi3 (`abi3-py311`) and include linux/aarch64 plus an sdist. Install with `pip install cite-right==0.4.0`.
