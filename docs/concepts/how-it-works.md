# How It Works

Understanding the internal mechanics of Cite-Right helps you use the library more effectively and tune it for your specific requirements. This page explains the complete pipeline from raw text input to citation output.

## The Alignment Problem

When a language model generates text based on retrieved documents, we face a fundamental challenge: determining which parts of the source material support which parts of the generated response. This is complicated by several factors.

The generated text rarely quotes sources verbatim. Language models paraphrase, condense, and restructure information. A sentence in the answer might combine facts from multiple source paragraphs or express a source fact using completely different words.

Source documents vary widely in length and structure. Some are brief snippets while others are lengthy articles. The relevant evidence might appear anywhere within a document.

Multiple sources might support the same claim with different wording. We need to find the best match while acknowledging that alternatives exist.

Cite-Right addresses these challenges through a combination of text segmentation, candidate retrieval, and local sequence alignment.

## The Smith-Waterman Algorithm

At the core of Cite-Right is the Smith-Waterman algorithm, a dynamic programming approach originally developed for biological sequence alignment in 1981 by Temple Smith and Michael Waterman. The algorithm was published in the Journal of Molecular Biology and has since become a fundamental tool in bioinformatics.

The algorithm finds the optimal local alignment between two sequences. Unlike global alignment which tries to align entire sequences end-to-end, local alignment identifies the best matching subsequences, ignoring regions that do not match well. This property makes it ideal for citation extraction where the relevant evidence may be a small portion of a larger document.

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

### Step 3: Tokenization

Both answer spans and source passages are tokenized using the same tokenizer instance. Using a consistent tokenizer ensures that the same word receives the same token ID throughout, enabling accurate comparison.

### Step 4: Candidate Selection

Aligning every answer span against every passage would be computationally expensive. Candidate selection reduces the search space by identifying passages likely to contain relevant evidence.

The lexical prefilter computes IDF-weighted token overlap between each answer span and passage. Passages with higher overlap are more likely to contain matching content. This filter is fast and effective for near-verbatim matching.

When an embedder is provided, semantic similarity provides a complementary signal. The answer span and passages are encoded as dense vectors, and passages with high cosine similarity are prioritized. This approach captures paraphrased content that shares meaning but not vocabulary.

### Step 5: Smith-Waterman Alignment

The selected candidates undergo full Smith-Waterman alignment against the answer span. The algorithm finds the best matching region within each passage.

The alignment returns a score indicating match quality along with the token positions of the matching region. Higher scores indicate better matches with more consecutive matching tokens and fewer gaps.

### Step 6: Character Offset Calculation

The token positions are converted back to character offsets in the original source document. This step is critical for accuracy: the passage window introduces its own offset within the document, and the token alignment introduces an offset within the passage.

The final character offsets account for both layers, pointing to the exact location in the original document text. Extracting `source.text[char_start:char_end]` yields the evidence string.

### Step 7: Ranking and Status Assignment

Citations are ranked by their alignment score, with ties broken by source order, character position, and evidence length. This deterministic ranking ensures reproducible results.

Each answer span receives a status based on its best citation score. Spans with high-quality matches are "supported", those with moderate matches are "partial", and those without adequate matches are "unsupported".

## Scoring Components

The final citation score combines several signals, each measuring a different aspect of match quality. The `components` dictionary in each citation breaks down these contributions.

The normalized alignment score from Smith-Waterman forms the base. Answer coverage measures what fraction of the answer tokens appear in the alignment. Evidence coverage measures what fraction of the evidence tokens are matched, penalizing overly long evidence spans that happen to contain the answer.

When embeddings are enabled, cosine similarity between the answer span and evidence provides additional signal. This component helps identify paraphrased content where lexical matching underperforms.

The citation weights configuration controls how these components combine. Applications requiring high precision should emphasize answer coverage, while those tolerating paraphrase should give more weight to embedding similarity.

## Determinism and Reproducibility

Cite-Right prioritizes deterministic behavior. Given the same inputs and configuration, the library produces identical outputs across runs. This property is essential for debugging, testing, and compliance requirements.

The pure Python implementation serves as the reference for correctness. The optional Rust extension reproduces Python's behavior exactly, including tie-breaking order and floating-point rounding. Tests verify this equivalence across a comprehensive suite of inputs.

## Performance Characteristics

The computational complexity of citation alignment depends on several factors.

Tokenization is linear in text length. The SimpleTokenizer processes text with a single pass through the input.

Candidate selection with the lexical filter is proportional to the number of passages times the vocabulary size, but the use of set operations keeps this efficient in practice.

Smith-Waterman alignment is quadratic in the length of the sequences being aligned. The candidate selection step limits the number of full alignments performed, making the total cost manageable.

The Rust extension provides substantial speedup for the alignment step through parallel processing with Rayon. For workloads with many sources or long documents, the extension reduces latency significantly while maintaining identical results.
