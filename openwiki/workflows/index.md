# Files

- [align_citations workflow](align-citations.md) - Step-by-step walkthrough of the main citation alignment API — signature, pipeline phases, return-shape interpretation, and status decisions.
- [Embedding-backed recall](embedding-recall.md) - How semantic similarity via `embedder=` expands candidate recall, how `_add_embedding_candidates` merges embedding scores with lexical candidates, and when embedding-only passages become RetrievalSupport instead of Citations.
- [Groundedness checks workflow](groundedness-checks.md) - Patterns for using span-level hallucination checks and claim-level fact verification in RAG post-processing pipelines.
- [High-precision tuning](high-precision-tuning.md) - How to bias the citation alignment pipeline toward fewer false positives — the benchmarked high-precision configuration, the role of each filtering knob, and how to adapt the recipe for domain-specific use cases.
- [Prepared corpus workflow](prepared-corpus.md) - When and how to use PreparedCitationCorpus.from_sources(...).align(answer) to amortize prepare cost across many answers.
- [Source input shapes](source-inputs.md) - The three accepted input types for source documents, how they are normalized into `NormalizedSource`, and how citation offsets are rebased relative to the original document.
- [Structured-field sources (data2txt)](structured-field-sources.md) - How _retry_structured_field_citations rescues field:value style sources where the answer reorders values; gap=0 retry on _looks_like_structured_source candidates.
