# Files

- [Citation alignment pipeline](pipeline-overview.md) - End-to-end map of how an answer span becomes a Citation or RetrievalSupport entry, from source normalization through candidate selection, alignment, and final ranking.
- [Result data model](result-types.md) - Reference Pydantic models returned by the cite-right API: SpanCitations, Citation, EvidenceSpan, RetrievalSupport, AnswerSpan, SourceDocument, SourceChunk, Alignment, TokenizedText, and Segment.
- [Candidate retrieval pipeline](retrieval-pipeline.md) - How inverted-index seeding, lexical IDF prefilter, embedding top-k, ranking, and limits select candidates for Smith-Waterman alignment.
- [Smith-Waterman Aligners](smith-waterman.md) - Pure-Python and Rust implementations of Smith-Waterman local alignment for token sequences, used to localize answer evidence in candidate documents.
