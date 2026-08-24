from __future__ import annotations

from typing import Dict, Sequence

class InvertedIndex:
    """Inverted index for token-to-candidate mapping (Rust object)."""

    def query(self, query_tokens: Sequence[int], max_candidates: int) -> list[int]:
        """Query the inverted index for candidate indices.

        Args:
            query_tokens: Token IDs to search for
            max_candidates: Maximum number of candidates to return

        Returns:
            List of candidate indices
        """
        ...

    def get_posting_count(self, token_id: int) -> int:
        """Get the number of postings for a token.

        Args:
            token_id: Token ID to query

        Returns:
            Number of postings for this token
        """
        ...

class PyEvidenceSpan:
    """Evidence span returned from Rust."""

    char_start: int
    char_end: int
    evidence: str

class PyCitation:
    """Citation result from Rust."""

    score: float
    source_id: str
    source_index: int
    candidate_index: int
    char_start: int
    char_end: int
    evidence: str
    evidence_spans: list[PyEvidenceSpan]
    components: Dict[str, float]

class PyRetrievalSupport:
    """Retrieval support from Rust."""

    retrieval_score: float
    source_id: str
    source_index: int
    candidate_index: int
    passage_char_start: int
    passage_char_end: int
    passage_text: str
    embedding_score: float
    lexical_score: float

class PyCitationResult:
    """Citation building result from Rust."""

    citations: list[PyCitation]
    supports: list[PyRetrievalSupport]
    num_alignments: int

class PreparedCorpus:
    """Prepared corpus kept in Rust (opaque to Python)."""

    def num_candidates(self) -> int:
        """Get the number of candidates."""
        ...

    def get_candidate_tokens(self, candidate_indices: Sequence[int]) -> list[list[int]]:
        """Get token_ids for specific candidate indices (for alignment)."""
        ...

    def get_candidate_metadata(
        self, candidate_indices: Sequence[int]
    ) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
        """Get candidate metadata (source_index, passage_start, passage_end, token_spans)."""
        ...

    def query_index(
        self, query_tokens: Sequence[int], max_candidates: int
    ) -> list[int]:
        """Query inverted index for seed candidates."""
        ...

    def get_idf(self) -> list[tuple[int, float]]:
        """Get IDF weights."""
        ...

    def get_vocab(self) -> list[tuple[str, int]]:
        """Get vocabulary."""
        ...

    def get_source_text(self, source_index: int) -> str | None:
        """Get source text by index."""
        ...

    def get_source_candidates(self, source_index: int) -> list[tuple[int, int, int]]:
        """Get all candidates for a specific source (global_idx, passage_start, passage_end)."""
        ...

    def get_all_candidate_info(
        self,
    ) -> list[tuple[int, int, int, int]]:
        """Get minimal info for all candidates (global_idx, source_idx, passage_start, passage_end)."""
        ...

    def build_citations(
        self,
        answer_tokens: Sequence[int],
        candidate_indices: Sequence[int],
        lexical_scores: Sequence[float],
        embed_scores: Sequence[float],
        source_id_map: Dict[int, str],
        base_offset_map: Dict[int, int],
        config_tuple: tuple[int, float, float, bool, int, float, float, float, float, float],
        multi_span_config: tuple[bool, int, int],
        match_score: int,
        mismatch_score: int,
        gap_score: int,
    ) -> PyCitationResult:
        """Build citations directly from PreparedCorpus without Python marshalling.

        This eliminates the overhead of copying full source texts to Python and back.
        """
        ...

def rust_tokenize_and_prepare(
    source_texts: Sequence[str], window_size: int, stride: int
) -> PreparedCorpus:
    """Tokenize and prepare sources, returning a Rust-side corpus object.

    Args:
        source_texts: List of source text strings
        window_size: Window size in sentences
        stride: Stride size in sentences

    Returns:
        PreparedCorpus object (opaque Rust object)
    """
    ...

def align_pair(
    seq1: Sequence[int],
    seq2: Sequence[int],
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> tuple[int, int, int]: ...
def align_pair_details(
    seq1: Sequence[int],
    seq2: Sequence[int],
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> tuple[int, int, int, int, int, int]: ...
def align_pair_blocks_details(
    seq1: Sequence[int],
    seq2: Sequence[int],
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> tuple[int, int, int, int, int, int, list[tuple[int, int]]]: ...
def align_best(
    seq1: Sequence[int],
    seqs: Sequence[Sequence[int]],
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> tuple[int, int, int, int] | None: ...
def align_best_details(
    seq1: Sequence[int],
    seqs: Sequence[Sequence[int]],
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> tuple[int, int, int, int, int, int, int] | None: ...
def align_topk_details(
    seq1: Sequence[int],
    seqs: Sequence[Sequence[int]],
    top_k: int = ...,
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> list[tuple[int, int, int, int, int, int, int]]: ...
def align_batch_details(
    seq1: Sequence[int],
    seqs: Sequence[Sequence[int]],
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> list[tuple[int, int, int, int, int, int]]: ...
def align_batch_blocks_details(
    seq1: Sequence[int],
    seqs: Sequence[Sequence[int]],
    match_score: int = ...,
    mismatch_score: int = ...,
    gap_score: int = ...,
) -> list[tuple[int, int, int, int, int, int, list[tuple[int, int]]]]: ...
