from __future__ import annotations

from typing import Sequence

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

def rust_tokenize_and_prepare(
    source_texts: Sequence[str],
    window_size: int,
    stride: int,
) -> tuple[
    list[list[tuple[int, int, list[int], list[tuple[int, int]]]]],
    list[tuple[int, float]],
    list[tuple[str, int]],
    InvertedIndex,
]:
    """Tokenize and prepare sources with inverted index.

    Args:
        source_texts: List of source text strings
        window_size: Window size in sentences
        stride: Stride size in sentences

    Returns:
        Tuple of (source_candidates, idf_vec, vocab_vec, inverted_index)
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
