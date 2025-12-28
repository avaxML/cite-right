from __future__ import annotations

from typing import Sequence

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
