from __future__ import annotations

from enum import IntEnum
from typing import Sequence

from cite_right.core.results import Alignment


class Direction(IntEnum):
    """Direction constants for Smith-Waterman traceback."""

    STOP = 0
    DIAGONAL = 1
    UP = 2
    LEFT = 3


class SmithWatermanAligner:
    """Smith–Waterman local aligner over token IDs.

    Args:
        match_score: Score for an exact token match.
        mismatch_score: Score for a token mismatch.
        gap_score: Score for a gap (insertion/deletion).
        return_match_blocks: If True, populate `Alignment.match_blocks` with token
            index ranges in `seq2` that correspond to contiguous runs of exact
            matches in the selected alignment.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_score: int = -1,
        gap_score: int = -1,
        *,
        return_match_blocks: bool = False,
    ) -> None:
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score
        self.return_match_blocks = return_match_blocks

    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment:
        """Align two token sequences and return the best local alignment.

        Args:
            seq1: Query token IDs.
            seq2: Candidate token IDs.

        Returns:
            An `Alignment` describing the best local alignment.
        """
        if not seq1 or not seq2:
            return Alignment(score=0, token_start=0, token_end=0)

        seq1_list = list(seq1)
        seq2_list = list(seq2)
        rows = len(seq1_list) + 1
        cols = len(seq2_list) + 1

        scores = [[0] * cols for _ in range(rows)]
        directions = [[Direction.STOP] * cols for _ in range(rows)]

        max_score = 0
        max_positions: list[tuple[int, int]] = []

        for i in range(1, rows):
            for j in range(1, cols):
                match = (
                    self.match_score
                    if seq1_list[i - 1] == seq2_list[j - 1]
                    else self.mismatch_score
                )
                score_diag = scores[i - 1][j - 1] + match
                score_up = scores[i - 1][j] + self.gap_score
                score_left = scores[i][j - 1] + self.gap_score

                best = max(0, score_diag, score_up, score_left)
                if best <= 0:
                    scores[i][j] = 0
                    directions[i][j] = Direction.STOP
                else:
                    scores[i][j] = best
                    directions[i][j] = _choose_direction(
                        best, score_diag, score_up, score_left
                    )

                if scores[i][j] > max_score:
                    max_score = scores[i][j]
                    max_positions = [(i, j)]
                elif scores[i][j] == max_score and scores[i][j] > 0:
                    max_positions.append((i, j))

        if max_score == 0:
            return Alignment(score=0, token_start=0, token_end=0)

        best_start = 0
        best_end = 0
        best_query_start = 0
        best_query_end = 0
        best_matches = 0
        best_match_blocks: list[tuple[int, int]] = []
        best_key: tuple[int, int, int, int, int] | None = None

        for i_end, j_end in max_positions:
            i_start, j_start, matches, match_blocks = _traceback_details(
                i_end,
                j_end,
                directions,
                scores,
                seq1_list,
                seq2_list,
                return_match_blocks=self.return_match_blocks,
            )
            span_len = j_end - j_start
            key = (j_start, -span_len, i_start, j_end, i_end)
            if best_key is None or key < best_key:
                best_key = key
                best_start = j_start
                best_end = j_end
                best_query_start = i_start
                best_query_end = i_end
                best_matches = matches
                best_match_blocks = match_blocks

        return Alignment(
            score=max_score,
            token_start=best_start,
            token_end=best_end,
            query_start=best_query_start,
            query_end=best_query_end,
            matches=best_matches,
            match_blocks=best_match_blocks,
        )


def _choose_direction(
    best: int, score_diag: int, score_up: int, score_left: int
) -> Direction:
    if best == score_diag:
        return Direction.DIAGONAL
    if best == score_up:
        return Direction.UP
    return Direction.LEFT


def _traceback_details(
    i: int,
    j: int,
    directions: list[list[Direction]],
    scores: list[list[int]],
    seq1: list[int],
    seq2: list[int],
    *,
    return_match_blocks: bool,
) -> tuple[int, int, int, list[tuple[int, int]]]:
    matches = 0
    match_positions: list[int] = []
    while i > 0 and j > 0 and directions[i][j] != Direction.STOP and scores[i][j] > 0:
        match directions[i][j]:
            case Direction.DIAGONAL:
                i -= 1
                j -= 1
                if seq1[i] == seq2[j]:
                    matches += 1
                    if return_match_blocks:
                        match_positions.append(j)
            case Direction.UP:
                i -= 1
            case Direction.LEFT:
                j -= 1

    if not return_match_blocks or not match_positions:
        return i, j, matches, []

    match_positions.reverse()
    blocks: list[tuple[int, int]] = []
    start = match_positions[0]
    prev = start
    for pos in match_positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        blocks.append((start, prev + 1))
        start = pos
        prev = pos
    blocks.append((start, prev + 1))
    return i, j, matches, blocks
