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
        """Align two token sequences and return the best local alignment."""
        if not seq1 or not seq2:
            return Alignment(score=0, token_start=0, token_end=0)

        seq1_list = list(seq1)
        seq2_list = list(seq2)

        scores, directions, max_score, max_positions = self._fill_matrix(
            seq1_list, seq2_list
        )

        if max_score == 0:
            return Alignment(score=0, token_start=0, token_end=0)

        return self._select_best_alignment(
            max_score, max_positions, directions, scores, seq1_list, seq2_list
        )

    def _fill_matrix(
        self, seq1: list[int], seq2: list[int]
    ) -> tuple[list[list[int]], list[list[Direction]], int, list[tuple[int, int]]]:
        """Fill the scoring matrix and track maximum positions."""
        rows = len(seq1) + 1
        cols = len(seq2) + 1

        scores = [[0] * cols for _ in range(rows)]
        directions = [[Direction.STOP] * cols for _ in range(rows)]
        max_score = 0
        max_positions: list[tuple[int, int]] = []

        for i in range(1, rows):
            for j in range(1, cols):
                cell_score, direction = self._compute_cell(i, j, seq1, seq2, scores)
                scores[i][j] = cell_score
                directions[i][j] = direction

                if cell_score > max_score:
                    max_score = cell_score
                    max_positions = [(i, j)]
                elif cell_score == max_score and cell_score > 0:
                    max_positions.append((i, j))

        return scores, directions, max_score, max_positions

    def _compute_cell(
        self, i: int, j: int, seq1: list[int], seq2: list[int], scores: list[list[int]]
    ) -> tuple[int, Direction]:
        """Compute score and direction for a single matrix cell."""
        match = self.match_score if seq1[i - 1] == seq2[j - 1] else self.mismatch_score
        score_diag = scores[i - 1][j - 1] + match
        score_up = scores[i - 1][j] + self.gap_score
        score_left = scores[i][j - 1] + self.gap_score

        best = max(0, score_diag, score_up, score_left)
        if best <= 0:
            return 0, Direction.STOP
        return best, _choose_direction(best, score_diag, score_up, score_left)

    def _select_best_alignment(
        self,
        max_score: int,
        max_positions: list[tuple[int, int]],
        directions: list[list[Direction]],
        scores: list[list[int]],
        seq1: list[int],
        seq2: list[int],
    ) -> Alignment:
        """Select the best alignment from all maximum positions."""
        best_key: tuple[int, int, int, int, int] | None = None
        best_result: tuple[int, int, int, int, int, list[tuple[int, int]]] | None = None

        for i_end, j_end in max_positions:
            i_start, j_start, matches, match_blocks = _traceback_details(
                i_end, j_end, directions, scores, seq1, seq2,
                return_match_blocks=self.return_match_blocks,
            )
            span_len = j_end - j_start
            key = (j_start, -span_len, i_start, j_end, i_end)
            if best_key is None or key < best_key:
                best_key = key
                best_result = (j_start, j_end, i_start, i_end, matches, match_blocks)

        assert best_result is not None
        return Alignment(
            score=max_score,
            token_start=best_result[0],
            token_end=best_result[1],
            query_start=best_result[2],
            query_end=best_result[3],
            matches=best_result[4],
            match_blocks=best_result[5],
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
    """Trace back through the alignment matrix to find match details."""
    matches = 0
    match_positions: list[int] = []

    while i > 0 and j > 0 and directions[i][j] != Direction.STOP and scores[i][j] > 0:
        i, j, is_match = _step_traceback(i, j, directions, seq1, seq2)
        if is_match:
            matches += 1
            if return_match_blocks:
                match_positions.append(j)

    blocks = _consolidate_match_blocks(match_positions) if return_match_blocks else []
    return i, j, matches, blocks


def _step_traceback(
    i: int,
    j: int,
    directions: list[list[Direction]],
    seq1: list[int],
    seq2: list[int],
) -> tuple[int, int, bool]:
    """Take one step in the traceback, returning new position and whether it was a match."""
    match directions[i][j]:
        case Direction.DIAGONAL:
            i -= 1
            j -= 1
            return i, j, seq1[i] == seq2[j]
        case Direction.UP:
            return i - 1, j, False
        case Direction.LEFT:
            return i, j - 1, False
    return i, j, False  # pragma: no cover


def _consolidate_match_blocks(match_positions: list[int]) -> list[tuple[int, int]]:
    """Convert match positions into contiguous blocks."""
    if not match_positions:
        return []

    match_positions.reverse()
    blocks: list[tuple[int, int]] = []
    start = match_positions[0]
    prev = start

    for pos in match_positions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            blocks.append((start, prev + 1))
            start = pos
            prev = pos

    blocks.append((start, prev + 1))
    return blocks
