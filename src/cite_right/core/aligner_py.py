"""Python implementation of the Smith-Waterman local aligner."""

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

        if self.return_match_blocks:
            scores, directions, max_score, best_end = self._fill_matrix(
                seq1_list, seq2_list
            )
        else:
            scores, directions, max_score, best_end = self._fill_matrix_reduced_state(
                seq1_list, seq2_list
            )

        if max_score == 0:
            return Alignment(score=0, token_start=0, token_end=0)

        assert best_end is not None
        return _build_alignment(
            max_score,
            best_end,
            directions,
            scores,
            seq1_list,
            seq2_list,
            return_match_blocks=self.return_match_blocks,
        )

    def align_batch(
        self, seq1: Sequence[int], seqs: Sequence[Sequence[int]]
    ) -> list[Alignment]:
        """Align one query against multiple candidates in input order."""
        return [self.align(seq1, seq2) for seq2 in seqs]

    def _fill_matrix(
        self, seq1: list[int], seq2: list[int]
    ) -> tuple[list[list[int]], list[list[Direction]], int, tuple[int, int] | None]:
        """Fill the scoring matrix and track the best-scoring endpoint."""
        rows = len(seq1) + 1
        cols = len(seq2) + 1

        scores = [[0] * cols for _ in range(rows)]
        directions = [[Direction.STOP] * cols for _ in range(rows)]
        match_counts = [[0] * cols for _ in range(rows)]
        query_starts = [[0] * cols for _ in range(rows)]
        token_starts = [[0] * cols for _ in range(rows)]
        max_score = 0
        best_end: tuple[int, int] | None = None
        best_key: tuple[int, int, int, int, int, int] | None = None

        for i in range(1, rows):
            for j in range(1, cols):
                cell_score, direction, matches, query_start, token_start = (
                    self._compute_cell(
                        i,
                        j,
                        seq1,
                        seq2,
                        scores,
                        match_counts,
                        query_starts,
                        token_starts,
                    )
                )
                scores[i][j] = cell_score
                directions[i][j] = direction
                match_counts[i][j] = matches
                query_starts[i][j] = query_start
                token_starts[i][j] = token_start

                if cell_score > max_score:
                    max_score = cell_score
                    best_end = (i, j)
                    best_key = _alignment_key(query_start, token_start, i, j, matches)
                elif cell_score == max_score and cell_score > 0:
                    candidate_key = _alignment_key(
                        query_start, token_start, i, j, matches
                    )
                    if best_key is None or candidate_key < best_key:
                        best_end = (i, j)
                        best_key = candidate_key

        return scores, directions, max_score, best_end

    def _fill_matrix_reduced_state(
        self, seq1: list[int], seq2: list[int]
    ) -> tuple[list[list[int]], list[list[Direction]], int, tuple[int, int] | None]:
        """Fill the default-path matrices with rolling tie-break metadata."""
        rows = len(seq1) + 1
        cols = len(seq2) + 1

        scores = [[0] * cols for _ in range(rows)]
        directions = [[Direction.STOP] * cols for _ in range(rows)]
        max_score = 0
        best_end: tuple[int, int] | None = None
        best_key: tuple[int, int, int, int, int, int] | None = None
        prev_match_counts = [0] * cols
        prev_query_starts = [0] * cols
        prev_token_starts = [0] * cols

        for i in range(1, rows):
            current_match_counts = [0] * cols
            current_query_starts = [0] * cols
            current_token_starts = [0] * cols
            for j in range(1, cols):
                score, direction, matches, query_start, token_start = (
                    self._compute_cell_reduced_state(
                        i,
                        j,
                        seq1,
                        seq2,
                        scores,
                        prev_match_counts,
                        current_match_counts,
                        prev_query_starts,
                        current_query_starts,
                        prev_token_starts,
                        current_token_starts,
                    )
                )
                scores[i][j] = score
                directions[i][j] = direction
                current_match_counts[j] = matches
                current_query_starts[j] = query_start
                current_token_starts[j] = token_start

                if score > max_score:
                    max_score = score
                    best_end = (i, j)
                    best_key = _alignment_key(query_start, token_start, i, j, matches)
                elif score == max_score and score > 0:
                    candidate_key = _alignment_key(
                        query_start, token_start, i, j, matches
                    )
                    if best_key is None or candidate_key < best_key:
                        best_end = (i, j)
                        best_key = candidate_key

            prev_match_counts = current_match_counts
            prev_query_starts = current_query_starts
            prev_token_starts = current_token_starts

        return scores, directions, max_score, best_end

    def _compute_cell(
        self,
        i: int,
        j: int,
        seq1: list[int],
        seq2: list[int],
        scores: list[list[int]],
        match_counts: list[list[int]],
        query_starts: list[list[int]],
        token_starts: list[list[int]],
    ) -> tuple[int, Direction, int, int, int]:
        """Compute score and direction for a single matrix cell."""
        is_match = seq1[i - 1] == seq2[j - 1]
        diag_delta = self.match_score if is_match else self.mismatch_score
        best: tuple[int, Direction, int, int, int, int] | None = None

        score_diag = scores[i - 1][j - 1] + diag_delta
        if score_diag > 0:
            if scores[i - 1][j - 1] > 0:
                diag_matches = match_counts[i - 1][j - 1] + int(is_match)
                diag_query_start = query_starts[i - 1][j - 1]
                diag_token_start = token_starts[i - 1][j - 1]
            else:
                diag_matches = int(is_match)
                diag_query_start = i - 1
                diag_token_start = j - 1
            best = _pick_better_cell_candidate(
                best,
                (
                    score_diag,
                    Direction.DIAGONAL,
                    diag_matches,
                    diag_query_start,
                    diag_token_start,
                    0,
                ),
            )

        score_up = scores[i - 1][j] + self.gap_score
        if score_up > 0:
            if scores[i - 1][j] > 0:
                up_matches = match_counts[i - 1][j]
                up_query_start = query_starts[i - 1][j]
                up_token_start = token_starts[i - 1][j]
            else:
                up_matches = 0
                up_query_start = i - 1
                up_token_start = j
            best = _pick_better_cell_candidate(
                best,
                (
                    score_up,
                    Direction.UP,
                    up_matches,
                    up_query_start,
                    up_token_start,
                    1,
                ),
            )

        score_left = scores[i][j - 1] + self.gap_score
        if score_left > 0:
            if scores[i][j - 1] > 0:
                left_matches = match_counts[i][j - 1]
                left_query_start = query_starts[i][j - 1]
                left_token_start = token_starts[i][j - 1]
            else:
                left_matches = 0
                left_query_start = i
                left_token_start = j - 1
            best = _pick_better_cell_candidate(
                best,
                (
                    score_left,
                    Direction.LEFT,
                    left_matches,
                    left_query_start,
                    left_token_start,
                    2,
                ),
            )

        if best is None:
            return 0, Direction.STOP, 0, 0, 0
        return best[0], best[1], best[2], best[3], best[4]

    def _compute_cell_reduced_state(
        self,
        i: int,
        j: int,
        seq1: list[int],
        seq2: list[int],
        scores: list[list[int]],
        prev_match_counts: list[int],
        current_match_counts: list[int],
        prev_query_starts: list[int],
        current_query_starts: list[int],
        prev_token_starts: list[int],
        current_token_starts: list[int],
    ) -> tuple[int, Direction, int, int, int]:
        """Compute score and direction for the reduced-state default path."""
        is_match = seq1[i - 1] == seq2[j - 1]
        diag_delta = self.match_score if is_match else self.mismatch_score
        best: tuple[int, Direction, int, int, int, int] | None = None

        score_diag = scores[i - 1][j - 1] + diag_delta
        if score_diag > 0:
            if scores[i - 1][j - 1] > 0:
                diag_matches = prev_match_counts[j - 1] + int(is_match)
                diag_query_start = prev_query_starts[j - 1]
                diag_token_start = prev_token_starts[j - 1]
            else:
                diag_matches = int(is_match)
                diag_query_start = i - 1
                diag_token_start = j - 1
            best = _pick_better_cell_candidate(
                best,
                (
                    score_diag,
                    Direction.DIAGONAL,
                    diag_matches,
                    diag_query_start,
                    diag_token_start,
                    0,
                ),
            )

        score_up = scores[i - 1][j] + self.gap_score
        if score_up > 0:
            if scores[i - 1][j] > 0:
                up_matches = prev_match_counts[j]
                up_query_start = prev_query_starts[j]
                up_token_start = prev_token_starts[j]
            else:
                up_matches = 0
                up_query_start = i - 1
                up_token_start = j
            best = _pick_better_cell_candidate(
                best,
                (
                    score_up,
                    Direction.UP,
                    up_matches,
                    up_query_start,
                    up_token_start,
                    1,
                ),
            )

        score_left = scores[i][j - 1] + self.gap_score
        if score_left > 0:
            if scores[i][j - 1] > 0:
                left_matches = current_match_counts[j - 1]
                left_query_start = current_query_starts[j - 1]
                left_token_start = current_token_starts[j - 1]
            else:
                left_matches = 0
                left_query_start = i
                left_token_start = j - 1
            best = _pick_better_cell_candidate(
                best,
                (
                    score_left,
                    Direction.LEFT,
                    left_matches,
                    left_query_start,
                    left_token_start,
                    2,
                ),
            )

        if best is None:
            return 0, Direction.STOP, 0, 0, 0
        return best[0], best[1], best[2], best[3], best[4]


def _pick_better_cell_candidate(
    current: tuple[int, Direction, int, int, int, int] | None,
    candidate: tuple[int, Direction, int, int, int, int],
) -> tuple[int, Direction, int, int, int, int]:
    """Pick the better per-cell candidate without container churn."""
    if current is None:
        return candidate
    if candidate[0] != current[0]:
        return candidate if candidate[0] > current[0] else current
    candidate_key = (-candidate[2], candidate[4], candidate[3], candidate[5])
    current_key = (-current[2], current[4], current[3], current[5])
    return candidate if candidate_key < current_key else current


def _build_alignment(
    max_score: int,
    best_end: tuple[int, int],
    directions: list[list[Direction]],
    scores: list[list[int]],
    seq1: list[int],
    seq2: list[int],
    *,
    return_match_blocks: bool,
) -> Alignment:
    """Build the selected alignment from the tracked best endpoint."""
    i_end, j_end = best_end
    i_start, j_start, matches, match_blocks = _traceback_details(
        i_end,
        j_end,
        directions,
        scores,
        seq1,
        seq2,
        return_match_blocks=return_match_blocks,
    )
    return Alignment(
        score=max_score,
        token_start=j_start,
        token_end=j_end,
        query_start=i_start,
        query_end=i_end,
        matches=matches,
        match_blocks=match_blocks,
    )


def _alignment_key(
    i_start: int, j_start: int, i_end: int, j_end: int, matches: int
) -> tuple[int, int, int, int, int, int]:
    span_len = j_end - j_start
    return (-matches, j_start, -span_len, i_start, j_end, i_end)


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
