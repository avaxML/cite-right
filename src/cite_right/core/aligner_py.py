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

        # Keep track of the best endpoint's characteristics for comparison
        max_matches = 0
        max_token_start = 0
        max_query_start = 0
        max_best_i = 0
        max_best_j = 0

        match_score = self.match_score
        mismatch_score = self.mismatch_score
        gap_score = self.gap_score

        for i in range(1, rows):
            row_scores = scores[i]
            prev_row_scores = scores[i - 1]
            row_directions = directions[i]

            row_match_counts = match_counts[i]
            prev_row_match_counts = match_counts[i - 1]

            row_query_starts = query_starts[i]
            prev_row_query_starts = query_starts[i - 1]

            row_token_starts = token_starts[i]
            prev_row_token_starts = token_starts[i - 1]

            val_seq1_prev = seq1[i - 1]

            for j in range(1, cols):
                is_match = val_seq1_prev == seq2[j - 1]
                diag_delta = match_score if is_match else mismatch_score

                best_score = 0
                best_dir = Direction.STOP
                best_matches = 0
                best_query_start = 0
                best_token_start = 0
                best_priority = 0

                # 1. DIAGONAL
                score_diag = prev_row_scores[j - 1] + diag_delta
                if score_diag > 0:
                    if prev_row_scores[j - 1] > 0:
                        diag_matches = prev_row_match_counts[j - 1] + int(is_match)
                        diag_query_start = prev_row_query_starts[j - 1]
                        diag_token_start = prev_row_token_starts[j - 1]
                    else:
                        diag_matches = int(is_match)
                        diag_query_start = i - 1
                        diag_token_start = j - 1

                    best_score = score_diag
                    best_dir = Direction.DIAGONAL
                    best_matches = diag_matches
                    best_query_start = diag_query_start
                    best_token_start = diag_token_start
                    best_priority = 0

                # 2. UP
                score_up = prev_row_scores[j] + gap_score
                if score_up > 0:
                    if prev_row_scores[j] > 0:
                        up_matches = prev_row_match_counts[j]
                        up_query_start = prev_row_query_starts[j]
                        up_token_start = prev_row_token_starts[j]
                    else:
                        up_matches = 0
                        up_query_start = i - 1
                        up_token_start = j

                    is_better = False
                    if score_up > best_score:
                        is_better = True
                    elif score_up == best_score and score_up > 0:
                        if up_matches != best_matches:
                            is_better = up_matches > best_matches
                        elif up_token_start != best_token_start:
                            is_better = up_token_start < best_token_start
                        elif up_query_start != best_query_start:
                            is_better = up_query_start < best_query_start
                        else:
                            is_better = 1 < best_priority

                    if is_better:
                        best_score = score_up
                        best_dir = Direction.UP
                        best_matches = up_matches
                        best_query_start = up_query_start
                        best_token_start = up_token_start
                        best_priority = 1

                # 3. LEFT
                score_left = row_scores[j - 1] + gap_score
                if score_left > 0:
                    if row_scores[j - 1] > 0:
                        left_matches = row_match_counts[j - 1]
                        left_query_start = row_query_starts[j - 1]
                        left_token_start = row_token_starts[j - 1]
                    else:
                        left_matches = 0
                        left_query_start = i
                        left_token_start = j - 1

                    is_better = False
                    if score_left > best_score:
                        is_better = True
                    elif score_left == best_score and score_left > 0:
                        if left_matches != best_matches:
                            is_better = left_matches > best_matches
                        elif left_token_start != best_token_start:
                            is_better = left_token_start < best_token_start
                        elif left_query_start != best_query_start:
                            is_better = left_query_start < best_query_start
                        else:
                            is_better = 2 < best_priority

                    if is_better:
                        best_score = score_left
                        best_dir = Direction.LEFT
                        best_matches = left_matches
                        best_query_start = left_query_start
                        best_token_start = left_token_start
                        best_priority = 2

                # Save computed values
                row_scores[j] = best_score
                row_directions[j] = best_dir
                row_match_counts[j] = best_matches
                row_query_starts[j] = best_query_start
                row_token_starts[j] = best_token_start

                # Update best-scoring endpoint
                if best_score > max_score:
                    max_score = best_score
                    best_end = (i, j)
                    max_matches = best_matches
                    max_token_start = best_token_start
                    max_query_start = best_query_start
                    max_best_i = i
                    max_best_j = j
                elif best_score == max_score and best_score > 0:
                    is_better_end = False
                    if best_matches != max_matches:
                        is_better_end = best_matches > max_matches
                    elif best_token_start != max_token_start:
                        is_better_end = best_token_start < max_token_start
                    else:
                        cand_span = j - best_token_start
                        best_span = max_best_j - max_token_start
                        if cand_span != best_span:
                            is_better_end = cand_span > best_span
                        elif best_query_start != max_query_start:
                            is_better_end = best_query_start < max_query_start
                        elif j != max_best_j:
                            is_better_end = j < max_best_j
                        else:
                            is_better_end = i < max_best_i

                    if is_better_end:
                        best_end = (i, j)
                        max_matches = best_matches
                        max_token_start = best_token_start
                        max_query_start = best_query_start
                        max_best_i = i
                        max_best_j = j

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

        # Keep track of the best endpoint's characteristics for comparison
        max_matches = 0
        max_token_start = 0
        max_query_start = 0
        max_best_i = 0
        max_best_j = 0

        prev_match_counts = [0] * cols
        prev_query_starts = [0] * cols
        prev_token_starts = [0] * cols

        match_score = self.match_score
        mismatch_score = self.mismatch_score
        gap_score = self.gap_score

        for i in range(1, rows):
            row_scores = scores[i]
            prev_row_scores = scores[i - 1]
            row_directions = directions[i]

            current_match_counts = [0] * cols
            current_query_starts = [0] * cols
            current_token_starts = [0] * cols

            val_seq1_prev = seq1[i - 1]

            for j in range(1, cols):
                is_match = val_seq1_prev == seq2[j - 1]
                diag_delta = match_score if is_match else mismatch_score

                best_score = 0
                best_dir = Direction.STOP
                best_matches = 0
                best_query_start = 0
                best_token_start = 0
                best_priority = 0

                # 1. DIAGONAL
                score_diag = prev_row_scores[j - 1] + diag_delta
                if score_diag > 0:
                    if prev_row_scores[j - 1] > 0:
                        diag_matches = prev_match_counts[j - 1] + int(is_match)
                        diag_query_start = prev_query_starts[j - 1]
                        diag_token_start = prev_token_starts[j - 1]
                    else:
                        diag_matches = int(is_match)
                        diag_query_start = i - 1
                        diag_token_start = j - 1

                    best_score = score_diag
                    best_dir = Direction.DIAGONAL
                    best_matches = diag_matches
                    best_query_start = diag_query_start
                    best_token_start = diag_token_start
                    best_priority = 0

                # 2. UP
                score_up = prev_row_scores[j] + gap_score
                if score_up > 0:
                    if prev_row_scores[j] > 0:
                        up_matches = prev_match_counts[j]
                        up_query_start = prev_query_starts[j]
                        up_token_start = prev_token_starts[j]
                    else:
                        up_matches = 0
                        up_query_start = i - 1
                        up_token_start = j

                    is_better = False
                    if score_up > best_score:
                        is_better = True
                    elif score_up == best_score and score_up > 0:
                        if up_matches != best_matches:
                            is_better = up_matches > best_matches
                        elif up_token_start != best_token_start:
                            is_better = up_token_start < best_token_start
                        elif up_query_start != best_query_start:
                            is_better = up_query_start < best_query_start
                        else:
                            is_better = 1 < best_priority

                    if is_better:
                        best_score = score_up
                        best_dir = Direction.UP
                        best_matches = up_matches
                        best_query_start = up_query_start
                        best_token_start = up_token_start
                        best_priority = 1

                # 3. LEFT
                score_left = row_scores[j - 1] + gap_score
                if score_left > 0:
                    if row_scores[j - 1] > 0:
                        left_matches = current_match_counts[j - 1]
                        left_query_start = current_query_starts[j - 1]
                        left_token_start = current_token_starts[j - 1]
                    else:
                        left_matches = 0
                        left_query_start = i
                        left_token_start = j - 1

                    is_better = False
                    if score_left > best_score:
                        is_better = True
                    elif score_left == best_score and score_left > 0:
                        if left_matches != best_matches:
                            is_better = left_matches > best_matches
                        elif left_token_start != best_token_start:
                            is_better = left_token_start < best_token_start
                        elif left_query_start != best_query_start:
                            is_better = left_query_start < best_query_start
                        else:
                            is_better = 2 < best_priority

                    if is_better:
                        best_score = score_left
                        best_dir = Direction.LEFT
                        best_matches = left_matches
                        best_query_start = left_query_start
                        best_token_start = left_token_start
                        best_priority = 2

                # Save computed values
                row_scores[j] = best_score
                row_directions[j] = best_dir
                current_match_counts[j] = best_matches
                current_query_starts[j] = best_query_start
                current_token_starts[j] = best_token_start

                # Update best-scoring endpoint
                if best_score > max_score:
                    max_score = best_score
                    best_end = (i, j)
                    max_matches = best_matches
                    max_token_start = best_token_start
                    max_query_start = best_query_start
                    max_best_i = i
                    max_best_j = j
                elif best_score == max_score and best_score > 0:
                    is_better_end = False
                    if best_matches != max_matches:
                        is_better_end = best_matches > max_matches
                    elif best_token_start != max_token_start:
                        is_better_end = best_token_start < max_token_start
                    else:
                        cand_span = j - best_token_start
                        best_span = max_best_j - max_token_start
                        if cand_span != best_span:
                            is_better_end = cand_span > best_span
                        elif best_query_start != max_query_start:
                            is_better_end = best_query_start < max_query_start
                        elif j != max_best_j:
                            is_better_end = j < max_best_j
                        else:
                            is_better_end = i < max_best_i

                    if is_better_end:
                        best_end = (i, j)
                        max_matches = best_matches
                        max_token_start = best_token_start
                        max_query_start = best_query_start
                        max_best_i = i
                        max_best_j = j

            prev_match_counts = current_match_counts
            prev_query_starts = current_query_starts
            prev_token_starts = current_token_starts

        return scores, directions, max_score, best_end


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
