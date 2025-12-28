from __future__ import annotations

from typing import Sequence

from cite_right.core.results import Alignment


class SmithWatermanAligner:
    def __init__(
        self,
        match_score: int = 2,
        mismatch_score: int = -1,
        gap_score: int = -1,
    ) -> None:
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score

    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment:
        if not seq1 or not seq2:
            return Alignment(score=0, token_start=0, token_end=0)

        seq1_list = list(seq1)
        seq2_list = list(seq2)
        rows = len(seq1_list) + 1
        cols = len(seq2_list) + 1

        scores = [[0] * cols for _ in range(rows)]
        directions = [[0] * cols for _ in range(rows)]

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
                    directions[i][j] = 0
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
        best_key: tuple[int, int, int, int, int] | None = None

        for i_end, j_end in max_positions:
            i_start, j_start, matches = _traceback_details(
                i_end,
                j_end,
                directions,
                scores,
                seq1_list,
                seq2_list,
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

        return Alignment(
            score=max_score,
            token_start=best_start,
            token_end=best_end,
            query_start=best_query_start,
            query_end=best_query_end,
            matches=best_matches,
        )


def _choose_direction(
    best: int, score_diag: int, score_up: int, score_left: int
) -> int:
    if best == score_diag:
        return 1
    if best == score_up:
        return 2
    return 3


def _traceback_details(
    i: int,
    j: int,
    directions: list[list[int]],
    scores: list[list[int]],
    seq1: list[int],
    seq2: list[int],
) -> tuple[int, int, int]:
    matches = 0
    while i > 0 and j > 0 and directions[i][j] != 0 and scores[i][j] > 0:
        move = directions[i][j]
        if move == 1:
            if seq1[i - 1] == seq2[j - 1]:
                matches += 1
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1
    return i, j, matches
