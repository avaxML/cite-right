from __future__ import annotations

from typing import Sequence

from cite_right.core.results import Alignment


class RustSmithWatermanAligner:
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

        try:
            from cite_right import _core  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - optional extension
            raise RuntimeError(
                "Rust extension is not available. Build it with: uv run maturin develop"
            ) from exc

        self._core = _core

    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment:
        if self.return_match_blocks:
            try:
                (
                    score,
                    token_start,
                    token_end,
                    query_start,
                    query_end,
                    matches,
                    match_blocks,
                ) = self._core.align_pair_blocks_details(
                    seq1,
                    seq2,
                    self.match_score,
                    self.mismatch_score,
                    self.gap_score,
                )
                return Alignment(
                    score=score,
                    token_start=token_start,
                    token_end=token_end,
                    query_start=query_start,
                    query_end=query_end,
                    matches=matches,
                    match_blocks=list(match_blocks),
                )
            except AttributeError:
                pass

        try:
            score, token_start, token_end, query_start, query_end, matches = (
                self._core.align_pair_details(
                    seq1,
                    seq2,
                    self.match_score,
                    self.mismatch_score,
                    self.gap_score,
                )
            )
            return Alignment(
                score=score,
                token_start=token_start,
                token_end=token_end,
                query_start=query_start,
                query_end=query_end,
                matches=matches,
            )
        except AttributeError:
            pass

        score, token_start, token_end = self._core.align_pair(
            seq1,
            seq2,
            self.match_score,
            self.mismatch_score,
            self.gap_score,
        )
        return Alignment(score=score, token_start=token_start, token_end=token_end)

    def align_best(
        self, seq1: Sequence[int], seqs: Sequence[Sequence[int]]
    ) -> tuple[int, int, int, int] | None:
        return self._core.align_best(
            seq1,
            seqs,
            self.match_score,
            self.mismatch_score,
            self.gap_score,
        )
