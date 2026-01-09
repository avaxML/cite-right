"""Rust implementation of the Smith-Waterman local aligner."""

from __future__ import annotations

from contextlib import suppress
from typing import Sequence

from cite_right.core.results import Alignment


class RustSmithWatermanAligner:
    """Smith-Waterman local aligner powered by a Rust extension module.

    Uses a high-performance Rust implementation for local alignment of token sequences.
    Methods fall back to simpler Python return values if the detailed interface is
    not available in the installed Rust extension.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_score: int = -1,
        gap_score: int = -1,
        *,
        return_match_blocks: bool = False,
    ) -> None:
        """Initializes the RustSmithWatermanAligner.

        Args:
            match_score (int, optional): Score for exact token matches. Defaults to 2.
            mismatch_score (int, optional): Score for token mismatches. Defaults to -1.
            gap_score (int, optional): Score for gaps (insertions/deletions). Defaults to -1.
            return_match_blocks (bool, optional): If True, output `Alignment.match_blocks`
                to specify runs of exact matches in the aligned tokens. Defaults to False.

        Raises:
            RuntimeError: If the Rust extension could not be imported or is unavailable.
        """
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
        """Align two token sequences and return the best local alignment.

        Args:
            seq1 (Sequence[int]): Query sequence of token IDs.
            seq2 (Sequence[int]): Candidate/document sequence of token IDs.

        Returns:
            Alignment: Alignment object with region indices and alignment statistics
                (including optional match blocks if supported).

        Raises:
            AttributeError: If required alignment methods are not present in the Rust extension.
                In general, this is suppressed; the function will try less-detailed versions.
        """
        if self.return_match_blocks:
            with suppress(AttributeError):
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

        with suppress(AttributeError):
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
        """Find the best-matching sequence from a list of candidates.

        Args:
            seq1 (Sequence[int]): Query sequence of token IDs.
            seqs (Sequence[Sequence[int]]): List of candidate/document sequences.

        Returns:
            Optional[Tuple[int, int, int, int]]: Tuple with
                (index, score, token_start, token_end) of the highest-scoring alignment,
                or None if no candidates are given.
        """
        return self._core.align_best(
            seq1,
            seqs,
            self.match_score,
            self.mismatch_score,
            self.gap_score,
        )
