"""Rust implementation of the Smith-Waterman local aligner."""

from __future__ import annotations

from typing import Sequence

from cite_right.core.results import Alignment


class RustSmithWatermanAligner:
    """Smith-Waterman local aligner powered by a Rust extension module.

    Uses a high-performance Rust implementation for local alignment of token sequences.
    Citation scoring requires detailed traceback outputs from the extension.
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
        if return_match_blocks:
            if not hasattr(self._core, "align_pair_blocks_details") or not hasattr(
                self._core, "align_batch_blocks_details"
            ):
                raise RuntimeError(
                    "Rust extension is missing detailed alignment with match blocks; "
                    "rebuild it or use backend='python'"
                )
        elif not hasattr(self._core, "align_pair_details") or not hasattr(
            self._core, "align_batch_details"
        ):
            raise RuntimeError(
                "Rust extension is missing detailed alignment outputs required for "
                "citation scoring; rebuild it or use backend='python'"
            )

    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment:
        """Align two token sequences and return the best local alignment.

        Args:
            seq1 (Sequence[int]): Query sequence of token IDs.
            seq2 (Sequence[int]): Candidate/document sequence of token IDs.

        Returns:
            Alignment: Alignment object with region indices and alignment statistics
                (including optional match blocks if supported).

        Raises:
            RuntimeError: If the installed Rust extension does not provide the detailed
                traceback outputs required by citation scoring.
        """
        if self.return_match_blocks:
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

    def align_batch(
        self, seq1: Sequence[int], seqs: Sequence[Sequence[int]]
    ) -> list[Alignment]:
        """Align one query against multiple candidates in input order."""
        if self.return_match_blocks:
            return [
                Alignment(
                    score=score,
                    token_start=token_start,
                    token_end=token_end,
                    query_start=query_start,
                    query_end=query_end,
                    matches=matches,
                    match_blocks=list(match_blocks),
                )
                for (
                    score,
                    token_start,
                    token_end,
                    query_start,
                    query_end,
                    matches,
                    match_blocks,
                ) in self._core.align_batch_blocks_details(
                    seq1,
                    seqs,
                    self.match_score,
                    self.mismatch_score,
                    self.gap_score,
                )
            ]

        return [
            Alignment(
                score=score,
                token_start=token_start,
                token_end=token_end,
                query_start=query_start,
                query_end=query_end,
                matches=matches,
            )
            for (
                score,
                token_start,
                token_end,
                query_start,
                query_end,
                matches,
            ) in self._core.align_batch_details(
                seq1,
                seqs,
                self.match_score,
                self.mismatch_score,
                self.gap_score,
            )
        ]

    def align_best(
        self, seq1: Sequence[int], seqs: Sequence[Sequence[int]]
    ) -> tuple[int, int, int, int] | None:
        """Find the best-matching sequence from a list of candidates.

        Args:
            seq1 (Sequence[int]): Query sequence of token IDs.
            seqs (Sequence[Sequence[int]]): List of candidate/document sequences.

        Returns:
            Optional[Tuple[int, int, int, int]]: Tuple with
                (score, index, token_start, token_end) of the highest-scoring alignment,
                or None if no candidates are given.
        """
        return self._core.align_best(
            seq1,
            seqs,
            self.match_score,
            self.mismatch_score,
            self.gap_score,
        )

    def align_best_details(
        self, seq1: Sequence[int], seqs: Sequence[Sequence[int]]
    ) -> tuple[int, int, int, int, int, int, int] | None:
        """Find the best-matching sequence with detailed alignment metadata.

        Args:
            seq1 (Sequence[int]): Query sequence of token IDs.
            seqs (Sequence[Sequence[int]]): List of candidate/document sequences.

        Returns:
            Optional[Tuple[int, int, int, int, int, int, int]]: Tuple with
                (score, index, token_start, token_end, query_start, query_end, matches)
                of the highest-scoring alignment, or None if no candidates are given.
        """
        return self._core.align_best_details(
            seq1,
            seqs,
            self.match_score,
            self.mismatch_score,
            self.gap_score,
        )
