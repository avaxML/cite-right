"""Tests for Python Smith-Waterman aligner implementation."""

from cite_right.core.aligner_py import SmithWatermanAligner


def test_alignment_basic() -> None:
    """Verify basic alignment finds correct subsequence."""
    aligner = SmithWatermanAligner()
    result = aligner.align([1, 2, 3], [0, 1, 2, 3, 4])

    assert result.score == 6, f"Expected score 6, got {result.score}"
    assert result.token_start == 1, f"Expected token_start 1, got {result.token_start}"
    assert result.token_end == 4, f"Expected token_end 4, got {result.token_end}"


def test_alignment_prefers_earlier_start() -> None:
    """Verify alignment prefers earlier start position for equal scores."""
    aligner = SmithWatermanAligner()
    result = aligner.align([1, 2], [1, 2, 1, 2])

    assert result.score == 4, f"Expected score 4, got {result.score}"
    assert result.token_start == 0, (
        f"Expected token_start 0 (earlier position), got {result.token_start}"
    )
    assert result.token_end == 2, f"Expected token_end 2, got {result.token_end}"


def test_alignment_no_match() -> None:
    """Verify alignment returns zero score when no match exists."""
    aligner = SmithWatermanAligner()
    result = aligner.align([1, 2], [3, 4])

    assert result.score == 0, f"Expected score 0 for no match, got {result.score}"
    assert result.token_start == 0, f"Expected token_start 0, got {result.token_start}"
    assert result.token_end == 0, f"Expected token_end 0, got {result.token_end}"


def test_alignment_empty_query() -> None:
    """Verify alignment handles empty query sequence."""
    aligner = SmithWatermanAligner()
    result = aligner.align([], [1, 2, 3])

    assert result.score == 0, "Empty query should have zero score"
    assert result.token_start == 0
    assert result.token_end == 0


def test_alignment_empty_target() -> None:
    """Verify alignment handles empty target sequence."""
    aligner = SmithWatermanAligner()
    result = aligner.align([1, 2, 3], [])

    assert result.score == 0, "Empty target should have zero score"
    assert result.token_start == 0
    assert result.token_end == 0


def test_alignment_exact_match() -> None:
    """Verify alignment finds exact match when sequences are identical."""
    aligner = SmithWatermanAligner()
    seq = [1, 2, 3, 4, 5]
    result = aligner.align(seq, seq)

    assert result.score == len(seq) * 2, f"Expected perfect score, got {result.score}"
    assert result.token_start == 0
    assert result.token_end == len(seq)


def test_alignment_partial_match() -> None:
    """Verify alignment finds partial match within longer sequence."""
    aligner = SmithWatermanAligner()
    result = aligner.align([2, 3], [1, 2, 3, 4])

    assert result.score == 4, f"Expected score 4, got {result.score}"
    assert result.token_start == 1
    assert result.token_end == 3


def test_alignment_single_element_match() -> None:
    """Verify alignment handles single-element match."""
    aligner = SmithWatermanAligner()
    result = aligner.align([5], [1, 2, 5, 3, 4])

    assert result.score == 2, f"Expected score 2, got {result.score}"
    assert result.token_start == 2
    assert result.token_end == 3
