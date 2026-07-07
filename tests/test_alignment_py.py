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


def test_alignment_prefers_more_matches_across_equal_score_endpoints() -> None:
    """Verify equal-score endpoints prefer the traceback with more exact matches."""
    aligner = SmithWatermanAligner(return_match_blocks=True)

    result = aligner.align([0, 1, 0], [0, 1, 1, 1, 0])

    assert result.score == 4
    assert result.token_start == 0
    assert result.token_end == 5
    assert result.query_start == 0
    assert result.query_end == 3
    assert result.matches == 3
    assert result.match_blocks == [(0, 1), (2, 3), (4, 5)]


def test_alignment_prefers_more_matches_within_single_optimal_endpoint() -> None:
    """Verify traceback explores equal-score predecessors at the same end cell."""
    aligner = SmithWatermanAligner(return_match_blocks=True)

    result = aligner.align([0, 1, 0], [0, 2, 1, 1, 0])

    assert result.score == 4
    assert result.token_start == 0
    assert result.token_end == 5
    assert result.query_start == 0
    assert result.query_end == 3
    assert result.matches == 3
    assert result.match_blocks == [(0, 1), (2, 3), (4, 5)]


def test_fill_matrix_tracks_single_best_endpoint() -> None:
    """Verify fill_matrix returns one winning endpoint instead of collecting all max cells."""
    aligner = SmithWatermanAligner()

    _, _, max_score, best_end = aligner._fill_matrix([1, 2], [1, 2, 9, 1, 2, 0])

    assert max_score == 4
    assert best_end == (2, 2)


def test_align_batch_preserves_single_alignment_results_in_order() -> None:
    """Verify batch alignment matches per-candidate results without reordering."""
    aligner = SmithWatermanAligner(return_match_blocks=True)
    query = [1, 2, 3]
    candidates = [[0, 1, 2, 3, 4], [1, 2, 9, 3], [8, 9, 10]]

    expected = [aligner.align(query, candidate) for candidate in candidates]

    assert aligner.align_batch(query, candidates) == expected


def test_reduced_state_fill_tracks_best_endpoint_for_default_path() -> None:
    """Verify the default path exposes a reduced-state fill result."""
    aligner = SmithWatermanAligner()

    scores, directions, max_score, best_end = aligner._fill_matrix_reduced_state(
        [1, 2], [1, 2, 9, 1, 2, 0]
    )

    assert max_score == 4
    assert best_end == (2, 2)
    assert len(scores) == 3
    assert len(scores[0]) == 7
    assert len(directions) == 3
    assert len(directions[0]) == 7


def test_default_path_matches_detailed_path_without_match_blocks() -> None:
    """Verify reduced-state and detailed paths pick the same single-span alignment."""
    simple = SmithWatermanAligner()
    detailed = SmithWatermanAligner(return_match_blocks=True)

    seq1 = [0, 1, 0]
    seq2 = [0, 1, 1, 1, 0]

    simple_result = simple.align(seq1, seq2)
    detailed_result = detailed.align(seq1, seq2)

    assert simple_result.score == detailed_result.score
    assert simple_result.token_start == detailed_result.token_start
    assert simple_result.token_end == detailed_result.token_end
    assert simple_result.query_start == detailed_result.query_start
    assert simple_result.query_end == detailed_result.query_end
    assert simple_result.matches == detailed_result.matches
    assert simple_result.match_blocks == []
