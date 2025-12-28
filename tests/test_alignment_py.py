from cite_right.core.aligner_py import SmithWatermanAligner


def test_alignment_basic() -> None:
    aligner = SmithWatermanAligner()
    result = aligner.align([1, 2, 3], [0, 1, 2, 3, 4])
    assert result.score == 6
    assert result.token_start == 1
    assert result.token_end == 4


def test_alignment_prefers_earlier_start() -> None:
    aligner = SmithWatermanAligner()
    result = aligner.align([1, 2], [1, 2, 1, 2])
    assert result.score == 4
    assert result.token_start == 0
    assert result.token_end == 2


def test_alignment_no_match() -> None:
    aligner = SmithWatermanAligner()
    result = aligner.align([1, 2], [3, 4])
    assert result.score == 0
    assert result.token_start == 0
    assert result.token_end == 0
