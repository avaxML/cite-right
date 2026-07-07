"""Tests for Rust/Python parity in Smith-Waterman alignment."""

from types import ModuleType

from cite_right.core.aligner_py import SmithWatermanAligner
from cite_right.core.aligner_rust import RustSmithWatermanAligner

from .conftest import requires_rust, requires_rust_blocks


def _python_alignment_sort_key(
    score: int,
    matches: int,
    token_start: int,
    token_end: int,
    query_start: int,
    query_end: int,
    index: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    span_len = token_end - token_start
    return (
        -score,
        -matches,
        token_start,
        -span_len,
        query_start,
        index,
        token_end,
        query_end,
    )


@requires_rust
def test_rust_parity(rust_core: ModuleType) -> None:
    """Verify Python and Rust implementations produce identical results."""
    aligner = SmithWatermanAligner()
    cases = [
        ([1, 2], [1, 2, 1, 2]),
        ([1, 2, 3], [0, 1, 2, 3, 4]),
        ([1, 2], [3, 4]),
    ]

    for seq1, seq2 in cases:
        py = aligner.align(seq1, seq2)
        rust = rust_core.align_pair_details(seq1, seq2, 2, -1, -1)
        assert rust == (
            py.score,
            py.token_start,
            py.token_end,
            py.query_start,
            py.query_end,
            py.matches,
        ), f"Mismatch for sequences {seq1}, {seq2}"


@requires_rust_blocks
def test_rust_parity_for_equal_score_more_matches_case(
    rust_core_with_blocks: ModuleType,
) -> None:
    """Verify Rust matches Python on the equal-score coverage regression."""
    aligner = SmithWatermanAligner(return_match_blocks=True)
    seq1 = [0, 1, 0]
    seq2 = [0, 1, 1, 1, 0]

    py = aligner.align(seq1, seq2)
    rust = rust_core_with_blocks.align_pair_blocks_details(seq1, seq2, 2, -1, -1)

    assert rust == (
        py.score,
        py.token_start,
        py.token_end,
        py.query_start,
        py.query_end,
        py.matches,
        py.match_blocks,
    ), "Rust equal-score traceback differs from Python"


@requires_rust
def test_rust_align_best_matches_python_selection(rust_core: ModuleType) -> None:
    """Verify Rust align_best matches Python selection logic."""
    aligner = SmithWatermanAligner()
    claim = [1, 2]
    candidates = [[3, 4], [1, 2, 1, 2], [1, 2], [0, 1, 2, 3]]

    rust = rust_core.align_best_details(claim, candidates, 2, -1, -1)
    assert rust is not None, "Rust align_best_details returned None unexpectedly"
    (
        rust_score,
        rust_index,
        rust_start,
        rust_end,
        rust_query_start,
        rust_query_end,
        rust_matches,
    ) = rust

    best_key: tuple[int, int, int, int, int, int, int, int] | None = None
    best: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    for index, seq2 in enumerate(candidates):
        py = aligner.align(claim, seq2)
        key = _python_alignment_sort_key(
            py.score,
            py.matches,
            py.token_start,
            py.token_end,
            py.query_start,
            py.query_end,
            index,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (
                py.score,
                index,
                py.token_start,
                py.token_end,
                py.query_start,
                py.query_end,
                py.matches,
            )

    assert (
        rust_score,
        rust_index,
        rust_start,
        rust_end,
        rust_query_start,
        rust_query_end,
        rust_matches,
    ) == best, "Rust best selection differs from Python"


@requires_rust
def test_rust_align_best_empty_returns_none(rust_core: ModuleType) -> None:
    """Verify Rust returns None for empty candidate list."""
    assert rust_core.align_best([1], [], 2, -1, -1) is None
    assert rust_core.align_best_details([1], [], 2, -1, -1) is None


@requires_rust
def test_rust_wrapper_align_best_details_matches_extension() -> None:
    """Verify the Rust wrapper exposes the detailed best-match API."""
    aligner = RustSmithWatermanAligner()
    claim = [1, 2]
    candidates = [[3, 4], [1, 2, 1, 2], [1, 2], [0, 1, 2, 3]]

    assert aligner.align_best_details(claim, candidates) == (4, 1, 0, 2, 0, 2, 2)


@requires_rust_blocks
def test_rust_wrapper_align_batch_matches_python_ordered_results() -> None:
    """Verify the Rust batch wrapper preserves input order and full details."""
    py_aligner = SmithWatermanAligner(return_match_blocks=True)
    rust_aligner = RustSmithWatermanAligner(return_match_blocks=True)
    claim = [1, 2, 3]
    candidates = [[0, 1, 2, 3, 4], [1, 2, 9, 3], [8, 9, 10]]

    expected = [py_aligner.align(claim, candidate) for candidate in candidates]

    assert rust_aligner.align_batch(claim, candidates) == expected


@requires_rust_blocks
def test_rust_align_pair_blocks_details_matches_python_blocks(
    rust_core_with_blocks: ModuleType,
) -> None:
    """Verify Rust align_pair_blocks_details matches Python blocks output."""
    aligner = SmithWatermanAligner(return_match_blocks=True)
    seq1 = [1, 2, 3, 4]
    seq2 = [1, 2, 9, 9, 3, 4]

    py = aligner.align(seq1, seq2)
    rust = rust_core_with_blocks.align_pair_blocks_details(seq1, seq2, 2, -1, -1)
    assert rust == (
        py.score,
        py.token_start,
        py.token_end,
        py.query_start,
        py.query_end,
        py.matches,
        py.match_blocks,
    ), "Rust match_blocks differs from Python"


@requires_rust_blocks
def test_rust_block_and_non_block_entrypoints_share_alignment(
    rust_core: ModuleType,
    rust_core_with_blocks: ModuleType,
) -> None:
    """Verify block collection does not change the chosen alignment."""
    seq1 = [0, 1, 0]
    seq2 = [0, 2, 1, 1, 0]

    without_blocks = rust_core.align_pair_details(seq1, seq2, 2, -1, -1)
    with_blocks = rust_core_with_blocks.align_pair_blocks_details(seq1, seq2, 2, -1, -1)

    assert with_blocks[:6] == without_blocks


@requires_rust
def test_rust_align_topk_matches_python_selection(rust_core: ModuleType) -> None:
    """Verify Rust top-k selection matches Python sorting logic."""
    aligner = SmithWatermanAligner()
    claim = [1, 2]
    candidates = [[3, 4], [1, 2, 1, 2], [1, 2], [0, 1, 2, 3]]

    top_k = 3
    rust = rust_core.align_topk_details(claim, candidates, top_k, 2, -1, -1)

    py_items: list[tuple[int, int, int, int, int, int, int]] = []
    for index, seq2 in enumerate(candidates):
        py = aligner.align(claim, seq2)
        py_items.append(
            (
                py.score,
                index,
                py.token_start,
                py.token_end,
                py.query_start,
                py.query_end,
                py.matches,
            )
        )

    py_items.sort(
        key=lambda item: _python_alignment_sort_key(
            item[0],
            item[6],
            item[2],
            item[3],
            item[4],
            item[5],
            item[1],
        )
    )
    assert rust == py_items[:top_k], "Rust top-k differs from Python selection"


@requires_rust
def test_rust_align_batch_extension_preserves_input_order(
    rust_core: ModuleType,
) -> None:
    """Verify ordered Rust batch results match repeated Python alignment."""
    aligner = SmithWatermanAligner()
    claim = [1, 2, 3]
    candidates = [[0, 1, 2, 3, 4], [1, 2, 9, 3], [8, 9, 10]]

    rust = rust_core.align_batch_details(claim, candidates, 2, -1, -1)
    expected = [aligner.align(claim, candidate) for candidate in candidates]

    assert rust == [
        (
            item.score,
            item.token_start,
            item.token_end,
            item.query_start,
            item.query_end,
            item.matches,
        )
        for item in expected
    ]


@requires_rust
def test_rust_align_best_prefers_more_matches_on_equal_scores(
    rust_core: ModuleType,
) -> None:
    """Verify Rust best selection follows Python's matches-first tie-break."""
    aligner = SmithWatermanAligner()
    claim = [0, 1, 0]
    candidates = [[0, 1, 1, 1, 0], [0, 2, 1, 1, 0]]

    rust = rust_core.align_best_details(claim, candidates, 2, -1, -1)
    assert rust is not None

    py_items: list[tuple[int, int, int, int, int, int, int]] = []
    for index, seq2 in enumerate(candidates):
        py = aligner.align(claim, seq2)
        py_items.append(
            (
                py.score,
                index,
                py.token_start,
                py.token_end,
                py.query_start,
                py.query_end,
                py.matches,
            )
        )

    py_items.sort(
        key=lambda item: _python_alignment_sort_key(
            item[0],
            item[6],
            item[2],
            item[3],
            item[4],
            item[5],
            item[1],
        )
    )
    assert rust == py_items[0], "Rust align_best_details differs on equal-score ties"


@requires_rust
def test_rust_align_best_matches_topk_first_item(rust_core: ModuleType) -> None:
    """Verify align_best remains equivalent to the first top-k result."""
    claim = [1, 2, 3]
    candidates = [[1, 2], [1, 2, 3], [0, 1, 2, 3], [1, 9, 2, 9, 3]]

    best = rust_core.align_best_details(claim, candidates, 2, -1, -1)
    top1 = rust_core.align_topk_details(claim, candidates, 1, 2, -1, -1)

    assert best is not None
    assert top1 == [best]
