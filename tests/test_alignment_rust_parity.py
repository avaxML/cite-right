"""Tests for Rust/Python parity in Smith-Waterman alignment."""

from types import ModuleType

from cite_right.core.aligner_py import SmithWatermanAligner

from .conftest import requires_rust, requires_rust_blocks


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

    best_key: tuple[int, int, int, int, int, int, int] | None = None
    best: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    for index, seq2 in enumerate(candidates):
        py = aligner.align(claim, seq2)
        span_len = py.token_end - py.token_start
        key = (
            -py.score,
            py.token_start,
            -span_len,
            py.query_start,
            index,
            py.token_end,
            py.query_end,
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
        key=lambda item: (
            -item[0],
            item[2],
            -(item[3] - item[2]),
            item[4],
            item[1],
            item[3],
            item[5],
        )
    )
    assert rust == py_items[:top_k], "Rust top-k differs from Python selection"
