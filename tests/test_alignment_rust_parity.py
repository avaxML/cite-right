import pytest

from cite_right.core.aligner_py import SmithWatermanAligner


def test_rust_parity() -> None:
    try:
        from cite_right import _core
    except ImportError:
        pytest.skip("Rust extension not built")

    aligner = SmithWatermanAligner()
    cases = [
        ([1, 2], [1, 2, 1, 2]),
        ([1, 2, 3], [0, 1, 2, 3, 4]),
        ([1, 2], [3, 4]),
    ]

    for seq1, seq2 in cases:
        py = aligner.align(seq1, seq2)
        rust = _core.align_pair_details(seq1, seq2, 2, -1, -1)
        assert rust == (
            py.score,
            py.token_start,
            py.token_end,
            py.query_start,
            py.query_end,
            py.matches,
        )


def test_rust_align_best_matches_python_selection() -> None:
    try:
        from cite_right import _core
    except ImportError:
        pytest.skip("Rust extension not built")

    aligner = SmithWatermanAligner()
    claim = [1, 2]
    candidates = [[3, 4], [1, 2, 1, 2], [1, 2], [0, 1, 2, 3]]

    rust = _core.align_best_details(claim, candidates, 2, -1, -1)
    assert rust is not None
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
    ) == best


def test_rust_align_best_empty_returns_none() -> None:
    try:
        from cite_right import _core
    except ImportError:
        pytest.skip("Rust extension not built")

    assert _core.align_best([1], [], 2, -1, -1) is None
    assert _core.align_best_details([1], [], 2, -1, -1) is None


def test_rust_align_pair_blocks_details_matches_python_blocks() -> None:
    try:
        from cite_right import _core
    except ImportError:
        pytest.skip("Rust extension not built")
    if not hasattr(_core, "align_pair_blocks_details"):
        pytest.skip(
            "Rust extension is missing align_pair_blocks_details (rebuild required)"
        )

    aligner = SmithWatermanAligner(return_match_blocks=True)
    seq1 = [1, 2, 3, 4]
    seq2 = [1, 2, 9, 9, 3, 4]

    py = aligner.align(seq1, seq2)
    rust = _core.align_pair_blocks_details(seq1, seq2, 2, -1, -1)
    assert rust == (
        py.score,
        py.token_start,
        py.token_end,
        py.query_start,
        py.query_end,
        py.matches,
        py.match_blocks,
    )


def test_rust_align_topk_matches_python_selection() -> None:
    try:
        from cite_right import _core
    except ImportError:
        pytest.skip("Rust extension not built")

    aligner = SmithWatermanAligner()
    claim = [1, 2]
    candidates = [[3, 4], [1, 2, 1, 2], [1, 2], [0, 1, 2, 3]]

    top_k = 3
    rust = _core.align_topk_details(claim, candidates, top_k, 2, -1, -1)

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
    assert rust == py_items[:top_k]
