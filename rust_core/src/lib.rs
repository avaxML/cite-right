use pyo3::prelude::*;

mod smith_waterman;

type MatchBlocks = Vec<(usize, usize)>;
type AlignmentDetails = (i32, usize, usize, usize, usize, usize, usize);
type AlignmentWithBlocks = (i32, usize, usize, usize, usize, usize, MatchBlocks);

#[pyfunction(signature = (seq1, seq2, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_pair(
    py: Python<'_>,
    seq1: Vec<u32>,
    seq2: Vec<u32>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> (i32, usize, usize) {
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    py.detach(|| {
        let alignment = smith_waterman::smith_waterman(&seq1, &seq2, params);
        (alignment.score, alignment.token_start, alignment.token_end)
    })
}

#[pyfunction(signature = (seq1, seq2, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_pair_details(
    py: Python<'_>,
    seq1: Vec<u32>,
    seq2: Vec<u32>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> (i32, usize, usize, usize, usize, usize) {
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    py.detach(|| {
        let alignment = smith_waterman::smith_waterman(&seq1, &seq2, params);
        (
            alignment.score,
            alignment.token_start,
            alignment.token_end,
            alignment.query_start,
            alignment.query_end,
            alignment.matches,
        )
    })
}

#[pyfunction(signature = (seq1, seq2, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_pair_blocks_details(
    py: Python<'_>,
    seq1: Vec<u32>,
    seq2: Vec<u32>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> AlignmentWithBlocks {
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    py.detach(|| {
        let (alignment, match_blocks) =
            smith_waterman::smith_waterman_match_blocks(&seq1, &seq2, params);
        (
            alignment.score,
            alignment.token_start,
            alignment.token_end,
            alignment.query_start,
            alignment.query_end,
            alignment.matches,
            match_blocks,
        )
    })
}

#[pyfunction(signature = (seq1, seqs, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_best(
    py: Python<'_>,
    seq1: Vec<u32>,
    seqs: Vec<Vec<u32>>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> Option<(i32, usize, usize, usize)> {
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    let best = py.detach(|| smith_waterman::align_best(&seq1, &seqs, params))?;
    Some((best.score, best.index, best.token_start, best.token_end))
}

#[pyfunction(signature = (seq1, seqs, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_best_details(
    py: Python<'_>,
    seq1: Vec<u32>,
    seqs: Vec<Vec<u32>>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> Option<AlignmentDetails> {
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    let best = py.detach(|| smith_waterman::align_best(&seq1, &seqs, params))?;
    Some((
        best.score,
        best.index,
        best.token_start,
        best.token_end,
        best.query_start,
        best.query_end,
        best.matches,
    ))
}

#[pyfunction(signature = (seq1, seqs, top_k=1, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_topk_details(
    py: Python<'_>,
    seq1: Vec<u32>,
    seqs: Vec<Vec<u32>>,
    top_k: usize,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> Vec<AlignmentDetails> {
    if top_k == 0 || seqs.is_empty() {
        return Vec::new();
    }
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    py.detach(|| {
        smith_waterman::align_topk(&seq1, &seqs, params, top_k)
            .into_iter()
            .map(|item| {
                (
                    item.score,
                    item.index,
                    item.token_start,
                    item.token_end,
                    item.query_start,
                    item.query_end,
                    item.matches,
                )
            })
            .collect()
    })
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(align_pair, module)?)?;
    module.add_function(wrap_pyfunction!(align_pair_details, module)?)?;
    module.add_function(wrap_pyfunction!(align_pair_blocks_details, module)?)?;
    module.add_function(wrap_pyfunction!(align_best, module)?)?;
    module.add_function(wrap_pyfunction!(align_best_details, module)?)?;
    module.add_function(wrap_pyfunction!(align_topk_details, module)?)?;
    Ok(())
}
