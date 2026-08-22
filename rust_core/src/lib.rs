use pyo3::prelude::*;

mod citation_fast;
mod prepare;
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

#[pyfunction(signature = (seq1, seqs, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_batch_details(
    py: Python<'_>,
    seq1: Vec<u32>,
    seqs: Vec<Vec<u32>>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> Vec<(i32, usize, usize, usize, usize, usize)> {
    if seqs.is_empty() {
        return Vec::new();
    }
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    py.detach(|| {
        smith_waterman::align_batch(&seq1, &seqs, params)
            .into_iter()
            .map(|alignment| {
                (
                    alignment.score,
                    alignment.token_start,
                    alignment.token_end,
                    alignment.query_start,
                    alignment.query_end,
                    alignment.matches,
                )
            })
            .collect()
    })
}

#[pyfunction(signature = (seq1, seqs, match_score=2, mismatch_score=-1, gap_score=-1))]
fn align_batch_blocks_details(
    py: Python<'_>,
    seq1: Vec<u32>,
    seqs: Vec<Vec<u32>>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> Vec<AlignmentWithBlocks> {
    if seqs.is_empty() {
        return Vec::new();
    }
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };
    py.detach(|| {
        smith_waterman::align_batch_with_match_blocks(&seq1, &seqs, params)
            .into_iter()
            .map(|(alignment, match_blocks)| {
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
            .collect()
    })
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn rust_tokenize_and_prepare(
    py: Python<'_>,
    source_texts: Vec<String>,
    window_size: usize,
    stride: usize,
) -> PyResult<(
    Vec<Vec<(usize, usize, Vec<u32>, Vec<(usize, usize)>)>>,
    Vec<(u32, f64)>,
    Vec<(String, u32)>,
)> {
    py.detach(|| {
        let mut tokenizer = prepare::SimpleTokenizer::new();
        let mut all_tokenized = Vec::new();

        // Tokenize all sources
        for text in &source_texts {
            all_tokenized.push(tokenizer.tokenize(text));
        }

        // Build candidates per source
        let mut source_candidates = Vec::new();
        let mut all_candidate_tokens = Vec::new();

        for (text, tokenized) in source_texts.iter().zip(&all_tokenized) {
            let segments = prepare::simple_segment(text);
            let passages = prepare::generate_passages(&segments, window_size, stride);
            let mut candidates = Vec::new();

            for (passage_start, passage_end) in passages {
                let (token_ids, token_spans) = prepare::slice_tokenized_text(
                    &tokenized.token_ids,
                    &tokenized.token_spans,
                    passage_start,
                    passage_end,
                );

                candidates.push((passage_start, passage_end, token_ids.clone(), token_spans));
                all_candidate_tokens.push(token_ids);
            }

            source_candidates.push(candidates);
        }

        // Compute IDF
        let idf = prepare::compute_idf(&all_candidate_tokens);
        let idf_vec: Vec<(u32, f64)> = idf.into_iter().collect();

        // Extract vocab
        let vocab_vec: Vec<(String, u32)> = tokenizer.get_vocab().into_iter().collect();

        Ok((source_candidates, idf_vec, vocab_vec))
    })
}

/// Batch align answer tokens against selected candidates and return alignment results.
/// Returns list of (candidate_index, score, token_start, token_end, query_start, query_end, matches)
#[pyfunction(signature = (answer_tokens, candidate_indices, candidate_token_lists, match_score=2, mismatch_score=-1, gap_score=-1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn rust_align_batch_candidates(
    py: Python<'_>,
    answer_tokens: Vec<u32>,
    candidate_indices: Vec<usize>,
    candidate_token_lists: Vec<Vec<u32>>,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> PyResult<Vec<(usize, i32, usize, usize, usize, usize, usize)>> {
    let params = smith_waterman::ScoreParams {
        match_score,
        mismatch_score,
        gap_score,
    };

    py.detach(|| {
        let alignments =
            smith_waterman::align_batch(&answer_tokens, &candidate_token_lists, params);

        Ok(candidate_indices
            .into_iter()
            .zip(alignments)
            .map(|(idx, alignment)| {
                (
                    idx,
                    alignment.score,
                    alignment.token_start,
                    alignment.token_end,
                    alignment.query_start,
                    alignment.query_end,
                    alignment.matches,
                )
            })
            .collect())
    })
}

/// Fast citation building - returns JSON string  
#[pyfunction(signature = (
    answer_tokens,
    candidates_data,
    candidate_indices,
    lexical_scores,
    embed_scores,
    config_tuple,
    match_score=2,
    mismatch_score=-1,
    gap_score=-1
))]
#[allow(clippy::too_many_arguments)]
fn rust_build_citations_fast(
    py: Python<'_>,
    answer_tokens: Vec<u32>,
    candidates_data: Vec<(
        usize,               // index
        String,              // source_id
        usize,               // source_index
        String,              // source_text
        Option<String>,      // source_full_text
        usize,               // base_offset
        usize,               // passage_start
        usize,               // passage_end
        Vec<u32>,            // token_ids
        Vec<(usize, usize)>, // token_spans
    )>,
    candidate_indices: Vec<usize>,
    lexical_scores: Vec<f64>,
    embed_scores: Vec<f64>,
    config_tuple: (i32, f64, f64, bool, i32, f64, f64, f64, f64, f64),
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> PyResult<String> {
    py.detach(|| {
        // Build candidates
        let candidates: Vec<citation_fast::Candidate> = candidates_data
            .into_iter()
            .map(
                |(
                    index,
                    source_id,
                    source_index,
                    source_text,
                    source_full_text,
                    base_offset,
                    passage_start,
                    passage_end,
                    token_ids,
                    token_spans,
                )| {
                    citation_fast::Candidate {
                        index,
                        source_id,
                        source_index,
                        source_text,
                        source_full_text,
                        base_offset,
                        passage_start,
                        passage_end,
                        token_ids,
                        token_spans,
                    }
                },
            )
            .collect();

        // Build config
        let cfg = citation_fast::Config {
            min_alignment_score: config_tuple.0,
            min_answer_coverage: config_tuple.1,
            min_final_score: config_tuple.2,
            require_all_tokens: config_tuple.3,
            match_score: config_tuple.4,
            weight_alignment: config_tuple.5,
            weight_answer_coverage: config_tuple.6,
            weight_evidence_coverage: config_tuple.7,
            weight_lexical: config_tuple.8,
            weight_embedding: config_tuple.9,
        };

        // Run SW alignments
        let candidate_token_lists: Vec<Vec<u32>> = candidate_indices
            .iter()
            .map(|&idx| {
                if idx < candidates.len() {
                    candidates[idx].token_ids.clone()
                } else {
                    Vec::new()
                }
            })
            .collect();

        let params = smith_waterman::ScoreParams {
            match_score,
            mismatch_score,
            gap_score,
        };

        let sw_alignments =
            smith_waterman::align_batch(&answer_tokens, &candidate_token_lists, params);

        // Convert to local alignment type
        let alignments: Vec<citation_fast::Alignment> = sw_alignments
            .into_iter()
            .map(|a| citation_fast::Alignment {
                score: a.score,
                token_start: a.token_start,
                token_end: a.token_end,
                query_start: a.query_start,
                query_end: a.query_end,
                matches: a.matches,
            })
            .collect();

        // Build citations and return JSON
        Ok(citation_fast::build_citations_json(
            &answer_tokens,
            candidates,
            &candidate_indices,
            alignments,
            &lexical_scores,
            &embed_scores,
            cfg,
        ))
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
    module.add_function(wrap_pyfunction!(align_batch_details, module)?)?;
    module.add_function(wrap_pyfunction!(align_batch_blocks_details, module)?)?;
    module.add_function(wrap_pyfunction!(rust_tokenize_and_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(rust_align_batch_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(rust_build_citations_fast, module)?)?;
    Ok(())
}
