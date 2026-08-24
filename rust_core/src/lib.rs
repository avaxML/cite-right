use pyo3::prelude::*;

mod citation_fast;
mod contradiction_check;
mod inverted_index;
mod prepare;
mod prepared_corpus;
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
fn rust_tokenize_and_prepare(
    py: Python<'_>,
    source_texts: Vec<String>,
    window_size: usize,
    stride: usize,
) -> PyResult<prepared_corpus::PreparedCorpus> {
    py.detach(|| {
        let mut tokenizer = prepare::SimpleTokenizer::new();
        let mut all_tokenized = Vec::new();

        // Tokenize all sources
        for text in &source_texts {
            all_tokenized.push(tokenizer.tokenize(text));
        }

        // Build candidates and index inline
        let mut candidates = Vec::new();
        let mut all_candidate_tokens = Vec::new();
        let mut index = inverted_index::InvertedIndex::new();
        let mut global_candidate_index = 0;

        for (source_index, (text, tokenized)) in source_texts.iter().zip(&all_tokenized).enumerate()
        {
            let segments = prepare::simple_segment(text);
            let passages = prepare::generate_passages(&segments, window_size, stride);

            for (passage_start, passage_end) in passages {
                let (token_ids, token_spans) = prepare::slice_tokenized_text(
                    &tokenized.token_ids,
                    &tokenized.token_spans,
                    passage_start,
                    passage_end,
                );

                // Add to index inline
                for (token_pos, (&token_id, &(char_start, char_end))) in
                    token_ids.iter().zip(token_spans.iter()).enumerate()
                {
                    index.add_posting(
                        token_id,
                        inverted_index::Posting {
                            candidate_index: global_candidate_index,
                            token_pos,
                            char_start,
                            char_end,
                        },
                    );
                }

                candidates.push(prepared_corpus::Candidate {
                    source_index,
                    passage_start,
                    passage_end,
                    token_ids: token_ids.clone(),
                    token_spans: token_spans.clone(),
                });
                all_candidate_tokens.push(token_ids);
                global_candidate_index += 1;
            }
        }

        // Compute IDF
        let idf = prepare::compute_idf(&all_candidate_tokens);

        // Extract vocab
        let vocab = tokenizer.get_vocab();

        Ok(prepared_corpus::PreparedCorpus {
            candidates,
            idf,
            vocab,
            inverted_index: index,
            source_texts,
        })
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
    multi_span_config,
    match_score=2,
    mismatch_score=-1,
    gap_score=-1
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
    multi_span_config: (bool, i32, usize),
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
            multi_span_evidence: multi_span_config.0,
            multi_span_merge_gap_chars: multi_span_config.1,
            multi_span_max_spans: multi_span_config.2,
        };

        // Run SW alignments (with or without match blocks)
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

        let alignments: Vec<citation_fast::Alignment> = if cfg.multi_span_evidence {
            // Use align_batch_with_match_blocks for multi-span support
            smith_waterman::align_batch_with_match_blocks(
                &answer_tokens,
                &candidate_token_lists,
                params,
            )
            .into_iter()
            .map(|(a, match_blocks)| citation_fast::Alignment {
                score: a.score,
                token_start: a.token_start,
                token_end: a.token_end,
                matches: a.matches,
                match_blocks,
            })
            .collect()
        } else {
            // Standard align_batch without match blocks
            smith_waterman::align_batch(&answer_tokens, &candidate_token_lists, params)
                .into_iter()
                .map(|a| citation_fast::Alignment {
                    score: a.score,
                    token_start: a.token_start,
                    token_end: a.token_end,
                    matches: a.matches,
                    match_blocks: Vec::new(),
                })
                .collect()
        };

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
    module.add_class::<inverted_index::InvertedIndex>()?;
    module.add_class::<prepared_corpus::PreparedCorpus>()?;
    Ok(())
}
