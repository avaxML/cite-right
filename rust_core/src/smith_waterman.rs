#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Copy)]
pub struct ScoreParams {
    pub match_score: i32,
    pub mismatch_score: i32,
    pub gap_score: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Alignment {
    pub score: i32,
    pub query_start: usize,
    pub query_end: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub matches: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CandidateAlignment {
    pub score: i32,
    pub index: usize,
    pub query_start: usize,
    pub query_end: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub matches: usize,
}

pub fn smith_waterman(seq1: &[u32], seq2: &[u32], params: ScoreParams) -> Alignment {
    if seq1.is_empty() || seq2.is_empty() {
        return Alignment {
            score: 0,
            query_start: 0,
            query_end: 0,
            token_start: 0,
            token_end: 0,
            matches: 0,
        };
    }

    let (scores, directions, max_score, best_end) = fill_matrices_reduced_state(seq1, seq2, params);

    if max_score == 0 {
        return Alignment {
            score: 0,
            query_start: 0,
            query_end: 0,
            token_start: 0,
            token_end: 0,
            matches: 0,
        };
    }

    let cols = seq2.len() + 1;
    build_alignment(
        max_score,
        best_end,
        &directions,
        &scores,
        cols,
        seq1,
        seq2,
        false,
    )
    .expect("max_positions is non-empty when max_score > 0")
    .0
}

pub fn smith_waterman_match_blocks(
    seq1: &[u32],
    seq2: &[u32],
    params: ScoreParams,
) -> (Alignment, Vec<(usize, usize)>) {
    if seq1.is_empty() || seq2.is_empty() {
        return (
            Alignment {
                score: 0,
                query_start: 0,
                query_end: 0,
                token_start: 0,
                token_end: 0,
                matches: 0,
            },
            Vec::new(),
        );
    }

    let (scores, directions, max_score, best_end) = fill_matrices(seq1, seq2, params);

    if max_score == 0 {
        return (
            Alignment {
                score: 0,
                query_start: 0,
                query_end: 0,
                token_start: 0,
                token_end: 0,
                matches: 0,
            },
            Vec::new(),
        );
    }

    let cols = seq2.len() + 1;
    build_alignment(
        max_score,
        best_end,
        &directions,
        &scores,
        cols,
        seq1,
        seq2,
        true,
    )
    .expect("max_positions is non-empty when max_score > 0")
}

pub fn align_topk(
    seq1: &[u32],
    seqs: &[Vec<u32>],
    params: ScoreParams,
    top_k: usize,
) -> Vec<CandidateAlignment> {
    if seqs.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut results: Vec<CandidateAlignment> = seqs
        .par_iter()
        .enumerate()
        .map(|(index, seq2)| {
            let alignment = smith_waterman(seq1, seq2, params);
            CandidateAlignment {
                score: alignment.score,
                index,
                query_start: alignment.query_start,
                query_end: alignment.query_end,
                token_start: alignment.token_start,
                token_end: alignment.token_end,
                matches: alignment.matches,
            }
        })
        .collect();

    results.sort_by(cmp_candidate);
    results.truncate(top_k.min(results.len()));
    results
}

pub fn align_batch(seq1: &[u32], seqs: &[Vec<u32>], params: ScoreParams) -> Vec<Alignment> {
    seqs.par_iter()
        .map(|seq2| smith_waterman(seq1, seq2, params))
        .collect()
}

pub fn align_batch_with_match_blocks(
    seq1: &[u32],
    seqs: &[Vec<u32>],
    params: ScoreParams,
) -> Vec<(Alignment, Vec<(usize, usize)>)> {
    seqs.par_iter()
        .map(|seq2| smith_waterman_match_blocks(seq1, seq2, params))
        .collect()
}

pub fn align_best(
    seq1: &[u32],
    seqs: &[Vec<u32>],
    params: ScoreParams,
) -> Option<CandidateAlignment> {
    seqs.par_iter()
        .enumerate()
        .map(|(index, seq2)| {
            let alignment = smith_waterman(seq1, seq2, params);
            CandidateAlignment {
                score: alignment.score,
                index,
                query_start: alignment.query_start,
                query_end: alignment.query_end,
                token_start: alignment.token_start,
                token_end: alignment.token_end,
                matches: alignment.matches,
            }
        })
        .reduce_with(|left, right| {
            if cmp_candidate(&left, &right) == Ordering::Less {
                left
            } else {
                right
            }
        })
}

#[allow(unused_assignments)]
fn fill_matrices(
    seq1: &[u32],
    seq2: &[u32],
    params: ScoreParams,
) -> (Vec<i32>, Vec<u8>, i32, Option<(usize, usize)>) {
    let rows = seq1.len() + 1;
    let cols = seq2.len() + 1;
    let mut scores = vec![0i32; rows * cols];
    let mut directions = vec![0u8; rows * cols];
    let mut match_counts = vec![0usize; rows * cols];
    let mut query_starts = vec![0usize; rows * cols];
    let mut token_starts = vec![0usize; rows * cols];

    let mut max_score = 0i32;
    let mut best_end: Option<(usize, usize)> = None;
    let mut best_key: Option<(usize, usize, usize, usize, usize, usize)> = None;

    for i in 1..rows {
        let row_offset = i * cols;
        let prev_row_offset = (i - 1) * cols;
        let s1_val = seq1[i - 1];

        for j in 1..cols {
            let is_match = s1_val == seq2[j - 1];
            let diag_delta = if is_match {
                params.match_score
            } else {
                params.mismatch_score
            };

            let mut best_score = 0i32;
            let mut best_direction = 0u8;
            let mut best_matches = 0usize;
            let mut best_query_start = 0usize;
            let mut best_token_start = 0usize;
            let mut best_priority = 0u8;

            // 1. DIAGONAL
            let score_diag = scores[prev_row_offset + j - 1] + diag_delta;
            if score_diag > 0 {
                let (matches, query_start, token_start) = if scores[prev_row_offset + j - 1] > 0 {
                    (
                        match_counts[prev_row_offset + j - 1] + usize::from(is_match),
                        query_starts[prev_row_offset + j - 1],
                        token_starts[prev_row_offset + j - 1],
                    )
                } else {
                    (usize::from(is_match), i - 1, j - 1)
                };
                best_score = score_diag;
                best_direction = 1;
                best_matches = matches;
                best_query_start = query_start;
                best_token_start = token_start;
                best_priority = 0;
            }

            // 2. UP
            let score_up = scores[prev_row_offset + j] + params.gap_score;
            if score_up > 0 {
                let (matches, query_start, token_start) = if scores[prev_row_offset + j] > 0 {
                    (
                        match_counts[prev_row_offset + j],
                        query_starts[prev_row_offset + j],
                        token_starts[prev_row_offset + j],
                    )
                } else {
                    (0, i - 1, j)
                };
                let mut is_better = false;
                if score_up > best_score {
                    is_better = true;
                } else if score_up == best_score && score_up > 0 {
                    if matches != best_matches {
                        is_better = matches > best_matches;
                    } else if token_start != best_token_start {
                        is_better = token_start < best_token_start;
                    } else if query_start != best_query_start {
                        is_better = query_start < best_query_start;
                    } else {
                        is_better = 1 < best_priority;
                    }
                }
                if is_better {
                    best_score = score_up;
                    best_direction = 2;
                    best_matches = matches;
                    best_query_start = query_start;
                    best_token_start = token_start;
                    best_priority = 1;
                }
            }

            // 3. LEFT
            let score_left = scores[row_offset + j - 1] + params.gap_score;
            if score_left > 0 {
                let (matches, query_start, token_start) = if scores[row_offset + j - 1] > 0 {
                    (
                        match_counts[row_offset + j - 1],
                        query_starts[row_offset + j - 1],
                        token_starts[row_offset + j - 1],
                    )
                } else {
                    (0, i, j - 1)
                };
                let mut is_better = false;
                if score_left > best_score {
                    is_better = true;
                } else if score_left == best_score && score_left > 0 {
                    if matches != best_matches {
                        is_better = matches > best_matches;
                    } else if token_start != best_token_start {
                        is_better = token_start < best_token_start;
                    } else if query_start != best_query_start {
                        is_better = query_start < best_query_start;
                    } else {
                        is_better = 2 < best_priority;
                    }
                }
                if is_better {
                    best_score = score_left;
                    best_direction = 3;
                    best_matches = matches;
                    best_query_start = query_start;
                    best_token_start = token_start;
                    best_priority = 2;
                }
            }

            scores[row_offset + j] = best_score;
            directions[row_offset + j] = best_direction;
            match_counts[row_offset + j] = best_matches;
            query_starts[row_offset + j] = best_query_start;
            token_starts[row_offset + j] = best_token_start;

            if best_score > max_score {
                max_score = best_score;
                best_end = Some((i, j));
                best_key = Some(alignment_key(
                    best_query_start,
                    best_token_start,
                    i,
                    j,
                    best_matches,
                ));
            } else if best_score == max_score && best_score > 0 {
                let candidate_key =
                    alignment_key(best_query_start, best_token_start, i, j, best_matches);
                if best_key.is_none_or(|current| candidate_key < current) {
                    best_end = Some((i, j));
                    best_key = Some(candidate_key);
                }
            }
        }
    }

    (scores, directions, max_score, best_end)
}

#[allow(unused_assignments)]
fn fill_matrices_reduced_state(
    seq1: &[u32],
    seq2: &[u32],
    params: ScoreParams,
) -> (Vec<i32>, Vec<u8>, i32, Option<(usize, usize)>) {
    let rows = seq1.len() + 1;
    let cols = seq2.len() + 1;
    let mut scores = vec![0i32; rows * cols];
    let mut directions = vec![0u8; rows * cols];

    let mut max_score = 0i32;
    let mut best_end: Option<(usize, usize)> = None;
    let mut best_key: Option<(usize, usize, usize, usize, usize, usize)> = None;

    let mut prev_match_counts = vec![0usize; cols];
    let mut prev_query_starts = vec![0usize; cols];
    let mut prev_token_starts = vec![0usize; cols];

    let mut current_match_counts = vec![0usize; cols];
    let mut current_query_starts = vec![0usize; cols];
    let mut current_token_starts = vec![0usize; cols];

    for i in 1..rows {
        let row_offset = i * cols;
        let prev_row_offset = (i - 1) * cols;
        let s1_val = seq1[i - 1];

        current_match_counts[0] = 0;
        current_query_starts[0] = 0;
        current_token_starts[0] = 0;

        for j in 1..cols {
            let is_match = s1_val == seq2[j - 1];
            let diag_delta = if is_match {
                params.match_score
            } else {
                params.mismatch_score
            };

            let mut best_score = 0i32;
            let mut best_direction = 0u8;
            let mut best_matches = 0usize;
            let mut best_query_start = 0usize;
            let mut best_token_start = 0usize;
            let mut best_priority = 0u8;

            // 1. DIAGONAL
            let score_diag = scores[prev_row_offset + j - 1] + diag_delta;
            if score_diag > 0 {
                let (matches, query_start, token_start) = if scores[prev_row_offset + j - 1] > 0 {
                    (
                        prev_match_counts[j - 1] + usize::from(is_match),
                        prev_query_starts[j - 1],
                        prev_token_starts[j - 1],
                    )
                } else {
                    (usize::from(is_match), i - 1, j - 1)
                };
                best_score = score_diag;
                best_direction = 1;
                best_matches = matches;
                best_query_start = query_start;
                best_token_start = token_start;
                best_priority = 0;
            }

            // 2. UP
            let score_up = scores[prev_row_offset + j] + params.gap_score;
            if score_up > 0 {
                let (matches, query_start, token_start) = if scores[prev_row_offset + j] > 0 {
                    (
                        prev_match_counts[j],
                        prev_query_starts[j],
                        prev_token_starts[j],
                    )
                } else {
                    (0, i - 1, j)
                };
                let mut is_better = false;
                if score_up > best_score {
                    is_better = true;
                } else if score_up == best_score && score_up > 0 {
                    if matches != best_matches {
                        is_better = matches > best_matches;
                    } else if token_start != best_token_start {
                        is_better = token_start < best_token_start;
                    } else if query_start != best_query_start {
                        is_better = query_start < best_query_start;
                    } else {
                        is_better = 1 < best_priority;
                    }
                }
                if is_better {
                    best_score = score_up;
                    best_direction = 2;
                    best_matches = matches;
                    best_query_start = query_start;
                    best_token_start = token_start;
                    best_priority = 1;
                }
            }

            // 3. LEFT
            let score_left = scores[row_offset + j - 1] + params.gap_score;
            if score_left > 0 {
                let (matches, query_start, token_start) = if scores[row_offset + j - 1] > 0 {
                    (
                        current_match_counts[j - 1],
                        current_query_starts[j - 1],
                        current_token_starts[j - 1],
                    )
                } else {
                    (0, i, j - 1)
                };
                let mut is_better = false;
                if score_left > best_score {
                    is_better = true;
                } else if score_left == best_score && score_left > 0 {
                    if matches != best_matches {
                        is_better = matches > best_matches;
                    } else if token_start != best_token_start {
                        is_better = token_start < best_token_start;
                    } else if query_start != best_query_start {
                        is_better = query_start < best_query_start;
                    } else {
                        is_better = 2 < best_priority;
                    }
                }
                if is_better {
                    best_score = score_left;
                    best_direction = 3;
                    best_matches = matches;
                    best_query_start = query_start;
                    best_token_start = token_start;
                    best_priority = 2;
                }
            }

            scores[row_offset + j] = best_score;
            directions[row_offset + j] = best_direction;
            current_match_counts[j] = best_matches;
            current_query_starts[j] = best_query_start;
            current_token_starts[j] = best_token_start;

            if best_score > max_score {
                max_score = best_score;
                best_end = Some((i, j));
                best_key = Some(alignment_key(
                    best_query_start,
                    best_token_start,
                    i,
                    j,
                    best_matches,
                ));
            } else if best_score == max_score && best_score > 0 {
                let candidate_key =
                    alignment_key(best_query_start, best_token_start, i, j, best_matches);
                if best_key.is_none_or(|current| candidate_key < current) {
                    best_end = Some((i, j));
                    best_key = Some(candidate_key);
                }
            }
        }

        std::mem::swap(&mut prev_match_counts, &mut current_match_counts);
        std::mem::swap(&mut prev_query_starts, &mut current_query_starts);
        std::mem::swap(&mut prev_token_starts, &mut current_token_starts);
    }

    (scores, directions, max_score, best_end)
}

fn build_alignment(
    max_score: i32,
    best_end: Option<(usize, usize)>,
    directions: &[u8],
    scores: &[i32],
    cols: usize,
    seq1: &[u32],
    seq2: &[u32],
    return_match_blocks: bool,
) -> Option<(Alignment, Vec<(usize, usize)>)> {
    let (i_end, j_end) = best_end?;
    let (i_start, j_start, matches, match_blocks) = traceback_details_common(
        i_end,
        j_end,
        directions,
        scores,
        cols,
        seq1,
        seq2,
        return_match_blocks,
    );
    Some((
        Alignment {
            score: max_score,
            query_start: i_start,
            query_end: i_end,
            token_start: j_start,
            token_end: j_end,
            matches,
        },
        match_blocks,
    ))
}

fn alignment_key(
    query_start: usize,
    token_start: usize,
    query_end: usize,
    token_end: usize,
    matches: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    let span_len = token_end - token_start;
    (
        usize::MAX - matches,
        token_start,
        usize::MAX - span_len,
        query_start,
        token_end,
        query_end,
    )
}

fn traceback_details_common(
    mut i: usize,
    mut j: usize,
    directions: &[u8],
    scores: &[i32],
    cols: usize,
    seq1: &[u32],
    seq2: &[u32],
    return_match_blocks: bool,
) -> (usize, usize, usize, Vec<(usize, usize)>) {
    let mut matches = 0usize;
    let mut match_positions: Vec<usize> = Vec::new();

    while i > 0 && j > 0 && directions[i * cols + j] != 0 && scores[i * cols + j] > 0 {
        match directions[i * cols + j] {
            1 => {
                i -= 1;
                j -= 1;
                if seq1[i] == seq2[j] {
                    matches += 1;
                    if return_match_blocks {
                        match_positions.push(j);
                    }
                }
            }
            2 => {
                i -= 1;
            }
            _ => {
                j -= 1;
            }
        }
    }

    let match_blocks = if return_match_blocks {
        consolidate_match_blocks(match_positions)
    } else {
        Vec::new()
    };

    (i, j, matches, match_blocks)
}

fn consolidate_match_blocks(mut match_positions: Vec<usize>) -> Vec<(usize, usize)> {
    if match_positions.is_empty() {
        return Vec::new();
    }

    match_positions.reverse();
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut start = match_positions[0];
    let mut prev = start;
    for pos in match_positions.into_iter().skip(1) {
        if pos == prev + 1 {
            prev = pos;
            continue;
        }
        blocks.push((start, prev + 1));
        start = pos;
        prev = pos;
    }
    blocks.push((start, prev + 1));
    blocks
}

fn cmp_candidate(left: &CandidateAlignment, right: &CandidateAlignment) -> Ordering {
    if left.score != right.score {
        return right.score.cmp(&left.score);
    }
    if left.matches != right.matches {
        return right.matches.cmp(&left.matches);
    }
    if left.token_start != right.token_start {
        return left.token_start.cmp(&right.token_start);
    }

    let left_span = left.token_end - left.token_start;
    let right_span = right.token_end - right.token_start;
    if left_span != right_span {
        return right_span.cmp(&left_span);
    }

    if left.query_start != right.query_start {
        return left.query_start.cmp(&right.query_start);
    }
    if left.index != right.index {
        return left.index.cmp(&right.index);
    }
    if left.token_end != right.token_end {
        return left.token_end.cmp(&right.token_end);
    }
    left.query_end.cmp(&right.query_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smith_waterman_prefers_earlier_start() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![1, 2];
        let seq2 = vec![1, 2, 1, 2];
        let alignment = smith_waterman(&seq1, &seq2, params);
        assert_eq!(alignment.score, 4);
        assert_eq!(alignment.token_start, 0);
        assert_eq!(alignment.token_end, 2);
        assert_eq!(alignment.matches, 2);
        assert_eq!(alignment.query_start, 0);
        assert_eq!(alignment.query_end, 2);
    }

    #[test]
    fn smith_waterman_match_blocks_returns_disjoint_blocks() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![1, 2, 3, 4];
        let seq2 = vec![1, 2, 9, 9, 3, 4];

        let (alignment, match_blocks) = smith_waterman_match_blocks(&seq1, &seq2, params);
        assert_eq!(alignment.score, 6);
        assert_eq!(alignment.token_start, 0);
        assert_eq!(alignment.token_end, 6);
        assert_eq!(alignment.query_start, 0);
        assert_eq!(alignment.query_end, 4);
        assert_eq!(alignment.matches, 4);
        assert_eq!(match_blocks, vec![(0, 2), (4, 6)]);
    }

    #[test]
    fn smith_waterman_prefers_more_matches_across_equal_score_endpoints() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![0, 1, 0];
        let seq2 = vec![0, 1, 1, 1, 0];

        let (alignment, match_blocks) = smith_waterman_match_blocks(&seq1, &seq2, params);
        assert_eq!(alignment.score, 4);
        assert_eq!(alignment.token_start, 0);
        assert_eq!(alignment.token_end, 5);
        assert_eq!(alignment.query_start, 0);
        assert_eq!(alignment.query_end, 3);
        assert_eq!(alignment.matches, 3);
        assert_eq!(match_blocks, vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn smith_waterman_prefers_more_matches_within_single_optimal_endpoint() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![0, 1, 0];
        let seq2 = vec![0, 2, 1, 1, 0];

        let (alignment, match_blocks) = smith_waterman_match_blocks(&seq1, &seq2, params);
        assert_eq!(alignment.score, 4);
        assert_eq!(alignment.token_start, 0);
        assert_eq!(alignment.token_end, 5);
        assert_eq!(alignment.query_start, 0);
        assert_eq!(alignment.query_end, 3);
        assert_eq!(alignment.matches, 3);
        assert_eq!(match_blocks, vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn fill_matrices_tracks_single_best_endpoint() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![1, 2];
        let seq2 = vec![1, 2, 9, 1, 2, 0];

        let (_, _, max_score, best_end) = fill_matrices(&seq1, &seq2, params);
        assert_eq!(max_score, 4);
        assert_eq!(best_end, Some((2, 2)));
    }

    #[test]
    fn reduced_state_fill_tracks_single_best_endpoint() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![1, 2];
        let seq2 = vec![1, 2, 9, 1, 2, 0];

        let (scores, directions, max_score, best_end) =
            fill_matrices_reduced_state(&seq1, &seq2, params);
        assert_eq!(max_score, 4);
        assert_eq!(best_end, Some((2, 2)));
        assert_eq!(scores.len(), 21);
        assert_eq!(directions.len(), 21);
    }

    #[test]
    fn default_path_matches_detailed_path_without_match_blocks() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![0, 1, 0];
        let seq2 = vec![0, 1, 1, 1, 0];

        let simple = smith_waterman(&seq1, &seq2, params);
        let (detailed, match_blocks) = smith_waterman_match_blocks(&seq1, &seq2, params);

        assert_eq!(simple.score, detailed.score);
        assert_eq!(simple.token_start, detailed.token_start);
        assert_eq!(simple.token_end, detailed.token_end);
        assert_eq!(simple.query_start, detailed.query_start);
        assert_eq!(simple.query_end, detailed.query_end);
        assert_eq!(simple.matches, detailed.matches);
        assert_eq!(match_blocks, vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn align_topk_is_deterministic_and_sorted() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![1, 2];
        let seqs = vec![vec![3, 4], vec![1, 2, 1, 2], vec![1, 2], vec![0, 1, 2, 3]];
        let top = align_topk(&seq1, &seqs, params, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].index, 1);
        assert_eq!(top[1].index, 2);
        assert_eq!(top[2].index, 3);
    }

    #[test]
    fn align_batch_preserves_input_order() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![1, 2, 3];
        let seqs = vec![vec![0, 1, 2, 3, 4], vec![1, 2, 9, 3], vec![8, 9, 10]];

        let batch = align_batch(&seq1, &seqs, params);

        assert_eq!(batch.len(), seqs.len());
        for (alignment, seq2) in batch.iter().zip(seqs.iter()) {
            let expected = smith_waterman(&seq1, seq2, params);
            assert_eq!(alignment.score, expected.score);
            assert_eq!(alignment.token_start, expected.token_start);
            assert_eq!(alignment.token_end, expected.token_end);
            assert_eq!(alignment.query_start, expected.query_start);
            assert_eq!(alignment.query_end, expected.query_end);
            assert_eq!(alignment.matches, expected.matches);
        }
    }

    #[test]
    fn align_batch_with_match_blocks_preserves_input_order() {
        let params = ScoreParams {
            match_score: 2,
            mismatch_score: -1,
            gap_score: -1,
        };
        let seq1 = vec![1, 2, 3, 4];
        let seqs = vec![vec![1, 2, 9, 9, 3, 4], vec![8, 9, 10]];

        let batch = align_batch_with_match_blocks(&seq1, &seqs, params);

        assert_eq!(batch.len(), seqs.len());
        for ((alignment, blocks), seq2) in batch.iter().zip(seqs.iter()) {
            let (expected_alignment, expected_blocks) =
                smith_waterman_match_blocks(&seq1, seq2, params);
            assert_eq!(alignment.score, expected_alignment.score);
            assert_eq!(alignment.token_start, expected_alignment.token_start);
            assert_eq!(alignment.token_end, expected_alignment.token_end);
            assert_eq!(alignment.query_start, expected_alignment.query_start);
            assert_eq!(alignment.query_end, expected_alignment.query_end);
            assert_eq!(alignment.matches, expected_alignment.matches);
            assert_eq!(blocks, &expected_blocks);
        }
    }
}
