use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, Copy)]
pub struct ScoreParams {
    pub match_score: i32,
    pub mismatch_score: i32,
    pub gap_score: i32,
}

#[derive(Clone, Copy, Debug)]
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

    let rows = seq1.len() + 1;
    let cols = seq2.len() + 1;
    let mut scores = vec![vec![0i32; cols]; rows];
    let mut directions = vec![vec![0u8; cols]; rows];

    let mut max_score = 0i32;
    let mut max_positions: Vec<(usize, usize)> = Vec::new();

    for i in 1..rows {
        for j in 1..cols {
            let match_score = if seq1[i - 1] == seq2[j - 1] {
                params.match_score
            } else {
                params.mismatch_score
            };
            let score_diag = scores[i - 1][j - 1] + match_score;
            let score_up = scores[i - 1][j] + params.gap_score;
            let score_left = scores[i][j - 1] + params.gap_score;

            let best = 0i32.max(score_diag).max(score_up).max(score_left);
            if best <= 0 {
                scores[i][j] = 0;
                directions[i][j] = 0;
            } else {
                scores[i][j] = best;
                directions[i][j] = choose_direction(best, score_diag, score_up, score_left);
            }

            if scores[i][j] > max_score {
                max_score = scores[i][j];
                max_positions.clear();
                if max_score > 0 {
                    max_positions.push((i, j));
                }
            } else if scores[i][j] == max_score && scores[i][j] > 0 {
                max_positions.push((i, j));
            }
        }
    }

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

    let mut best: Option<Alignment> = None;
    for (i_end, j_end) in max_positions {
        let (i_start, j_start, matches) =
            traceback_details(i_end, j_end, &directions, &scores, seq1, seq2);
        let candidate = Alignment {
            score: max_score,
            query_start: i_start,
            query_end: i_end,
            token_start: j_start,
            token_end: j_end,
            matches,
        };
        if best.is_none() {
            best = Some(candidate);
            continue;
        }
        if cmp_alignment(&candidate, &best.unwrap()) == Ordering::Less {
            best = Some(candidate);
        }
    }

    best.expect("max_positions is non-empty when max_score > 0")
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

    let rows = seq1.len() + 1;
    let cols = seq2.len() + 1;
    let mut scores = vec![vec![0i32; cols]; rows];
    let mut directions = vec![vec![0u8; cols]; rows];

    let mut max_score = 0i32;
    let mut max_positions: Vec<(usize, usize)> = Vec::new();

    for i in 1..rows {
        for j in 1..cols {
            let match_score = if seq1[i - 1] == seq2[j - 1] {
                params.match_score
            } else {
                params.mismatch_score
            };
            let score_diag = scores[i - 1][j - 1] + match_score;
            let score_up = scores[i - 1][j] + params.gap_score;
            let score_left = scores[i][j - 1] + params.gap_score;

            let best = 0i32.max(score_diag).max(score_up).max(score_left);
            if best <= 0 {
                scores[i][j] = 0;
                directions[i][j] = 0;
            } else {
                scores[i][j] = best;
                directions[i][j] = choose_direction(best, score_diag, score_up, score_left);
            }

            if scores[i][j] > max_score {
                max_score = scores[i][j];
                max_positions.clear();
                if max_score > 0 {
                    max_positions.push((i, j));
                }
            } else if scores[i][j] == max_score && scores[i][j] > 0 {
                max_positions.push((i, j));
            }
        }
    }

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

    let mut best: Option<(Alignment, Vec<(usize, usize)>)> = None;
    for (i_end, j_end) in max_positions {
        let (i_start, j_start, matches, match_blocks) =
            traceback_details_with_match_blocks(i_end, j_end, &directions, &scores, seq1, seq2);
        let candidate = Alignment {
            score: max_score,
            query_start: i_start,
            query_end: i_end,
            token_start: j_start,
            token_end: j_end,
            matches,
        };
        match best.as_ref() {
            None => {
                best = Some((candidate, match_blocks));
            }
            Some((best_alignment, _)) => {
                if cmp_alignment(&candidate, best_alignment) == Ordering::Less {
                    best = Some((candidate, match_blocks));
                }
            }
        }
    }

    best.expect("max_positions is non-empty when max_score > 0")
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

pub fn align_best(
    seq1: &[u32],
    seqs: &[Vec<u32>],
    params: ScoreParams,
) -> Option<CandidateAlignment> {
    align_topk(seq1, seqs, params, 1).into_iter().next()
}

fn choose_direction(best: i32, score_diag: i32, score_up: i32, _score_left: i32) -> u8 {
    if best == score_diag {
        return 1;
    }
    if best == score_up {
        return 2;
    }
    3
}

fn traceback_details(
    mut i: usize,
    mut j: usize,
    directions: &[Vec<u8>],
    scores: &[Vec<i32>],
    seq1: &[u32],
    seq2: &[u32],
) -> (usize, usize, usize) {
    let mut matches = 0usize;
    while i > 0 && j > 0 && directions[i][j] != 0 && scores[i][j] > 0 {
        match directions[i][j] {
            1 => {
                if seq1[i - 1] == seq2[j - 1] {
                    matches += 1;
                }
                i -= 1;
                j -= 1;
            }
            2 => {
                i -= 1;
            }
            _ => {
                j -= 1;
            }
        }
    }
    (i, j, matches)
}

fn traceback_details_with_match_blocks(
    mut i: usize,
    mut j: usize,
    directions: &[Vec<u8>],
    scores: &[Vec<i32>],
    seq1: &[u32],
    seq2: &[u32],
) -> (usize, usize, usize, Vec<(usize, usize)>) {
    let mut matches = 0usize;
    let mut match_positions: Vec<usize> = Vec::new();

    while i > 0 && j > 0 && directions[i][j] != 0 && scores[i][j] > 0 {
        match directions[i][j] {
            1 => {
                i -= 1;
                j -= 1;
                if seq1[i] == seq2[j] {
                    matches += 1;
                    match_positions.push(j);
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

    if match_positions.is_empty() {
        return (i, j, matches, Vec::new());
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

    (i, j, matches, blocks)
}

fn cmp_alignment(left: &Alignment, right: &Alignment) -> Ordering {
    if left.score != right.score {
        return right.score.cmp(&left.score);
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
    if left.token_end != right.token_end {
        return left.token_end.cmp(&right.token_end);
    }
    left.query_end.cmp(&right.query_end)
}

fn cmp_candidate(left: &CandidateAlignment, right: &CandidateAlignment) -> Ordering {
    if left.score != right.score {
        return right.score.cmp(&left.score);
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
}
