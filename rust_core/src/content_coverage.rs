//! Content-word overlap for paraphrase citation gating.
//!
//! Smith-Waterman still localizes evidence. Sequential match count alone drops
//! restatements when shared content words are scattered or reordered.

use std::collections::{HashMap, HashSet};

use crate::term_aliases::canonical_token_id;

/// Match the admission rule used by the Python builder.
pub fn adequate_coverage(answer_coverage: f64, content_coverage: f64, minimum: f64) -> bool {
    content_coverage > 0.0 && (answer_coverage >= minimum || content_coverage >= minimum)
}

pub fn content_token_coverage(
    answer_tokens: &[u32],
    passage_tokens: &[u32],
    stopword_ids: &HashSet<u32>,
    alias_ids: Option<&HashMap<u32, u32>>,
) -> f64 {
    let answer_content: Vec<u32> = answer_tokens
        .iter()
        .copied()
        .filter(|token| !stopword_ids.contains(token))
        .map(|token| canonical_token_id(token, alias_ids))
        .collect();
    let answer_len = answer_content.len();
    if answer_len == 0 {
        return 0.0;
    }

    let mut available: HashMap<u32, usize> = HashMap::new();
    for token in passage_tokens
        .iter()
        .copied()
        .filter(|token| !stopword_ids.contains(token))
        .map(|token| canonical_token_id(token, alias_ids))
    {
        *available.entry(token).or_insert(0) += 1;
    }

    let mut hits = 0usize;
    for token in answer_content {
        if let Some(count) = available.get_mut(&token) {
            if *count > 0 {
                *count -= 1;
                hits += 1;
            }
        }
    }
    hits as f64 / answer_len as f64
}
