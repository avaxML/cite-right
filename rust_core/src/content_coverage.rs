//! Content-word overlap for paraphrase citation gating.
//!
//! Smith-Waterman still localizes evidence. Sequential match count alone drops
//! restatements when shared content words are scattered or reordered.

use std::collections::{HashMap, HashSet};

/// Function words only. Polarity markers (not, no, never) stay as content.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "of", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "what", "which", "who", "whom",
];

pub fn stopword_token_ids(vocab: &HashMap<String, u32>) -> HashSet<u32> {
    STOPWORDS
        .iter()
        .filter_map(|word| vocab.get(*word).copied())
        .collect()
}

pub fn content_token_coverage(
    answer_tokens: &[u32],
    passage_tokens: &[u32],
    stopword_ids: &HashSet<u32>,
) -> f64 {
    let answer_content: Vec<u32> = answer_tokens
        .iter()
        .copied()
        .filter(|token| !stopword_ids.contains(token))
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
