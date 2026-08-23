//! Inverted index for fast token-level retrieval
//!
//! Maps token IDs to posting lists containing candidate and position information.
//! Used to quickly find seed candidates for Smith-Waterman alignment.

use std::collections::HashMap;

/// A single posting in the inverted index
#[derive(Clone, Debug)]
pub struct Posting {
    /// Global candidate index
    pub candidate_index: usize,
    /// Token position within the candidate
    pub token_pos: usize,
    /// Character start in passage
    pub char_start: usize,
    /// Character end in passage
    pub char_end: usize,
}

/// Inverted index mapping tokens to postings
pub struct InvertedIndex {
    /// Map from token_id to list of postings
    index: HashMap<u32, Vec<Posting>>,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Add a posting for a token
    pub fn add_posting(&mut self, token_id: u32, posting: Posting) {
        self.index.entry(token_id).or_default().push(posting);
    }

    /// Get postings for a token
    pub fn get_postings(&self, token_id: u32) -> &[Posting] {
        self.index.get(&token_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Find candidate indices that contain any of the query tokens
    pub fn find_seed_candidates(&self, query_tokens: &[u32], max_candidates: usize) -> Vec<usize> {
        let mut candidate_scores: HashMap<usize, usize> = HashMap::new();

        // Count how many query tokens each candidate contains
        for &token_id in query_tokens {
            if let Some(postings) = self.index.get(&token_id) {
                for posting in postings {
                    *candidate_scores
                        .entry(posting.candidate_index)
                        .or_insert(0) += 1;
                }
            }
        }

        // Sort by score (descending), then by candidate index for determinism
        let mut scored: Vec<(usize, usize)> = candidate_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Return top candidates
        scored
            .into_iter()
            .take(max_candidates)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get size statistics for debugging
    #[allow(dead_code)]
    pub fn stats(&self) -> IndexStats {
        let num_tokens = self.index.len();
        let total_postings: usize = self.index.values().map(|v| v.len()).sum();
        let max_postings = self.index.values().map(|v| v.len()).max().unwrap_or(0);

        IndexStats {
            num_tokens,
            total_postings,
            max_postings,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct IndexStats {
    pub num_tokens: usize,
    pub total_postings: usize,
    pub max_postings: usize,
}
