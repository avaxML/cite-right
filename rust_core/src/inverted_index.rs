//! Inverted index for fast token-level retrieval
//!
//! Maps token IDs to posting lists containing candidate and position information.
//! Used to quickly find seed candidates for Smith-Waterman alignment.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

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

/// Inverted index mapping tokens to postings (kept in Rust, opaque to Python)
#[pyclass]
pub struct InvertedIndex {
    /// Map from token_id to list of postings
    index: HashMap<u32, Vec<Posting>>,
}

#[pymethods]
impl InvertedIndex {
    /// Query index with conjunctive (AND) query using rarest tokens
    /// Falls back to small union if intersection is empty
    pub fn query(
        &self,
        py: Python<'_>,
        query_tokens: Vec<u32>,
        max_candidates: usize,
    ) -> Vec<usize> {
        py.detach(|| self.query_internal(&query_tokens, max_candidates))
    }

    /// Get posting count for a token (for IDF-based sorting)
    pub fn get_posting_count(&self, token_id: u32) -> usize {
        self.index.get(&token_id).map(|v| v.len()).unwrap_or(0)
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
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
        self.index
            .get(&token_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Internal query implementation (called with GIL released)
    fn query_internal(&self, query_tokens: &[u32], max_candidates: usize) -> Vec<usize> {
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Sort query tokens by rarity (ascending posting count)
        let mut token_counts: Vec<(u32, usize)> = query_tokens
            .iter()
            .map(|&token_id| {
                let count = self.index.get(&token_id).map(|v| v.len()).unwrap_or(0);
                (token_id, count)
            })
            .collect();
        token_counts.sort_by_key(|(_, count)| *count);

        // Filter to tokens that exist in the index
        let existing_tokens: Vec<(u32, usize)> = token_counts
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .collect();

        if existing_tokens.is_empty() {
            return Vec::new();
        }

        // Start with rarest token candidates
        let mut candidate_scores: HashMap<usize, usize> = HashMap::new();
        for posting in self.get_postings(existing_tokens[0].0) {
            candidate_scores.insert(posting.candidate_index, 1);
        }

        // Try intersecting with second rarest if we have multiple tokens
        if existing_tokens.len() > 1 {
            let mut intersected: HashMap<usize, usize> = HashMap::new();
            let second_candidates: HashSet<usize> = self
                .get_postings(existing_tokens[1].0)
                .iter()
                .map(|p| p.candidate_index)
                .collect();

            for (&cand_idx, &score) in &candidate_scores {
                if second_candidates.contains(&cand_idx) {
                    intersected.insert(cand_idx, score + 1);
                }
            }

            // Accept smaller intersections (≥4 instead of ≥8) for better recall
            if intersected.len() >= 4 || existing_tokens.len() == 2 {
                candidate_scores = intersected;
            }
            // Otherwise keep the rarest token candidates and OR-expand
        }

        // Expand with more tokens if we have too few candidates
        if candidate_scores.len() < 24 && existing_tokens.len() > 2 {
            // Add candidates from additional rare tokens (up to 4 more)
            for &(token_id, _) in existing_tokens.iter().skip(2).take(4) {
                for posting in self.get_postings(token_id) {
                    *candidate_scores.entry(posting.candidate_index).or_insert(0) += 1;
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
