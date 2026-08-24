//! Rust-side prepared corpus that keeps all data in Rust memory

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::inverted_index::InvertedIndex;

/// Type alias for candidate metadata tuple
type CandidateMetadata = (usize, usize, usize, Vec<(usize, usize)>);

/// A single candidate passage
#[derive(Clone)]
pub struct Candidate {
    pub source_index: usize,
    pub passage_start: usize,
    pub passage_end: usize,
    pub token_ids: Vec<u32>,
    pub token_spans: Vec<(usize, usize)>,
}

/// Prepared corpus kept in Rust (opaque to Python)
#[pyclass]
pub struct PreparedCorpus {
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) idf: HashMap<u32, f64>,
    pub(crate) vocab: HashMap<String, u32>,
    pub(crate) inverted_index: InvertedIndex,
    pub(crate) source_texts: Vec<String>,
}

#[pymethods]
impl PreparedCorpus {
    /// Get the number of candidates
    pub fn num_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Get token_ids for specific candidate indices (for alignment)
    pub fn get_candidate_tokens(
        &self,
        py: Python<'_>,
        candidate_indices: Vec<usize>,
    ) -> Vec<Vec<u32>> {
        py.detach(|| {
            candidate_indices
                .iter()
                .filter_map(|&idx| self.candidates.get(idx).map(|c| c.token_ids.clone()))
                .collect()
        })
    }

    /// Get candidate metadata (source_index, passage_start, passage_end, token_spans)
    /// for specific indices
    pub fn get_candidate_metadata(
        &self,
        py: Python<'_>,
        candidate_indices: Vec<usize>,
    ) -> Vec<CandidateMetadata> {
        py.detach(|| {
            candidate_indices
                .iter()
                .filter_map(|&idx| {
                    self.candidates.get(idx).map(|c| {
                        (
                            c.source_index,
                            c.passage_start,
                            c.passage_end,
                            c.token_spans.clone(),
                        )
                    })
                })
                .collect()
        })
    }

    /// Query inverted index for seed candidates
    pub fn query_index(
        &self,
        py: Python<'_>,
        query_tokens: Vec<u32>,
        max_candidates: usize,
    ) -> Vec<usize> {
        self.inverted_index.query(py, query_tokens, max_candidates)
    }

    /// Get IDF weights
    pub fn get_idf(&self) -> Vec<(u32, f64)> {
        self.idf.iter().map(|(&k, &v)| (k, v)).collect()
    }

    /// Get vocabulary
    pub fn get_vocab(&self) -> Vec<(String, u32)> {
        self.vocab.iter().map(|(k, &v)| (k.clone(), v)).collect()
    }

    /// Get source text by index
    pub fn get_source_text(&self, source_index: usize) -> Option<String> {
        self.source_texts.get(source_index).cloned()
    }

    /// Get all candidates for a specific source (for passage generation in Python)
    pub fn get_source_candidates(&self, source_index: usize) -> Vec<(usize, usize, usize)> {
        self.candidates
            .iter()
            .enumerate()
            .filter_map(|(idx, c)| {
                if c.source_index == source_index {
                    Some((idx, c.passage_start, c.passage_end))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get minimal candidate info for all candidates (without token data)
    /// Returns list of (global_idx, source_idx, passage_start, passage_end)
    pub fn get_all_candidate_info(&self) -> Vec<(usize, usize, usize, usize)> {
        self.candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, c.source_index, c.passage_start, c.passage_end))
            .collect()
    }
}
