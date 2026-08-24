//! Rust-side prepared corpus that keeps all data in Rust memory

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::inverted_index::InvertedIndex;
use crate::smith_waterman;

/// Type alias for candidate metadata tuple
type CandidateMetadata = (usize, usize, usize, Vec<(usize, usize)>);

/// Evidence span for Python
#[pyclass]
#[derive(Clone)]
pub struct PyEvidenceSpan {
    #[pyo3(get)]
    pub char_start: usize,
    #[pyo3(get)]
    pub char_end: usize,
    #[pyo3(get)]
    pub evidence: String,
}

/// Citation result for Python
#[pyclass]
#[derive(Clone)]
pub struct PyCitation {
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub source_id: String,
    #[pyo3(get)]
    pub source_index: usize,
    #[pyo3(get)]
    pub candidate_index: usize,
    #[pyo3(get)]
    pub char_start: usize,
    #[pyo3(get)]
    pub char_end: usize,
    #[pyo3(get)]
    pub evidence: String,
    #[pyo3(get)]
    pub evidence_spans: Vec<PyEvidenceSpan>,
    #[pyo3(get)]
    pub components: HashMap<String, f64>,
}

/// Retrieval support for Python
#[pyclass]
#[derive(Clone)]
pub struct PyRetrievalSupport {
    #[pyo3(get)]
    pub retrieval_score: f64,
    #[pyo3(get)]
    pub source_id: String,
    #[pyo3(get)]
    pub source_index: usize,
    #[pyo3(get)]
    pub candidate_index: usize,
    #[pyo3(get)]
    pub passage_char_start: usize,
    #[pyo3(get)]
    pub passage_char_end: usize,
    #[pyo3(get)]
    pub passage_text: String,
    #[pyo3(get)]
    pub embedding_score: f64,
    #[pyo3(get)]
    pub lexical_score: f64,
}

/// Citation building result
#[pyclass]
pub struct PyCitationResult {
    #[pyo3(get)]
    pub citations: Vec<PyCitation>,
    #[pyo3(get)]
    pub supports: Vec<PyRetrievalSupport>,
    #[pyo3(get)]
    pub num_alignments: usize,
}

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

    /// Build citations directly from PreparedCorpus without Python marshalling
    /// This eliminates the overhead of copying full source texts to Python and back
    #[allow(clippy::too_many_arguments)]
    pub fn build_citations(
        &self,
        py: Python<'_>,
        answer_tokens: Vec<u32>,
        candidate_indices: Vec<usize>,
        lexical_scores: Vec<f64>,
        embed_scores: Vec<f64>,
        source_id_map: HashMap<usize, String>, // source_index -> source_id
        base_offset_map: HashMap<usize, usize>, // source_index -> base_offset
        config_tuple: (i32, f64, f64, bool, i32, f64, f64, f64, f64, f64),
        multi_span_config: (bool, i32, usize),
        match_score: i32,
        mismatch_score: i32,
        gap_score: i32,
    ) -> PyResult<PyCitationResult> {
        py.detach(|| {
            use rayon::prelude::*;

            let (
                min_alignment_score,
                min_answer_coverage,
                min_final_score,
                require_all_tokens,
                _match_score_cfg,
                weight_alignment,
                weight_answer_coverage,
                weight_evidence_coverage,
                weight_lexical,
                weight_embedding,
            ) = config_tuple;

            let (multi_span_evidence, multi_span_merge_gap_chars, multi_span_max_spans) =
                multi_span_config;

            // Align answer_tokens against each candidate in parallel
            let params = smith_waterman::ScoreParams {
                match_score,
                mismatch_score,
                gap_score,
            };

            // Use match_blocks only when multi_span_evidence is enabled
            // This matches v8 behavior where plain SW was used for single-span mode
            let alignments: Vec<_> = candidate_indices
                .par_iter()
                .map(|&idx| {
                    if let Some(candidate) = self.candidates.get(idx) {
                        if multi_span_evidence {
                            let (alignment, match_blocks) =
                                smith_waterman::smith_waterman_match_blocks(
                                    &answer_tokens,
                                    &candidate.token_ids,
                                    params,
                                );
                            Some((alignment, Some(match_blocks)))
                        } else {
                            let alignment = smith_waterman::smith_waterman(
                                &answer_tokens,
                                &candidate.token_ids,
                                params,
                            );
                            Some((alignment, None))
                        }
                    } else {
                        None
                    }
                })
                .collect();

            let num_alignments = alignments.iter().filter(|a| a.is_some()).count();

            // Build citations from alignments
            let results: Vec<_> = candidate_indices
                .iter()
                .zip(&alignments)
                .zip(&lexical_scores)
                .zip(&embed_scores)
                .filter_map(|(((&cand_idx, alignment), &lex), &emb)| {
                    alignment.as_ref().and_then(|(align, match_blocks_opt)| {
                        self.process_alignment_internal(
                            &answer_tokens,
                            cand_idx,
                            align,
                            match_blocks_opt.as_deref(),
                            lex,
                            emb,
                            &source_id_map,
                            &base_offset_map,
                            min_alignment_score,
                            min_answer_coverage,
                            min_final_score,
                            require_all_tokens,
                            match_score,
                            weight_alignment,
                            weight_answer_coverage,
                            weight_evidence_coverage,
                            weight_lexical,
                            weight_embedding,
                            multi_span_evidence,
                            multi_span_merge_gap_chars,
                            multi_span_max_spans,
                        )
                    })
                })
                .collect();

            let (citations, supports): (Vec<_>, Vec<_>) = results
                .into_iter()
                .partition(|r| matches!(r, CitationOrSupport::Citation(_)));

            let citations: Vec<PyCitation> = citations
                .into_iter()
                .filter_map(|r| match r {
                    CitationOrSupport::Citation(c) => Some(c),
                    _ => None,
                })
                .collect();

            let supports: Vec<PyRetrievalSupport> = supports
                .into_iter()
                .filter_map(|r| match r {
                    CitationOrSupport::Support(s) => Some(s),
                    _ => None,
                })
                .collect();

            Ok(PyCitationResult {
                citations,
                supports,
                num_alignments,
            })
        })
    }
}

enum CitationOrSupport {
    Citation(PyCitation),
    Support(PyRetrievalSupport),
}

impl PreparedCorpus {
    #[allow(clippy::too_many_arguments)]
    fn process_alignment_internal(
        &self,
        answer_tokens: &[u32],
        candidate_index: usize,
        alignment: &smith_waterman::Alignment,
        match_blocks: Option<&[(usize, usize)]>,
        lexical_score: f64,
        embed_score: f64,
        source_id_map: &HashMap<usize, String>,
        base_offset_map: &HashMap<usize, usize>,
        min_alignment_score: i32,
        min_answer_coverage: f64,
        min_final_score: f64,
        require_all_tokens: bool,
        match_score: i32,
        weight_alignment: f64,
        weight_answer_coverage: f64,
        weight_evidence_coverage: f64,
        weight_lexical: f64,
        weight_embedding: f64,
        multi_span_evidence: bool,
        multi_span_merge_gap_chars: i32,
        multi_span_max_spans: usize,
    ) -> Option<CitationOrSupport> {
        let candidate = self.candidates.get(candidate_index)?;
        let source_text = self.source_texts.get(candidate.source_index)?;
        let source_id = source_id_map
            .get(&candidate.source_index)
            .cloned()
            .unwrap_or_else(|| candidate.source_index.to_string());
        let base_offset = base_offset_map
            .get(&candidate.source_index)
            .copied()
            .unwrap_or(0);

        let answer_len = answer_tokens.len();
        let evidence_len = (alignment.token_end - alignment.token_start).max(1);

        let answer_coverage = alignment.matches as f64 / answer_len.max(1) as f64;
        let evidence_coverage = alignment.matches as f64 / evidence_len as f64;
        let normalized_alignment =
            alignment.score as f64 / (match_score as f64 * answer_len.max(1) as f64);

        // Check thresholds
        if alignment.score < min_alignment_score
            || alignment.token_start >= alignment.token_end
            || answer_coverage < min_answer_coverage
        {
            return self.build_support_internal(
                candidate,
                candidate_index,
                source_text,
                &source_id,
                base_offset,
                normalized_alignment,
                answer_coverage,
                evidence_coverage,
                lexical_score,
                embed_score,
                weight_alignment,
                weight_answer_coverage,
                weight_evidence_coverage,
                weight_lexical,
                weight_embedding,
            );
        }

        let final_score = weight_alignment * normalized_alignment
            + weight_answer_coverage * answer_coverage
            + weight_evidence_coverage * evidence_coverage
            + weight_lexical * lexical_score
            + weight_embedding * embed_score.max(0.0);

        if final_score < min_final_score {
            return None;
        }

        // Check all tokens requirement
        if require_all_tokens && alignment.matches != answer_len {
            let evidence_tokens = &candidate.token_ids[alignment.token_start..alignment.token_end];
            if !tokens_match(answer_tokens, evidence_tokens) {
                return self.build_support_internal(
                    candidate,
                    candidate_index,
                    source_text,
                    &source_id,
                    base_offset,
                    normalized_alignment,
                    answer_coverage,
                    evidence_coverage,
                    lexical_score,
                    embed_score,
                    weight_alignment,
                    weight_answer_coverage,
                    weight_evidence_coverage,
                    weight_lexical,
                    weight_embedding,
                );
            }
        }

        // Extract evidence spans
        let evidence_spans = self.extract_evidence_spans_internal(
            candidate,
            source_text,
            base_offset,
            alignment,
            match_blocks,
            multi_span_evidence,
            multi_span_merge_gap_chars,
            multi_span_max_spans,
        )?;

        if evidence_spans.is_empty() {
            return None;
        }

        // Build full evidence text spanning from first to last span
        // This matches v8 behavior: slice from first span start to last span end
        let char_start = evidence_spans.first().unwrap().char_start;
        let char_end = evidence_spans.last().unwrap().char_end;

        // Convert absolute positions to passage-relative
        let passage_rel_start = char_start - base_offset - candidate.passage_start;
        let passage_rel_end = char_end - base_offset - candidate.passage_start;

        let evidence = source_text
            .get(
                (candidate.passage_start + passage_rel_start)
                    ..(candidate.passage_start + passage_rel_end),
            )
            .unwrap_or("")
            .to_string();

        // Match v8 component keys and values
        let mut components = HashMap::new();
        components.insert("alignment_score".to_string(), alignment.score as f64);
        components.insert("normalized_alignment".to_string(), normalized_alignment);
        components.insert("matches".to_string(), alignment.matches as f64);
        components.insert("answer_coverage".to_string(), answer_coverage);
        components.insert("evidence_coverage".to_string(), evidence_coverage);
        components.insert("lexical_score".to_string(), lexical_score);
        components.insert("embedding_score".to_string(), embed_score.max(0.0));
        components.insert(
            "num_evidence_spans".to_string(),
            evidence_spans.len() as f64,
        );
        components.insert(
            "passage_char_start".to_string(),
            (base_offset + candidate.passage_start) as f64,
        );
        components.insert(
            "passage_char_end".to_string(),
            (base_offset + candidate.passage_end) as f64,
        );

        Some(CitationOrSupport::Citation(PyCitation {
            score: final_score,
            source_id,
            source_index: candidate.source_index,
            candidate_index,
            char_start,
            char_end,
            evidence,
            evidence_spans,
            components,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_support_internal(
        &self,
        candidate: &Candidate,
        candidate_index: usize,
        source_text: &str,
        source_id: &str,
        base_offset: usize,
        normalized_alignment: f64,
        answer_coverage: f64,
        evidence_coverage: f64,
        lexical_score: f64,
        embed_score: f64,
        weight_alignment: f64,
        weight_answer_coverage: f64,
        weight_evidence_coverage: f64,
        weight_lexical: f64,
        weight_embedding: f64,
    ) -> Option<CitationOrSupport> {
        // Match v8 behavior: filter out low-quality supports
        if lexical_score <= 0.0 && embed_score < 0.3 {
            return None;
        }

        let retrieval_score = weight_alignment * normalized_alignment
            + weight_answer_coverage * answer_coverage
            + weight_evidence_coverage * evidence_coverage
            + weight_lexical * lexical_score
            + weight_embedding * embed_score.max(0.0);

        let passage_start_abs = base_offset + candidate.passage_start;
        let passage_end_abs = base_offset + candidate.passage_end;
        let passage_text = source_text
            .get(candidate.passage_start..candidate.passage_end)
            .unwrap_or("")
            .to_string();

        Some(CitationOrSupport::Support(PyRetrievalSupport {
            retrieval_score,
            source_id: source_id.to_string(),
            source_index: candidate.source_index,
            candidate_index,
            passage_char_start: passage_start_abs,
            passage_char_end: passage_end_abs,
            passage_text,
            embedding_score: embed_score.max(0.0),
            lexical_score,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_evidence_spans_internal(
        &self,
        candidate: &Candidate,
        source_text: &str,
        base_offset: usize,
        alignment: &smith_waterman::Alignment,
        match_blocks: Option<&[(usize, usize)]>,
        multi_span_evidence: bool,
        merge_gap_chars: i32,
        max_spans: usize,
    ) -> Option<Vec<PyEvidenceSpan>> {
        if !multi_span_evidence {
            // Single span mode - use alignment token range directly
            if alignment.token_start >= candidate.token_spans.len()
                || alignment.token_end > candidate.token_spans.len()
            {
                return None;
            }

            let (char_start_rel, _) = candidate.token_spans[alignment.token_start];
            let (_, char_end_rel) = candidate.token_spans[alignment.token_end - 1];

            let char_start_abs = base_offset + candidate.passage_start + char_start_rel;
            let char_end_abs = base_offset + candidate.passage_start + char_end_rel;

            let evidence = source_text
                .get(
                    (candidate.passage_start + char_start_rel)
                        ..(candidate.passage_start + char_end_rel),
                )
                .unwrap_or("")
                .to_string();

            return Some(vec![PyEvidenceSpan {
                char_start: char_start_abs,
                char_end: char_end_abs,
                evidence,
            }]);
        }

        // Multi-span mode: merge consecutive match blocks
        let blocks = match_blocks?;
        let mut spans = Vec::new();
        let mut current_start: Option<usize> = None;
        let mut current_end: Option<usize> = None;

        for &(token_start, token_end) in blocks {
            if token_start >= candidate.token_spans.len() || token_end > candidate.token_spans.len()
            {
                continue;
            }

            let (span_start, _) = candidate.token_spans[token_start];
            let (_, span_end) = candidate.token_spans[token_end - 1];

            match (current_start, current_end) {
                (Some(cs), Some(ce)) => {
                    let gap = span_start.saturating_sub(ce);
                    if gap <= merge_gap_chars as usize {
                        current_end = Some(span_end);
                    } else {
                        // Finalize current span
                        let char_start_abs = base_offset + candidate.passage_start + cs;
                        let char_end_abs = base_offset + candidate.passage_start + ce;
                        let evidence = source_text
                            .get((candidate.passage_start + cs)..(candidate.passage_start + ce))
                            .unwrap_or("")
                            .to_string();
                        spans.push(PyEvidenceSpan {
                            char_start: char_start_abs,
                            char_end: char_end_abs,
                            evidence,
                        });

                        current_start = Some(span_start);
                        current_end = Some(span_end);
                    }
                }
                _ => {
                    current_start = Some(span_start);
                    current_end = Some(span_end);
                }
            }
        }

        // Finalize last span
        if let (Some(cs), Some(ce)) = (current_start, current_end) {
            let char_start_abs = base_offset + candidate.passage_start + cs;
            let char_end_abs = base_offset + candidate.passage_start + ce;
            let evidence = source_text
                .get((candidate.passage_start + cs)..(candidate.passage_start + ce))
                .unwrap_or("")
                .to_string();
            spans.push(PyEvidenceSpan {
                char_start: char_start_abs,
                char_end: char_end_abs,
                evidence,
            });
        }

        if spans.len() > max_spans {
            spans.truncate(max_spans);
        }

        if spans.is_empty() {
            None
        } else {
            Some(spans)
        }
    }
}

fn tokens_match(answer_tokens: &[u32], evidence_tokens: &[u32]) -> bool {
    // Match v8 behavior: check if all answer tokens are in evidence (with multiplicity)
    if answer_tokens == evidence_tokens {
        return true;
    }

    let mut counts: HashMap<u32, isize> = HashMap::new();
    for &token in answer_tokens {
        *counts.entry(token).or_insert(0) += 1;
    }
    for &token in evidence_tokens {
        if let Some(count) = counts.get_mut(&token) {
            *count -= 1;
            if *count == 0 {
                counts.remove(&token);
            }
        }
    }
    counts.is_empty()
}
