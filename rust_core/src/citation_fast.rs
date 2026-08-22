//! Fast citation building using JSON for Python interop

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct CitationResult {
    pub score: f64,
    pub source_id: String,
    pub source_index: usize,
    pub candidate_index: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub evidence: String,
    pub evidence_spans: Vec<EvidenceSpan>,
    pub components: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvidenceSpan {
    pub char_start: usize,
    pub char_end: usize,
    pub evidence: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RetrievalSupport {
    pub retrieval_score: f64,
    pub source_id: String,
    pub source_index: usize,
    pub candidate_index: usize,
    pub passage_char_start: usize,
    pub passage_char_end: usize,
    pub passage_text: String,
    pub embedding_score: f64,
    pub lexical_score: f64,
}

// Lightweight config for citation building
pub struct Config {
    pub min_alignment_score: i32,
    pub min_answer_coverage: f64,
    pub min_final_score: f64,
    pub require_all_tokens: bool,
    pub match_score: i32,
    pub weight_alignment: f64,
    pub weight_answer_coverage: f64,
    pub weight_evidence_coverage: f64,
    pub weight_lexical: f64,
    pub weight_embedding: f64,
}

// Candidate metadata needed for citation building
pub struct Candidate {
    pub index: usize,
    pub source_id: String,
    pub source_index: usize,
    pub source_text: String,
    pub source_full_text: Option<String>,
    pub base_offset: usize,
    pub passage_start: usize,
    pub passage_end: usize,
    pub token_ids: Vec<u32>,
    pub token_spans: Vec<(usize, usize)>,
}

pub struct Alignment {
    pub score: i32,
    pub token_start: usize,
    pub token_end: usize,
    pub query_start: usize,
    pub query_end: usize,
    pub matches: usize,
}

// Build citations from alignments
pub fn build_citations_json(
    answer_tokens: &[u32],
    candidates: Vec<Candidate>,
    candidate_indices: &[usize],
    alignments: Vec<Alignment>,
    lexical_scores: &[f64],
    embed_scores: &[f64],
    cfg: Config,
) -> String {
    let results: Vec<_> = candidate_indices
        .par_iter()
        .zip(&alignments)
        .zip(lexical_scores)
        .zip(embed_scores)
        .filter_map(|(((&cand_idx, alignment), &lex), &emb)| {
            process_alignment(
                answer_tokens,
                &candidates[cand_idx],
                alignment,
                lex,
                emb,
                &cfg,
            )
        })
        .collect();

    let (citations, supports): (Vec<_>, Vec<_>) = results
        .into_iter()
        .partition(|r| matches!(r, Either::Citation(_)));

    let citations: Vec<CitationResult> = citations
        .into_iter()
        .filter_map(|r| match r {
            Either::Citation(c) => Some(c),
            _ => None,
        })
        .collect();

    let supports: Vec<RetrievalSupport> = supports
        .into_iter()
        .filter_map(|r| match r {
            Either::Support(s) => Some(s),
            _ => None,
        })
        .collect();

    serde_json::json!({
        "citations": citations,
        "supports": supports,
    })
    .to_string()
}

enum Either {
    Citation(CitationResult),
    Support(RetrievalSupport),
}

fn process_alignment(
    answer_tokens: &[u32],
    candidate: &Candidate,
    alignment: &Alignment,
    lexical_score: f64,
    embed_score: f64,
    cfg: &Config,
) -> Option<Either> {
    let answer_len = answer_tokens.len();
    let evidence_len = (alignment.token_end - alignment.token_start).max(1);

    let answer_coverage = alignment.matches as f64 / answer_len.max(1) as f64;
    let evidence_coverage = alignment.matches as f64 / evidence_len as f64;
    let normalized_alignment =
        alignment.score as f64 / (cfg.match_score as f64 * answer_len.max(1) as f64);

    // Check thresholds
    if alignment.score < cfg.min_alignment_score
        || alignment.token_start >= alignment.token_end
        || answer_coverage < cfg.min_answer_coverage
    {
        return build_support(
            candidate,
            normalized_alignment,
            answer_coverage,
            evidence_coverage,
            lexical_score,
            embed_score,
            cfg,
        );
    }

    let final_score = cfg.weight_alignment * normalized_alignment
        + cfg.weight_answer_coverage * answer_coverage
        + cfg.weight_evidence_coverage * evidence_coverage
        + cfg.weight_lexical * lexical_score
        + cfg.weight_embedding * embed_score.max(0.0);

    if final_score < cfg.min_final_score {
        return None;
    }

    // Check all tokens requirement
    if cfg.require_all_tokens && alignment.matches != answer_len {
        let evidence_tokens = &candidate.token_ids[alignment.token_start..alignment.token_end];
        if !tokens_match(answer_tokens, evidence_tokens) {
            return build_support(
                candidate,
                normalized_alignment,
                answer_coverage,
                evidence_coverage,
                lexical_score,
                embed_score,
                cfg,
            );
        }
    }

    // Extract evidence
    let (char_start, char_end) = token_span_to_char_span(
        &candidate.token_spans,
        alignment.token_start,
        alignment.token_end,
    )?;

    let abs_start = candidate.base_offset + candidate.passage_start + char_start;
    let abs_end = candidate.base_offset + candidate.passage_start + char_end;

    if abs_start >= abs_end {
        return None;
    }

    let evidence = slice_source(candidate, abs_start, abs_end);

    let evidence_span = EvidenceSpan {
        char_start: abs_start,
        char_end: abs_end,
        evidence: evidence.clone(),
    };

    let mut components = HashMap::new();
    components.insert("alignment_score".to_string(), alignment.score as f64);
    components.insert("normalized_alignment".to_string(), normalized_alignment);
    components.insert("matches".to_string(), alignment.matches as f64);
    components.insert("answer_coverage".to_string(), answer_coverage);
    components.insert("evidence_coverage".to_string(), evidence_coverage);
    components.insert("lexical_score".to_string(), lexical_score);
    components.insert("embedding_score".to_string(), embed_score);
    components.insert("num_evidence_spans".to_string(), 1.0);
    components.insert(
        "evidence_chars_total".to_string(),
        (abs_end - abs_start) as f64,
    );
    components.insert(
        "passage_char_start".to_string(),
        (candidate.base_offset + candidate.passage_start) as f64,
    );
    components.insert(
        "passage_char_end".to_string(),
        (candidate.base_offset + candidate.passage_end) as f64,
    );

    Some(Either::Citation(CitationResult {
        score: final_score,
        source_id: candidate.source_id.clone(),
        source_index: candidate.source_index,
        candidate_index: candidate.index,
        char_start: abs_start,
        char_end: abs_end,
        evidence,
        evidence_spans: vec![evidence_span],
        components,
    }))
}

fn build_support(
    candidate: &Candidate,
    normalized_alignment: f64,
    answer_coverage: f64,
    evidence_coverage: f64,
    lexical_score: f64,
    embed_score: f64,
    cfg: &Config,
) -> Option<Either> {
    if lexical_score <= 0.0 && embed_score < 0.3 {
        return None;
    }

    let retrieval_score = cfg.weight_alignment * normalized_alignment
        + cfg.weight_answer_coverage * answer_coverage
        + cfg.weight_evidence_coverage * evidence_coverage
        + cfg.weight_lexical * lexical_score
        + cfg.weight_embedding * embed_score.max(0.0);

    let abs_start = candidate.base_offset + candidate.passage_start;
    let abs_end = candidate.base_offset + candidate.passage_end;

    Some(Either::Support(RetrievalSupport {
        retrieval_score,
        source_id: candidate.source_id.clone(),
        source_index: candidate.source_index,
        candidate_index: candidate.index,
        passage_char_start: abs_start,
        passage_char_end: abs_end,
        passage_text: slice_source(candidate, abs_start, abs_end),
        embedding_score: embed_score,
        lexical_score,
    }))
}

fn tokens_match(answer: &[u32], evidence: &[u32]) -> bool {
    if answer == evidence {
        return true;
    }

    let mut counts: HashMap<u32, isize> = HashMap::new();
    for &token in answer {
        *counts.entry(token).or_insert(0) += 1;
    }
    for &token in evidence {
        if let Some(count) = counts.get_mut(&token) {
            *count -= 1;
            if *count == 0 {
                counts.remove(&token);
            }
        }
    }
    counts.is_empty()
}

fn token_span_to_char_span(
    token_spans: &[(usize, usize)],
    token_start: usize,
    token_end: usize,
) -> Option<(usize, usize)> {
    if token_start >= token_end || token_end > token_spans.len() {
        return None;
    }
    Some((token_spans[token_start].0, token_spans[token_end - 1].1))
}

fn slice_source(candidate: &Candidate, abs_start: usize, abs_end: usize) -> String {
    if let Some(ref full) = candidate.source_full_text {
        full.chars()
            .skip(abs_start)
            .take(abs_end - abs_start)
            .collect()
    } else {
        let local_start = abs_start - candidate.base_offset;
        let local_end = abs_end - candidate.base_offset;
        candidate
            .source_text
            .chars()
            .skip(local_start)
            .take(local_end - local_start)
            .collect()
    }
}
