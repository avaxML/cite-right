//! Rust-accelerated corpus preparation that maintains quality.
//!
//! This module moves tokenization, passage generation, and candidate building to Rust
//! while keeping Smith-Waterman alignment in Python for quality.

use std::collections::HashMap;

#[derive(Clone)]
pub struct RustTokenizedText {
    pub token_ids: Vec<u32>,
    pub token_spans: Vec<(usize, usize)>,
}

pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    next_id: u32,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn tokenize(&mut self, text: &str) -> RustTokenizedText {
        let mut token_ids = Vec::new();
        let mut token_spans = Vec::new();

        for (start, end) in iter_token_spans(text) {
            let raw = &text[start..end];
            let normalized = normalize_token_simple(raw);
            if normalized.is_empty() {
                continue;
            }
            
            let token_id = *self.vocab.entry(normalized).or_insert_with(|| {
                let id = self.next_id;
                self.next_id += 1;
                id
            });
            
            token_ids.push(token_id);
            token_spans.push((start, end));
        }

        RustTokenizedText {
            token_ids,
            token_spans,
        }
    }
}

fn iter_token_spans(text: &str) -> Vec<(usize, usize)> {
    // Simple whitespace + punctuation tokenization matching Python SimpleTokenizer
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let mut idx = 0;

    while idx < bytes.len() {
        // Skip whitespace
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= bytes.len() {
            break;
        }

        let start = idx;
        
        // Consume token
        if bytes[idx].is_ascii_digit() {
            // Number token
            while idx < bytes.len() && (bytes[idx].is_ascii_digit() || bytes[idx] == b'.' || bytes[idx] == b',') {
                idx += 1;
            }
        } else if bytes[idx].is_ascii_alphanumeric() {
            // Word token
            while idx < bytes.len() {
                let b = bytes[idx];
                if b.is_ascii_alphanumeric() || b == b'\'' || b == b'-' || b == b'_' {
                    idx += 1;
                } else {
                    break;
                }
            }
        } else {
            // Single character token (punctuation)
            idx += 1;
        }
        
        spans.push((start, idx));
    }

    spans
}

fn normalize_token_simple(token: &str) -> String {
    // Simple lowercase normalization matching Python
    token.to_lowercase()
}

pub fn simple_segment(text: &str) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut i = 0;

    while i < bytes.len() {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            let next_is_space = i + 1 < bytes.len() && bytes[i + 1].is_ascii_whitespace();
            let is_end = i + 1 == bytes.len();
            if next_is_space || is_end {
                let end = if next_is_space { i + 2 } else { i + 1 };
                segments.push((start, end.min(bytes.len())));
                start = end.min(bytes.len());
                i = end;
                continue;
            }
        }
        i += 1;
    }

    if start < bytes.len() {
        segments.push((start, bytes.len()));
    }

    if segments.is_empty() && !text.is_empty() {
        segments.push((0, text.len()));
    }

    segments
}

pub fn generate_passages(
    segments: &[(usize, usize)],
    window_size: usize,
    stride: usize,
) -> Vec<(usize, usize)> {
    if segments.is_empty() {
        return Vec::new();
    }

    let window = window_size.max(1);
    let stride = stride.max(1);
    let mut passages = Vec::new();
    let mut idx = 0;

    while idx < segments.len() {
        let end_idx = (idx + window).min(segments.len());
        let start = segments[idx].0;
        let end = segments[end_idx - 1].1;
        passages.push((start, end));
        if end_idx == segments.len() {
            break;
        }
        idx += stride;
    }

    passages
}

pub fn slice_tokenized_text(
    token_ids: &[u32],
    token_spans: &[(usize, usize)],
    start: usize,
    end: usize,
) -> (Vec<u32>, Vec<(usize, usize)>) {
    let mut result_ids = Vec::new();
    let mut result_spans = Vec::new();

    for (i, &(token_start, token_end)) in token_spans.iter().enumerate() {
        if token_end <= start {
            continue;
        }
        if token_start >= end {
            break;
        }
        
        let local_start = token_start.max(start) - start;
        let local_end = token_end.min(end) - start;
        
        if local_start < local_end {
            result_ids.push(token_ids[i]);
            result_spans.push((local_start, local_end));
        }
    }

    (result_ids, result_spans)
}

pub fn compute_idf(candidate_token_sets: &[Vec<u32>]) -> HashMap<u32, f64> {
    let mut df: HashMap<u32, usize> = HashMap::new();
    
    for token_ids in candidate_token_sets {
        let unique: std::collections::HashSet<u32> = token_ids.iter().copied().collect();
        for token_id in unique {
            *df.entry(token_id).or_insert(0) += 1;
        }
    }
    
    let n = candidate_token_sets.len();
    df.into_iter()
        .map(|(token_id, count)| {
            (token_id, ((n + 1) as f64 / (count + 1) as f64).ln() + 1.0)
        })
        .collect()
}
