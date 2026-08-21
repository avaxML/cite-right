//! Rust-accelerated corpus preparation that maintains quality.
//!
//! This module moves tokenization, passage generation, and candidate building to Rust
//! while keeping Smith-Waterman alignment in Python for quality.
//!
//! IMPORTANT: All indices returned to Python are CHARACTER indices, not byte indices,
//! because Python string slicing uses character offsets.

use std::collections::HashMap;

/// Convert byte index to character index in a UTF-8 string
fn byte_to_char_index(text: &str, byte_index: usize) -> usize {
    text.char_indices()
        .take_while(|(i, _)| *i < byte_index)
        .count()
}

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

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    pub fn tokenize(&mut self, text: &str) -> RustTokenizedText {
        let mut token_ids = Vec::new();
        let mut token_spans = Vec::new();

        for (start_byte, end_byte) in iter_token_spans(text) {
            let raw = &text[start_byte..end_byte];
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
            // Convert byte indices to char indices for Python
            let start_char = byte_to_char_index(text, start_byte);
            let end_char = byte_to_char_index(text, end_byte);
            token_spans.push((start_char, end_char));
        }

        RustTokenizedText {
            token_ids,
            token_spans,
        }
    }
}

fn iter_token_spans(text: &str) -> Vec<(usize, usize)> {
    // Unicode-aware tokenization matching Python SimpleTokenizer
    // Only yields spans for: numbers, words, and special symbols (%, $, €, £)
    let mut spans = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut idx = 0;

    while idx < chars.len() {
        let c = chars[idx].1;
        
        if c.is_whitespace() {
            idx += 1;
            continue;
        }

        let start_byte = chars[idx].0;
        
        // Check if it's a number
        if c.is_numeric() {
            idx += 1;
            while idx < chars.len() {
                let ch = chars[idx].1;
                if ch.is_numeric() {
                    idx += 1;
                } else if (ch == '.' || ch == ',') && idx > 0 && idx + 1 < chars.len() {
                    // Only include . or , if between digits
                    let prev_is_digit = idx > 0 && chars[idx - 1].1.is_numeric();
                    let next_is_digit = idx + 1 < chars.len() && chars[idx + 1].1.is_numeric();
                    if prev_is_digit && next_is_digit {
                        idx += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        // Check if it's a special symbol (%, $, €, £) - after NFKC normalization
        else if matches!(c, '%' | '$' | '€' | '£' | '％' | '＄') {
            idx += 1;
        }
        // Check if it's a word character
        else if c.is_alphanumeric() {
            idx += 1;
            while idx < chars.len() {
                let ch = chars[idx].1;
                if ch.is_alphanumeric() || ch == '\'' || ch == '-' || ch == '_' {
                    idx += 1;
                } else {
                    break;
                }
            }
        }
        // Skip other punctuation
        else {
            idx += 1;
            continue;
        }
        
        let end_byte = if idx < chars.len() {
            chars[idx].0
        } else {
            text.len()
        };
        
        spans.push((start_byte, end_byte));
    }

    spans
}

fn normalize_token_simple(token: &str) -> String {
    // Simple lowercase normalization matching Python
    token.to_lowercase()
}

pub fn simple_segment(text: &str) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut start_byte = 0;
    let mut i = 0;

    while i < chars.len() {
        let (_, c) = chars[i];
        if matches!(c, '.' | '!' | '?') {
            let next_is_space = i + 1 < chars.len() && chars[i + 1].1.is_whitespace();
            let is_end = i + 1 == chars.len();
            if next_is_space || is_end {
                // End at the punctuation, not after the space
                let end_byte = if i + 1 < chars.len() {
                    chars[i + 1].0
                } else {
                    text.len()
                };
                
                // Convert to char indices for Python
                let start_char = byte_to_char_index(text, start_byte);
                let end_char = byte_to_char_index(text, end_byte);
                segments.push((start_char, end_char));
                
                // Start next segment after the space
                start_byte = if next_is_space && i + 2 < chars.len() {
                    chars[i + 2].0
                } else {
                    end_byte
                };
                i = if next_is_space { i + 2 } else { i + 1 };
                continue;
            }
        }
        i += 1;
    }

    if start_byte < text.len() {
        let start_char = byte_to_char_index(text, start_byte);
        let end_char = text.chars().count();
        segments.push((start_char, end_char));
    }

    if segments.is_empty() && !text.is_empty() {
        segments.push((0, text.chars().count()));
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
