//! Rust-accelerated corpus preparation that maintains quality.
//!
//! This module moves tokenization, passage generation, and candidate building to Rust
//! while keeping Smith-Waterman alignment in Python for quality.
//!
//! IMPORTANT: All indices returned to Python are CHARACTER indices, not byte indices,
//! because Python string slicing uses character offsets.

use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

use crate::inverted_index::{InvertedIndex, Posting};

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
    pub next_id: u32,
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

    #[allow(dead_code)]
    pub fn get_next_id(&self) -> u32 {
        self.next_id
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
                // Include alphanumeric, apostrophes (all variants), hyphens (all variants), underscores, AND combining marks
                if ch.is_alphanumeric()
                    || is_apostrophe_variant(ch)
                    || is_dash_variant(ch)
                    || ch == '_'
                    || is_combining_mark(ch)
                {
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
    // Apply NFKC normalization (fullwidth→halfwidth, etc.) + casefold
    let normalized: String = token.nfkc().collect();
    let casefolded = normalized
        .chars()
        .flat_map(|c| c.to_lowercase())
        .collect::<String>();

    // Normalize punctuation (quotes and dashes) to ASCII equivalents
    let punct_normalized = normalize_punctuation(&casefolded);

    // Normalize percent symbol (matching Python's normalize_percent)
    if punct_normalized == "%" {
        "percent".to_string()
    } else {
        punct_normalized
    }
}

fn normalize_punctuation(text: &str) -> String {
    // Map quote and dash variants that should not affect matching
    // Matches Python's _normalize_punctuation function
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{02bc}' => '\'', // Curly quotes, modifier apostrophe → ASCII
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2212}' => '-', // Various dashes → ASCII hyphen
            _ => c,
        })
        .collect()
}

fn is_apostrophe_variant(ch: char) -> bool {
    // ASCII apostrophe and Unicode variants
    matches!(ch, '\'' | '\u{2018}' | '\u{2019}' | '\u{02bc}')
}

fn is_dash_variant(ch: char) -> bool {
    // ASCII hyphen and Unicode variants
    matches!(
        ch,
        '-' | '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2212}'
    )
}

fn is_combining_mark(ch: char) -> bool {
    // Unicode combining diacritical marks (category Mn, Me, Mc)
    // Range U+0300 to U+036F is the main combining diacriticals block
    matches!(ch, '\u{0300}'..='\u{036F}' | '\u{1AB0}'..='\u{1AFF}' | '\u{1DC0}'..='\u{1DFF}' | '\u{20D0}'..='\u{20FF}' | '\u{FE20}'..='\u{FE2F}')
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

            // Check if this is likely an abbreviation (e.g., U.S., Dr., etc.)
            let is_abbreviation = if i > 0 && c == '.' {
                let prev_char = chars[i - 1].1;
                // Pattern: uppercase letter followed by period (U. in U.S.)
                prev_char.is_uppercase()
            } else {
                false
            };

            if !is_abbreviation && (next_is_space || is_end) {
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
        .map(|(token_id, count)| (token_id, ((n + 1) as f64 / (count + 1) as f64).ln() + 1.0))
        .collect()
}

type CandidateData = (usize, usize, Vec<u32>, Vec<(usize, usize)>);

/// Build inverted index from candidates
#[allow(clippy::type_complexity)]
pub fn build_inverted_index(candidates: &[CandidateData]) -> InvertedIndex {
    let mut index = InvertedIndex::new();

    for (global_candidate_index, candidate_tokens_spans) in candidates.iter().enumerate() {
        let (_passage_start, _passage_end, token_ids, token_spans) = candidate_tokens_spans;

        for (token_pos, (&token_id, &(char_start, char_end))) in
            token_ids.iter().zip(token_spans.iter()).enumerate()
        {
            index.add_posting(
                token_id,
                Posting {
                    candidate_index: global_candidate_index,
                    token_pos,
                    char_start,
                    char_end,
                },
            );
        }
    }

    index
}
