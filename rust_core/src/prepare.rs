//! Rust-accelerated corpus preparation that maintains quality.
//!
//! This module moves tokenization, passage generation, and candidate building to Rust
//! while keeping Smith-Waterman alignment in Python for quality.
//!
//! IMPORTANT: All indices returned to Python are CHARACTER indices, not byte indices,
//! because Python string slicing uses character offsets.

use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

/// Build a byte-to-char index mapping for fast lookups
fn build_byte_to_char_map(text: &str) -> Vec<usize> {
    let char_positions: Vec<(usize, usize)> = text
        .char_indices()
        .enumerate()
        .map(|(char_idx, (byte_idx, _))| (byte_idx, char_idx))
        .collect();

    let mut map = vec![0; text.len() + 1];

    // Fill in byte positions that start characters
    for &(byte_idx, char_idx) in &char_positions {
        map[byte_idx] = char_idx;
    }

    // Fill in gaps (middle of multi-byte chars) with forward fill
    let mut last_char_idx = 0;
    for (i, entry) in map.iter_mut().enumerate() {
        if *entry > 0 || i == 0 {
            last_char_idx = *entry;
        } else {
            *entry = last_char_idx;
        }
    }

    // Set the final position (text.len()) to total char count
    if let Some(&(_, last_char_idx)) = char_positions.last() {
        map[text.len()] = last_char_idx + 1;
    }

    map
}

#[derive(Clone)]
pub struct RustTokenizedText {
    pub token_ids: Vec<u32>,
    pub token_spans: Vec<(usize, usize)>,
}

/// Mirrors the three normalization flags of Python's `TokenizerConfig`.
///
/// The caller supplies every value; Python's `TokenizerConfig` is the only
/// owner of the defaults.
#[derive(Clone, Copy, Debug)]
pub struct NormalizeFlags {
    pub numbers: bool,
    pub percent: bool,
    pub currency: bool,
}

pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    pub next_id: u32,
    flags: NormalizeFlags,
}

impl SimpleTokenizer {
    pub fn new(flags: NormalizeFlags) -> Self {
        Self {
            vocab: HashMap::new(),
            next_id: 1,
            flags,
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

        // Build byte-to-char mapping once for the entire document
        let byte_to_char = build_byte_to_char_map(text);

        for (start_byte, end_byte) in iter_token_spans(text) {
            let raw = &text[start_byte..end_byte];
            let normalized = normalize_token_simple(raw, self.flags);
            if normalized.is_empty() {
                continue;
            }

            let token_id = *self.vocab.entry(normalized).or_insert_with(|| {
                let id = self.next_id;
                self.next_id += 1;
                id
            });

            token_ids.push(token_id);
            // Convert byte indices to char indices using precomputed map
            let start_char = byte_to_char[start_byte];
            let end_char = byte_to_char[end_byte];
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
        else if is_symbol_char(c) {
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

/// Mirrors Python's `_normalize_token`: NFKC, casefold, punctuation, then the
/// flag-gated comma strip and percent/currency word mapping, in that order.
fn normalize_token_simple(token: &str, flags: NormalizeFlags) -> String {
    let normalized: String = token.nfkc().collect();
    let casefolded = normalized
        .chars()
        .flat_map(|c| c.to_lowercase())
        .collect::<String>();
    let mut normalized = normalize_punctuation(&casefolded);

    if flags.numbers && normalized.chars().next().is_some_and(char::is_numeric) {
        normalized.retain(|c| c != ',');
    }

    match (normalized.as_str(), flags) {
        ("%", NormalizeFlags { percent: true, .. }) => "percent".to_string(),
        ("$", NormalizeFlags { currency: true, .. }) => "dollar".to_string(),
        ("\u{20ac}", NormalizeFlags { currency: true, .. }) => "euro".to_string(),
        ("\u{a3}", NormalizeFlags { currency: true, .. }) => "pound".to_string(),
        _ => normalized,
    }
}

/// True when `c` normalizes (NFKC) to one of the symbols Python's
/// `_iter_token_spans` emits as a standalone token: `%`, `$`, `€`, `£`.
fn is_symbol_char(c: char) -> bool {
    let mut folded = std::iter::once(c).nfkc();
    matches!(
        (folded.next(), folded.next()),
        (Some('%' | '$' | '\u{20ac}' | '\u{a3}'), None)
    )
}

fn normalize_punctuation(text: &str) -> String {
    // Map quote and dash variants that should not affect matching
    // Matches Python's _normalize_punctuation function
    text.chars().map(map_punctuation_char).collect()
}

/// Single source of truth for the quote and dash variants Python's
/// `_normalize_punctuation` folds to ASCII.
fn map_punctuation_char(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{02bc}' => '\'', // Curly quotes, modifier apostrophe → ASCII
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2212}' => '-', // Various dashes → ASCII hyphen
        _ => c,
    }
}

fn is_apostrophe_variant(ch: char) -> bool {
    map_punctuation_char(ch) == '\''
}

fn is_dash_variant(ch: char) -> bool {
    map_punctuation_char(ch) == '-'
}

fn is_combining_mark(ch: char) -> bool {
    // Unicode combining diacritical marks (category Mn, Me, Mc)
    // Range U+0300 to U+036F is the main combining diacriticals block
    matches!(ch, '\u{0300}'..='\u{036F}' | '\u{1AB0}'..='\u{1AFF}' | '\u{1DC0}'..='\u{1DFF}' | '\u{20D0}'..='\u{20FF}' | '\u{FE20}'..='\u{FE2F}')
}

pub fn simple_segment(text: &str) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();

    // Build byte-to-char mapping once
    let byte_to_char = build_byte_to_char_map(text);

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

                // Convert to char indices using precomputed map
                let start_char = byte_to_char[start_byte];
                let end_char = byte_to_char[end_byte];
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
        let start_char = byte_to_char[start_byte];
        let end_char = byte_to_char[text.len()];
        segments.push((start_char, end_char));
    }

    if segments.is_empty() && !text.is_empty() {
        segments.push((0, byte_to_char[text.len()]));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_variants_map_to_ascii() {
        for c in ['\u{2018}', '\u{2019}', '\u{02bc}', '\''] {
            assert!(is_apostrophe_variant(c));
            assert!(!is_dash_variant(c));
            assert_eq!(map_punctuation_char(c), '\'');
        }
        for c in [
            '\u{2010}',
            '\u{2011}',
            '\u{2012}',
            '\u{2013}',
            '\u{2212}',
            '-',
        ] {
            assert!(is_dash_variant(c));
            assert!(!is_apostrophe_variant(c));
            assert_eq!(map_punctuation_char(c), '-');
        }
        assert!(!is_apostrophe_variant('a') && !is_dash_variant('a'));
        assert_eq!(
            normalize_punctuation("don\u{2019}t re\u{2011}enter"),
            "don't re-enter"
        );
    }

    const ALL_ON: NormalizeFlags = NormalizeFlags {
        numbers: true,
        percent: true,
        currency: true,
    };
    const ALL_OFF: NormalizeFlags = NormalizeFlags {
        numbers: false,
        percent: false,
        currency: false,
    };

    fn only_numbers_off() -> NormalizeFlags {
        NormalizeFlags {
            numbers: false,
            ..ALL_ON
        }
    }

    fn only_percent_off() -> NormalizeFlags {
        NormalizeFlags {
            percent: false,
            ..ALL_ON
        }
    }

    fn only_currency_off() -> NormalizeFlags {
        NormalizeFlags {
            currency: false,
            ..ALL_ON
        }
    }

    #[test]
    fn comma_grouping_is_stripped_only_when_numbers_on() {
        assert_eq!(normalize_token_simple("2,410.12", ALL_ON), "2410.12");
        assert_eq!(
            normalize_token_simple("2,410.12", only_numbers_off()),
            "2,410.12"
        );
        assert_eq!(normalize_token_simple("2,410.12", ALL_OFF), "2,410.12");
    }

    #[test]
    fn comma_strip_follows_first_normalized_char_being_a_digit() {
        // Fullwidth digits fold to ASCII under NFKC; Arabic-Indic digits stay as is.
        assert_eq!(
            normalize_token_simple("\u{ff12},\u{ff14}\u{ff11}\u{ff10}", ALL_ON),
            "2410"
        );
        assert_eq!(
            normalize_token_simple("\u{662},\u{664}\u{661}\u{660}", ALL_ON),
            "\u{662}\u{664}\u{661}\u{660}"
        );
        assert_eq!(normalize_token_simple("a,b", ALL_ON), "a,b");
    }

    #[test]
    fn currency_symbols_map_only_when_currency_on() {
        for (raw, word) in [
            ("$", "dollar"),
            ("\u{20ac}", "euro"),
            ("\u{a3}", "pound"),
            ("\u{ff04}", "dollar"),
        ] {
            assert_eq!(normalize_token_simple(raw, ALL_ON), word);
            let folded: String = raw.nfkc().collect();
            assert_eq!(normalize_token_simple(raw, only_currency_off()), folded);
        }
    }

    #[test]
    fn percent_maps_only_when_percent_on() {
        assert_eq!(normalize_token_simple("%", ALL_ON), "percent");
        assert_eq!(normalize_token_simple("\u{ff05}", ALL_ON), "percent");
        assert_eq!(normalize_token_simple("%", only_percent_off()), "%");
        assert_eq!(normalize_token_simple("\u{ff05}", only_percent_off()), "%");
    }

    #[test]
    fn symbol_spans_follow_nfkc_form() {
        for text in [
            "\u{ffe1}",
            "\u{fe69}",
            "\u{fe6a}",
            "%",
            "$",
            "\u{20ac}",
            "\u{a3}",
        ] {
            assert_eq!(iter_token_spans(text), vec![(0, text.len())], "{text:?}");
        }
        assert!(iter_token_spans("\u{ff5e}").is_empty());
    }

    #[test]
    fn tokenizer_applies_flags_to_vocab() {
        let mut tokenizer = SimpleTokenizer::new(ALL_ON);
        let out = tokenizer.tokenize("Fee 1,204.50 \u{ffe1}5");
        assert_eq!(out.token_ids.len(), 4);
        let vocab = tokenizer.get_vocab();
        assert!(vocab.contains_key("1204.50"));
        assert!(vocab.contains_key("pound"));
        assert!(!vocab.contains_key("1,204.50"));
    }
}
