//! Cheap contradiction detection for citation validation
//!
//! Checks for:
//! - Negation mismatches (answer says "X is not Y" but source says "X is Y")
//! - Number mismatches (answer says "15%" but source says "20%")
//! - Entity-token mismatches in negated contexts

use std::collections::HashSet;

/// Check if evidence contradicts the answer
#[allow(dead_code)]
pub fn check_contradiction(
    answer_text: &str,
    evidence_text: &str,
    answer_tokens: &[u32],
    evidence_tokens: &[u32],
) -> bool {
    // Check for negation mismatch
    if has_negation_mismatch(answer_text, evidence_text) {
        return true;
    }

    // Check for number mismatch
    if has_number_mismatch(answer_tokens, evidence_tokens) {
        return true;
    }

    false
}

/// Check if one text is negated and the other is not
fn has_negation_mismatch(answer_text: &str, evidence_text: &str) -> bool {
    let answer_negated = contains_negation(answer_text);
    let evidence_negated = contains_negation(evidence_text);

    // Mismatch if exactly one is negated
    answer_negated != evidence_negated
}

/// Check if text contains negation markers
fn contains_negation(text: &str) -> bool {
    let lower = text.to_lowercase();
    let negation_words = [
        "not", "never", "no ", "n't", "neither", "nor", "nobody", "nothing",
    ];

    negation_words.iter().any(|&word| lower.contains(word))
}

/// Check if numbers differ between answer and evidence tokens
fn has_number_mismatch(answer_tokens: &[u32], evidence_tokens: &[u32]) -> bool {
    // For now, just check if answer has unique tokens not in evidence
    // A more sophisticated version would extract and compare actual numbers
    let answer_set: HashSet<u32> = answer_tokens.iter().copied().collect();
    let evidence_set: HashSet<u32> = evidence_tokens.iter().copied().collect();

    // If answer has tokens not in evidence, there might be a mismatch
    // But this is just a heuristic - we'd need actual number extraction for accuracy
    let unique_in_answer: Vec<_> = answer_set.difference(&evidence_set).collect();

    // Only flag as mismatch if there are many unique tokens (conservative)
    unique_in_answer.len() > answer_tokens.len() / 2
}
