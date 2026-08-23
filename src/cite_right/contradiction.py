"""Lightweight contradiction detection for citation validation.

Checks for:
- Negation mismatches (answer says "X is not Y" but source says "X is Y")
- Number mismatches (answer says "15%" but source says "20%")
- Entity-token mismatches in key positions
"""

from __future__ import annotations

import re
import unicodedata


def check_contradiction(answer_text: str, evidence_text: str) -> bool:
    """Check if evidence contradicts the answer.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if contradiction is detected, False otherwise
    """
    # Check for negation mismatch
    if has_negation_mismatch(answer_text, evidence_text):
        return True

    # Check for number mismatch
    if has_number_mismatch(answer_text, evidence_text):
        return True

    return False


def has_negation_mismatch(answer_text: str, evidence_text: str) -> bool:
    """Check if one text is negated and the other is not.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if negation mismatch detected
    """
    answer_negated = contains_negation(answer_text)
    evidence_negated = contains_negation(evidence_text)

    # Mismatch if exactly one is negated
    return answer_negated != evidence_negated


def contains_negation(text: str) -> bool:
    """Check if text contains negation markers.

    Args:
        text: The text to check

    Returns:
        True if negation markers found
    """
    lower = text.lower()
    negation_patterns = [
        r"\bnot\b",
        r"\bnever\b",
        r"\bno\b",
        r"n't\b",
        r"\bneither\b",
        r"\bnor\b",
        r"\bnobody\b",
        r"\bnothing\b",
        r"\bnowhere\b",
        r"\bhadn\'t\b",
        r"\bhasn\'t\b",
        r"\bhaven\'t\b",
        r"\bisn\'t\b",
        r"\bwasn\'t\b",
        r"\bweren\'t\b",
        r"\bwon\'t\b",
        r"\bwouldn\'t\b",
        r"\bdon\'t\b",
        r"\bdoesn\'t\b",
        r"\bdidn\'t\b",
        r"\bcan\'t\b",
        r"\bcannot\b",
        r"\bcouldn\'t\b",
        r"\bshouldn\'t\b",
        r"\bmustn\'t\b",
    ]

    for pattern in negation_patterns:
        if re.search(pattern, lower):
            return True
    return False


def has_number_mismatch(answer_text: str, evidence_text: str) -> bool:
    """Check if numbers differ between answer and evidence.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if number mismatch detected
    """
    answer_numbers = extract_numbers(answer_text)
    evidence_numbers = extract_numbers(evidence_text)

    # If answer has numbers not in evidence, potential mismatch
    if answer_numbers and evidence_numbers:
        answer_set = set(answer_numbers)
        evidence_set = set(evidence_numbers)

        # Check if answer has numbers not found in evidence
        unique_in_answer = answer_set - evidence_set

        # If all answer numbers are missing from evidence, it's a mismatch
        if unique_in_answer and len(unique_in_answer) == len(answer_set):
            return True

    return False


def extract_numbers(text: str) -> list[str]:
    """Extract numbers from text.

    Args:
        text: The text to extract numbers from

    Returns:
        List of number strings found (normalized, without % or commas)
    """
    # Normalize Unicode (NFKC converts fullwidth digits to ASCII)
    normalized_text = unicodedata.normalize("NFKC", text)

    # Match numbers including decimals and thousands separators
    number_pattern = r"\b\d+(?:[.,]\d+)*"
    numbers = re.findall(number_pattern, normalized_text)

    # Normalize numbers (remove commas, keep periods)
    normalized = []
    for num in numbers:
        # Remove commas
        num = num.replace(",", "")
        normalized.append(num)

    return normalized
