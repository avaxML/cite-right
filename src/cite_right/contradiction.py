"""Lightweight contradiction detection for citation validation.

Checks for:
- Negation mismatches (answer says "X is not Y" but source says "X is Y")
- Number mismatches (answer says "15%" but source says "20%")
- Entity-token mismatches in key positions
- Temporal/polarity markers (BC vs ago, oppose vs support, etc.)
- Number context mismatches (same number, different context)
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

    # Check for entity swaps (proper nouns that differ)
    if has_entity_swap(answer_text, evidence_text):
        return True

    # Check for temporal/polarity contradictions
    if has_temporal_polarity_mismatch(answer_text, evidence_text):
        return True

    # Check for number context mismatches (same number, different meaning)
    if has_number_context_mismatch(answer_text, evidence_text):
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


def has_entity_swap(answer_text: str, evidence_text: str) -> bool:
    """Check if entities (proper nouns) differ between answer and evidence.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if entity swap detected
    """
    # Extract capitalized words (likely proper nouns/entities)
    # Match words that start with capital letter (excluding sentence starts)
    answer_entities = extract_entities(answer_text)
    evidence_entities = extract_entities(evidence_text)

    if not answer_entities or not evidence_entities:
        return False

    # Check if answer has entities not in evidence
    # This catches swaps like India→France, American→Indian
    answer_set = set(answer_entities)
    evidence_set = set(evidence_entities)

    # If answer has entities not in evidence, and evidence has entities not in answer,
    # it's likely an entity swap
    unique_in_answer = answer_set - evidence_set
    unique_in_evidence = evidence_set - answer_set

    # Entity swap: both have unique entities (not just missing)
    if unique_in_answer and unique_in_evidence:
        return True

    return False


def extract_entities(text: str) -> list[str]:
    """Extract likely entity names (capitalized words) from text.

    Args:
        text: The text to extract entities from

    Returns:
        List of capitalized words (lowercased and normalized for comparison)
    """
    # Normalize Unicode to handle fullwidth/halfwidth differences
    normalized_text = unicodedata.normalize("NFKC", text)

    # Match capitalized words (2+ chars to avoid initials)
    # Skip first word (might be capitalized due to sentence start)
    words = normalized_text.split()
    entities = []

    for i, word in enumerate(words):
        # Clean word of punctuation at end
        clean_word = re.sub(r"[.,;:!?]+$", "", word)

        # Check if word starts with capital and has 2+ letters
        if len(clean_word) >= 2 and clean_word[0].isupper():
            # Skip if it's the first word and all subsequent chars are lowercase
            # (likely sentence start, not entity)
            if i == 0 and clean_word[1:].islower():
                continue
            entities.append(clean_word.lower())

    return entities


def has_temporal_polarity_mismatch(answer_text: str, evidence_text: str) -> bool:
    """Check for temporal or polarity contradictions.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if temporal/polarity mismatch detected
    """
    answer_lower = answer_text.lower()
    evidence_lower = evidence_text.lower()

    # Temporal markers that contradict each other
    temporal_contradictions = [
        (r"\bbc\b", r"\bago\b"),  # BC (2000+ years) vs ago (recent)
        (r"\bce\b", r"\bbc\b"),  # CE vs BC
        (r"\bad\b", r"\bbc\b"),  # AD vs BC
        (r"\bbce\b", r"\bce\b"),  # BCE vs CE
    ]

    for pattern1, pattern2 in temporal_contradictions:
        if re.search(pattern1, evidence_lower) and re.search(pattern2, answer_lower):
            return True
        if re.search(pattern2, evidence_lower) and re.search(pattern1, answer_lower):
            return True

    # Polarity/sentiment markers that contradict each other
    polarity_contradictions = [
        (r"\boppose[ds]?\b", r"\b(?:support|urge[ds]?|advocate[ds]?|promote[ds]?)\b"),
        (r"\bsupport[s]?\b", r"\b(?:oppose[ds]?|reject[s]?|resist[s]?)\b"),
        (r"\breject[s]?\b", r"\b(?:accept[s]?|approve[s]?|endorse[ds]?)\b"),
        (r"\bdenied\b", r"\b(?:confirmed|admitted|acknowledged)\b"),
        (r"\bfailed\b", r"\b(?:succeeded|achieved|accomplished)\b"),
    ]

    for neg_pattern, pos_pattern in polarity_contradictions:
        if re.search(neg_pattern, evidence_lower) and re.search(
            pos_pattern, answer_lower
        ):
            return True
        if re.search(pos_pattern, evidence_lower) and re.search(
            neg_pattern, answer_lower
        ):
            return True

    return False


def has_number_context_mismatch(answer_text: str, evidence_text: str) -> bool:
    """Check if numbers appear in different contexts (leftover n-gram issue).

    When the same number appears in both texts but the surrounding context words
    differ significantly, it's likely a number context mismatch (e.g., "10 rebounds"
    vs "10 of which came in the first half").

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if number context mismatch detected
    """
    answer_numbers = extract_numbers(answer_text)
    evidence_numbers = extract_numbers(evidence_text)

    # Need at least one shared number
    shared_numbers = set(answer_numbers) & set(evidence_numbers)
    if not shared_numbers:
        return False

    # For each shared number, check if context words differ
    for number in shared_numbers:
        answer_context = extract_number_context(answer_text, number)
        evidence_context = extract_number_context(evidence_text, number)

        # Case 1: Evidence has no context (truncated) but answer has significant context
        # This is the "leftover n-gram" problem
        if answer_context and not evidence_context:
            # Check if the evidence actually ends with the number (truncation indicator)
            # vs having punctuation after it (like "123%" where % isn't captured)
            if evidence_ends_with_number(evidence_text, number):
                # Evidence truly truncated at the number
                common_words = {
                    "a",
                    "an",
                    "the",
                    "of",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "with",
                    "by",
                    "from",
                    "and",
                    "or",
                }
                significant_answer_words = set(answer_context) - common_words
                # If answer has significant context, it's suspicious
                if len(significant_answer_words) >= 1:
                    return True

        # Case 2: Both have context but they differ significantly
        if answer_context and evidence_context:
            # Check for key context words that differ
            answer_words = set(answer_context)
            evidence_words = set(evidence_context)

            # If there are context words in answer not in evidence, it's suspicious
            unique_answer_words = answer_words - evidence_words

            # Filter out common words (articles, prepositions, etc.)
            common_words = {
                "a",
                "an",
                "the",
                "of",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "by",
                "from",
                "and",
                "or",
            }
            significant_unique = unique_answer_words - common_words

            # If answer has significant unique context words (likely different meaning)
            if len(significant_unique) >= 1:
                return True

    return False


def evidence_ends_with_number(text: str, number: str) -> bool:
    """Check if evidence text ends with the given number (or whitespace after it).

    Args:
        text: The text to check
        number: The number to look for at the end

    Returns:
        True if text ends with the number (possibly followed by whitespace/punctuation)
    """
    # Escape special regex chars in number
    escaped_number = re.escape(number)

    # Check if text ends with the number (possibly followed by whitespace or sentence-ending punctuation)
    pattern = escaped_number + r"[\s,.;!?]*$"
    return bool(re.search(pattern, text))


def extract_number_context(text: str, number: str) -> list[str]:
    """Extract context words around a number.

    Args:
        text: The text containing the number
        number: The number to find context for

    Returns:
        List of context words (1-2 words before and after the number)
    """
    # Escape special regex chars in number
    escaped_number = re.escape(number)

    # Find the number with context (up to 2 words before/after)
    pattern = r"(\w+\s+)?(\w+\s+)?" + escaped_number + r"(\s+\w+)?(\s+\w+)?"
    match = re.search(pattern, text)

    if not match:
        return []

    # Extract context words (excluding the number itself)
    context_words = []
    for group in match.groups():
        if group:
            words = group.strip().split()
            context_words.extend(w.lower() for w in words if not re.match(r"^\d", w))

    return context_words
