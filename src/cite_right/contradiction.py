"""Lightweight contradiction detection for citation validation.

Checks for:
- Negation mismatches (answer says "X is not Y" but source says "X is Y")
- Number mismatches (answer says "15%" but source says "20%")
- Entity-token mismatches in key positions
- Temporal/polarity markers (BC vs ago, oppose vs support, etc.)
- Number context mismatches (same number, different slot)
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

from cite_right.text.content_coverage import STOPWORDS
from cite_right.text.segmenter_simple import SimpleSegmenter

_CURRENCY_CODES = {
    "usd": "usd",
    "eur": "€",
    "gbp": "£",
    "cad": "cad",
    "aud": "aud",
    "nzd": "nzd",
}
_CURRENCY_RESTATEMENT_UNITS = {
    "$": ("",),
    "€": ("",),
    "£": ("",),
    **{code: ("", "$") for code in ("usd", "cad", "aud", "nzd")},
}
_CURRENCY_UNITS = {
    **_CURRENCY_CODES,
    "$": "$",
    "dollar": "$",
    "dollars": "$",
    "€": "€",
    "euro": "€",
    "euros": "€",
    "£": "£",
    "pound": "£",
    "pounds": "£",
}
_PERCENT_UNITS = {"%", "percent", "percentage", "pct"}
_UNIT_WORDS = "|".join(unit for unit in _CURRENCY_UNITS if unit.isalpha())
_CODE_WORDS = "|".join(_CURRENCY_CODES)
_QUANTITY = re.compile(
    r"(?:(?<![\w.,])([-+−])[ \t]*)?"
    rf"(?:([$€£]|\b(?:{_CODE_WORDS})\b)(?:[ \t]*[$€£])?[ \t]*(?:([-+−])[ \t]*)?)?"
    r"\b(\d+(?:[.,]\d+)*)"
    rf"(?:[ \t]*([%$€£]|\b(?:percentage[ \t]+points|percent|percentage|pct|{_UNIT_WORDS})\b))?",
    re.IGNORECASE,
)
_GROUPED_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_WORDS_BEFORE_QUANTITY = re.compile(r"(?:\b[^\W\d_]+[\s:]+){1,4}$")
_WORD_AFTER_QUANTITY = re.compile(r"[ \t]+([^\W\d_]+)")
_OPPOSING_QUANTITY_LABELS = (
    ({"credit", "credits"}, {"debit", "debits", "debt", "debts"}),
    ({"gain", "gains", "yield", "yields"}, {"loss", "losses"}),
    ({"increase", "increases", "higher"}, {"decrease", "decreases", "lower"}),
)

_FUNCTION_WORDS = frozenset(
    {
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
        # Tokenizer/source normalization variants, not slot labels.
        "percent",
        "percentage",
        "pct",
    }
)

# Sentence-initial words that are not entities even when capitalized.
_SENTENCE_START_STOPWORDS = _FUNCTION_WORDS | {
    "this",
    "that",
    "these",
    "those",
    "we",
    "they",
    "he",
    "she",
    "it",
    "i",
    "if",
    "when",
    "while",
    "after",
    "before",
    "as",
    "so",
    "then",
    "there",
    "here",
    "however",
    "therefore",
    "but",
}


def check_contradiction(answer_text: str, evidence_text: str) -> bool:
    """Check if evidence contradicts the answer.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if contradiction is detected, False otherwise
    """
    return next(iter_contradiction_checks(answer_text, evidence_text), None) is not None


def has_negation_mismatch(answer_text: str, evidence_text: str) -> bool:
    """Check if one text is negated and the other is not.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if negation mismatch detected
    """
    return contains_negation(answer_text) != contains_negation(evidence_text)


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
    """Check whether the evidence contains each quantity in the answer.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if number mismatch detected
    """
    answer_values = {value for value, _, _ in _iter_quantities(answer_text)}
    evidence_values = {
        restatement
        for value, _, _ in _iter_quantities(evidence_text)
        for restatement in _quantity_restatements(value)
    }
    if not answer_values or not evidence_values:
        return False

    return bool(answer_values - evidence_values)


def _quantity_restatements(value: tuple[str, str]) -> Iterator[tuple[str, str]]:
    """Allow a restatement to omit a monetary amount's denomination."""
    yield value
    number, unit = value
    for restated_unit in _CURRENCY_RESTATEMENT_UNITS.get(unit, ()):
        yield number, restated_unit


def _iter_quantities(
    text: str,
) -> Iterator[tuple[tuple[str, str], re.Match[str], bool]]:
    """Yield the normalized quantity, its match, and whether it has grouping or units."""
    text = unicodedata.normalize("NFKC", text)
    previous_end = None
    for match in _QUANTITY.finditer(text):
        sign, prefix, prefix_sign, number, suffix = match.groups()
        # An intervening dash between quantities is a range delimiter.
        if (
            sign
            and previous_end is not None
            and not text[previous_end : match.start()].strip()
        ):
            sign = None
        previous_end = match.end()

        grouped = _GROUPED_NUMBER.fullmatch(number) is not None
        if grouped:
            number = number.replace(",", "")
        if (sign or prefix_sign) in ("-", "−"):
            number = "-" + number

        prefix_unit = (prefix or "").lower()
        suffix_unit = (suffix or "").lower()
        raw_unit = prefix_unit or suffix_unit
        if prefix_unit not in _CURRENCY_CODES and suffix_unit in _CURRENCY_CODES:
            raw_unit = suffix_unit
        unit = _CURRENCY_UNITS.get(raw_unit, "")
        if raw_unit in _PERCENT_UNITS:
            unit = "%"
        elif raw_unit.startswith("percentage"):
            unit = "percentage points"
        yield (number, unit), match, grouped or bool(unit)


def extract_numbers(text: str) -> list[str]:
    """Extract numbers from text.

    Args:
        text: The text to extract numbers from

    Returns:
        List of number strings found (normalized, without % or commas)
    """
    normalized_text = unicodedata.normalize("NFKC", text)
    number_pattern = r"\b\d+(?:[.,]\d+)*"
    numbers = re.findall(number_pattern, normalized_text)
    return [num.replace(",", "") for num in numbers]


def has_entity_swap(answer_text: str, evidence_text: str) -> bool:
    """Check if entities (proper nouns) differ between answer and evidence.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if entity swap detected
    """
    answer_entities = extract_entities(answer_text)
    evidence_entities = extract_entities(evidence_text)
    if not answer_entities or not evidence_entities:
        return False

    answer_set = set(answer_entities)
    evidence_set = set(evidence_entities)
    unique_in_answer = answer_set - evidence_set
    unique_in_evidence = evidence_set - answer_set
    return bool(unique_in_answer and unique_in_evidence)


def extract_entities(text: str) -> list[str]:
    """Extract likely entity names (capitalized words) from text.

    Args:
        text: The text to extract entities from

    Returns:
        List of capitalized words (lowercased and normalized for comparison)
    """
    normalized_text = unicodedata.normalize("NFKC", text)
    words = normalized_text.split()
    entities: list[str] = []

    for i, word in enumerate(words):
        clean_word = re.sub(r"[.,;:!?]+$", "", word)
        if len(clean_word) < 2 or not clean_word[0].isupper():
            continue
        if i == 0 and clean_word.lower() in _SENTENCE_START_STOPWORDS:
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

    temporal_contradictions = [
        (r"\bbc\b", r"\bago\b"),
        (r"\bce\b", r"\bbc\b"),
        (r"\bad\b", r"\bbc\b"),
        (r"\bbce\b", r"\bce\b"),
    ]
    if _paired_marker_mismatch(answer_lower, evidence_lower, temporal_contradictions):
        return True
    if _has_reversed_before_after(answer_lower, evidence_lower):
        return True

    polarity_contradictions = [
        (r"\boppose[ds]?\b", r"\b(?:support|urge[ds]?|advocate[ds]?|promote[ds]?)\b"),
        (r"\bsupport[s]?\b", r"\b(?:oppose[ds]?|reject[s]?|resist[s]?)\b"),
        (r"\breject[s]?\b", r"\b(?:accept[s]?|approve[s]?|endorse[ds]?)\b"),
        (r"\bdenied\b", r"\b(?:confirmed|admitted|acknowledged)\b"),
        (r"\bfailed\b", r"\b(?:succeeded|achieved|accomplished)\b"),
    ]
    return _paired_marker_mismatch(
        answer_lower, evidence_lower, polarity_contradictions
    )


def _has_reversed_before_after(answer_lower: str, evidence_lower: str) -> bool:
    """Detect a flipped temporal marker between the same two clauses."""
    clauses = _split_temporal_claim(answer_lower)
    if len(clauses) != 3:
        return False
    first, marker, second = clauses
    opposite = "after" if marker == "before" else "before"
    reversed_claim = [first, opposite, second]
    return any(
        _split_temporal_claim(segment.text) == reversed_claim
        for segment in SimpleSegmenter().segment(evidence_lower)
    )


def _split_temporal_claim(text: str) -> list[str]:
    normalized = " ".join(text.split()).rstrip(".!?;")
    return re.split(r",? (before|after) ", normalized)


def _paired_marker_mismatch(
    answer_lower: str,
    evidence_lower: str,
    pairs: list[tuple[str, str]],
) -> bool:
    """Return True if exactly one side of a contradictory marker pair matches."""
    for left, right in pairs:
        answer_left = bool(re.search(left, answer_lower))
        answer_right = bool(re.search(right, answer_lower))
        evidence_left = bool(re.search(left, evidence_lower))
        evidence_right = bool(re.search(right, evidence_lower))
        if (evidence_left and answer_right) or (evidence_right and answer_left):
            return True
    return False


def has_number_context_mismatch(answer_text: str, evidence_text: str) -> bool:
    """Check if numbers appear in different slots (leftover n-gram issue).

    Shared numbers with different content words around them (e.g. "10 rebounds"
    vs "10 of which came in the first half") are a contradiction.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if number context mismatch detected
    """
    answer_numbers = extract_numbers(answer_text)
    evidence_numbers = extract_numbers(evidence_text)
    shared_numbers = set(answer_numbers) & set(evidence_numbers)
    if not shared_numbers:
        return False

    for number in shared_numbers:
        if _number_slot_mismatch(
            extract_number_context(answer_text, number),
            extract_number_context(evidence_text, number),
            evidence_text,
            number,
        ):
            return True
    return _has_marked_number_slot_mismatch(answer_text, evidence_text)


def _has_marked_number_slot_mismatch(answer_text: str, evidence_text: str) -> bool:
    evidence_slots: dict[tuple[str, str], set[tuple[str, str, bool]]] = {}
    for value, slots in _marked_number_slots(evidence_text).items():
        for restatement in _quantity_restatements(value):
            evidence_slots.setdefault(restatement, set()).update(slots)

    for quantity, words in _marked_number_slots(answer_text).items():
        if quantity not in evidence_slots:
            continue
        # Require a shared nearby qualifier before treating a different label
        # as a conflict. A quantity may appear in several slots in the passage.
        alternatives = evidence_slots[quantity]
        for qualifier, label, marked in words:
            labels = {
                word
                for anchor, word, other_marked in alternatives
                if anchor == qualifier and (marked or other_marked)
            }
            if label in labels:
                continue
            if any(
                (label in left and labels & right) or (label in right and labels & left)
                for left, right in _OPPOSING_QUANTITY_LABELS
            ):
                return True
    return False


def _marked_number_slots(
    text: str,
) -> dict[tuple[str, str], set[tuple[str, str, bool]]]:
    slots: dict[tuple[str, str], set[tuple[str, str, bool]]] = {}
    previous_end = 0
    for value, quantity_match, marked in _iter_quantities(text):
        before = _WORDS_BEFORE_QUANTITY.search(
            quantity_match.string, previous_end, quantity_match.start()
        )
        previous_end = quantity_match.end()
        if before:
            words = [
                word
                for word in before[0].casefold().replace(":", " ").split()
                if word not in STOPWORDS
            ]
            if len(words) >= 2:
                slots.setdefault(value, set()).add((words[-2], words[-1], marked))
        if value[1] in ("%", "percentage points"):
            after = _WORD_AFTER_QUANTITY.match(
                quantity_match.string, quantity_match.end()
            )
            if after:
                slots.setdefault(value, set()).add(("", after[1].casefold(), True))
    return slots


def _content_words(words: list[str]) -> set[str]:
    return {
        word
        for word in words
        if word not in _FUNCTION_WORDS and word not in _CURRENCY_CODES
    }


def _number_slot_mismatch(
    answer_context: list[str],
    evidence_context: list[str],
    evidence_text: str,
    number: str,
) -> bool:
    """True when a shared number is attached to different content words."""
    if not answer_context:
        return False
    if not evidence_context:
        return bool(
            evidence_ends_with_number(evidence_text, number)
            and _content_words(answer_context)
        )
    return bool(_content_words(answer_context) - _content_words(evidence_context))


def evidence_ends_with_number(text: str, number: str) -> bool:
    """Check if evidence text ends with the given number (or whitespace after it).

    Args:
        text: The text to check
        number: The number to look for at the end

    Returns:
        True if text ends with the number (possibly followed by whitespace/punctuation)
    """
    escaped_number = re.escape(number)
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
    escaped_number = re.escape(number)
    pattern = r"(\w+\s+)?(\w+\s+)?" + escaped_number + r"(\s+\w+)?(\s+\w+)?"
    match = re.search(pattern, text)
    if not match:
        return []

    context_words = []
    for group in match.groups():
        if group:
            words = group.strip().split()
            context_words.extend(w.lower() for w in words if not re.match(r"^\d", w))
    return context_words


_CONTRADICTION_CHECKS = (
    has_negation_mismatch,
    has_number_mismatch,
    has_entity_swap,
    has_temporal_polarity_mismatch,
    has_number_context_mismatch,
)


def iter_contradiction_checks(answer_text: str, evidence_text: str) -> Iterator[str]:
    """Yield the names of triggered checks in evaluation order."""
    for check in _CONTRADICTION_CHECKS:
        if check(answer_text, evidence_text):
            yield check.__name__
