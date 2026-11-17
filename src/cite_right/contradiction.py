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
    if has_negation_mismatch(answer_text, evidence_text):
        return True
    if has_number_mismatch(answer_text, evidence_text):
        return True
    if has_entity_swap(answer_text, evidence_text):
        return True
    if has_temporal_polarity_mismatch(answer_text, evidence_text):
        return True
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
    """Check if numbers differ between answer and evidence.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if number mismatch detected
    """
    answer_numbers = extract_numbers(answer_text)
    evidence_numbers = extract_numbers(evidence_text)

    if answer_numbers and evidence_numbers:
        answer_set = set(answer_numbers)
        evidence_set = set(evidence_numbers)
        unique_in_answer = answer_set - evidence_set
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
    normalized_text = unicodedata.normalize("NFKC", text)
    number_pattern = r"\b\d+(?:[.,]\d+)*"
    numbers = re.findall(number_pattern, normalized_text)
    return [num.replace(",", "") for num in numbers]


_NAME_TOKEN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*")
_CLAUSE_BOUNDARY = re.compile(r"[:;.!?•\n]")
_LIST_MARKERS = frozenset({"-", "*", "+", "–", "—", "−"})
_NameRole = tuple[str, str]
_UNIT_NAMES = frozenset(
    {"eur", "usd", "gbp", "cad", "aud", "jpy", "kg", "mg", "km", "cm", "mm", "kwh"}
)
_CONTROL_VERBS = frozenset(
    {"click", "choose", "hit", "open", "press", "select", "tap", "toggle", "use"}
)
_CONTROL_LABELS = frozenset(
    {
        "button",
        "field",
        "menu",
        "option",
        "options",
        "panel",
        "screen",
        "tab",
        "toggle",
        "toolbar",
    }
)
_DISPLAY_VERBS = frozenset(
    {
        "display",
        "displays",
        "displayed",
        "list",
        "lists",
        "listed",
        "show",
        "shows",
        "showed",
    }
)
_PRODUCT_ROLES = frozenset({"edition", "plan", "product", "subscription", "tier"})
_DECLARED_NAME_ROLES = _PRODUCT_ROLES | {
    "approver",
    "author",
    "owner",
    "recipient",
    "reviewer",
    "winner",
}
_COPULAS = frozenset({"is", "was", "are", "were"})
_ARTICLES = frozenset({"a", "an", "the"})
_CLAUSE_START_ADVERBS = frozenset(
    {
        "accidentally",
        "already",
        "automatically",
        "currently",
        "finally",
        "immediately",
        "just",
        "manually",
        "now",
        "previously",
        "quickly",
        "recently",
        "slowly",
        "successfully",
        "temporarily",
        "unexpectedly",
        "usually",
    }
)
_NAME_PREPOSITIONS = frozenset({"at", "by", "for", "from", "in", "of", "to", "with"})
_NON_NAME_WORDS = (
    _SENTENCE_START_STOPWORDS
    | _UNIT_NAMES
    | _CONTROL_VERBS
    | _CONTROL_LABELS
    | _COPULAS
    | {
        "also",
        "step",
        "up",
        "view",
    }
)


def has_entity_swap(answer_text: str, evidence_text: str) -> bool:
    """Check for absent names occupying a corresponding role on both sides."""
    answer_words, answer_names = _find_named_roles(answer_text)
    evidence_words, evidence_names = _find_named_roles(evidence_text)
    answer_roles = {
        role
        for name, roles in answer_names
        if name not in evidence_words
        for role in roles
    }
    evidence_roles = {
        role
        for name, roles in evidence_names
        if name not in answer_words
        for role in roles
    }
    return not answer_roles.isdisjoint(evidence_roles)


def extract_entities(text: str) -> list[str]:
    """Return casefolded mentions that can participate in entity comparison."""
    _, names = _find_named_roles(text)
    return [name for name, _ in names]


def _find_named_roles(
    text: str,
) -> tuple[set[str], list[tuple[str, list[_NameRole]]]]:
    normalized = unicodedata.normalize("NFKC", text)
    matches = list(_NAME_TOKEN.finditer(normalized))
    raw_words = [match.group() for match in matches]
    words = [word.casefold() for word in raw_words]
    heading = _looks_like_heading(normalized, raw_words, words)

    mentions: list[tuple[str, list[_NameRole]]] = []
    clause_start = previous_end = 0
    for index, (raw, word) in enumerate(zip(raw_words, words, strict=True)):
        leading = normalized[previous_end : matches[index].start()]
        previous_end = matches[index].end()
        if _is_clause_gap(leading):
            clause_start = index
        if not _looks_like_name(raw, word):
            continue
        after = _right_anchor(normalized, matches, words, index)
        roles = _local_name_roles(words, index, leading, after, heading, clause_start)
        if roles:
            mentions.append((word, roles))
    return set(words), mentions


def _looks_like_heading(text: str, raw_words: list[str], words: list[str]) -> bool:
    if (
        len(words) >= 4
        and _ends_with_assertion(text)
        and raw_words[0].isupper()
        and not _looks_like_adverb(words[0])
        and raw_words[1].isupper()
        and words[1].endswith("ed")
        and words[2] in _ARTICLES
    ):
        return False
    return all(
        raw[0].isupper() or word in _FUNCTION_WORDS
        for raw, word in zip(raw_words, words, strict=True)
    )


def _ends_with_assertion(text: str) -> bool:
    trimmed = text.rstrip().rstrip('”’»)]}"').rstrip("'").rstrip()
    ending = trimmed[len(trimmed.rstrip(".!?")) :]
    return bool(ending) and "?" not in ending


def _looks_like_adverb(word: str) -> bool:
    return word in _CLAUSE_START_ADVERBS or word.endswith(
        ("ately", "ably", "edly", "ently", "ibly", "ically", "ously")
    )


def _looks_like_name(raw: str, folded: str) -> bool:
    return len(raw) >= 2 and raw[0].isupper() and folded not in _NON_NAME_WORDS


def _local_name_roles(
    words: list[str],
    index: int,
    leading: str,
    after: str,
    heading: bool,
    clause_start: int,
) -> list[_NameRole]:
    before = words[index - 1] if index else ""
    explicit = _explicit_name_role(words, index, before, after, leading, clause_start)
    if explicit is not None:
        return [explicit]
    if heading or _interface_or_layout(
        words, index, leading, before, after, clause_start
    ):
        return []

    roles: list[_NameRole] = []
    if _has_left_anchor(before, after, leading):
        roles.append(("after", before))
    if after and after not in _SENTENCE_START_STOPWORDS:
        roles.append(("before", after))
    return roles


def _is_clause_gap(gap: str) -> bool:
    return _CLAUSE_BOUNDARY.search(gap) is not None or gap.strip() in _LIST_MARKERS


def _has_left_anchor(before: str, after: str, leading: str) -> bool:
    if _is_clause_gap(leading):
        return False
    if before in _NAME_PREPOSITIONS or (before.endswith("ed") and after in _ARTICLES):
        return True
    return not after and before.isalpha() and before not in _NON_NAME_WORDS


def _right_anchor(
    text: str, matches: list[re.Match[str]], words: list[str], index: int
) -> str:
    next_index = index + 1
    if next_index == len(words):
        return ""
    if _is_clause_gap(text[matches[index].end() : matches[next_index].start()]):
        return ""
    after = words[next_index]
    if after.endswith("ly") and next_index + 1 < len(words):
        gap = text[matches[next_index].end() : matches[next_index + 1].start()]
        return "" if _is_clause_gap(gap) else words[next_index + 1]
    return after


def _explicit_name_role(
    words: list[str],
    index: int,
    before: str,
    after: str,
    leading: str,
    clause_start: int,
) -> _NameRole | None:
    if after in _PRODUCT_ROLES:
        return ("before", after)
    if before in _DECLARED_NAME_ROLES and ":" in leading:
        return ("after", before)
    subject = _role_before_copula(words, index, clause_start)
    return ("after", subject) if subject in _DECLARED_NAME_ROLES else None


def _role_before_copula(words: list[str], index: int, clause_start: int) -> str:
    if index - clause_start < 2:
        return ""
    verb, before_verb = words[index - 1], words[index - 2]
    if verb in _COPULAS:
        return before_verb
    if (
        verb == "been"
        and index - clause_start >= 3
        and before_verb in {"has", "have", "had"}
    ):
        return words[index - 3]
    return ""


def _interface_or_layout(
    words: list[str],
    index: int,
    leading: str,
    before: str,
    after: str,
    clause_start: int,
) -> bool:
    if any(separator in leading for separator in (":", "•", "\n")):
        return True
    if leading.strip() in _LIST_MARKERS:
        return True
    if before.isdigit() and leading.strip() in {".", ")"}:
        return True
    if after in _CONTROL_LABELS:
        return True
    if index == clause_start + 1 and before in _COPULAS:
        return True
    recent_words = words[max(clause_start, index - 3) : index]
    return _has_control_context(recent_words, before)


def _has_control_context(recent_words: list[str], before: str) -> bool:
    return any(word in _CONTROL_VERBS for word in recent_words) or (
        before in _DISPLAY_VERBS
        and any(word in _CONTROL_LABELS for word in recent_words[:-1])
    )


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
    return False


def _content_words(words: list[str]) -> set[str]:
    return {word for word in words if word not in _FUNCTION_WORDS}


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
