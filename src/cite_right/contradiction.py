"""Lightweight contradiction detection for citation validation.

Checks for:
- Negation mismatches (answer says "X is not Y" but source says "X is Y")
- Number mismatches (answer says "15%" but source says "20%")
- Entity-token mismatches in key positions
- Control targets that differ in matching instructions
- Temporal/polarity markers (BC vs ago, oppose vs support, etc.)
- Number context mismatches (same number, different slot)
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterator

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


_NEGATION_PATTERNS = (
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
)

_TEMPORAL_CONTRADICTIONS = (
    (r"\bbc\b", r"\bago\b"),
    (r"\bce\b", r"\bbc\b"),
    (r"\bad\b", r"\bbc\b"),
    (r"\bbce\b", r"\bce\b"),
)

_POLARITY_CONTRADICTIONS = (
    (r"\boppose[ds]?\b", r"\b(?:support|urge[ds]?|advocate[ds]?|promote[ds]?)\b"),
    (r"\bsupport[s]?\b", r"\b(?:oppose[ds]?|reject[s]?|resist[s]?)\b"),
    (r"\breject[s]?\b", r"\b(?:accept[s]?|approve[s]?|endorse[ds]?)\b"),
    (r"\bdenied\b", r"\b(?:confirmed|admitted|acknowledged)\b"),
    (r"\bfailed\b", r"\b(?:succeeded|achieved|accomplished)\b"),
)


def is_protected_alias(term: str) -> bool:
    """Return whether an alias would equate a contradiction marker or number."""
    return bool(
        extract_numbers(term)
        or contains_negation(term)
        or any(
            re.search(pattern, term)
            for pair in (*_TEMPORAL_CONTRADICTIONS, *_POLARITY_CONTRADICTIONS)
            for pattern in pair
        )
    )


def check_contradiction(
    answer_text: str,
    evidence_text: str,
    *,
    entity_aliases: Callable[[str], str] | None = None,
    normalize_negation: Callable[[str], str] | None = None,
    normalize_temporal_clause: Callable[[str], str] | None = None,
) -> bool:
    """Check if evidence contradicts the answer.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if contradiction is detected, False otherwise
    """
    checks = iter_contradiction_checks(
        answer_text,
        evidence_text,
        entity_aliases=entity_aliases,
        normalize_negation=normalize_negation,
        normalize_temporal_clause=normalize_temporal_clause,
    )
    return next(checks, None) is not None


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

    for pattern in _NEGATION_PATTERNS:
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


_NAME_TOKEN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*")
_CLAUSE_BOUNDARY = re.compile(r"[:;.!?•\n]")
_LIST_MARKERS = frozenset({"-", "*", "+", "–", "—", "−"})
_NameRole = tuple[str, str]
_UNIT_NAMES = frozenset(
    {"eur", "usd", "gbp", "cad", "aud", "jpy", "kg", "mg", "km", "cm", "mm", "kwh"}
)
_CONTROL_ACTIONS = frozenset({"click", "clicked", "clicks", "press", "tap"})
_CONTROL_VERBS = _CONTROL_ACTIONS | {
    "choose",
    "hit",
    "open",
    "select",
    "toggle",
    "use",
}
_CONTROL_INSTRUCTION = re.compile(
    r"""(?:^|(?<=[.!?,;\n])|\b(?:and|then)\b)[ \t]*(?:please[ \t]+)?
    (?:click|press|tap)[ \t]+(?:the[ \t]+)?
    (?:["“'‘`]([^"”'’`\n]+)["”'’`]|([^,.:;!?\n]+?))
    (?=[ \t]+(?:to|when|after|before|if|on|in|under|then|and)\b|[,.:;!?\n]|$)
    """,
    re.VERBOSE,
)
_CONTROL_CONTEXT_END = re.compile(r"[,.:;!?\n]|\b(?:and[ \t]+)?then\b")
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
_APPOSITIVE = re.compile(r"[ \t]*,[ \t]*(?:a|an|the)[ \t]+[^,.:;!?•\n]+,", re.I)
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
_RECIPIENT_VERBS = frozenset(
    {
        "assigned",
        "awarded",
        "brought",
        "emailed",
        "gave",
        "granted",
        "handed",
        "lent",
        "mailed",
        "offered",
        "paid",
        "promised",
        "sent",
        "showed",
        "taught",
        "told",
        "wrote",
    }
)
_CLAUSE_START_ADVERBS = frozenset(
    {
        "accidentally",
        "already",
        "automatically",
        "currently",
        "finally",
        "frequently",
        "immediately",
        "just",
        "manually",
        "now",
        "often",
        "previously",
        "quickly",
        "rarely",
        "recently",
        "seldom",
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


def has_entity_swap(
    answer_text: str,
    evidence_text: str,
    *,
    entity_aliases: Callable[[str], str] | None = None,
) -> bool:
    """Check for different names occupying corresponding roles."""
    answer_words, answer_names = _find_named_roles(answer_text)
    evidence_words, evidence_names = _find_named_roles(evidence_text)
    if entity_aliases is not None:
        answer_words = set(map(entity_aliases, answer_words))
        evidence_words = set(map(entity_aliases, evidence_words))
        answer_names = [(entity_aliases(name), roles) for name, roles in answer_names]
        evidence_names = [
            (entity_aliases(name), roles) for name, roles in evidence_names
        ]
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
    return not answer_roles.isdisjoint(evidence_roles) or _names_exchanged_roles(
        answer_names, evidence_names
    )


def _names_exchanged_roles(
    answer_names: list[tuple[str, list[_NameRole]]],
    evidence_names: list[tuple[str, list[_NameRole]]],
) -> bool:
    def by_role(
        mentions: list[tuple[str, list[_NameRole]]],
    ) -> dict[_NameRole, set[str]]:
        result: dict[_NameRole, set[str]] = {}
        for name, roles in mentions:
            for role in roles:
                result.setdefault(role, set()).add(name)
        return result

    answer = by_role(answer_names)
    evidence = by_role(evidence_names)
    replacements = set()
    for role in answer.keys() & evidence.keys():
        if len(answer[role]) == len(evidence[role]) == 1:
            (answer_name,) = answer[role]
            (evidence_name,) = evidence[role]
            if answer_name == evidence_name:
                continue
            if (evidence_name, answer_name) in replacements:
                return True
            replacements.add((answer_name, evidence_name))
    return False


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
    right_anchors = _right_anchors(normalized, matches, raw_words, words)

    mentions: list[tuple[str, list[_NameRole]]] = []
    clause_start = previous_end = 0
    for index, (raw, word) in enumerate(zip(raw_words, words, strict=True)):
        leading = normalized[previous_end : matches[index].start()]
        previous_end = matches[index].end()
        if _is_clause_gap(leading):
            clause_start = index
        if not _looks_like_name(raw, word):
            continue
        roles = _local_name_roles(
            words, index, leading, right_anchors[index], heading, clause_start
        )
        if roles:
            mentions.append((word, roles))
    return set(words), mentions


def _looks_like_heading(text: str, raw_words: list[str], words: list[str]) -> bool:
    if not all(
        raw[0].isupper() or word in _FUNCTION_WORDS
        for raw, word in zip(raw_words, words, strict=True)
    ):
        return False
    if len(words) < 4 or not _ends_with_assertion(text):
        return True
    if words[0] in _ARTICLES and _role_before_copula(words, len(words) - 1, 0):
        return False
    return not (
        raw_words[0].isupper()
        and not _looks_like_adverb(words[0])
        and raw_words[1].isupper()
        and words[1].endswith("ed")
        and words[2] in _ARTICLES
    )


def _ends_with_assertion(text: str) -> bool:
    end = len(text)
    while end:
        char = text[end - 1]
        if not (
            char.isspace()
            or char in "\"'"
            or unicodedata.category(char) in {"Pe", "Pf", "Pi"}
        ):
            break
        end -= 1
    statement = text[:end]
    ending = statement[len(statement.rstrip(".!?")) :]
    return bool(ending) and "?" not in ending


def _looks_like_adverb(word: str) -> bool:
    return word in _CLAUSE_START_ADVERBS or word.endswith(
        ("ately", "ably", "edly", "ently", "etely", "fully", "ibly", "ically", "ously")
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
    subject = _role_before_copula(words, index, clause_start)
    if subject:
        roles.append(("after", subject))
    elif _has_left_anchor(before, after, leading):
        roles.append(("after", before))
    if after and after not in _SENTENCE_START_STOPWORDS:
        roles.append(("before", after))
    return roles


def _is_clause_gap(gap: str) -> bool:
    return _CLAUSE_BOUNDARY.search(gap) is not None or gap.strip() in _LIST_MARKERS


def _has_left_anchor(before: str, after: str, leading: str) -> bool:
    if _is_clause_gap(leading):
        return False
    if before in _NAME_PREPOSITIONS or (
        before in _RECIPIENT_VERBS and after in _ARTICLES
    ):
        return True
    return (
        (not after or after in _NAME_PREPOSITIONS)
        and before.isalpha()
        and (before not in _NON_NAME_WORDS or before in _CONTROL_VERBS)
    )


def _right_anchors(
    text: str, matches: list[re.Match[str]], raw_words: list[str], words: list[str]
) -> list[str]:
    # Work backwards so each name in a coordinated subject can reuse the next anchor.
    anchors = [""] * len(words)
    for index in range(len(words) - 2, -1, -1):
        end = matches[index].end()
        following = index + 1
        gap = text[end : matches[following].start()]
        if _is_clause_gap(gap):
            continue
        if gap.strip() == "," and words[following] in _ARTICLES:
            appositive = _APPOSITIVE.match(text, end)
            if appositive is not None:
                end = appositive.end()
                while following < len(words) and matches[following].start() < end:
                    following += 1
                if following == len(words) or _is_clause_gap(
                    text[end : matches[following].start()]
                ):
                    continue
        after = words[following]
        if after in {"and", "or"} and following + 1 < len(words):
            if not _is_clause_gap(
                text[matches[following].end() : matches[following + 1].start()]
            ) and _looks_like_name(raw_words[following + 1], words[following + 1]):
                anchors[index] = anchors[following + 1]
                continue
        if after.endswith("ly") and following + 1 < len(words):
            gap = text[matches[following].end() : matches[following + 1].start()]
            after = "" if _is_clause_gap(gap) else words[following + 1]
        anchors[index] = after
    return anchors


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
    return _has_control_context(recent_words, before, words[clause_start])


def _has_control_context(recent_words: list[str], before: str, start: str) -> bool:
    return (
        any(word in _CONTROL_ACTIONS for word in recent_words)
        or (start in _CONTROL_VERBS and before in _CONTROL_VERBS)
        or (
            before in _DISPLAY_VERBS
            and any(word in _CONTROL_LABELS for word in recent_words[:-1])
        )
    )


def _control_steps(
    text: str, aliases: Callable[[str], str] | None
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    matches = list(_CONTROL_INSTRUCTION.finditer(normalized))
    steps = []
    for index, match in enumerate(matches):
        quoted, plain = match.groups()
        words = _NAME_TOKEN.findall(quoted or plain)
        if plain and len(words) > 1 and words[-1] in _CONTROL_LABELS:
            words.pop()
        if not words:
            continue
        end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        )
        boundary = _CONTROL_CONTEXT_END.search(normalized, match.end(), end)
        context = _NAME_TOKEN.findall(
            normalized, match.end(), boundary.start() if boundary else end
        )
        if aliases is not None:
            words = list(map(aliases, words))
            context = list(map(aliases, context))
        steps.append((tuple(context), tuple(words)))
    return steps


def has_control_target_mismatch(
    answer_text: str,
    evidence_text: str,
    *,
    entity_aliases: Callable[[str], str] | None = None,
) -> bool:
    """Detect substituted targets in explicit click, press, or tap instructions."""
    answer_steps = _control_steps(answer_text, entity_aliases)
    if not answer_steps:
        return False
    evidence_steps = _control_steps(evidence_text, entity_aliases)
    if [context for context, _ in answer_steps] == [
        context for context, _ in evidence_steps
    ]:
        return answer_steps != evidence_steps

    # A target can occur in the passage and still belong to a different step.
    evidence_by_context: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    for context, target in evidence_steps:
        if context:
            evidence_by_context.setdefault(context, set()).add(target)
    if any(
        context in evidence_by_context and target not in evidence_by_context[context]
        for context, target in answer_steps
    ):
        return True

    answer_targets = {target for _, target in answer_steps}
    evidence_targets = {target for _, target in evidence_steps}
    return bool(answer_targets - evidence_targets and evidence_targets - answer_targets)


def has_temporal_polarity_mismatch(
    answer_text: str,
    evidence_text: str,
    *,
    normalize_clause: Callable[[str], str] | None = None,
) -> bool:
    """Check for temporal or polarity contradictions.

    Args:
        answer_text: The answer claim text
        evidence_text: The evidence text from source

    Returns:
        True if temporal/polarity mismatch detected
    """
    answer_lower = answer_text.lower()
    evidence_lower = evidence_text.lower()

    if _paired_marker_mismatch(answer_lower, evidence_lower, _TEMPORAL_CONTRADICTIONS):
        return True
    if _has_reversed_before_after(answer_lower, evidence_lower, normalize_clause):
        return True

    return _paired_marker_mismatch(
        answer_lower, evidence_lower, _POLARITY_CONTRADICTIONS
    )


def _has_reversed_before_after(
    answer_lower: str,
    evidence_lower: str,
    normalize_clause: Callable[[str], str] | None,
) -> bool:
    """Detect a flipped temporal marker between the same two clauses."""
    clauses = _split_temporal_claim(answer_lower, normalize_clause)
    if len(clauses) != 3:
        return False
    first, marker, second = clauses
    opposite = "after" if marker == "before" else "before"
    reversed_claim = [first, opposite, second]
    return any(
        _split_temporal_claim(segment.text, normalize_clause) == reversed_claim
        for segment in SimpleSegmenter().segment(evidence_lower)
    )


def _split_temporal_claim(
    text: str, normalize_clause: Callable[[str], str] | None
) -> list[str]:
    normalized = " ".join(text.split()).rstrip(".!?;")
    clauses = re.split(r",? (before|after) ", normalized)
    if normalize_clause is not None and len(clauses) == 3:
        first, marker, second = clauses
        return [normalize_clause(first), marker, normalize_clause(second)]
    return clauses


def _paired_marker_mismatch(
    answer_lower: str,
    evidence_lower: str,
    pairs: tuple[tuple[str, str], ...],
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


def iter_contradiction_checks(
    answer_text: str,
    evidence_text: str,
    *,
    entity_aliases: Callable[[str], str] | None = None,
    normalize_negation: Callable[[str], str] | None = None,
    normalize_temporal_clause: Callable[[str], str] | None = None,
) -> Iterator[str]:
    """Yield the names of triggered checks in evaluation order."""
    negation_answer, negation_evidence = answer_text, evidence_text
    if normalize_negation is not None:
        negation_answer = normalize_negation(answer_text)
        negation_evidence = normalize_negation(evidence_text)
    if has_negation_mismatch(negation_answer, negation_evidence):
        yield "has_negation_mismatch"
    if has_number_mismatch(answer_text, evidence_text):
        yield "has_number_mismatch"
    if has_entity_swap(answer_text, evidence_text, entity_aliases=entity_aliases):
        yield "has_entity_swap"
    if has_temporal_polarity_mismatch(
        answer_text, evidence_text, normalize_clause=normalize_temporal_clause
    ):
        yield "has_temporal_polarity_mismatch"
    if has_number_context_mismatch(answer_text, evidence_text):
        yield "has_number_context_mismatch"
    if has_control_target_mismatch(
        answer_text, evidence_text, entity_aliases=entity_aliases
    ):
        yield "has_control_target_mismatch"
