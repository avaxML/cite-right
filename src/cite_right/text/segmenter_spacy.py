"""Sentence segmenter using spaCy with additional clause splitting on coordinating conjunctions."""
from __future__ import annotations

from typing import Any

from cite_right.core.results import Segment


class SpacySegmenter:
    """Sentence segmenter using spaCy with additional clause splitting on coordinating conjunctions.

    This segmenter uses a spaCy language model to split text into sentences, and then further divides
    sentences at clause-level conjunctions (such as "and", "or", "but") for finer granularity.
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """Initializes the SpacySegmenter with a specified spaCy language model.

        Args:
            model (str, optional): The name of the spaCy language model to load. Defaults to "en_core_web_sm".

        Raises:
            RuntimeError: If spaCy or the specified spaCy model is not installed.
        """
        try:
            import spacy  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "spaCy is not installed. Install with 'cite-right[spacy]'."
            ) from exc

        try:
            self._nlp = spacy.load(model)
        except OSError as exc:  # pragma: no cover - model guard
            raise RuntimeError(
                f"spaCy model '{model}' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc

    def segment(self, text: str) -> list[Segment]:
        """Segments the input text into sentences and further splits sentences at specific conjunctions.

        Args:
            text (str): The input text to be segmented.

        Returns:
            list[Segment]: A list of Segment objects representing the detected spans in the text.
        """
        doc = self._nlp(text)
        segments: list[Segment] = []

        for sent in doc.sents:
            segments.extend(_split_sentence(text, sent))

        return segments


def _split_sentence(text: str, sent: Any) -> list[Segment]:
    """Further splits a spaCy sentence at clause-level coordinating conjunctions.

    For a given `sent`, finds conjunction tokens ("and", "or", "but") functioning as clause connectors,
    and splits the sentence into smaller spans at these tokens.

    Args:
        text (str): The original document text.
        sent (Any): A spaCy Span object (sentence).

    Returns:
        list[Segment]: List of Segment objects corresponding to the finer-grained sentence segments.
    """
    markers: list[tuple[int, int]] = []
    for token in sent:
        if token.dep_ != "cc":
            continue
        if token.lower_ not in {"and", "or", "but"}:
            continue
        if not _is_clause_conjunction(token, sent):
            continue
        start = token.idx
        end = token.idx + len(token)
        if start <= sent.start_char or end >= sent.end_char:
            continue
        markers.append((start, end))

    markers.sort()
    segments: list[Segment] = []
    cursor = sent.start_char

    for start, end in markers:
        _add_segment(text, cursor, start, segments)
        cursor = _skip_whitespace(text, end)

    _add_segment(text, cursor, sent.end_char, segments)
    return segments


def _is_clause_conjunction(token: Any, sent: Any) -> bool:
    """Determines if a conjunction token is connecting clauses within a sentence.

    Args:
        token (Any): The spaCy Token with dependency label "cc".
        sent (Any): The parent spaCy Span (sentence).

    Returns:
        bool: True if the conjunction is likely to split clauses, False otherwise.
    """
    head = token.head
    if head == sent.root:
        return head.pos_ in {"VERB", "AUX", "ADJ"}
    if head.dep_ == "conj" and head.head == sent.root:
        return head.pos_ in {"VERB", "AUX", "ADJ"}
    if head.dep_ == "ROOT":
        return head.pos_ in {"VERB", "AUX", "ADJ"}
    return False


def _skip_whitespace(text: str, idx: int) -> int:
    """Advances an index past any contiguous whitespace.

    Args:
        text (str): The text being scanned.
        idx (int): The starting index.

    Returns:
        int: The index of the first non-whitespace character after `idx`.
    """
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _add_segment(text: str, start: int, end: int, segments: list[Segment]) -> None:
    """Appends a Segment for the substring [start:end] if it contains non-whitespace.

    Strips leading and trailing whitespace inside the range before creating the segment.

    Args:
        text (str): The original document text.
        start (int): Start index of the segment span.
        end (int): End index (exclusive) of the segment span.
        segments (list[Segment]): The list of segments to append to.
    """
    if start >= end:
        return
    snippet = text[start:end]
    stripped = snippet.strip()
    if not stripped:
        return
    left_trim = len(snippet) - len(snippet.lstrip())
    right_trim = len(snippet) - len(snippet.rstrip())
    seg_start = start + left_trim
    seg_end = end - right_trim
    if seg_start >= seg_end:
        return
    segments.append(
        Segment(
            text=text[seg_start:seg_end], doc_char_start=seg_start, doc_char_end=seg_end
        )
    )
