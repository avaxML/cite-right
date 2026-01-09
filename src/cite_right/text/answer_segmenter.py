"""Segments text into sentences within paragraphs using simple heuristics."""
from __future__ import annotations

import re

from cite_right.core.results import AnswerSpan
from cite_right.text.segmenter_simple import SimpleSegmenter


class SimpleAnswerSegmenter:
    """Segments text into sentences within paragraphs using simple heuristics.

    This class uses a simple rule-based sentence segmenter to split
    text into answer spans (sentences), preserving paragraph boundaries.
    """

    def __init__(self) -> None:
        """Initializes the SimpleAnswerSegmenter."""
        self._sentence_segmenter = SimpleSegmenter(split_on_newlines=False)

    def segment(self, text: str) -> list[AnswerSpan]:
        """Segments the input text into answer spans (sentences).

        The input is split into paragraphs (delimited by two or more linebreaks),
        and each paragraph is then segmented into sentences. Each AnswerSpan
        represents a sentence with its character offsets in the original text.

        Args:
            text (str): The input text to segment.

        Returns:
            list[AnswerSpan]: A list of AnswerSpan objects, each representing a
                detected sentence, with `char_start` and `char_end` referencing
                the span in the original text, and paragraph/sentence indices.
        """
        spans: list[AnswerSpan] = []
        sentence_index = 0

        for paragraph_index, (para_start, para_end) in enumerate(
            _iter_paragraph_spans(text)
        ):
            paragraph_text = text[para_start:para_end]
            sentences = self._sentence_segmenter.segment(paragraph_text)
            for sentence in sentences:
                spans.append(
                    AnswerSpan(
                        text=sentence.text,
                        char_start=para_start + sentence.doc_char_start,
                        char_end=para_start + sentence.doc_char_end,
                        kind="sentence",
                        paragraph_index=paragraph_index,
                        sentence_index=sentence_index,
                    )
                )
                sentence_index += 1

        return spans


_PARA_BREAK_RE = re.compile(r"\n[ \t]*\n+")


def _iter_paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Yields the character spans for each paragraph in the text.

    Paragraphs are separated by two or more consecutive linebreaks. Each
    span excludes leading or trailing whitespace.

    Args:
        text (str): The input text.

    Returns:
        list[tuple[int, int]]: A list of (start, end) tuples giving the
            character offsets of each paragraph in the original text,
            with whitespace trimmed.
    """
    spans: list[tuple[int, int]] = []
    start = 0

    for match in _PARA_BREAK_RE.finditer(text):
        end = match.start()
        paragraph = _trim_span(text, start, end)
        if paragraph is not None:
            spans.append(paragraph)
        start = match.end()

    paragraph = _trim_span(text, start, len(text))
    if paragraph is not None:
        spans.append(paragraph)

    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trims leading/trailing whitespace from a substring range.

    Args:
        text (str): The full text.
        start (int): Start character index (inclusive).
        end (int): End character index (exclusive).

    Returns:
        tuple[int, int] | None: Returns (trimmed_start, trimmed_end) if the
            span is non-empty after trimming, otherwise None.
    """
    if start >= end:
        return None
    snippet = text[start:end]
    if not snippet.strip():
        return None
    left_trim = len(snippet) - len(snippet.lstrip())
    right_trim = len(snippet) - len(snippet.rstrip())
    trimmed_start = start + left_trim
    trimmed_end = end - right_trim
    if trimmed_start >= trimmed_end:
        return None
    return trimmed_start, trimmed_end
