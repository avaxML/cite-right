from __future__ import annotations

import re

from cite_right.core.results import AnswerSpan
from cite_right.text.segmenter_simple import SimpleSegmenter


class SimpleAnswerSegmenter:
    def __init__(self) -> None:
        self._sentence_segmenter = SimpleSegmenter(split_on_newlines=False)

    def segment(self, text: str) -> list[AnswerSpan]:
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
