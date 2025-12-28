from __future__ import annotations

import re

from cite_right.core.results import AnswerSpan
from cite_right.text.segmenter_spacy import _split_sentence


class SpacyAnswerSegmenter:
    def __init__(
        self,
        model: str = "en_core_web_sm",
        *,
        split_clauses: bool = False,
    ) -> None:
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

        self._split_clauses = split_clauses

    def segment(self, text: str) -> list[AnswerSpan]:
        spans: list[AnswerSpan] = []
        sentence_index = 0

        for paragraph_index, (para_start, para_end) in enumerate(
            _iter_paragraph_spans(text)
        ):
            paragraph_text = text[para_start:para_end]
            doc = self._nlp(paragraph_text)

            for sent in doc.sents:
                if self._split_clauses:
                    clauses = _split_sentence(paragraph_text, sent)
                    for clause in clauses:
                        spans.append(
                            AnswerSpan(
                                text=clause.text,
                                char_start=para_start + clause.doc_char_start,
                                char_end=para_start + clause.doc_char_end,
                                kind="clause",
                                paragraph_index=paragraph_index,
                                sentence_index=sentence_index,
                            )
                        )
                        sentence_index += 1
                    continue

                trimmed = _trim_span(paragraph_text, sent.start_char, sent.end_char)
                if trimmed is None:
                    continue
                start, end = trimmed
                spans.append(
                    AnswerSpan(
                        text=paragraph_text[start:end],
                        char_start=para_start + start,
                        char_end=para_start + end,
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
