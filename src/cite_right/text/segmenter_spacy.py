from __future__ import annotations

from typing import Any

from cite_right.core.results import Segment


class SpacySegmenter:
    def __init__(self, model: str = "en_core_web_sm") -> None:
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
        doc = self._nlp(text)
        segments: list[Segment] = []

        for sent in doc.sents:
            segments.extend(_split_sentence(text, sent))

        return segments


def _split_sentence(text: str, sent: Any) -> list[Segment]:
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
    head = token.head
    if head == sent.root:
        return head.pos_ in {"VERB", "AUX", "ADJ"}
    if head.dep_ == "conj" and head.head == sent.root:
        return head.pos_ in {"VERB", "AUX", "ADJ"}
    if head.dep_ == "ROOT":
        return head.pos_ in {"VERB", "AUX", "ADJ"}
    return False


def _skip_whitespace(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _add_segment(text: str, start: int, end: int, segments: list[Segment]) -> None:
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
