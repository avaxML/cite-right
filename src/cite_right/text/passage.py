from __future__ import annotations

from dataclasses import dataclass

from cite_right.core.interfaces import Segmenter
from cite_right.core.results import Segment


@dataclass(frozen=True, slots=True)
class Passage:
    text: str
    doc_char_start: int
    doc_char_end: int
    segment_start: int
    segment_end: int


def generate_passages(
    text: str,
    *,
    segmenter: Segmenter,
    window_size_sentences: int = 1,
    window_stride_sentences: int = 1,
) -> list[Passage]:
    segments = segmenter.segment(text)
    if not segments:
        return []

    window = max(1, window_size_sentences)
    stride = max(1, window_stride_sentences)

    passages: list[Passage] = []
    idx = 0

    while idx < len(segments):
        end_idx = min(len(segments), idx + window)
        passages.append(_window_from_segments(text, segments, idx, end_idx))
        if end_idx == len(segments):
            break
        idx += stride

    return passages


def _window_from_segments(
    text: str, segments: list[Segment], start_idx: int, end_idx: int
) -> Passage:
    start = segments[start_idx].doc_char_start
    end = segments[end_idx - 1].doc_char_end
    return Passage(
        text=text[start:end],
        doc_char_start=start,
        doc_char_end=end,
        segment_start=start_idx,
        segment_end=end_idx,
    )
