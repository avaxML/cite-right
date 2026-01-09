"""Generates passages of text from segmented content."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from cite_right.core.interfaces import Segmenter
from cite_right.core.results import Segment


class Passage(BaseModel):
    """A passage of text corresponding to consecutive segments.

    Attributes:
        text (str): The passage text.
        doc_char_start (int): Start character index in the original document.
        doc_char_end (int): End character index (exclusive) in the original document.
        segment_start (int): Start segment index (inclusive).
        segment_end (int): End segment index (exclusive).
    """

    model_config = ConfigDict(frozen=True)

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
    """Generate a list of passages from the text using a sliding window over segments.

    Args:
        text (str): The input text to be segmented and grouped.
        segmenter (Segmenter): The segmenter to use for splitting text into segments (e.g., sentences).
        window_size_sentences (int, optional): The number of segments per passage window. Defaults to 1.
        window_stride_sentences (int, optional): The stride for the sliding window, in segments. Defaults to 1.

    Returns:
        list[Passage]: A list of Passage objects, each containing `window_size_sentences` consecutive segments,
            sliding by `window_stride_sentences`.
    """
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
    """Create a Passage object from a consecutive window of segments.

    Args:
        text (str): The original text.
        segments (list[Segment]): List of segment objects for the text.
        start_idx (int): Start index for the window (inclusive).
        end_idx (int): End index for the window (exclusive).

    Returns:
        Passage: The passage corresponding to the concatenated window of segments.
    """
    start = segments[start_idx].doc_char_start
    end = segments[end_idx - 1].doc_char_end
    return Passage(
        text=text[start:end],
        doc_char_start=start,
        doc_char_end=end,
        segment_start=start_idx,
        segment_end=end_idx,
    )
