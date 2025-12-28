from __future__ import annotations

from cite_right.core.results import Segment


class SimpleSegmenter:
    def __init__(self, split_on_newlines: bool = True) -> None:
        self.split_on_newlines = split_on_newlines

    def segment(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        start = 0
        idx = 0
        length = len(text)

        while idx < length:
            char = text[idx]
            if char == "\n" and self.split_on_newlines:
                _add_segment(text, start, idx, segments)
                start = idx + 1
                idx += 1
                continue

            if char in ".?!" and _is_boundary(text, idx):
                end = idx + 1
                while end < length and text[end] in ".?!":
                    end += 1
                _add_segment(text, start, end, segments)
                start = end
                idx = end
                continue

            if char == ";":
                _add_segment(text, start, idx + 1, segments)
                start = idx + 1
                idx += 1
                continue

            idx += 1

        _add_segment(text, start, length, segments)
        return segments


def _is_boundary(text: str, idx: int) -> bool:
    if idx + 1 >= len(text):
        return True
    return text[idx + 1].isspace()


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
