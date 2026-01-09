"""Simple rule-based sentence segmenter."""

from __future__ import annotations

from cite_right.core.results import Segment


class SimpleSegmenter:
    """Simple rule-based sentence segmenter.

    This segmenter splits text at sentence boundaries defined by periods, question marks,
    exclamation marks (with whitespace following), and semicolons. Optionally, it can also
    split on newline characters.
    """

    def __init__(self, split_on_newlines: bool = True) -> None:
        """Initializes the SimpleSegmenter.

        Args:
            split_on_newlines (bool, optional): If True, segments are split on newline ('\\n') characters.
                                                Defaults to True.
        """
        self.split_on_newlines = split_on_newlines

    def segment(self, text: str) -> list[Segment]:
        """Segments the input text into sentence-like segments.

        Sentences are detected based on '.', '?', '!', or ';', with some rules to check for
        boundary conditions (such as punctuation followed by whitespace). Optionally splits
        also on newlines, depending on the `split_on_newlines` setting.

        Args:
            text (str): The input text to segment.

        Returns:
            list[Segment]: A list of Segment objects, each containing a text span and its
                start and end character positions in the original text.
        """
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
    """Determines if the character at the given index is at a sentence boundary.

    Args:
        text (str): The input text.
        idx (int): The index of the punctuation character.

    Returns:
        bool: True if the punctuation is at the end of the text or is followed by whitespace.
    """
    if idx + 1 >= len(text):
        return True
    return text[idx + 1].isspace()


def _add_segment(text: str, start: int, end: int, segments: list[Segment]) -> None:
    """Adds a text segment to the segments list if it contains non-whitespace characters.

    Strips leading and trailing whitespace from the detected segment and calculates its
    positions in the original text.

    Args:
        text (str): The original text being segmented.
        start (int): The start character index of the candidate segment.
        end (int): The end character index of the candidate segment.
        segments (list[Segment]): The list to which a new Segment will be appended.
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
