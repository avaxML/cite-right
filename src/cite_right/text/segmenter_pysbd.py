"""Sentence segmenter using pySBD (Python Sentence Boundary Disambiguation)."""

from __future__ import annotations

from functools import lru_cache

from cite_right.core.results import Segment


@lru_cache(maxsize=2000)
def _segment_pysbd_cached(
    text: str, language: str, clean: bool
) -> list[tuple[str, int, int]]:
    """Cached sentence boundary disambiguation using pySBD.

    Args:
        text (str): Input text to segment.
        language (str): Language code.
        clean (bool): Whether pySBD cleans text first.

    Returns:
        list[tuple[str, int, int]]: List of (sentence_text, start_offset, end_offset).
    """
    import pysbd  # pyright: ignore[reportMissingImports]

    segmenter = pysbd.Segmenter(language=language, clean=clean)
    sentences = segmenter.segment(text)
    segments: list[tuple[str, int, int]] = []
    cursor = 0

    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start == -1:
            stripped = sentence.strip()
            start = text.find(stripped, cursor)
            if start == -1:
                continue
            sentence = stripped

        end = start + len(sentence)

        snippet = text[start:end]
        stripped = snippet.strip()
        if not stripped:
            cursor = end
            continue

        left_trim = len(snippet) - len(snippet.lstrip())
        right_trim = len(snippet) - len(snippet.rstrip())
        seg_start = start + left_trim
        seg_end = end - right_trim

        if seg_start < seg_end:
            segments.append((text[seg_start:seg_end], seg_start, seg_end))

        cursor = end

    return segments


class PySBDSegmenter:
    """Sentence segmenter using pySBD (Python Sentence Boundary Disambiguation).

    This class utilizes pySBD, a rule-based sentence boundary disambiguation library,
    to split text into sentences. pySBD efficiently handles abbreviations, URLs, emails,
    and other edge cases, and does not require a full NLP pipeline. It is significantly
    faster than spaCy while maintaining high accuracy.

    To use this segmenter, install the required dependency:
        pip install cite-right[pysbd]
    """

    def __init__(self, language: str = "en", clean: bool = False) -> None:
        """Initializes the PySBDSegmenter.

        Args:
            language (str, optional): The language code for segmentation rules (default: "en").
            clean (bool, optional):
                If True, pySBD will clean the text before segmentation.
                Default is False, which preserves original text offsets for accurate mapping.
        Raises:
            RuntimeError: If pysbd is not installed.
        """
        try:
            import pysbd  # pyright: ignore[reportMissingImports]

            _ = pysbd
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "pysbd is not installed. Install with 'pip install cite-right[pysbd]'."
            ) from exc

        self._language = language
        self._clean = clean

    def segment(self, text: str) -> list[Segment]:
        """Segments the input text into sentences with accurate character offsets.

        Args:
            text (str): The input text to be segmented into sentences.

        Returns:
            list[Segment]:
                A list of Segment objects, each representing a detected sentence,
                with text and its character offsets in the original text.
        """
        cached_segments = _segment_pysbd_cached(text, self._language, self._clean)
        return [
            Segment(text=text_val, doc_char_start=start, doc_char_end=end)
            for text_val, start, end in cached_segments
        ]
