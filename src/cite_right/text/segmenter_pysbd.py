from __future__ import annotations

from cite_right.core.results import Segment


class PySBDSegmenter:
    """Sentence segmenter using pySBD (Python Sentence Boundary Disambiguation).

    pySBD is a rule-based sentence boundary detection library that handles
    abbreviations, URLs, emails, and other edge cases without requiring
    a full NLP pipeline. It is significantly faster than spaCy while
    maintaining high accuracy.

    Install with: pip install cite-right[pysbd]
    """

    def __init__(self, language: str = "en", clean: bool = False) -> None:
        """Initialize the PySBD segmenter.

        Args:
            language: Language code for segmentation rules (default: "en").
            clean: If True, pySBD will clean the text before segmentation.
                   Default is False to preserve original text offsets.
        """
        try:
            import pysbd  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "pysbd is not installed. Install with 'pip install cite-right[pysbd]'."
            ) from exc

        self._segmenter = pysbd.Segmenter(language=language, clean=clean)
        self._language = language
        self._clean = clean

    def segment(self, text: str) -> list[Segment]:
        """Segment text into sentences.

        Args:
            text: The input text to segment.

        Returns:
            A list of Segment objects with accurate character offsets.
        """
        sentences = self._segmenter.segment(text)
        segments: list[Segment] = []
        cursor = 0

        for sentence in sentences:
            # Find the sentence in the original text starting from cursor
            start = text.find(sentence, cursor)
            if start == -1:
                # Handle edge case where pySBD may have modified whitespace
                stripped = sentence.strip()
                start = text.find(stripped, cursor)
                if start == -1:
                    continue
                sentence = stripped

            end = start + len(sentence)

            # Trim whitespace while preserving accurate offsets
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
                segments.append(
                    Segment(
                        text=text[seg_start:seg_end],
                        doc_char_start=seg_start,
                        doc_char_end=seg_end,
                    )
                )

            cursor = end

        return segments
