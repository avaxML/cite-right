"""Interfaces for the core components of the citation alignment pipeline."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from cite_right.core.results import Alignment, AnswerSpan, Segment, TokenizedText


@runtime_checkable
class Tokenizer(Protocol):
    """Interface for text tokenization.

    Methods:
        tokenize(text): Tokenizes the input text into tokens and character spans.

    Example:
        >>> tokenizer: Tokenizer
        >>> result = tokenizer.tokenize("Example sentence.")
    """

    def tokenize(self, text: str) -> TokenizedText:
        """Tokenize the given text string.

        Args:
            text (str): The text to tokenize.

        Returns:
            TokenizedText: Tokenized representation of the input.
        """
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Interface for segmenting text into cohesive units such as sentences.

    Methods:
        segment(text): Splits the input text into segments.

    Example:
        >>> segmenter: Segmenter
        >>> segments = segmenter.segment("Sentence one. Sentence two.")
    """

    def segment(self, text: str) -> list[Segment]:
        """Segment the text string into a list of segments.

        Args:
            text (str): The text to segment.

        Returns:
            list[Segment]: List of text segments.
        """
        ...


@runtime_checkable
class AnswerSegmenter(Protocol):
    """Interface for segmenting text into answer spans.

    Methods:
        segment(text): Identifies and returns answer spans within the text.

    Example:
        >>> answer_segmenter: AnswerSegmenter
        >>> answer_spans = answer_segmenter.segment("Some answer text.")
    """

    def segment(self, text: str) -> list[AnswerSpan]:
        """Segment the text string into a list of answer spans.

        Args:
            text (str): The text to segment for answer spans.

        Returns:
            list[AnswerSpan]: List of answer spans within the text.
        """
        ...


@runtime_checkable
class Aligner(Protocol):
    """Interface for aligning two integer token sequences.

    Methods:
        align(seq1, seq2): Aligns two sequences and returns an Alignment result.

    Example:
        >>> aligner: Aligner
        >>> result = aligner.align([1, 2, 3], [2, 3, 4])
    """

    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment:
        """Align two token sequences and return the alignment result.

        Args:
            seq1 (Sequence[int]): The query or reference token sequence.
            seq2 (Sequence[int]): The candidate or document token sequence.

        Returns:
            Alignment: Result of aligning the two sequences.
        """
        ...
