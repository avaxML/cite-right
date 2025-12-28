from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from cite_right.core.results import Alignment, AnswerSpan, Segment, TokenizedText


@runtime_checkable
class Tokenizer(Protocol):
    def tokenize(self, text: str) -> TokenizedText: ...


@runtime_checkable
class Segmenter(Protocol):
    def segment(self, text: str) -> list[Segment]: ...


@runtime_checkable
class AnswerSegmenter(Protocol):
    def segment(self, text: str) -> list[AnswerSpan]: ...


@runtime_checkable
class Aligner(Protocol):
    def align(self, seq1: Sequence[int], seq2: Sequence[int]) -> Alignment: ...
