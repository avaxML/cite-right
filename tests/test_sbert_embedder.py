"""Tests for the sentence-transformer embedding cache."""

from collections import OrderedDict
from typing import Sequence

import pytest

import cite_right.models.sbert_embedder as sbert_module
from cite_right.models.sbert_embedder import SentenceTransformerEmbedder


class _EncodedRows:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class _FakeModel:
    def encode(self, texts: Sequence[str]) -> _EncodedRows:
        return _EncodedRows([[float(len(text))] for text in texts])


def test_embedding_cache_is_bounded_and_evicts_least_recently_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sbert_module, "_MAX_EMBEDDING_CACHE_SIZE", 2)
    monkeypatch.setattr(sbert_module, "_EMBEDDING_CACHE", OrderedDict())
    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder._model = _FakeModel()
    embedder.model_name = "fake"

    embedder.encode(["first", "second"])
    embedder.encode(["first"])
    embedder.encode(["third"])

    assert list(sbert_module._EMBEDDING_CACHE) == [
        ("first", "fake"),
        ("third", "fake"),
    ]
