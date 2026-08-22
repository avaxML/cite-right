"""Tests for ONNX embedder."""

import os
from typing import Sequence

import pytest

# Only run these tests if ONNX dependencies are available
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
pytest.importorskip("tokenizers", reason="tokenizers not installed")

from cite_right.models.onnx_embedder import OnnxMiniLmEmbedder


@pytest.mark.skipif(
    os.environ.get("CITE_RIGHT_RUN_ONNX_TESTS") != "1",
    reason="Set CITE_RIGHT_RUN_ONNX_TESTS=1 to run ONNX tests (downloads model)",
)
def test_onnx_embedder_encode_smoke() -> None:
    """Smoke test that ONNX embedder can encode text."""
    embedder = OnnxMiniLmEmbedder()

    texts = ["Hello world", "Goodbye world"]
    result = embedder.encode(texts)

    assert len(result) == 2
    assert all(isinstance(vec, list) for vec in result)
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    assert all(len(vec) == 384 for vec in result)
    assert all(isinstance(val, float) for vec in result for val in vec)


@pytest.mark.skipif(
    os.environ.get("CITE_RIGHT_RUN_ONNX_TESTS") != "1",
    reason="Set CITE_RIGHT_RUN_ONNX_TESTS=1 to run ONNX tests (downloads model)",
)
def test_onnx_embedder_cache_reuses_embeddings() -> None:
    """Test that the embedder caches embeddings for identical texts."""
    embedder = OnnxMiniLmEmbedder()

    text = "Climate change impacts coastal regions."
    first = embedder.encode([text])[0]
    second = embedder.encode([text])[0]

    # Should return identical cached results
    assert first == second


@pytest.mark.skipif(
    os.environ.get("CITE_RIGHT_RUN_ONNX_TESTS") != "1",
    reason="Set CITE_RIGHT_RUN_ONNX_TESTS=1 to run ONNX tests (downloads model)",
)
def test_onnx_embedder_batching() -> None:
    """Test that the embedder handles batch encoding."""
    embedder = OnnxMiniLmEmbedder()

    texts = [
        "First sentence.",
        "Second sentence with more words.",
        "Third.",
    ]
    result = embedder.encode(texts)

    assert len(result) == 3
    assert all(len(vec) == 384 for vec in result)


def test_onnx_embedder_requires_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OnnxMiniLmEmbedder raises if onnxruntime is not installed."""
    import sys

    # Remove onnxruntime from sys.modules to simulate it not being installed
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
        OnnxMiniLmEmbedder()


class _DummyEmbedder:
    """Dummy embedder for testing without downloading model."""

    def __init__(self) -> None:
        self.model_name = "dummy"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        # Return fixed-size dummy embeddings
        return [[float(i) for i in range(384)] for _ in texts]


@pytest.fixture
def dummy_embedder() -> _DummyEmbedder:
    """Provide a dummy embedder for tests that don't need real embeddings."""
    return _DummyEmbedder()
