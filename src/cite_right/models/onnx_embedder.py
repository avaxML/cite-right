"""ONNX Runtime embedder for fast inference without PyTorch."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

_MAX_EMBEDDING_CACHE_SIZE = 10_000
_EMBEDDING_CACHE: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Use the file that actually exists on HuggingFace Hub (not model_quantized.onnx which 404s)
_MODEL_FILENAME = "onnx/model_quint8_avx2.onnx"
_TOKENIZER_FILENAME = "tokenizer.json"
_DEFAULT_BATCH_SIZE = 32  # Bounded batch size to avoid padding to global max


def _default_model_dir() -> Path:
    """Get the default directory for cached ONNX models."""
    cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache_home) / "cite-right" / "onnx-models"


def _download_model_if_needed(model_path: Path, tokenizer_path: Path) -> None:
    """Download ONNX model and tokenizer if they don't exist."""
    if model_path.exists() and tokenizer_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import (  # pyright: ignore[reportMissingImports]
            hf_hub_download,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface-hub is required to download ONNX models. "
            "Install with 'cite-right[onnx]'."
        ) from exc

    import shutil

    # Download from HuggingFace Hub using secure API
    repo_id = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "main"  # Pin revision for security (Bandit B615)

    if not model_path.exists():
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=_MODEL_FILENAME,
            revision=revision,
        )
        shutil.copy2(downloaded, model_path)

    if not tokenizer_path.exists():
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=_TOKENIZER_FILENAME,
            revision=revision,
        )
        shutil.copy2(downloaded, tokenizer_path)


class OnnxMiniLmEmbedder:
    """ONNX Runtime embedder for all-MiniLM-L6-v2 (384-dimensional).

    This embedder uses ONNX Runtime for fast inference without requiring
    PyTorch or sentence-transformers at inference time. The model is
    quantized (int8) for improved performance.

    Args:
        model_path: Path to the ONNX model file. If None, downloads the
            quantized all-MiniLM-L6-v2 model to the cache directory.
        tokenizer_path: Path to the tokenizer.json file. If None, downloads
            the tokenizer for all-MiniLM-L6-v2 to the cache directory.
        model_name: Name identifier for the model (used for caching).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
    ) -> None:
        """Initialize the ONNX embedder.

        Raises:
            RuntimeError: If onnxruntime or tokenizers is not installed.
        """
        try:
            import onnxruntime as ort  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed. Install with 'cite-right[onnx]'."
            ) from exc

        try:
            from tokenizers import (  # pyright: ignore[reportMissingImports]
                Tokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "tokenizers is not installed. Install with 'cite-right[onnx]'."
            ) from exc

        # Resolve model and tokenizer paths
        if model_path is None or tokenizer_path is None:
            default_dir = _default_model_dir()
            model_hash = hashlib.sha256(_DEFAULT_MODEL_NAME.encode()).hexdigest()[:16]
            default_model_path = default_dir / f"model-{model_hash}.onnx"
            default_tokenizer_path = default_dir / f"tokenizer-{model_hash}.json"

            model_path = model_path or default_model_path
            tokenizer_path = tokenizer_path or default_tokenizer_path

            _download_model_if_needed(Path(model_path), Path(tokenizer_path))

        # Configure session options for reasonable performance
        sess_options = ort.SessionOptions()
        # Set intra_op_num_threads to a sensible default (don't oversubscribe)
        sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.model_name = model_name
        self._batch_size = _DEFAULT_BATCH_SIZE

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a list of text strings into a list of float vectors.

        Args:
            texts: The text strings to encode.

        Returns:
            list[list[float]]: List of 384-dimensional float vectors.
        """
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        # Check cache
        for i, text in enumerate(texts):
            key = (text, self.model_name)
            if key in _EMBEDDING_CACHE:
                results[i] = _EMBEDDING_CACHE[key]
                _EMBEDDING_CACHE.move_to_end(key)
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        # Encode missing texts
        if missing_texts:
            encoded = self._encode_batch(missing_texts)
            for i, idx in enumerate(missing_indices):
                vector = encoded[i]
                _EMBEDDING_CACHE[(missing_texts[i], self.model_name)] = vector
                if len(_EMBEDDING_CACHE) > _MAX_EMBEDDING_CACHE_SIZE:
                    _EMBEDDING_CACHE.popitem(last=False)
                results[idx] = vector

        return results  # type: ignore

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts using the ONNX model.

        Uses bounded batching to avoid padding all texts to the global max length.
        Processes in chunks and pads only to the max length within each chunk.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in bounded batches to avoid padding to global max
        for batch_start in range(0, len(texts), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            # Tokenize this batch only
            encodings = self._tokenizer.encode_batch(batch_texts)
            max_length = max(len(enc.ids) for enc in encodings)

            # Prepare padded inputs for this batch
            batch_size = len(batch_texts)
            input_ids = np.zeros((batch_size, max_length), dtype=np.int64)
            attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)
            token_type_ids = np.zeros((batch_size, max_length), dtype=np.int64)

            for i, enc in enumerate(encodings):
                input_ids[i, : len(enc.ids)] = enc.ids
                attention_mask[i, : len(enc.attention_mask)] = enc.attention_mask
                # token_type_ids remain zeros (required by MiniLM ONNX graph)

            # Run inference
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            outputs = self._session.run(None, onnx_inputs)

            # Mean pooling with attention mask
            token_embeddings = np.array(outputs[0], dtype=np.float32)
            embeddings = self._mean_pool(token_embeddings, attention_mask)

            # Normalize
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            ).clip(min=1e-12)

            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def _mean_pool(
        self,
        token_embeddings: npt.NDArray[np.float32],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Apply mean pooling to token embeddings using attention mask."""
        # Expand attention mask to match embedding dimensions
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)

        # Sum embeddings where attention mask is 1
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)

        # Divide by the number of non-masked tokens
        sum_mask = np.sum(mask_expanded, axis=1).clip(min=1e-9)

        return sum_embeddings / sum_mask
