"""Document-level span embedder for efficient passage embedding."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt


class DocumentPassageSpan(Protocol):
    """Protocol for passage spans within a document."""

    doc_char_start: int
    doc_char_end: int


class DocumentSpanEmbedder:
    """Document-level embedder that pools token embeddings for passage spans.

    Instead of encoding each passage independently, this embedder:
    1. Encodes the source document to get token-level embeddings
    2. Maps passage character spans to token spans using tokenizer offsets
    3. Mean-pools the token embeddings within each passage span

    This approach is more efficient for documents with many passages.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the DocumentSpanEmbedder.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                SentenceTransformer,
            )
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with 'cite-right[embeddings]'."
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a list of text strings (fallback for Embedder protocol).

        This method exists to satisfy the Embedder protocol but is not
        the primary interface. Use encode_document_spans for efficiency.

        Args:
            texts (Sequence[str]): The text strings to encode.

        Returns:
            list[list[float]]: List of float vectors for each input text.
        """
        encoded = self._model.encode(texts)
        return encoded.tolist()

    def encode_document_spans(
        self,
        document_text: str,
        passages: Sequence[DocumentPassageSpan],
    ) -> list[list[float]]:
        """Encode passages by pooling token embeddings from document encoding.

        Args:
            document_text (str): The full source document text.
            passages (Sequence[DocumentPassageSpan]): Passage spans with char offsets.

        Returns:
            list[list[float]]: List of embedding vectors, one per passage.
        """
        if not passages:
            return []

        # Estimate character length per chunk to avoid OOM
        # max_seq_length is in tokens; approximate 4-5 chars per token
        max_seq_length = self._model.max_seq_length
        if max_seq_length is None:
            max_seq_length = 256  # Default for MiniLM
        max_chars_per_chunk = max_seq_length * 4

        # If document is short enough, encode as one chunk
        if len(document_text) <= max_chars_per_chunk:
            return self._encode_single_chunk(document_text, passages)

        # For long documents, chunk at character level
        return self._encode_chunked(document_text, passages, max_chars_per_chunk)

    def _encode_single_chunk(
        self,
        document_text: str,
        passages: Sequence[DocumentPassageSpan],
    ) -> list[list[float]]:
        """Encode passages from a single document chunk."""
        # Use the documented SentenceTransformer API for token embeddings
        token_embeddings = self._model.encode(
            [document_text],
            output_value="token_embeddings",
            convert_to_tensor=True,
        )

        # Get character offsets via tokenizer
        tokenizer = self._model.tokenizer
        tokenized = tokenizer(
            document_text,
            padding=False,
            truncation=True,
            max_length=self._model.max_seq_length,
            return_offsets_mapping=True,
        )
        token_offsets = tokenized["offset_mapping"]

        # Convert to numpy
        token_embeddings_np = token_embeddings[0].cpu().numpy()
        token_offsets_np = np.array(token_offsets)

        # Pool embeddings for each passage
        passage_embeddings: list[list[float]] = []
        for passage in passages:
            embedding = self._pool_passage_tokens(
                passage.doc_char_start,
                passage.doc_char_end,
                token_embeddings_np,
                token_offsets_np,
                chunk_char_offset=0,
            )
            passage_embeddings.append(embedding)

        return passage_embeddings

    def _encode_chunked(
        self,
        document_text: str,
        passages: Sequence[DocumentPassageSpan],
        max_chars_per_chunk: int,
    ) -> list[list[float]]:
        """Encode passages from a document that requires multiple chunks."""
        # Split document into character-level chunks to avoid OOM
        text_chunks = []
        chunk_char_offsets = []
        pos = 0

        while pos < len(document_text):
            end_pos = min(pos + max_chars_per_chunk, len(document_text))
            text_chunks.append(document_text[pos:end_pos])
            chunk_char_offsets.append(pos)
            pos = end_pos

        # Encode all text chunks using the proper API
        # This gives us proper CLS/SEP tokens and attention masks
        token_embeddings_list = self._model.encode(
            text_chunks,
            output_value="token_embeddings",
            convert_to_tensor=True,
        )

        # Get character offsets for each chunk
        tokenizer = self._model.tokenizer
        all_token_offsets = []

        for text_chunk in text_chunks:
            tokenized = tokenizer(
                text_chunk,
                padding=False,
                truncation=True,
                max_length=self._model.max_seq_length,
                return_offsets_mapping=True,
            )
            all_token_offsets.append(np.array(tokenized["offset_mapping"]))

        # Convert embeddings to numpy
        all_token_embeddings = [emb.cpu().numpy() for emb in token_embeddings_list]

        # Pool embeddings for each passage
        passage_embeddings: list[list[float]] = []
        for passage in passages:
            embedding = self._pool_passage_from_chunks(
                passage.doc_char_start,
                passage.doc_char_end,
                all_token_embeddings,
                all_token_offsets,
                chunk_char_offsets,
            )
            passage_embeddings.append(embedding)

        return passage_embeddings

    def _pool_passage_tokens(
        self,
        passage_start: int,
        passage_end: int,
        token_embeddings: npt.NDArray[np.float32],
        token_offsets: npt.NDArray[np.int_],
        chunk_char_offset: int,
    ) -> list[float]:
        """Pool token embeddings for a passage span."""
        # Find tokens that overlap with the passage
        tokens_in_passage = []
        for i, (start_offset, end_offset) in enumerate(token_offsets):
            # Adjust offsets relative to the document
            actual_start = start_offset + chunk_char_offset
            actual_end = end_offset + chunk_char_offset

            # Check if token overlaps with passage (any overlap counts)
            if actual_end > passage_start and actual_start < passage_end:
                tokens_in_passage.append(i)

        if not tokens_in_passage:
            # Return zero vector only if no tokens found
            embedding_dim = token_embeddings.shape[1]
            return [0.0] * embedding_dim

        # Mean pool the token embeddings
        passage_token_embeddings = token_embeddings[tokens_in_passage]
        mean_embedding = np.mean(passage_token_embeddings, axis=0)
        return mean_embedding.tolist()

    def _pool_passage_from_chunks(
        self,
        passage_start: int,
        passage_end: int,
        all_token_embeddings: list[npt.NDArray[np.float32]],
        all_token_offsets: list[npt.NDArray[np.int_]],
        chunk_char_offsets: list[int],
    ) -> list[float]:
        """Pool token embeddings for a passage that may span multiple chunks."""
        all_passage_embeddings = []

        for chunk_idx, (token_embeddings, token_offsets) in enumerate(
            zip(all_token_embeddings, all_token_offsets, strict=False)
        ):
            chunk_char_offset = chunk_char_offsets[chunk_idx]

            for i, (start_offset, end_offset) in enumerate(token_offsets):
                # Adjust offsets relative to the full document
                actual_start = start_offset + chunk_char_offset
                actual_end = end_offset + chunk_char_offset

                # Check if token overlaps with passage (any overlap counts)
                if actual_end > passage_start and actual_start < passage_end:
                    all_passage_embeddings.append(token_embeddings[i])

        if not all_passage_embeddings:
            # Return zero vector only if truly no tokens found
            embedding_dim = all_token_embeddings[0].shape[1]
            return [0.0] * embedding_dim

        # Mean pool all token embeddings from all chunks
        mean_embedding = np.mean(np.array(all_passage_embeddings), axis=0)
        return mean_embedding.tolist()

    def _fallback_encode_passages(
        self, passages: Sequence[DocumentPassageSpan]
    ) -> list[list[float]]:
        """Fallback: encode each passage text independently."""
        from cite_right.text.passage import Passage

        texts = []
        for passage in passages:
            if isinstance(passage, Passage):
                texts.append(passage.text)
            elif hasattr(passage, "source_text"):
                source_text_attr = getattr(passage, "source_text", None)
                if isinstance(source_text_attr, str):
                    texts.append(
                        source_text_attr[
                            passage.doc_char_start : passage.doc_char_end
                        ]
                    )
                else:
                    texts.append("")
            else:
                # Unknown passage type, use empty string
                texts.append("")

        if not texts:
            return []
        encoded = self._model.encode(texts)
        return encoded.tolist()

    def supports_span_pooling(self) -> bool:
        """Check if this embedder supports span pooling."""
        return True
