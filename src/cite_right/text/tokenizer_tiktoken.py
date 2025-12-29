"""Tiktoken-based tokenizer for cite-right.

This module provides a tokenizer that uses OpenAI's tiktoken library for
byte-pair encoding (BPE) tokenization. This is useful when you want to
align citations using the same tokenization as GPT models.

Example:
    >>> from cite_right.text.tokenizer_tiktoken import TiktokenTokenizer
    >>> tokenizer = TiktokenTokenizer()  # defaults to cl100k_base
    >>> result = tokenizer.tokenize("Hello, world!")
    >>> result.token_ids
    [9906, 11, 1917, 0]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cite_right.core.results import TokenizedText

if TYPE_CHECKING:
    import tiktoken


class TiktokenTokenizer:
    """Tokenizer using OpenAI's tiktoken BPE encoding.

    This tokenizer wraps tiktoken encodings to provide character-accurate
    token spans suitable for citation alignment.

    Args:
        encoding_name: Name of the tiktoken encoding to use.
            Common options:
            - "cl100k_base": Used by GPT-4, GPT-3.5-turbo, text-embedding-ada-002
            - "p50k_base": Used by Codex models
            - "r50k_base": Used by GPT-3 models (davinci, curie, etc.)
            Defaults to "cl100k_base".
        encoding: Pre-initialized tiktoken Encoding object. If provided,
            encoding_name is ignored.

    Raises:
        ImportError: If tiktoken is not installed.

    Example:
        >>> tokenizer = TiktokenTokenizer("cl100k_base")
        >>> result = tokenizer.tokenize("Hello world")
        >>> len(result.token_ids)
        2
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        *,
        encoding: tiktoken.Encoding | None = None,
    ) -> None:
        try:
            import tiktoken as _tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. "
                "Install it with: pip install cite-right[tiktoken]"
            ) from e

        if encoding is not None:
            self._encoding = encoding
        else:
            self._encoding = _tiktoken.get_encoding(encoding_name)

    def tokenize(self, text: str) -> TokenizedText:
        """Tokenize text using tiktoken encoding.

        Args:
            text: The text to tokenize.

        Returns:
            TokenizedText with token IDs and character-accurate spans.
        """
        if not text:
            return TokenizedText(text=text, token_ids=[], token_spans=[])

        # Encode text to get token IDs
        token_ids = self._encoding.encode(text, allowed_special="all")

        if not token_ids:
            return TokenizedText(text=text, token_ids=[], token_spans=[])

        # Compute character spans by decoding each token and tracking position
        token_spans: list[tuple[int, int]] = []
        text_bytes = text.encode("utf-8")
        byte_offset = 0

        for token_id in token_ids:
            # Get the bytes for this token
            token_bytes = self._encoding.decode_single_token_bytes(token_id)
            token_byte_len = len(token_bytes)

            # Convert byte offset to character offset
            char_start = len(text_bytes[:byte_offset].decode("utf-8"))
            char_end = len(text_bytes[: byte_offset + token_byte_len].decode("utf-8"))

            token_spans.append((char_start, char_end))
            byte_offset += token_byte_len

        return TokenizedText(
            text=text,
            token_ids=list(token_ids),
            token_spans=token_spans,
        )

    @property
    def encoding_name(self) -> str:
        """Return the name of the tiktoken encoding."""
        return self._encoding.name
