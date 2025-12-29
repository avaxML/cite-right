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

        # Compute character spans using byte-to-character mapping
        # BPE tokens operate on bytes and can split multi-byte UTF-8 characters,
        # so we need to map byte positions to character positions
        text_bytes = text.encode("utf-8")

        # Build mapping from byte offset to character offset
        byte_to_char: list[int] = []
        char_idx = 0
        byte_idx = 0

        while byte_idx < len(text_bytes):
            byte_to_char.append(char_idx)
            # Determine UTF-8 character byte length
            byte_val = text_bytes[byte_idx]
            if byte_val < 0x80:  # 1-byte character (ASCII)
                char_len_bytes = 1
            elif byte_val < 0xE0:  # 2-byte character
                char_len_bytes = 2
            elif byte_val < 0xF0:  # 3-byte character
                char_len_bytes = 3
            else:  # 4-byte character
                char_len_bytes = 4

            # Map continuation bytes to the same character index
            for _ in range(1, char_len_bytes):
                byte_idx += 1
                if byte_idx < len(text_bytes):
                    byte_to_char.append(char_idx)

            byte_idx += 1
            char_idx += 1

        byte_to_char.append(char_idx)  # Map for position after last byte

        # Convert token byte positions to character positions
        token_spans: list[tuple[int, int]] = []
        byte_offset = 0

        for token_id in token_ids:
            token_bytes = self._encoding.decode_single_token_bytes(token_id)
            byte_start = byte_offset
            byte_end = byte_offset + len(token_bytes)

            char_start = byte_to_char[byte_start]
            char_end = byte_to_char[byte_end]

            token_spans.append((char_start, char_end))
            byte_offset = byte_end

        return TokenizedText(
            text=text,
            token_ids=list(token_ids),
            token_spans=token_spans,
        )

    @property
    def encoding_name(self) -> str:
        """Return the name of the tiktoken encoding."""
        return self._encoding.name
