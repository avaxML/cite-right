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
    """Tokenizer using OpenAI's tiktoken BPE encoding to generate character-accurate token spans.

    This tokenizer wraps `tiktoken` encodings to provide character-accurate
    token spans suitable for citation alignment. It supports several encoding
    schemes used by OpenAI GPT and Codex models.

    Attributes:
        _encoding: The tiktoken Encoding instance used for tokenization.

    Examples:
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
        """Initializes the TiktokenTokenizer.

        Args:
            encoding_name (str, optional): Name of the tiktoken encoding to use.
                Common options include:
                    - "cl100k_base": Used by GPT-4, GPT-3.5-turbo, text-embedding-ada-002
                    - "p50k_base": Used by Codex models
                    - "r50k_base": Used by GPT-3 models (davinci, curie, etc.)
                Defaults to "cl100k_base".
            encoding (tiktoken.Encoding, optional):
                A pre-initialized tiktoken Encoding object. If provided,
                `encoding_name` is ignored.

        Raises:
            ImportError: If tiktoken is not installed.
        """
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
        """Tokenizes input text with tiktoken BPE, mapping byte spans to character spans.

        Args:
            text (str): The text to tokenize.

        Returns:
            TokenizedText: TokenizedText instance containing:
                - text (str): The original input text.
                - token_ids (list of int): List of BPE token IDs.
                - token_spans (list of tuple[int, int]):
                    List of (char_start, char_end) spans mapping each token to the original text.

        Notes:
            - Token spans are computed to be character-accurate, compensating for possible
              splitting at arbitrary byte boundaries from UTF-8.
            - If the input text is empty, an empty TokenizedText is returned.
        """
        if not text:
            return TokenizedText(text=text, token_ids=[], token_spans=[])

        token_ids = self._encoding.encode(text, allowed_special="all")

        if not token_ids:
            return TokenizedText(text=text, token_ids=[], token_spans=[])

        text_bytes = text.encode("utf-8")

        byte_to_char: list[int] = []
        char_idx = 0
        byte_idx = 0

        while byte_idx < len(text_bytes):
            byte_to_char.append(char_idx)
            byte_val = text_bytes[byte_idx]
            if byte_val < 0x80:  # 1-byte character (ASCII)
                char_len_bytes = 1
            elif byte_val < 0xE0:  # 2-byte character
                char_len_bytes = 2
            elif byte_val < 0xF0:  # 3-byte character
                char_len_bytes = 3
            else:  # 4-byte character
                char_len_bytes = 4

            for _ in range(1, char_len_bytes):
                byte_idx += 1
                if byte_idx < len(text_bytes):
                    byte_to_char.append(char_idx)

            byte_idx += 1
            char_idx += 1

        byte_to_char.append(char_idx)

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
        """Returns the name of the tiktoken encoding.

        Returns:
            str: The name of the encoding being used.
        """
        return self._encoding.name
