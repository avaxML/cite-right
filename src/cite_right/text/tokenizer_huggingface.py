"""HuggingFace tokenizers integration for cite-right.

This module provides tokenizers that use HuggingFace's tokenizers library
for subword tokenization. This is useful when you want to align citations
using the same tokenization as transformer models like BERT, RoBERTa, etc.

Examples:
    >>> from cite_right.text.tokenizer_huggingface import HuggingFaceTokenizer
    >>> tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
    >>> result = tokenizer.tokenize("Hello, world!")
    >>> result.token_ids
    [101, 7592, 1010, 2088, 999, 102]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cite_right.core.results import TokenizedText

if TYPE_CHECKING:
    from tokenizers import Tokenizer as HFTokenizer
    from transformers import PreTrainedTokenizerBase


class HuggingFaceTokenizer:
    """Tokenizer using HuggingFace's tokenizers library.

    This tokenizer wraps HuggingFace tokenizers to provide character-accurate
    token spans suitable for citation alignment. It supports both the fast
    `tokenizers` library and `transformers` tokenizers.

    Attributes:
        _tokenizer: Underlying HuggingFace tokenizer instance.
        _add_special_tokens: Whether or not to add special tokens.
        _is_transformers: Whether the tokenizer is a transformers tokenizer.

    Example:
        >>> from transformers import AutoTokenizer
        >>> hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> tokenizer = HuggingFaceTokenizer(hf_tokenizer)
        >>> result = tokenizer.tokenize("Hello world")
    """

    def __init__(
        self,
        tokenizer: "HFTokenizer | PreTrainedTokenizerBase",
        *,
        add_special_tokens: bool = False,
    ) -> None:
        """Initializes a HuggingFaceTokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer instance. Can be either:
                - A `tokenizers.Tokenizer` from the `tokenizers` library
                - A `PreTrainedTokenizer` from the `transformers` library
            add_special_tokens: Whether to add special tokens (e.g., [CLS], [SEP]).
                Defaults to False for alignment tasks.

        Raises:
            TypeError: If the tokenizer type is not supported.
        """
        self._tokenizer = tokenizer
        self._add_special_tokens = add_special_tokens
        self._is_transformers = self._check_tokenizer_type(tokenizer)

    @staticmethod
    def _check_tokenizer_type(tokenizer: object) -> bool:
        """Check if tokenizer is from the transformers library or tokenizers library.

        Args:
            tokenizer: The tokenizer object to check.

        Returns:
            bool: True if a transformers tokenizer, False if a tokenizers library tokenizer.

        Raises:
            TypeError: If tokenizer type is not supported.
        """
        try:
            from transformers import PreTrainedTokenizerBase

            if isinstance(tokenizer, PreTrainedTokenizerBase):
                return True
        except ImportError:
            pass

        try:
            from tokenizers import Tokenizer as HFTok

            if isinstance(tokenizer, HFTok):
                return False
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported tokenizer type: {type(tokenizer)}. "
            "Expected tokenizers.Tokenizer or transformers.PreTrainedTokenizer"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        add_special_tokens: bool = False,
        use_fast: bool = True,
    ) -> "HuggingFaceTokenizer":
        """Load a tokenizer from a pretrained model.

        This is a convenience method that loads a tokenizer from the
        HuggingFace Hub or a local path.

        Args:
            model_name_or_path (str): Model identifier (e.g., "bert-base-uncased")
                or path to local tokenizer files.
            add_special_tokens (bool): Whether to add special tokens. Defaults to False.
            use_fast (bool): Whether to use the fast Rust-based tokenizer. Defaults to True.

        Returns:
            HuggingFaceTokenizer: A HuggingFaceTokenizer instance.

        Raises:
            ImportError: If transformers is not installed.

        Example:
            >>> tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for from_pretrained(). "
                "Install it with: pip install cite-right[huggingface]"
            ) from e

        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=use_fast
        )
        return cls(hf_tokenizer, add_special_tokens=add_special_tokens)

    def tokenize(self, text: str) -> TokenizedText:
        """Tokenize input text and return token IDs and (char_start, char_end) spans.

        Args:
            text (str): The text to tokenize.

        Returns:
            TokenizedText: Object containing token IDs and character-accurate spans.

        Example:
            >>> tokenizer = HuggingFaceTokenizer.from_pretrained("bert-base-uncased")
            >>> result = tokenizer.tokenize("Hello, world!")
            >>> result.token_ids
            [101, 7592, 1010, 2088, 999, 102]
        """
        if not text:
            return TokenizedText(text=text, token_ids=[], token_spans=[])

        if self._is_transformers:
            return self._tokenize_transformers(text)
        return self._tokenize_tokenizers(text)

    def _tokenize_transformers(self, text: str) -> TokenizedText:
        """Tokenize text using a transformers tokenizer.

        Args:
            text (str): The text to tokenize.

        Returns:
            TokenizedText: Object containing filtered token IDs and their character spans.

        Notes:
            This method internally uses the tokenizer with 'return_offsets_mapping=True' to
            obtain character spans, and will filter special tokens with empty spans.
        """
        encoding = self._tokenizer(  # type: ignore[operator]
            text,
            return_offsets_mapping=True,
            add_special_tokens=self._add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        token_ids: list[int] = encoding["input_ids"]  # type: ignore[index]
        offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]  # type: ignore[index]

        # Filter out special tokens (they have (0, 0) offsets)
        token_spans: list[tuple[int, int]] = []
        filtered_ids: list[int] = []

        for token_id, (start, end) in zip(token_ids, offset_mapping, strict=True):
            # Skip tokens with no character span (special tokens)
            if start == end == 0 and self._add_special_tokens:
                # Keep special tokens if explicitly requested
                if token_id in self._get_special_token_ids():
                    continue
            token_spans.append((start, end))
            filtered_ids.append(token_id)

        return TokenizedText(
            text=text,
            token_ids=filtered_ids,
            token_spans=token_spans,
        )

    def _tokenize_tokenizers(self, text: str) -> TokenizedText:
        """Tokenize text using a tokenizers library tokenizer.

        Args:
            text (str): The text to tokenize.

        Returns:
            TokenizedText: Object containing filtered token IDs and their character spans.

        Notes:
            Filters out any tokens with empty (start == end) spans.
        """
        from tokenizers import Tokenizer as HFTok

        tokenizer: HFTok = self._tokenizer  # type: ignore[assignment]

        encoding = tokenizer.encode(text, add_special_tokens=self._add_special_tokens)

        token_ids: list[int] = encoding.ids
        offsets: list[tuple[int, int]] = encoding.offsets

        token_spans: list[tuple[int, int]] = []
        filtered_ids: list[int] = []

        for token_id, (start, end) in zip(token_ids, offsets, strict=True):
            if start != end:
                token_spans.append((start, end))
                filtered_ids.append(token_id)

        return TokenizedText(
            text=text,
            token_ids=filtered_ids,
            token_spans=token_spans,
        )

    def _get_special_token_ids(self) -> set[int]:
        """Get the set of special token IDs from the tokenizer.

        Returns:
            set[int]: Set of special token IDs. Returns empty set if not available.
        """
        special_ids: set[int] = set()
        if hasattr(self._tokenizer, "all_special_ids"):
            special_ids = set(self._tokenizer.all_special_ids)  # type: ignore[union-attr]
        return special_ids
