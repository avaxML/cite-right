"""Simple rule-based tokenizer supporting normalization of numbers, percents, and currency."""
from __future__ import annotations

import unicodedata
from functools import lru_cache

from cite_right.core.results import TokenizedText


class TokenizerConfig:
    """Configuration for the SimpleTokenizer.

    Attributes:
        normalize_numbers (bool): Whether to normalize number tokens (e.g., remove commas).
        normalize_percent (bool): Whether to normalize percent tokens (convert '%' to 'percent').
        normalize_currency (bool): Whether to normalize currency symbols (e.g., '$' to 'dollar').
    """

    def __init__(
        self,
        *,
        normalize_numbers: bool = True,
        normalize_percent: bool = True,
        normalize_currency: bool = True,
    ) -> None:
        """Initializes the TokenizerConfig.

        Args:
            normalize_numbers (bool, optional): Whether to normalize number tokens. Defaults to True.
            normalize_percent (bool, optional): Whether to normalize percent tokens. Defaults to True.
            normalize_currency (bool, optional): Whether to normalize currency symbols. Defaults to True.
        """
        self.normalize_numbers = normalize_numbers
        self.normalize_percent = normalize_percent
        self.normalize_currency = normalize_currency

    def __hash__(self) -> int:
        """Returns a hash value of the TokenizerConfig.

        Returns:
            int: Hash based on config attributes.
        """
        return hash(
            (self.normalize_numbers, self.normalize_percent, self.normalize_currency)
        )

    def __eq__(self, other: object) -> bool:
        """Checks equality with another TokenizerConfig.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if the config objects are equal, False otherwise.
        """
        if not isinstance(other, TokenizerConfig):
            return NotImplemented
        return (
            self.normalize_numbers == other.normalize_numbers
            and self.normalize_percent == other.normalize_percent
            and self.normalize_currency == other.normalize_currency
        )


class SimpleTokenizer:
    """A simple rule-based tokenizer supporting normalization of numbers, percents, and currency.

    Attributes:
        _config (TokenizerConfig): Tokenization and normalization configuration.
        _vocab (dict[str, int]): Mapping from normalized token to token id.
        _next_id (int): Next available token id.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        """Initializes the SimpleTokenizer.

        Args:
            config (TokenizerConfig, optional): Configuration for normalization.
                If None, uses the default config.
        """
        self._config = config or TokenizerConfig()
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def tokenize(self, text: str) -> TokenizedText:
        """Tokenizes the input text into normalized token ids and spans.

        Args:
            text (str): The text to tokenize.

        Returns:
            TokenizedText: Object containing the input text,
                token ids (list of int), and token spans (list of (start, end) tuples).
        """
        token_ids: list[int] = []
        token_spans: list[tuple[int, int]] = []

        for start, end in _iter_token_spans(text):
            raw = text[start:end]
            normalized = _normalize_token_cached(raw, self._config)
            if not normalized:
                continue
            token_id = self._vocab.get(normalized)
            if token_id is None:
                token_id = self._next_id
                self._vocab[normalized] = token_id
                self._next_id += 1
            token_ids.append(token_id)
            token_spans.append((start, end))

        return TokenizedText(text=text, token_ids=token_ids, token_spans=token_spans)


def _iter_token_spans(text: str) -> list[tuple[int, int]]:
    """Yield the (start, end) spans of each token in the input string.

    Splits on numbers, percent/currency symbols, and alphanumeric words
    (including words with certain internal punctuation).

    Args:
        text (str): The text to segment into token spans.

    Returns:
        list[tuple[int, int]]: List of (start, end) indices for each token found in the input text.
    """
    spans: list[tuple[int, int]] = []
    idx = 0

    while idx < len(text):
        char = text[idx]
        if char.isdigit():
            end = _consume_number(text, idx)
            spans.append((idx, end))
            idx = end
        elif char in {"%", "$", "€", "£"}:
            spans.append((idx, idx + 1))
            idx += 1
        elif char.isalnum():
            end = _consume_word(text, idx)
            spans.append((idx, end))
            idx = end
        else:
            idx += 1

    return spans


def _consume_number(text: str, start: int) -> int:
    """Consume a number token, including decimal separators and commas.

    Args:
        text (str): The input text.
        start (int): Starting index of the number token.

    Returns:
        int: Index immediately after the last character of the number token.
    """
    idx = start + 1
    while idx < len(text):
        char = text[idx]
        if char.isdigit():
            idx += 1
        elif (
            char in {".", ","}
            and idx + 1 < len(text)
            and text[idx - 1].isdigit()
            and text[idx + 1].isdigit()
        ):
            idx += 1
        else:
            break
    return idx


def _consume_word(text: str, start: int) -> int:
    """Consume an alphanumeric word token, supporting internal apostrophes and hyphens.

    Args:
        text (str): The input text.
        start (int): Index at which the word token starts.

    Returns:
        int: Index immediately after the last character of the word token.
    """
    idx = start + 1
    while idx < len(text):
        char = text[idx]
        if char.isalnum():
            idx += 1
        elif _is_internal_punctuation(text, idx, char):
            idx += 1
        else:
            break
    return idx


def _is_internal_punctuation(text: str, idx: int, char: str) -> bool:
    """Check if the punctuation character is internal to a word (apostrophe or hyphen).

    An internal punctuation is defined as a character that is both
    preceded and followed by an alphanumeric character.

    Args:
        text (str): The input text.
        idx (int): Index of the punctuation character in the text.
        char (str): The punctuation character at `text[idx]`.

    Returns:
        bool: True if the character at idx is internal punctuation, False otherwise.
    """
    if idx + 1 >= len(text):
        return False
    if not text[idx - 1].isalnum() or not text[idx + 1].isalnum():
        return False
    return char in {"'", "\u2019", "-"}


@lru_cache(maxsize=10000)
def _normalize_token_cached(token: str, config: TokenizerConfig) -> str:
    """Normalize a token using the passed configuration, with LRU caching.

    Args:
        token (str): The token string to normalize.
        config (TokenizerConfig): Normalization options.

    Returns:
        str: The normalized token.
    """
    return _normalize_token(token, config)


def _normalize_token(token: str, config: TokenizerConfig) -> str:
    """Apply normalization rules to a token using the provided configuration.

    Applies NFKC unicode normalization, lowercasing, apostrophe replacement,
    number formatting removal, and percent/currency normalization as configured.

    Args:
        token (str): The token to normalize.
        config (TokenizerConfig): The normalization configuration.

    Returns:
        str: The normalized token string.
    """
    normalized = unicodedata.normalize("NFKC", token).casefold()
    normalized = normalized.replace("\u2019", "'")

    if config.normalize_numbers and normalized and normalized[0].isdigit():
        normalized = normalized.replace(",", "")

    if config.normalize_percent and normalized == "%":
        return "percent"

    if config.normalize_currency:
        if normalized == "$":
            return "dollar"
        if normalized == "€":
            return "euro"
        if normalized == "£":
            return "pound"

    return normalized
