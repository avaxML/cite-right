from __future__ import annotations

import unicodedata
from functools import lru_cache

from cite_right.core.results import TokenizedText


class TokenizerConfig:
    def __init__(
        self,
        *,
        normalize_numbers: bool = True,
        normalize_percent: bool = True,
        normalize_currency: bool = True,
    ) -> None:
        self.normalize_numbers = normalize_numbers
        self.normalize_percent = normalize_percent
        self.normalize_currency = normalize_currency

    def __hash__(self) -> int:
        return hash(
            (self.normalize_numbers, self.normalize_percent, self.normalize_currency)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenizerConfig):
            return NotImplemented
        return (
            self.normalize_numbers == other.normalize_numbers
            and self.normalize_percent == other.normalize_percent
            and self.normalize_currency == other.normalize_currency
        )


class SimpleTokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self._config = config or TokenizerConfig()
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def tokenize(self, text: str) -> TokenizedText:
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
    spans: list[tuple[int, int]] = []
    idx = 0

    while idx < len(text):
        char = text[idx]
        if char.isdigit():
            start = idx
            idx += 1
            while idx < len(text):
                char = text[idx]
                if char.isdigit():
                    idx += 1
                    continue
                if (
                    char in {".", ","}
                    and idx + 1 < len(text)
                    and text[idx - 1].isdigit()
                    and text[idx + 1].isdigit()
                ):
                    idx += 1
                    continue
                break
            spans.append((start, idx))
            continue

        if char in {"%", "$", "€", "£"}:
            spans.append((idx, idx + 1))
            idx += 1
            continue

        if char.isalnum():
            start = idx
            idx += 1
            while idx < len(text):
                char = text[idx]
                if char.isalnum():
                    idx += 1
                    continue
                if (
                    char in {"'", "\u2019"}
                    and idx + 1 < len(text)
                    and text[idx - 1].isalnum()
                    and text[idx + 1].isalnum()
                ):
                    idx += 1
                    continue
                if (
                    char == "-"
                    and idx + 1 < len(text)
                    and text[idx - 1].isalnum()
                    and text[idx + 1].isalnum()
                ):
                    idx += 1
                    continue
                break
            spans.append((start, idx))
            continue

        idx += 1

    return spans


@lru_cache(maxsize=10000)
def _normalize_token_cached(token: str, config: TokenizerConfig) -> str:
    """Normalize a token with caching for repeated tokens."""
    return _normalize_token(token, config)


def _normalize_token(token: str, config: TokenizerConfig) -> str:
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
