"""Call-local token equivalence and lexical retrieval for declared aliases."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import log
from unicodedata import normalize

from cite_right.text.tokenizer import (
    SimpleTokenizer,
    _iter_token_spans,
    _normalize_punctuation,
)


def normalize_alias(term: str) -> str:
    return _normalize_punctuation(normalize("NFKC", term).casefold())


def is_single_word(term: str) -> bool:
    return any(char.isalpha() for char in term) and _iter_token_spans(term) == [
        (0, len(term))
    ]


def canonical_token_id(token: int, aliases: Mapping[int, int]) -> int:
    return aliases.get(token, token)


def remap_ids(tokens: list[int], aliases: Mapping[int, int] | None) -> list[int]:
    if not aliases:
        return tokens
    return [canonical_token_id(token, aliases) for token in tokens]


@dataclass(frozen=True)
class TermAliases:
    ids: dict[int, int]
    members: dict[int, tuple[int, ...]]
    stopword_ids: frozenset[int]
    active_ids: frozenset[int]
    candidate_count: int
    lexemes: dict[str, str]
    postings: Callable[[list[int]], list[list[int]]]

    @classmethod
    def compile(
        cls,
        groups: tuple[tuple[str, ...], ...],
        tokenizer: SimpleTokenizer,
        postings: Callable[[list[int]], list[list[int]]],
        *,
        source_idf: Mapping[int, float],
        stopwords: frozenset[str],
        candidate_count: int,
    ) -> "TermAliases":
        lexemes = {}
        for group in groups:
            canonical = min(group)
            lexemes.update((word, canonical) for word in group)
        # Intern the configured terms in a stable order before reading the
        # native vocabulary, whose casefolding can differ from Python's.
        term_ids = {
            word: tokenizer.tokenize(word).token_ids[0] for word in sorted(lexemes)
        }
        ids = _resolve_alias_ids(tokenizer.vocabulary(), lexemes, term_ids)
        members = _source_members(ids, source_idf)
        stopword_groups = {
            term_ids[lexemes[word]] for word in lexemes.keys() & stopwords
        }
        return cls(
            ids=ids,
            members=members,
            stopword_ids=frozenset(
                raw for raw, canonical in ids.items() if canonical in stopword_groups
            ),
            active_ids=frozenset(
                raw
                for raw, canonical in ids.items()
                if (present := members.get(canonical)) and present != (raw,)
            ),
            candidate_count=candidate_count,
            lexemes=lexemes,
            postings=postings,
        )

    def entity_id(self, entity: str) -> str:
        return self.lexemes.get(normalize_alias(entity), entity)

    def lexical_scores(
        self, raw_answer: list[int], idf: Mapping[int, float]
    ) -> dict[int, float] | None:
        if self.active_ids.isdisjoint(raw_answer):
            return None

        answer = sorted(set(remap_ids(raw_answer, self.ids)))
        query_ids = [
            raw for token in answer for raw in self.members.get(token, (token,))
        ]
        hits: dict[int, set[int]] = defaultdict(set)
        for raw, candidates in zip(query_ids, self.postings(query_ids), strict=True):
            hits[canonical_token_id(raw, self.ids)].update(candidates)

        scores: dict[int, float] = defaultdict(float)
        total = 0.0
        for token in answer:
            candidates = hits[token]
            weight = idf.get(token, 1.0)
            if token in self.members:
                weight = log((self.candidate_count + 1) / (len(candidates) + 1)) + 1
            total += weight
            for candidate in candidates:
                scores[candidate] += weight
        return {candidate: score / total for candidate, score in scores.items()}


def _resolve_alias_ids(
    vocabulary: Mapping[str, int],
    lexemes: Mapping[str, str],
    term_ids: Mapping[str, int],
) -> dict[int, int]:
    resolved = {}
    for word, token_id in vocabulary.items():
        # Native preparation can retain a trailing dash or apostrophe.
        normalized = normalize_alias(word).rstrip("-'")
        representative = lexemes.get(normalized)
        if representative is not None:
            resolved[token_id] = term_ids[representative]
    return resolved


def _source_members(
    ids: Mapping[int, int], source_idf: Mapping[int, float]
) -> dict[int, tuple[int, ...]]:
    members: dict[int, list[int]] = defaultdict(list)
    for raw, canonical in ids.items():
        if raw in source_idf:
            members[canonical].append(raw)
    return {key: tuple(values) for key, values in members.items()}


def build_postings(
    token_sets: Iterable[frozenset[int]],
) -> Callable[[list[int]], list[list[int]]]:
    index: dict[int, list[int]] = defaultdict(list)
    for candidate, tokens in enumerate(token_sets):
        for token in tokens:
            index[token].append(candidate)

    def postings(query_ids: list[int]) -> list[list[int]]:
        return [index.get(token, []) for token in query_ids]

    return postings
