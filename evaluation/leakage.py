"""Cross-split leakage detection for evaluation corpora."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import cast

from evaluation.canonical import sha256_hex
from evaluation.schema import EvaluationCase, Split
from evaluation.splitting import (
    CaseLineage,
    _validate_cases,
    _validate_explicit_lineage,
)

SHINGLE_SIZE = 3
MIN_SHINGLE_TOKENS = 8
SHINGLE_ERROR_THRESHOLD = 0.8
SHINGLE_WARNING_THRESHOLD = 0.4

_PUNCT_TRANSLATION = str.maketrans({character: " " for character in string.punctuation})
_TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class LeakageFinding:
    severity: str
    code: str
    case_ids: tuple[str, ...]
    split_names: tuple[Split, ...]
    shared_fingerprint: str
    evidence: str
    similarity: float | None = None


@dataclass(frozen=True)
class LeakageReport:
    findings: tuple[LeakageFinding, ...]
    error_count: int
    warning_count: int


@dataclass(frozen=True)
class _TextArtifact:
    case_id: str
    split_name: Split
    text: str
    kind: str


@dataclass(frozen=True)
class _TextGroup:
    exact_text: str
    normalized_text: str
    case_ids: tuple[str, ...]
    case_splits: Mapping[str, Split]
    split_names: tuple[Split, ...]
    tokens: tuple[str, ...]
    shingles: frozenset[tuple[str, ...]]


class _UnionFind:
    def __init__(self, members: Iterable[str]) -> None:
        self._parent = {member: member for member in members}

    def find(self, member: str) -> str:
        parent = self._parent[member]
        if parent != member:
            parent = self.find(parent)
            self._parent[member] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


def detect_leakage(
    cases: Iterable[EvaluationCase],
    *,
    explicit_lineage: Iterable[CaseLineage] | None = None,
) -> LeakageReport:
    ordered_cases = _validate_cases(cases)
    cases_by_id = {case.case_id: case for case in ordered_cases}
    lineage_by_case_id = _validate_explicit_lineage(cases_by_id, explicit_lineage)

    findings: list[LeakageFinding] = []
    findings.extend(_lineage_findings(ordered_cases, lineage_by_case_id))

    text_groups = _build_text_groups(ordered_cases)
    covered_case_pairs: set[tuple[str, str]] = set()
    findings.extend(
        _duplicate_findings(
            text_groups=text_groups,
            grouping_key=lambda group: group.exact_text,
            code="exact_duplicate_cross_split",
            evidence_label="exact duplicate",
            covered_case_pairs=covered_case_pairs,
        )
    )
    findings.extend(
        _duplicate_findings(
            text_groups=text_groups,
            grouping_key=lambda group: _nfkc_casefold(group.exact_text),
            code="unicode_normalized_duplicate_cross_split",
            evidence_label="unicode-normalized duplicate",
            covered_case_pairs=covered_case_pairs,
            require_distinct_group_values=True,
        )
    )
    findings.extend(
        _duplicate_findings(
            text_groups=text_groups,
            grouping_key=lambda group: group.normalized_text,
            code="normalized_duplicate_cross_split",
            evidence_label="whitespace/punctuation-normalized duplicate",
            covered_case_pairs=covered_case_pairs,
            require_distinct_group_values=True,
        )
    )
    findings.extend(
        _shingle_findings(
            text_groups=text_groups,
            covered_case_pairs=covered_case_pairs,
        )
    )

    ordered_findings = tuple(sorted(findings, key=_finding_sort_key))
    return LeakageReport(
        findings=ordered_findings,
        error_count=sum(
            1 for finding in ordered_findings if finding.severity == "error"
        ),
        warning_count=sum(
            1 for finding in ordered_findings if finding.severity == "warning"
        ),
    )


def _lineage_findings(
    cases: tuple[EvaluationCase, ...],
    lineage_by_case_id: Mapping[str, CaseLineage],
) -> list[LeakageFinding]:
    buckets: defaultdict[str, list[str]] = defaultdict(list)
    union_find = _UnionFind(case.case_id for case in cases)
    case_ids = {case.case_id for case in cases}
    cases_by_id = {case.case_id: case for case in cases}

    for case in cases:
        buckets[f"document-family:{case.document_family_id}"].append(case.case_id)
        lineage = lineage_by_case_id.get(case.case_id)
        if lineage is None:
            continue
        for lineage_id in lineage.template_lineage_ids:
            buckets[f"template-lineage:{lineage_id}"].append(case.case_id)
        for lineage_id in lineage.transformation_lineage_ids:
            buckets[f"transformation-lineage:{lineage_id}"].append(case.case_id)

    for grouped_case_ids in buckets.values():
        unique_case_ids = tuple(sorted(set(grouped_case_ids)))
        if len(unique_case_ids) < 2:
            continue
        first_case_id = unique_case_ids[0]
        for case_id in unique_case_ids[1:]:
            union_find.union(first_case_id, case_id)

    grouped_components: defaultdict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        grouped_components[union_find.find(case_id)].append(case_id)

    findings: list[LeakageFinding] = []
    for component_case_ids in grouped_components.values():
        ordered_case_ids = tuple(sorted(component_case_ids))
        split_names = cast(
            tuple[Split, ...],
            tuple(sorted({cases_by_id[case_id].split for case_id in ordered_case_ids})),
        )
        if len(split_names) < 2:
            continue
        findings.append(
            LeakageFinding(
                severity="error",
                code="lineage_cross_split",
                case_ids=ordered_case_ids,
                split_names=split_names,
                shared_fingerprint=f"lineage:{sha256_hex('|'.join(ordered_case_ids).encode('utf-8'))[:20]}",
                evidence=", ".join(
                    sorted(
                        {
                            cases_by_id[case_id].document_family_id
                            for case_id in ordered_case_ids
                        }
                    )
                ),
            )
        )
    return findings


def _build_text_groups(cases: tuple[EvaluationCase, ...]) -> tuple[_TextGroup, ...]:
    grouped_artifacts: defaultdict[str, list[_TextArtifact]] = defaultdict(list)
    for case in cases:
        grouped_artifacts[case.answer].append(
            _TextArtifact(
                case_id=case.case_id,
                split_name=case.split,
                text=case.answer,
                kind="answer",
            )
        )
        for source in case.sources:
            grouped_artifacts[source.text].append(
                _TextArtifact(
                    case_id=case.case_id,
                    split_name=case.split,
                    text=source.text,
                    kind="source",
                )
            )

    groups: list[_TextGroup] = []
    for artifacts in grouped_artifacts.values():
        text = artifacts[0].text
        case_splits: dict[str, Split] = {
            artifact.case_id: artifact.split_name
            for artifact in sorted(artifacts, key=lambda artifact: artifact.case_id)
        }
        case_ids = tuple(case_splits)
        split_names = cast(tuple[Split, ...], tuple(sorted(set(case_splits.values()))))
        normalized_text = _punctuation_normalized(text)
        tokens = tuple(_TOKEN_RE.findall(normalized_text))
        shingles = (
            frozenset(
                zip(
                    *(tokens[index:] for index in range(SHINGLE_SIZE)),
                    strict=False,
                )
            )
            if len(tokens) >= SHINGLE_SIZE
            else frozenset()
        )
        groups.append(
            _TextGroup(
                exact_text=text,
                normalized_text=normalized_text,
                case_ids=case_ids,
                case_splits=cast(Mapping[str, Split], MappingProxyType(case_splits)),
                split_names=split_names,
                tokens=tokens,
                shingles=shingles,
            )
        )
    return tuple(sorted(groups, key=lambda group: (group.exact_text, group.case_ids)))


def _duplicate_findings(
    *,
    text_groups: tuple[_TextGroup, ...],
    grouping_key: Callable[[_TextGroup], str],
    code: str,
    evidence_label: str,
    covered_case_pairs: set[tuple[str, str]],
    require_distinct_group_values: bool = False,
) -> list[LeakageFinding]:
    buckets: defaultdict[str, list[_TextGroup]] = defaultdict(list)
    for group in text_groups:
        buckets[grouping_key(group)].append(group)

    findings: list[LeakageFinding] = []
    for bucket_key, groups in buckets.items():
        if len(groups) < 1:
            continue
        if (
            require_distinct_group_values
            and len({group.exact_text for group in groups}) < 2
        ):
            continue
        case_pairs = _cross_split_case_pairs(groups)
        uncovered_pairs = sorted(
            pair for pair in case_pairs if pair not in covered_case_pairs
        )
        if not uncovered_pairs:
            continue
        for pair in uncovered_pairs:
            covered_case_pairs.add(pair)
        case_ids = tuple(
            sorted({case_id for pair in uncovered_pairs for case_id in pair})
        )
        split_names = cast(
            tuple[Split, ...],
            tuple(
                sorted(
                    {
                        split_name
                        for group in groups
                        if set(group.case_ids) & set(case_ids)
                        for split_name in group.split_names
                    }
                )
            ),
        )
        sample_texts = " || ".join(
            _excerpt(text)
            for text in sorted({group.exact_text for group in groups})[:2]
        )
        findings.append(
            LeakageFinding(
                severity="error",
                code=code,
                case_ids=case_ids,
                split_names=split_names,
                shared_fingerprint=f"{code}:{sha256_hex(bucket_key.encode('utf-8'))[:20]}",
                evidence=f"{evidence_label}: {sample_texts}",
            )
        )
    return findings


def _shingle_findings(
    *,
    text_groups: tuple[_TextGroup, ...],
    covered_case_pairs: set[tuple[str, str]],
) -> list[LeakageFinding]:
    candidate_pairs: set[tuple[int, int]] = set()
    shingle_index: defaultdict[tuple[str, ...], set[int]] = defaultdict(set)

    for index, group in enumerate(text_groups):
        if len(group.tokens) < MIN_SHINGLE_TOKENS or not group.shingles:
            continue
        for shingle in group.shingles:
            shingle_index[shingle].add(index)

    for indexes in shingle_index.values():
        for left, right in combinations(sorted(indexes), 2):
            candidate_pairs.add((left, right))

    findings: list[LeakageFinding] = []
    for left_index, right_index in sorted(candidate_pairs):
        left_group = text_groups[left_index]
        right_group = text_groups[right_index]
        if left_group.normalized_text == right_group.normalized_text:
            continue
        case_pairs = _cross_split_case_pairs((left_group, right_group))
        uncovered_pairs = sorted(
            pair for pair in case_pairs if pair not in covered_case_pairs
        )
        if not uncovered_pairs:
            continue
        similarity = _jaccard_similarity(left_group.shingles, right_group.shingles)
        if similarity < SHINGLE_WARNING_THRESHOLD:
            continue
        severity = "error" if similarity >= SHINGLE_ERROR_THRESHOLD else "warning"
        code = (
            "shingle_overlap_error"
            if severity == "error"
            else "shingle_overlap_warning"
        )
        case_ids = tuple(
            sorted({case_id for pair in uncovered_pairs for case_id in pair})
        )
        split_names = cast(
            tuple[Split, ...],
            tuple(
                sorted(
                    {
                        split_name
                        for group in (left_group, right_group)
                        if set(group.case_ids) & set(case_ids)
                        for split_name in group.split_names
                    }
                )
            ),
        )
        findings.append(
            LeakageFinding(
                severity=severity,
                code=code,
                case_ids=case_ids,
                split_names=split_names,
                shared_fingerprint=(
                    f"shingle:{sha256_hex((left_group.normalized_text + '|' + right_group.normalized_text).encode('utf-8'))[:20]}"
                ),
                evidence=f"{_excerpt(left_group.exact_text)} || {_excerpt(right_group.exact_text)}",
                similarity=round(similarity, 3),
            )
        )
    return findings


def _cross_split_case_pairs(groups: Iterable[_TextGroup]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    group_list = list(groups)
    if len(group_list) == 1:
        group = group_list[0]
        for left_case_id, right_case_id in combinations(group.case_ids, 2):
            if group.case_splits[left_case_id] == group.case_splits[right_case_id]:
                continue
            ordered_pair = (
                (left_case_id, right_case_id)
                if left_case_id < right_case_id
                else (right_case_id, left_case_id)
            )
            pairs.add(ordered_pair)
        return pairs

    for left_group, right_group in combinations(group_list, 2):
        for left_case_id in left_group.case_ids:
            for right_case_id in right_group.case_ids:
                if (
                    left_group.case_splits[left_case_id]
                    == right_group.case_splits[right_case_id]
                ):
                    continue
                ordered_pair = (
                    (left_case_id, right_case_id)
                    if left_case_id < right_case_id
                    else (right_case_id, left_case_id)
                )
                pairs.add(ordered_pair)
    return pairs


def _jaccard_similarity(
    left: frozenset[tuple[str, ...]],
    right: frozenset[tuple[str, ...]],
) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _finding_sort_key(finding: LeakageFinding) -> tuple[object, ...]:
    severity_rank = 0 if finding.severity == "error" else 1
    code_rank = {
        "lineage_cross_split": 0,
        "exact_duplicate_cross_split": 1,
        "unicode_normalized_duplicate_cross_split": 2,
        "normalized_duplicate_cross_split": 3,
        "shingle_overlap_error": 4,
        "shingle_overlap_warning": 5,
    }.get(finding.code, 99)
    similarity = -finding.similarity if finding.similarity is not None else 0.0
    return (
        severity_rank,
        code_rank,
        finding.code,
        finding.case_ids,
        finding.split_names,
        finding.shared_fingerprint,
        similarity,
        finding.evidence,
    )


def _nfkc_casefold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _punctuation_normalized(text: str) -> str:
    normalized = _nfkc_casefold(text).translate(_PUNCT_TRANSLATION)
    return " ".join(normalized.split())


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= 80:
        return collapsed
    return f"{collapsed[:77]}..."


__all__ = [
    "LeakageFinding",
    "LeakageReport",
    "MIN_SHINGLE_TOKENS",
    "SHINGLE_ERROR_THRESHOLD",
    "SHINGLE_SIZE",
    "SHINGLE_WARNING_THRESHOLD",
    "detect_leakage",
]
