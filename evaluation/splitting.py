"""Grouped evaluation splitting with lineage-aware component assignment."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase, Split

_SPLIT_NAMES: tuple[Split, Split, Split] = ("train", "dev", "holdout")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CaseLineage:
    case_id: str
    template_lineage_ids: tuple[str, ...] = ()
    transformation_lineage_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LineageComponent:
    component_id: str
    case_ids: tuple[str, ...]
    document_family_ids: tuple[str, ...]
    template_lineage_ids: tuple[str, ...]
    transformation_lineage_ids: tuple[str, ...]

    @property
    def case_count(self) -> int:
        return len(self.case_ids)


@dataclass(frozen=True)
class SplitAssignment:
    case_id: str
    split: Split
    component_id: str


@dataclass(frozen=True)
class SplitAssignmentReport:
    assignments: tuple[SplitAssignment, ...]
    components: tuple[LineageComponent, ...]
    assignment_by_case_id: Mapping[str, Split]
    split_case_counts: Mapping[Split, int]
    target_case_counts: Mapping[Split, float]
    deviation_by_split: Mapping[Split, float]
    distributions_by_split: Mapping[Split, Mapping[str, Mapping[str, int]]]
    assignment_hash: str


@dataclass(frozen=True)
class _ComponentProfile:
    component: LineageComponent
    metrics: Mapping[str, Counter[str]]

    @property
    def case_count(self) -> int:
        return self.component.case_count


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


def build_lineage_components(
    cases: Iterable[EvaluationCase],
    *,
    explicit_lineage: Iterable[CaseLineage] | None = None,
) -> tuple[LineageComponent, ...]:
    ordered_cases = _validate_cases(cases)
    cases_by_id = {case.case_id: case for case in ordered_cases}
    lineage_by_case_id = _validate_explicit_lineage(cases_by_id, explicit_lineage)
    union_find = _UnionFind(cases_by_id)
    buckets: defaultdict[str, list[str]] = defaultdict(list)

    for case in ordered_cases:
        buckets[f"document-family:{case.document_family_id}"].append(case.case_id)
        # transformation_family_id values are global names like "negation", so the
        # default lineage key is scoped by document family to avoid collapsing the corpus.
        buckets[
            f"scoped-transformation:{case.document_family_id}:{case.transformation_family_id}"
        ].append(case.case_id)
        for source in case.sources:
            exact_hash = sha256_hex(source.text.encode("utf-8"))
            buckets[f"source-sha256:{exact_hash}"].append(case.case_id)
            normalized_hash = sha256_hex(
                _normalized_source_fingerprint(source.text).encode("utf-8")
            )
            buckets[f"source-normalized:{normalized_hash}"].append(case.case_id)
        lineage = lineage_by_case_id.get(case.case_id)
        if lineage is None:
            continue
        for lineage_id in lineage.template_lineage_ids:
            buckets[f"template-lineage:{lineage_id}"].append(case.case_id)
        for lineage_id in lineage.transformation_lineage_ids:
            buckets[f"transformation-lineage:{lineage_id}"].append(case.case_id)

    for case_ids in buckets.values():
        if len(case_ids) < 2:
            continue
        unique_case_ids = tuple(sorted(set(case_ids)))
        first_case_id = unique_case_ids[0]
        for case_id in unique_case_ids[1:]:
            union_find.union(first_case_id, case_id)

    grouped_case_ids: defaultdict[str, list[str]] = defaultdict(list)
    for case_id in cases_by_id:
        grouped_case_ids[union_find.find(case_id)].append(case_id)

    components: list[LineageComponent] = []
    for root_case_id, grouped_ids in grouped_case_ids.items():
        del root_case_id
        case_ids = tuple(sorted(grouped_ids))
        member_cases = [cases_by_id[case_id] for case_id in case_ids]
        template_lineage_ids = sorted(
            {
                lineage_id
                for case_id in case_ids
                for lineage_id in lineage_by_case_id.get(
                    case_id, CaseLineage(case_id=case_id)
                ).template_lineage_ids
            }
        )
        transformation_lineage_ids = sorted(
            {
                lineage_id
                for case_id in case_ids
                for lineage_id in lineage_by_case_id.get(
                    case_id, CaseLineage(case_id=case_id)
                ).transformation_lineage_ids
            }
        )
        component_id = _stable_component_id(case_ids)
        components.append(
            LineageComponent(
                component_id=component_id,
                case_ids=case_ids,
                document_family_ids=tuple(
                    sorted({case.document_family_id for case in member_cases})
                ),
                template_lineage_ids=tuple(template_lineage_ids),
                transformation_lineage_ids=tuple(transformation_lineage_ids),
            )
        )

    return tuple(
        sorted(
            components,
            key=lambda component: (component.case_ids[0], component.component_id),
        )
    )


def assign_splits(
    cases: Iterable[EvaluationCase],
    *,
    seed: int = 20260717,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    explicit_lineage: Iterable[CaseLineage] | None = None,
) -> SplitAssignmentReport:
    ordered_cases = _validate_cases(cases)
    if not ordered_cases:
        raise ValueError("cases must not be empty")
    validated_seed = _validate_seed(seed)
    validated_ratios = _validate_ratios(ratios)
    components = build_lineage_components(
        ordered_cases, explicit_lineage=explicit_lineage
    )
    cases_by_id = {case.case_id: case for case in ordered_cases}
    profiles = {
        component.component_id: _component_profile(component, cases_by_id)
        for component in components
    }

    total_case_count = len(ordered_cases)
    target_case_counts: dict[Split, float] = {
        split_name: total_case_count * validated_ratios[index]
        for index, split_name in enumerate(_SPLIT_NAMES)
    }
    global_metrics = _aggregate_metrics(profiles.values())
    split_case_counts: dict[Split, int] = {split_name: 0 for split_name in _SPLIT_NAMES}
    split_metrics: dict[Split, dict[str, Counter[str]]] = {
        split_name: _empty_metrics_like(global_metrics) for split_name in _SPLIT_NAMES
    }
    assigned_component_ids: set[str] = set()
    assignments_by_case_id: dict[str, Split] = {}
    assignments: list[SplitAssignment] = []

    ordered_components = tuple(
        sorted(
            components,
            key=lambda component: (
                -component.case_count,
                _seeded_order_key(validated_seed, component.component_id),
                component.component_id,
            ),
        )
    )
    _seed_domain_coverage(
        ordered_components=ordered_components,
        profiles=profiles,
        split_case_counts=split_case_counts,
        split_metrics=split_metrics,
        global_metrics=global_metrics,
        target_case_counts=target_case_counts,
        ratios=validated_ratios,
        total_case_count=total_case_count,
        assigned_component_ids=assigned_component_ids,
        assignments_by_case_id=assignments_by_case_id,
        assignments=assignments,
        seed=validated_seed,
    )

    for component in ordered_components:
        if component.component_id in assigned_component_ids:
            continue
        profile = profiles[component.component_id]
        chosen_split = min(
            _SPLIT_NAMES,
            key=lambda split_name: _assignment_sort_key(
                split_name=split_name,
                profile=profile,
                split_case_counts=split_case_counts,
                split_metrics=split_metrics,
                global_metrics=global_metrics,
                target_case_counts=target_case_counts,
                ratios=validated_ratios,
                total_case_count=total_case_count,
            ),
        )
        _record_component_assignment(
            profile=profile,
            split_name=chosen_split,
            split_case_counts=split_case_counts,
            split_metrics=split_metrics,
            assigned_component_ids=assigned_component_ids,
            assignments_by_case_id=assignments_by_case_id,
            assignments=assignments,
        )

    assignments.sort(key=lambda assignment: assignment.case_id)
    assignment_hash = sha256_hex(
        canonical_json_bytes(
            {
                "assignments": [
                    {"case_id": assignment.case_id, "split": assignment.split}
                    for assignment in assignments
                ]
            }
        )
    )
    deviation_by_split: dict[Split, float] = {
        split_name: (
            split_case_counts[split_name] / total_case_count - validated_ratios[index]
        )
        for index, split_name in enumerate(_SPLIT_NAMES)
    }

    return SplitAssignmentReport(
        assignments=tuple(assignments),
        components=components,
        assignment_by_case_id=MappingProxyType(dict(assignments_by_case_id)),
        split_case_counts=cast(
            Mapping[Split, int], MappingProxyType(dict(split_case_counts))
        ),
        target_case_counts=cast(
            Mapping[Split, float], MappingProxyType(dict(target_case_counts))
        ),
        deviation_by_split=cast(
            Mapping[Split, float], MappingProxyType(dict(deviation_by_split))
        ),
        distributions_by_split=_freeze_distributions(split_metrics),
        assignment_hash=assignment_hash,
    )


def apply_split_assignments(
    cases: Iterable[EvaluationCase],
    assignments: Mapping[str, Split],
) -> tuple[EvaluationCase, ...]:
    ordered_cases = _validate_cases(cases)
    assignment_keys = set(assignments)
    case_ids = {case.case_id for case in ordered_cases}
    if assignment_keys != case_ids:
        raise ValueError("assignments must cover exactly the provided case ids")

    assigned_cases: list[EvaluationCase] = []
    for case in ordered_cases:
        split_name = assignments[case.case_id]
        if split_name not in _SPLIT_NAMES:
            raise ValueError(f"invalid split assignment {split_name!r}")
        updated_case = case.model_copy(update={"split": split_name})
        validated_case = EvaluationCase.model_validate(
            updated_case.model_dump(mode="python", round_trip=True)
        )
        assigned_cases.append(validated_case)
    return tuple(assigned_cases)


def _validate_cases(cases: Iterable[EvaluationCase]) -> tuple[EvaluationCase, ...]:
    ordered_cases = tuple(cases)
    seen_case_ids: set[str] = set()
    for case in ordered_cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case id {case.case_id!r}")
        seen_case_ids.add(case.case_id)
    return ordered_cases


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return seed


def _validate_ratios(ratios: tuple[float, ...]) -> tuple[float, float, float]:
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must define exactly three positive values")
    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("ratios must sum to 1.0")
    return cast(tuple[float, float, float], ratios)


def _validate_explicit_lineage(
    cases_by_id: Mapping[str, EvaluationCase],
    explicit_lineage: Iterable[CaseLineage] | None,
) -> Mapping[str, CaseLineage]:
    if explicit_lineage is None:
        return MappingProxyType({})

    lineage_by_case_id: dict[str, CaseLineage] = {}
    for lineage in explicit_lineage:
        if lineage.case_id not in cases_by_id:
            raise ValueError("unknown case id in explicit lineage metadata")
        if lineage.case_id in lineage_by_case_id:
            raise ValueError(
                f"duplicate lineage metadata for case id {lineage.case_id!r}"
            )
        if len(set(lineage.template_lineage_ids)) != len(lineage.template_lineage_ids):
            raise ValueError(
                "duplicate template lineage id in explicit lineage metadata"
            )
        if len(set(lineage.transformation_lineage_ids)) != len(
            lineage.transformation_lineage_ids
        ):
            raise ValueError(
                "duplicate transformation lineage id in explicit lineage metadata"
            )
        lineage_by_case_id[lineage.case_id] = lineage
    return MappingProxyType(lineage_by_case_id)


def _component_profile(
    component: LineageComponent,
    cases_by_id: Mapping[str, EvaluationCase],
) -> _ComponentProfile:
    metrics = _empty_metrics()
    for case_id in component.case_ids:
        case = cases_by_id[case_id]
        metrics["provenance_kind"][case.provenance.kind] += 1
        metrics["transformation_family"][case.transformation_family_id] += 1
        metrics["domain"][_case_domain(case)] += 1
        for unit in case.evaluation_units:
            metrics["expected_status"][unit.expected_status] += 1
            for claim in unit.claims:
                metrics["support_label"][claim.label] += 1
    return _ComponentProfile(component=component, metrics=metrics)


def _aggregate_metrics(
    profiles: Iterable[_ComponentProfile],
) -> Mapping[str, Counter[str]]:
    aggregate = _empty_metrics()
    for profile in profiles:
        _merge_metrics(aggregate, profile.metrics)
    return aggregate


def _empty_metrics() -> dict[str, Counter[str]]:
    return {
        "expected_status": Counter(),
        "support_label": Counter(),
        "transformation_family": Counter(),
        "domain": Counter(),
        "provenance_kind": Counter(),
    }


def _empty_metrics_like(
    metrics: Mapping[str, Counter[str]],
) -> dict[str, Counter[str]]:
    return {name: Counter() for name in metrics}


def _merge_metrics(
    target: dict[str, Counter[str]],
    source: Mapping[str, Counter[str]],
) -> None:
    for group_name, counts in source.items():
        target[group_name].update(counts)


def _seed_domain_coverage(
    *,
    ordered_components: tuple[LineageComponent, ...],
    profiles: Mapping[str, _ComponentProfile],
    split_case_counts: dict[Split, int],
    split_metrics: dict[Split, dict[str, Counter[str]]],
    global_metrics: Mapping[str, Counter[str]],
    target_case_counts: Mapping[Split, float],
    ratios: tuple[float, float, float],
    total_case_count: int,
    assigned_component_ids: set[str],
    assignments_by_case_id: dict[str, Split],
    assignments: list[SplitAssignment],
    seed: int,
) -> None:
    eligible_domain_counts = _eligible_domain_component_counts(
        ordered_components=ordered_components,
        profiles=profiles,
    )
    if not eligible_domain_counts:
        return

    eligible_domains = tuple(
        sorted(
            eligible_domain_counts,
            key=lambda domain: (eligible_domain_counts[domain], domain),
        )
    )
    for domain in eligible_domains:
        missing_splits: list[Split] = [
            split_name
            for split_name in _SPLIT_NAMES
            if split_metrics[split_name]["domain"][domain] == 0
        ]
        if not missing_splits:
            continue
        candidate_profiles = [
            profiles[component.component_id]
            for component in ordered_components
            if component.component_id not in assigned_component_ids
            and domain in profiles[component.component_id].metrics["domain"]
        ]
        if len(candidate_profiles) < len(missing_splits):
            continue
        ordered_missing_split_entries = cast(
            list[tuple[float, int, Split]],
            sorted(
                (
                    (
                        split_case_counts[split_name]
                        / max(target_case_counts[split_name], 1.0),
                        _SPLIT_NAMES.index(split_name),
                        split_name,
                    )
                    for split_name in missing_splits
                ),
            ),
        )
        for _, _, raw_split_name in ordered_missing_split_entries:
            validated_split_name: Split = cast(Split, raw_split_name)
            available_profiles = [
                profile
                for profile in candidate_profiles
                if profile.component.component_id not in assigned_component_ids
            ]
            if not available_profiles:
                break
            chosen_profile = min(
                available_profiles,
                key=lambda profile: _domain_seed_sort_key(
                    split_name=validated_split_name,
                    profile=profile,
                    split_case_counts=split_case_counts,
                    split_metrics=split_metrics,
                    global_metrics=global_metrics,
                    target_case_counts=target_case_counts,
                    ratios=ratios,
                    total_case_count=total_case_count,
                    eligible_domain_counts=eligible_domain_counts,
                    seed=seed,
                ),
            )
            _record_component_assignment(
                profile=chosen_profile,
                split_name=validated_split_name,
                split_case_counts=split_case_counts,
                split_metrics=split_metrics,
                assigned_component_ids=assigned_component_ids,
                assignments_by_case_id=assignments_by_case_id,
                assignments=assignments,
            )


def _eligible_domain_component_counts(
    *,
    ordered_components: tuple[LineageComponent, ...],
    profiles: Mapping[str, _ComponentProfile],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for component in ordered_components:
        counts.update(_profile_domains(profiles[component.component_id]))
    return {
        domain: count for domain, count in counts.items() if count >= len(_SPLIT_NAMES)
    }


def _profile_domains(profile: _ComponentProfile) -> tuple[str, ...]:
    return tuple(sorted(profile.metrics["domain"]))


def _domain_seed_sort_key(
    *,
    split_name: Split,
    profile: _ComponentProfile,
    split_case_counts: Mapping[Split, int],
    split_metrics: Mapping[Split, Mapping[str, Counter[str]]],
    global_metrics: Mapping[str, Counter[str]],
    target_case_counts: Mapping[Split, float],
    ratios: tuple[float, float, float],
    total_case_count: int,
    eligible_domain_counts: Mapping[str, int],
    seed: int,
) -> tuple[int, int, float, float, float, float, str]:
    uncovered_domain_count = sum(
        1
        for domain in _profile_domains(profile)
        if domain in eligible_domain_counts
        and split_metrics[split_name]["domain"][domain] == 0
    )
    assignment_key = _assignment_sort_key(
        split_name=split_name,
        profile=profile,
        split_case_counts=split_case_counts,
        split_metrics=split_metrics,
        global_metrics=global_metrics,
        target_case_counts=target_case_counts,
        ratios=ratios,
        total_case_count=total_case_count,
    )
    return (
        -uncovered_domain_count,
        profile.case_count,
        assignment_key[0],
        assignment_key[1],
        assignment_key[2],
        assignment_key[3],
        _seeded_order_key(seed, profile.component.component_id),
    )


def _record_component_assignment(
    *,
    profile: _ComponentProfile,
    split_name: Split,
    split_case_counts: dict[Split, int],
    split_metrics: dict[Split, dict[str, Counter[str]]],
    assigned_component_ids: set[str],
    assignments_by_case_id: dict[str, Split],
    assignments: list[SplitAssignment],
) -> None:
    component = profile.component
    assigned_component_ids.add(component.component_id)
    split_case_counts[split_name] += profile.case_count
    _merge_metrics(split_metrics[split_name], profile.metrics)
    for case_id in component.case_ids:
        assignments_by_case_id[case_id] = split_name
        assignments.append(
            SplitAssignment(
                case_id=case_id,
                split=split_name,
                component_id=component.component_id,
            )
        )


def _assignment_sort_key(
    *,
    split_name: Split,
    profile: _ComponentProfile,
    split_case_counts: Mapping[Split, int],
    split_metrics: Mapping[Split, Mapping[str, Counter[str]]],
    global_metrics: Mapping[str, Counter[str]],
    target_case_counts: Mapping[Split, float],
    ratios: tuple[float, float, float],
    total_case_count: int,
) -> tuple[float, float, float, float, int]:
    ratio = ratios[_SPLIT_NAMES.index(split_name)]
    new_case_count = split_case_counts[split_name] + profile.case_count
    target_case_count = target_case_counts[split_name]
    overshoot_ratio = max(0.0, new_case_count - target_case_count) / max(
        target_case_count, 1.0
    )
    fill_ratio = new_case_count / max(target_case_count, 1.0)
    distance_ratio = abs(new_case_count - target_case_count) / max(
        target_case_count, 1.0
    )
    distribution_penalty = 0.0
    weights = {
        "expected_status": 2.0,
        "support_label": 1.0,
        "transformation_family": 0.75,
        "domain": 1.5,
        "provenance_kind": 2.5,
    }
    for group_name, weight in weights.items():
        current_counts = split_metrics[split_name][group_name]
        global_counts = global_metrics[group_name]
        for label, increment in profile.metrics[group_name].items():
            new_value = current_counts[label] + increment
            target_value = global_counts[label] * ratio
            distribution_penalty += (
                weight * abs(new_value - target_value) / max(global_counts[label], 1)
            )
    return (
        overshoot_ratio,
        fill_ratio,
        distribution_penalty,
        distance_ratio,
        _SPLIT_NAMES.index(split_name),
    )


def _freeze_distributions(
    distributions: Mapping[Split, Mapping[str, Counter[str]]],
) -> Mapping[Split, Mapping[str, Mapping[str, int]]]:
    frozen: dict[Split, Mapping[str, Mapping[str, int]]] = {}
    for split_name, groups in distributions.items():
        frozen_groups: dict[str, Mapping[str, int]] = {}
        for group_name, counts in groups.items():
            frozen_groups[group_name] = MappingProxyType(dict(sorted(counts.items())))
        frozen[split_name] = MappingProxyType(frozen_groups)
    return cast(
        Mapping[Split, Mapping[str, Mapping[str, int]]], MappingProxyType(frozen)
    )


def _case_domain(case: EvaluationCase) -> str:
    if case.difficulty_tags:
        return case.difficulty_tags[0]
    return "__unknown__"


def _seeded_order_key(seed: int, component_id: str) -> str:
    return sha256_hex(f"{seed}:{component_id}".encode("utf-8"))


def _stable_component_id(case_ids: tuple[str, ...]) -> str:
    return f"component-{sha256_hex('|'.join(case_ids).encode('utf-8'))[:20]}"


def _normalized_source_fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    collapsed = _WHITESPACE_RE.sub(" ", normalized).strip()
    return collapsed


__all__ = [
    "CaseLineage",
    "LineageComponent",
    "SplitAssignment",
    "SplitAssignmentReport",
    "apply_split_assignments",
    "assign_splits",
    "build_lineage_components",
]
