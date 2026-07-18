"""Deterministic authored source catalog for evaluation-case generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Generic, Literal, Self, TypeVar
from unicodedata import normalize

from pydantic import BaseModel as PydanticBaseModel
from pydantic import (
    ConfigDict,
    GetCoreSchemaHandler,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

from evaluation.schema import CharSpan

Domain = Literal["science", "finance", "policy", "technology", "health", "history"]
_ValueT = TypeVar("_ValueT")

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
REQUIRED_TRANSFORMATION_NAMES = (
    "negation",
    "number",
    "unit",
    "date",
    "entity",
    "relation",
    "modality",
    "unsupported_clause",
    "unicode",
    "duplicate_distractor",
    "multi_span",
    "multi_source",
)
RETRIEVAL_DATE = date(2026, 7, 17)


class FrozenMapping(Mapping[str, _ValueT], Generic[_ValueT]):
    """Concrete immutable mapping preserved by Pydantic validation."""

    __slots__ = ("_items", "_lookup")
    _items: tuple[tuple[str, _ValueT], ...]
    _lookup: dict[str, _ValueT]

    def __init__(
        self,
        items: Mapping[str, _ValueT],
    ) -> None:
        tuple_items = tuple((key, value) for key, value in items.items())
        self._items = tuple_items
        self._lookup = dict(tuple_items)
        if len(self._lookup) != len(self._items):
            raise ValueError("frozen mappings must not contain duplicate keys")

    def __getitem__(self, key: str) -> _ValueT:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        del source_type, handler
        return core_schema.no_info_plain_validator_function(cls._validate_input)

    @classmethod
    def _validate_input(cls, value: object) -> Self:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("frozen mappings must be provided as mappings")
        return cls(value)

    def to_dict(self) -> dict[str, _ValueT]:
        return {key: value for key, value in self._items}


class Evidence(PydanticBaseModel):
    model_config = STRICT_MODEL_CONFIG

    slot_id: str
    text: str
    span: CharSpan

    @field_validator("slot_id", "text")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        return _require_non_empty(value, "evidence fields must be non-empty")


class Fact(PydanticBaseModel):
    model_config = STRICT_MODEL_CONFIG

    fact_id: str
    claim_template: str
    # FrozenMapping preserves `str.format(**fact.slots)` ergonomics while
    # preventing post-construction mutation of the authored slot inventory.
    slots: FrozenMapping[str]
    answer_slots: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    adversarial_variants: FrozenMapping[object]

    @field_validator("fact_id", "claim_template")
    @classmethod
    def _validate_non_empty_fields(cls, value: str) -> str:
        return _require_non_empty(value, "fact identifiers and templates must be non-empty")

    @field_validator("slots", mode="before")
    @classmethod
    def _freeze_slots(cls, value: object) -> FrozenMapping[str]:
        return _freeze_string_mapping(
            value,
            empty_message="facts must define at least one slot",
            key_message="slot ids must be non-empty",
            value_message="slot text must be non-empty",
        )

    @field_validator("answer_slots")
    @classmethod
    def _validate_answer_slots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("facts must define at least one answer slot")
        cleaned = tuple(_require_non_empty(slot, "answer slots must be non-empty") for slot in value)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("answer slots must be unique")
        return cleaned

    @field_validator("adversarial_variants", mode="before")
    @classmethod
    def _freeze_adversarial_variants(
        cls,
        value: object,
    ) -> FrozenMapping[object]:
        return _freeze_variant_mapping(value)

    @field_serializer("slots")
    def _serialize_slots(self, value: FrozenMapping[str]) -> dict[str, str]:
        return value.to_dict()

    @field_serializer("adversarial_variants")
    def _serialize_adversarial_variants(
        self,
        value: FrozenMapping[object],
    ) -> dict[str, dict[str, object]]:
        return {
            key: _materialize_json_like(config)
            for key, config in value.items()
        }

    @model_validator(mode="after")
    def _validate_fact(self) -> Fact:
        if not self.evidence:
            raise ValueError("facts must define at least one evidence span")

        slot_ids = set(self.slots)
        if not set(self.answer_slots).issubset(slot_ids):
            raise ValueError("answer slots must reference defined slots")

        try:
            self.claim_template.format(**dict(self.slots))
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "claim templates must resolve using defined slots"
            ) from exc

        evidence_slot_ids: set[str] = set()
        previous_order_key: tuple[int, int, str] | None = None
        for evidence in self.evidence:
            if evidence.slot_id not in slot_ids:
                raise ValueError("evidence slot ids must reference defined slots")
            if evidence.slot_id in evidence_slot_ids:
                raise ValueError("evidence slot ids must be unique within a fact")
            if self.slots[evidence.slot_id] != evidence.text:
                raise ValueError("evidence text must equal the referenced slot text")
            order_key = (evidence.span.start, evidence.span.end, evidence.slot_id)
            if previous_order_key is not None and order_key <= previous_order_key:
                raise ValueError("fact evidence must be ordered by source span")
            evidence_slot_ids.add(evidence.slot_id)
            previous_order_key = order_key

        return self


class FactTemplate(PydanticBaseModel):
    model_config = STRICT_MODEL_CONFIG

    family_id: str
    domain: Domain
    source_text: str
    facts: tuple[Fact, ...]
    provenance_title: str
    provenance_origin: str
    provenance_publisher: str
    provenance_license: str
    provenance_retrieval_date: date

    @field_validator(
        "family_id",
        "source_text",
        "provenance_title",
        "provenance_origin",
        "provenance_publisher",
        "provenance_license",
    )
    @classmethod
    def _validate_non_empty_template_strings(cls, value: str) -> str:
        return _require_non_empty(value, "template text fields must be non-empty")

    @model_validator(mode="after")
    def _validate_template(self) -> FactTemplate:
        if not self.family_id.startswith(f"{self.domain}-"):
            raise ValueError("family ids must be prefixed with the template domain")
        if not self.facts:
            raise ValueError("fact templates must define at least one fact")

        fact_ids = tuple(fact.fact_id for fact in self.facts)
        if fact_ids != tuple(sorted(fact_ids)):
            raise ValueError("fact ids must be sorted within a template")
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("fact ids must be unique within a template")

        advertised_transformations: set[str] = set()
        for fact in self.facts:
            advertised_transformations.update(fact.adversarial_variants)
            for evidence in fact.evidence:
                if evidence.span.end > len(self.source_text):
                    raise ValueError(
                        "evidence spans must stay within the source text bounds"
                    )
                if self.source_text[evidence.span.start : evidence.span.end] != evidence.text:
                    raise ValueError(
                        "evidence text must equal the referenced source slice"
                    )

        if advertised_transformations != set(REQUIRED_TRANSFORMATION_NAMES):
            raise ValueError(
                "fact templates must advertise every required transformation family"
            )
        return self


@dataclass(frozen=True)
class _DomainSpec:
    domain: Domain
    cycle_claim_template: str
    negated_cycle_template: str
    unit_cycle_template: str
    relation_cycle_template: str
    multi_span_source_template: str
    multi_span_citation_templates: tuple[str, str]
    date_claim_template: str
    modality_claim_template: str
    modality_variant_template: str
    unicode_claim_template: str
    conjunction_claim_template: str
    secondary_source_template: str
    unsupported_suffix: str
    publisher: str


@dataclass(frozen=True)
class _FamilySeed:
    slug: str
    subject: str
    alternate_subject: str
    event_label: str
    program_label: str
    period: str
    year: str
    end_year: str
    unicode_term_nfc: str


_DOMAIN_SPECS: Mapping[Domain, _DomainSpec] = MappingProxyType(
    {
        "science": _DomainSpec(
            domain="science",
            cycle_claim_template="{subject} completes one orbit every {period} days.",
            negated_cycle_template=(
                "{subject} does not complete one orbit every {period} days."
            ),
            unit_cycle_template="{subject} completes one orbit every {period} weeks.",
            relation_cycle_template="{subject} documents one orbit every {period} days.",
            multi_span_source_template=(
                "{subject} completes one orbit. "
                "Archive staff track {event_label} telescope windows separately. "
                "That orbit lasts {period} days."
            ),
            multi_span_citation_templates=(
                "{subject} completes one orbit.",
                "That orbit lasts {period} days.",
            ),
            date_claim_template="The {event_label} bulletin was issued in {year}.",
            modality_claim_template=(
                "The report states the {program_label} should remain active through "
                "{end_year}."
            ),
            modality_variant_template=(
                "The report states the {program_label} will remain active through "
                "{end_year}."
            ),
            unicode_claim_template=(
                "Editors label the archival note {mode_name} in the margin."
            ),
            conjunction_claim_template=(
                "{subject} completes one orbit every {period} days and the "
                "{event_label} bulletin was issued in {year}."
            ),
            secondary_source_template=(
                "Mission logs confirm the {event_label} bulletin was issued in {year}."
            ),
            unsupported_suffix=" It cites an unverified helium canal.",
            publisher="Cite-Right Science Desk",
        ),
        "finance": _DomainSpec(
            domain="finance",
            cycle_claim_template="{subject} closes the reporting cycle every {period} days.",
            negated_cycle_template=(
                "{subject} does not close the reporting cycle every {period} days."
            ),
            unit_cycle_template="{subject} closes the reporting cycle every {period} weeks.",
            relation_cycle_template="{subject} audits the reporting cycle every {period} days.",
            multi_span_source_template=(
                "{subject} closes the reporting cycle. "
                "Desk staff publish {event_label} compliance notes separately. "
                "That reporting cycle lasts {period} days."
            ),
            multi_span_citation_templates=(
                "{subject} closes the reporting cycle.",
                "That reporting cycle lasts {period} days.",
            ),
            date_claim_template="The {event_label} fund opened in {year}.",
            modality_claim_template=(
                "The desk memo states the {program_label} should remain active through "
                "{end_year}."
            ),
            modality_variant_template=(
                "The desk memo states the {program_label} will remain active through "
                "{end_year}."
            ),
            unicode_claim_template=(
                "Analysts describe the control note as {mode_name} in desk annotations."
            ),
            conjunction_claim_template=(
                "{subject} closes the reporting cycle every {period} days and the "
                "{event_label} fund opened in {year}."
            ),
            secondary_source_template=(
                "Fund records confirm the {event_label} fund opened in {year}."
            ),
            unsupported_suffix=" It mentions an unverified mezzanine reserve.",
            publisher="Cite-Right Finance Desk",
        ),
        "policy": _DomainSpec(
            domain="policy",
            cycle_claim_template="{subject} renews the review cycle every {period} days.",
            negated_cycle_template=(
                "{subject} does not renew the review cycle every {period} days."
            ),
            unit_cycle_template="{subject} renews the review cycle every {period} weeks.",
            relation_cycle_template="{subject} funds the review cycle every {period} days.",
            multi_span_source_template=(
                "{subject} renews the review cycle. Clerks archive {event_label} agenda packets separately. "
                "That review cycle lasts {period} days."
            ),
            multi_span_citation_templates=(
                "{subject} renews the review cycle.",
                "That review cycle lasts {period} days.",
            ),
            date_claim_template="The {event_label} ordinance took effect in {year}.",
            modality_claim_template=(
                "The committee memo states the {program_label} should remain active "
                "through {end_year}."
            ),
            modality_variant_template=(
                "The committee memo states the {program_label} will remain active "
                "through {end_year}."
            ),
            unicode_claim_template=(
                "Staff annotate the guidance note as {mode_name} in the margin."
            ),
            conjunction_claim_template=(
                "{subject} renews the review cycle every {period} days and the "
                "{event_label} ordinance took effect in {year}."
            ),
            secondary_source_template=(
                "Council minutes confirm the {event_label} ordinance took effect in "
                "{year}."
            ),
            unsupported_suffix=" It adds an unapproved annex clause.",
            publisher="Cite-Right Policy Desk",
        ),
        "technology": _DomainSpec(
            domain="technology",
            cycle_claim_template="{subject} completes one backup cycle every {period} days.",
            negated_cycle_template=(
                "{subject} does not complete one backup cycle every {period} days."
            ),
            unit_cycle_template="{subject} completes one backup cycle every {period} weeks.",
            relation_cycle_template="{subject} archives one backup cycle every {period} days.",
            multi_span_source_template=(
                "{subject} completes one backup cycle. "
                "Operators rotate {event_label} storage checks weekly. "
                "That backup cycle lasts {period} days."
            ),
            multi_span_citation_templates=(
                "{subject} completes one backup cycle.",
                "That backup cycle lasts {period} days.",
            ),
            date_claim_template="The {event_label} platform launched in {year}.",
            modality_claim_template=(
                "The operations memo states the {program_label} should remain active "
                "through {end_year}."
            ),
            modality_variant_template=(
                "The operations memo states the {program_label} will remain active "
                "through {end_year}."
            ),
            unicode_claim_template=(
                "Engineers tag the fallback mode {mode_name} in release notes."
            ),
            conjunction_claim_template=(
                "{subject} completes one backup cycle every {period} days and the "
                "{event_label} platform launched in {year}."
            ),
            secondary_source_template=(
                "Release logs confirm the {event_label} platform launched in {year}."
            ),
            unsupported_suffix=" It references an undocumented failover rack.",
            publisher="Cite-Right Technology Desk",
        ),
        "health": _DomainSpec(
            domain="health",
            cycle_claim_template="{subject} completes one screening cycle every {period} days.",
            negated_cycle_template=(
                "{subject} does not complete one screening cycle every {period} days."
            ),
            unit_cycle_template="{subject} completes one screening cycle every {period} weeks.",
            relation_cycle_template="{subject} schedules one screening cycle every {period} days.",
            multi_span_source_template=(
                "{subject} completes one screening cycle. Nurses log {event_label} intake notes separately. "
                "That screening cycle lasts {period} days."
            ),
            multi_span_citation_templates=(
                "{subject} completes one screening cycle.",
                "That screening cycle lasts {period} days.",
            ),
            date_claim_template="The {event_label} clinic opened in {year}.",
            modality_claim_template=(
                "The care memo states the {program_label} should remain active through "
                "{end_year}."
            ),
            modality_variant_template=(
                "The care memo states the {program_label} will remain active through "
                "{end_year}."
            ),
            unicode_claim_template=(
                "Clinicians label the handoff note {mode_name} in training notes."
            ),
            conjunction_claim_template=(
                "{subject} completes one screening cycle every {period} days and the "
                "{event_label} clinic opened in {year}."
            ),
            secondary_source_template=(
                "Clinic records confirm the {event_label} clinic opened in {year}."
            ),
            unsupported_suffix=" It adds an unsupported dosage note.",
            publisher="Cite-Right Health Desk",
        ),
        "history": _DomainSpec(
            domain="history",
            cycle_claim_template="{subject} rotates through the gallery every {period} days.",
            negated_cycle_template=(
                "{subject} does not rotate through the gallery every {period} days."
            ),
            unit_cycle_template="{subject} rotates through the gallery every {period} weeks.",
            relation_cycle_template="{subject} curates the gallery every {period} days.",
            multi_span_source_template=(
                "{subject} rotates through the gallery. "
                "Curators catalog {event_label} placards separately. "
                "That rotation recurs every {period} days."
            ),
            multi_span_citation_templates=(
                "{subject} rotates through the gallery.",
                "That rotation recurs every {period} days.",
            ),
            date_claim_template="The {event_label} exhibit opened in {year}.",
            modality_claim_template=(
                "The curator memo states the {program_label} should remain active "
                "through {end_year}."
            ),
            modality_variant_template=(
                "The curator memo states the {program_label} will remain active through "
                "{end_year}."
            ),
            unicode_claim_template=(
                "Curators label the catalogue note {mode_name} in margin notes."
            ),
            conjunction_claim_template=(
                "{subject} rotates through the gallery every {period} days and the "
                "{event_label} exhibit opened in {year}."
            ),
            secondary_source_template=(
                "Archive logs confirm the {event_label} exhibit opened in {year}."
            ),
            unsupported_suffix=" It mentions an unverified donor ledger.",
            publisher="Cite-Right History Desk",
        ),
    }
)

_DOMAIN_FAMILY_SEEDS: Mapping[Domain, tuple[_FamilySeed, ...]] = MappingProxyType(
    {
        "science": (
            _FamilySeed("01-mercury-archive", "Mercury", "Venus", "helios launch", "orbital probe", "118", "1977", "2035", "café-safe"),
            _FamilySeed("02-europa-circular", "Europa", "Io", "europa survey", "ice mapper", "121", "1980", "2036", "naïve-lock"),
            _FamilySeed("03-ceres-docket", "Ceres", "Vesta", "ceres relay", "mineral scanner", "124", "1983", "2037", "façade-ready"),
            _FamilySeed("04-titan-notice", "Titan", "Rhea", "titan ingress", "methane array", "127", "1986", "2038", "jalapeño-mode"),
            _FamilySeed("05-ganymede-log", "Ganymede", "Callisto", "ganymede survey", "magnetics suite", "131", "1989", "2040", "résumé-check"),
            _FamilySeed("06-enceladus-file", "Enceladus", "Dione", "enceladus drift", "plume sampler", "134", "1992", "2043", "piñata-guard"),
            _FamilySeed("07-vesta-ledger", "Vesta", "Pallas", "vesta scan", "regolith beacon", "137", "1995", "2044", "coöperate-path"),
            _FamilySeed("08-ariel-register", "Ariel", "Umbriel", "ariel trace", "polar mapper", "142", "1998", "2045", "touché-state"),
            _FamilySeed("09-triton-brief", "Triton", "Nereid", "triton relay", "thermal lattice", "146", "2001", "2046", "protégé-flag"),
            _FamilySeed("10-oberon-ledger", "Oberon", "Titania", "oberon survey", "shadow monitor", "149", "2004", "2047", "élan-mark"),
        ),
        "finance": (
            _FamilySeed("01-harbor-fund", "Harbor Growth Fund", "Beacon Income Fund", "harbor growth", "liquidity sleeve", "154", "2007", "2033", "café-stable"),
            _FamilySeed("02-cedar-fund", "Cedar Yield Fund", "Maple Credit Fund", "cedar yield", "hedge overlay", "157", "2010", "2034", "naïve-check"),
            _FamilySeed("03-river-fund", "River Macro Fund", "Summit Macro Fund", "river macro", "risk corridor", "161", "2011", "2039", "façade-lock"),
            _FamilySeed("04-orbit-fund", "Orbit Value Fund", "Atlas Value Fund", "orbit value", "carry sleeve", "164", "2012", "2041", "jalapeño-note"),
            _FamilySeed("05-meridian-fund", "Meridian Index Fund", "Vertex Index Fund", "meridian index", "rebalance window", "167", "2013", "2042", "résumé-guard"),
            _FamilySeed("06-bright-fund", "Bright Alpha Fund", "North Alpha Fund", "bright alpha", "settlement rail", "171", "2014", "2048", "piñata-screen"),
            _FamilySeed("07-lattice-fund", "Lattice Credit Fund", "Signal Credit Fund", "lattice credit", "treasury buffer", "174", "2015", "2049", "coöperate-loop"),
            _FamilySeed("08-prairie-fund", "Prairie Blend Fund", "Sierra Blend Fund", "prairie blend", "coupon ladder", "178", "2016", "2050", "touché-mark"),
            _FamilySeed("09-anchor-fund", "Anchor Reserve Fund", "Harbor Reserve Fund", "anchor reserve", "compliance rail", "181", "2017", "2051", "protégé-step"),
            _FamilySeed("10-pivot-fund", "Pivot Income Fund", "Summit Income Fund", "pivot income", "capital sleeve", "184", "2018", "2052", "élan-switch"),
        ),
        "policy": (
            _FamilySeed("01-air-review", "Air Quality Board", "Water Quality Board", "clean air", "review docket", "186", "1999", "2032", "café-verified"),
            _FamilySeed("02-housing-review", "Housing Appeals Panel", "Transit Appeals Panel", "housing access", "compliance roster", "188", "2000", "2036", "naïve-marker"),
            _FamilySeed("03-water-review", "Water Standards Office", "Energy Standards Office", "river safety", "permit ledger", "191", "2002", "2037", "façade-signoff"),
            _FamilySeed("04-ethics-review", "Ethics Review Unit", "Audits Review Unit", "ethics reform", "appeals register", "194", "2003", "2038", "jalapeño-check"),
            _FamilySeed("05-zoning-review", "Zoning Review Council", "Budget Review Council", "downtown zoning", "variance program", "197", "2005", "2039", "résumé-lock"),
            _FamilySeed("06-transit-review", "Transit Licensing Desk", "Aviation Licensing Desk", "transit route", "waiver notice", "202", "2006", "2040", "piñata-path"),
            _FamilySeed("07-energy-review", "Energy Permits Office", "Climate Permits Office", "grid resilience", "monitoring charter", "206", "2008", "2043", "coöperate-flag"),
            _FamilySeed("08-coastal-review", "Coastal Claims Panel", "River Claims Panel", "coastal access", "appeals bureau", "209", "2009", "2044", "touché-proof"),
            _FamilySeed("09-labor-review", "Labor Standards Unit", "Benefits Standards Unit", "shift safety", "inspection circle", "212", "2011", "2045", "protégé-note"),
            _FamilySeed("10-civic-review", "Civic Records Office", "Parks Records Office", "civic records", "renewal archive", "214", "2012", "2046", "élan-audit"),
        ),
        "technology": (
            _FamilySeed("01-orbit-platform", "Orbit Backup Cluster", "Atlas Backup Cluster", "orbit backup", "telemetry daemon", "216", "2013", "2031", "café-buffer"),
            _FamilySeed("02-signal-platform", "Signal Archive Grid", "Beacon Archive Grid", "signal archive", "storage relay", "219", "2014", "2035", "naïve-fallback"),
            _FamilySeed("03-cedar-platform", "Cedar Recovery Mesh", "Maple Recovery Mesh", "cedar recovery", "retention agent", "223", "2015", "2036", "façade-cache"),
            _FamilySeed("04-nimbus-platform", "Nimbus Snapshot Farm", "Cirrus Snapshot Farm", "nimbus snapshot", "routing kernel", "226", "2016", "2037", "jalapeño-guard"),
            _FamilySeed("05-matrix-platform", "Matrix Logging Ring", "Vector Logging Ring", "matrix logging", "control plane", "229", "2017", "2038", "résumé-flip"),
            _FamilySeed("06-polar-platform", "Polar Sync Cluster", "Aurora Sync Cluster", "polar sync", "observer loop", "232", "2018", "2039", "piñata-cache"),
            _FamilySeed("07-rivet-platform", "Rivet Build Array", "Forge Build Array", "rivet build", "deployment lane", "236", "2019", "2040", "coöperate-signal"),
            _FamilySeed("08-lantern-platform", "Lantern Patch Service", "Beacon Patch Service", "lantern patch", "repair runner", "239", "2020", "2041", "touché-gate"),
            _FamilySeed("09-harbor-platform", "Harbor Mirror Queue", "Anchor Mirror Queue", "harbor mirror", "audit reactor", "242", "2021", "2042", "protégé-shield"),
            _FamilySeed("10-summit-platform", "Summit Restore Hub", "Vertex Restore Hub", "summit restore", "fallback switch", "244", "2022", "2043", "élan-latch"),
        ),
        "health": (
            _FamilySeed("01-aurora-clinic", "Aurora Screening Team", "Beacon Screening Team", "aurora care", "immunization roster", "246", "2004", "2033", "café-clear"),
            _FamilySeed("02-cedar-clinic", "Cedar Cardio Unit", "Maple Cardio Unit", "cedar cardiology", "followup tracker", "248", "2005", "2034", "naïve-checkin"),
            _FamilySeed("03-harbor-clinic", "Harbor Wellness Group", "Anchor Wellness Group", "harbor wellness", "triage ladder", "251", "2007", "2035", "façade-note"),
            _FamilySeed("04-lumen-clinic", "Lumen Vision Team", "Prism Vision Team", "lumen vision", "intake relay", "253", "2008", "2036", "jalapeño-screen"),
            _FamilySeed("05-prairie-clinic", "Prairie Nutrition Desk", "Sierra Nutrition Desk", "prairie nutrition", "care corridor", "256", "2010", "2037", "résumé-stage"),
            _FamilySeed("06-signal-clinic", "Signal Rehab Office", "Beacon Rehab Office", "signal rehab", "mobility route", "259", "2011", "2038", "piñata-guard"),
            _FamilySeed("07-river-clinic", "River Oncology Team", "Summit Oncology Team", "river oncology", "support charter", "262", "2013", "2039", "coöperate-note"),
            _FamilySeed("08-summit-clinic", "Summit Pediatric Desk", "Vertex Pediatric Desk", "summit pediatrics", "growth register", "264", "2014", "2040", "touché-frame"),
            _FamilySeed("09-lattice-clinic", "Lattice Pulmonary Unit", "Signal Pulmonary Unit", "lattice pulmonary", "monitoring lane", "267", "2016", "2041", "protégé-loop"),
            _FamilySeed("10-meridian-clinic", "Meridian Care Circle", "North Care Circle", "meridian care", "handoff archive", "269", "2017", "2042", "élan-screen"),
        ),
        "history": (
            _FamilySeed("01-river-exhibit", "River Trade Exhibit", "Harbor Trade Exhibit", "river trade", "gallery program", "271", "1984", "2031", "café-ledger"),
            _FamilySeed("02-mosaic-exhibit", "Mosaic Letters Exhibit", "Archive Letters Exhibit", "mosaic letters", "catalog program", "274", "1985", "2032", "naïve-catalog"),
            _FamilySeed("03-foundry-exhibit", "Foundry Tools Exhibit", "Workshop Tools Exhibit", "foundry tools", "rotation ledger", "277", "1987", "2034", "façade-ribbon"),
            _FamilySeed("04-marble-exhibit", "Marble Maps Exhibit", "Granite Maps Exhibit", "marble maps", "preservation file", "281", "1988", "2035", "jalapeño-index"),
            _FamilySeed("05-lantern-exhibit", "Lantern Posters Exhibit", "Beacon Posters Exhibit", "lantern posters", "docent program", "284", "1990", "2036", "résumé-stamp"),
            _FamilySeed("06-harvest-exhibit", "Harvest Songs Exhibit", "Prairie Songs Exhibit", "harvest songs", "rotation memo", "287", "1991", "2037", "piñata-signal"),
            _FamilySeed("07-copper-exhibit", "Copper Coins Exhibit", "Silver Coins Exhibit", "copper coins", "catalog relay", "289", "1993", "2038", "coöperate-label"),
            _FamilySeed("08-orbit-exhibit", "Orbit Instruments Exhibit", "Signal Instruments Exhibit", "orbit instruments", "exhibit charter", "293", "1994", "2039", "touché-tally"),
            _FamilySeed("09-cedar-exhibit", "Cedar Journals Exhibit", "Maple Journals Exhibit", "cedar journals", "archive program", "296", "1996", "2040", "protégé-plaque"),
            _FamilySeed("10-meridian-exhibit", "Meridian Textile Exhibit", "Vertex Textile Exhibit", "meridian textiles", "gallery docket", "299", "1997", "2041", "élan-marker"),
        ),
    }
)


def _build_template(*, domain: Domain, spec: _DomainSpec, seed: _FamilySeed) -> FactTemplate:
    slots = {
        "subject": seed.subject,
        "period": seed.period,
        "year": seed.year,
        "end_year": seed.end_year,
        "program_label": seed.program_label,
        "event_label": seed.event_label,
        "mode_name": normalize("NFD", seed.unicode_term_nfc),
    }

    cycle_sentence = spec.cycle_claim_template.format(
        subject=slots["subject"],
        period=slots["period"],
    )
    date_sentence = spec.date_claim_template.format(
        event_label=slots["event_label"],
        year=slots["year"],
    )
    modality_sentence = spec.modality_claim_template.format(
        program_label=slots["program_label"],
        end_year=slots["end_year"],
    )
    unicode_sentence = spec.unicode_claim_template.format(
        mode_name=slots["mode_name"],
    )
    source_text = " ".join(
        (cycle_sentence, date_sentence, modality_sentence, unicode_sentence)
    )

    return FactTemplate(
        family_id=f"{domain}-{seed.slug}",
        domain=domain,
        source_text=source_text,
        facts=(
            Fact(
                fact_id="fact-conjunction",
                claim_template=spec.conjunction_claim_template,
                slots=_freeze_string_mapping(
                    {
                        "subject": seed.subject,
                        "period": seed.period,
                        "year": seed.year,
                        "event_label": seed.event_label,
                    },
                    empty_message="facts must define at least one slot",
                    key_message="slot ids must be non-empty",
                    value_message="slot text must be non-empty",
                ),
                answer_slots=("subject", "period", "year"),
                evidence=(
                    _evidence(source_text, "subject", seed.subject),
                    _evidence(source_text, "period", seed.period),
                    _evidence(source_text, "year", seed.year),
                ),
                adversarial_variants=_freeze_variant_mapping(
                    {
                        "multi_source": {
                            "answer_text": (
                                spec.cycle_claim_template.format(
                                    subject=seed.subject,
                                    period=seed.period,
                                )
                                + " "
                                + spec.date_claim_template.format(
                                    event_label=seed.event_label,
                                    year=seed.year,
                                )
                            ),
                            "primary_claim_text": spec.cycle_claim_template.format(
                                subject=seed.subject,
                                period=seed.period,
                            ),
                            "primary_source_text": spec.cycle_claim_template.format(
                                subject=seed.subject,
                                period=seed.period,
                            ),
                            "secondary_claim_text": spec.date_claim_template.format(
                                event_label=seed.event_label,
                                year=seed.year,
                            ),
                            "secondary_source_text": spec.secondary_source_template.format(
                                event_label=seed.event_label,
                                year=seed.year,
                            )
                        }
                    }
                ),
            ),
            Fact(
                fact_id="fact-cycle",
                claim_template=spec.cycle_claim_template,
                slots=_freeze_string_mapping(
                    {
                        "subject": seed.subject,
                        "period": seed.period,
                    },
                    empty_message="facts must define at least one slot",
                    key_message="slot ids must be non-empty",
                    value_message="slot text must be non-empty",
                ),
                answer_slots=("subject", "period"),
                evidence=(
                    _evidence(source_text, "subject", seed.subject),
                    _evidence(source_text, "period", seed.period),
                ),
                adversarial_variants=_freeze_variant_mapping(
                    {
                        "negation": {"claim_template": spec.negated_cycle_template},
                        "number": {"slots": {"period": str(int(seed.period) + 1)}},
                        "unit": {"claim_template": spec.unit_cycle_template},
                        "entity": {"slots": {"subject": seed.alternate_subject}},
                        "relation": {"claim_template": spec.relation_cycle_template},
                        "multi_span": {
                            "citation_texts": tuple(
                                template.format(
                                    subject=seed.subject,
                                    event_label=seed.event_label,
                                    period=seed.period,
                                )
                                for template in spec.multi_span_citation_templates
                            ),
                            "primary_source_text": spec.multi_span_source_template.format(
                                subject=seed.subject,
                                event_label=seed.event_label,
                                period=seed.period,
                            ),
                        },
                    }
                ),
            ),
            Fact(
                fact_id="fact-date",
                claim_template=spec.date_claim_template,
                slots=_freeze_string_mapping(
                    {
                        "event_label": seed.event_label,
                        "year": seed.year,
                    },
                    empty_message="facts must define at least one slot",
                    key_message="slot ids must be non-empty",
                    value_message="slot text must be non-empty",
                ),
                answer_slots=("year",),
                evidence=(
                    _evidence(source_text, "year", seed.year),
                ),
                adversarial_variants=_freeze_variant_mapping(
                    {
                        "date": {"slots": {"year": str(int(seed.year) + 1)}},
                        "duplicate_distractor": {
                            "distractor_source_text": spec.date_claim_template.format(
                                event_label=seed.event_label,
                                year=str(int(seed.year) + 1),
                            )
                        },
                    }
                ),
            ),
            Fact(
                fact_id="fact-modality",
                claim_template=spec.modality_claim_template,
                slots=_freeze_string_mapping(
                    {
                        "program_label": seed.program_label,
                        "end_year": seed.end_year,
                    },
                    empty_message="facts must define at least one slot",
                    key_message="slot ids must be non-empty",
                    value_message="slot text must be non-empty",
                ),
                answer_slots=("end_year",),
                evidence=(
                    _evidence(source_text, "end_year", seed.end_year),
                ),
                adversarial_variants=_freeze_variant_mapping(
                    {
                        "modality": {
                            "claim_template": spec.modality_variant_template
                        },
                        "unsupported_clause": {
                            "unsupported_suffix": spec.unsupported_suffix
                        },
                    }
                ),
            ),
            Fact(
                fact_id="fact-unicode",
                claim_template=spec.unicode_claim_template,
                slots=_freeze_string_mapping(
                    {
                        "mode_name": normalize("NFD", seed.unicode_term_nfc),
                    },
                    empty_message="facts must define at least one slot",
                    key_message="slot ids must be non-empty",
                    value_message="slot text must be non-empty",
                ),
                answer_slots=("mode_name",),
                evidence=(
                    _evidence(source_text, "mode_name", normalize("NFD", seed.unicode_term_nfc)),
                ),
                adversarial_variants=_freeze_variant_mapping(
                    {
                        "unicode": {"slots": {"mode_name": seed.unicode_term_nfc}}
                    }
                ),
            ),
        ),
        provenance_title=f"{seed.event_label.title()} reference packet",
        provenance_origin="authored-evaluation",
        provenance_publisher=spec.publisher,
        provenance_license="CC-BY-4.0",
        provenance_retrieval_date=RETRIEVAL_DATE,
    )


def _evidence(source_text: str, slot_id: str, text: str) -> Evidence:
    return Evidence(slot_id=slot_id, text=text, span=_find_unique_span(source_text, text))


def _find_unique_span(source_text: str, fragment: str) -> CharSpan:
    if source_text.count(fragment) != 1:
        raise ValueError("authored evidence fragments must appear exactly once in source text")
    start = source_text.index(fragment)
    return CharSpan(start=start, end=start + len(fragment))


def _freeze_string_mapping(
    value: object,
    *,
    empty_message: str,
    key_message: str,
    value_message: str,
) -> FrozenMapping[str]:
    if not isinstance(value, Mapping):
        raise ValueError(empty_message)
    copied: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(key_message)
        if not isinstance(item, str):
            raise ValueError(value_message)
        copied[key] = _require_non_empty(item, value_message)
    if not copied:
        raise ValueError(empty_message)
    for key in copied:
        if not key.strip():
            raise ValueError(key_message)
    return FrozenMapping(copied)


def _freeze_variant_mapping(value: object) -> FrozenMapping[object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("facts must define at least one adversarial variant")
    frozen_variants: dict[str, object] = {}
    for raw_name, raw_config in value.items():
        if not isinstance(raw_name, str):
            raise ValueError("adversarial variant names must be non-empty")
        name = _require_non_empty(raw_name, "adversarial variant names must be non-empty")
        if not isinstance(raw_config, Mapping) or not raw_config:
            raise ValueError("adversarial variants must define non-empty configuration")
        frozen_variants[name] = _deep_freeze_mapping(raw_config)
    return FrozenMapping(frozen_variants)


def _deep_freeze_mapping(value: Mapping[object, object]) -> FrozenMapping[object]:
    frozen: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(
                "adversarial variant configuration keys must be non-empty"
            )
        key = _require_non_empty(
            raw_key, "adversarial variant configuration keys must be non-empty"
        )
        frozen[key] = _deep_freeze_value(raw_value)
    return FrozenMapping(frozen)


def _deep_freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _deep_freeze_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze_value(item) for item in value)
    return value


def _materialize_json_like(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping-shaped JSON-like value")
    materialized: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            materialized[key] = _materialize_json_like(item)
        elif isinstance(item, tuple):
            materialized[key] = [
                _materialize_json_like(entry) if isinstance(entry, Mapping) else entry
                for entry in item
            ]
        else:
            materialized[key] = item
    return materialized


def _require_non_empty(value: str, message: str) -> str:
    if not value.strip():
        raise ValueError(message)
    return value
AUTHORED_FACT_TEMPLATES: tuple[FactTemplate, ...] = tuple(
    sorted(
        (
            _build_template(domain=domain, spec=_DOMAIN_SPECS[domain], seed=seed)
            for domain, seeds in _DOMAIN_FAMILY_SEEDS.items()
            for seed in seeds
        ),
        key=lambda template: template.family_id,
    )
)
