"""Tests for Data2txt field:value paraphrase support (issue #50)."""

from cite_right import CitationConfig, align_citations
from cite_right.citations import _looks_like_structured_source


class TestData2txtFieldValueSupport:
    """Faithful rewrites of flattened field:value lines should not be unsupported."""

    def test_business_stars_and_wifi_field_rewrite_is_supported(self) -> None:
        source = """business_stars: 4.5
attributes.WiFi: free"""
        answer = "The business has a rating of 4.5 stars and offers free WiFi."

        results = align_citations(answer, [source], config=CitationConfig())

        assert results
        assert results[0].status in {"supported", "partial"}
        assert results[0].citations

    def test_hours_field_rewrite_is_supported(self) -> None:
        source = """hours.Monday: 9:0-17:0
hours.Tuesday: 9:0-17:0
hours.Wednesday: 9:0-17:0
hours.Thursday: 9:0-17:0
hours.Friday: 9:0-17:0"""
        answer = "Monday–Friday 9:00 AM–5:00 PM"

        results = align_citations(answer, [source], config=CitationConfig())

        assert results
        assert results[0].status != "unsupported"
        assert results[0].citations

    def test_null_wifi_with_invented_amenity_stays_unsupported(self) -> None:
        source = """business_stars: 3.5
attributes.WiFi: null
attributes.OutdoorSeating: null"""
        answer = "This place offers free Wi-Fi."

        results = align_citations(answer, [source], config=CitationConfig())

        assert results
        assert results[0].status == "unsupported"

    def test_platform_mismatch_not_fully_supported(self) -> None:
        source = """business_stars: 3.5
name: Best Restaurant"""
        answer = "The restaurant has a 3.5 star rating on Google."

        results = align_citations(answer, [source], config=CitationConfig())

        assert results
        # Star value is grounded; Google is invented. Do not mark supported.
        assert results[0].status != "supported"

    def test_mixed_field_source_with_review_text(self) -> None:
        source = """business_stars: 4.5
attributes.WiFi: free
attributes.OutdoorSeating: true

Review: Great place with nice outdoor seating."""
        answer = (
            "The business has a 4.5 star rating and offers free WiFi. "
            "Great outdoor seating area."
        )

        results = align_citations(answer, [source], config=CitationConfig())

        assert results
        assert results[0].status in {"supported", "partial"}

    def test_field_rewrite_still_works_beside_unrelated_prose(self) -> None:
        structured = """business_stars: 4.5
attributes.WiFi: free"""
        prose = "Unrelated weather report about rain in Seattle yesterday."
        answer = "The business has a rating of 4.5 stars and offers free WiFi."

        results = align_citations(answer, [structured, prose], config=CitationConfig())

        assert results
        assert results[0].status in {"supported", "partial"}
        assert results[0].citations[0].source_index == 0

    def test_structured_leniency_does_not_relax_prose_coverage(self) -> None:
        structured = """business_stars: 4.5
attributes.WiFi: free"""
        prose = "Alpha beta gamma delta epsilon zeta."
        answer = "Alpha something invented entirely."
        config = CitationConfig(min_answer_coverage=0.8)

        results = align_citations(answer, [structured, prose], config=config)

        assert results
        assert results[0].status == "unsupported"
        assert results[0].citations == []


def test_field_value_heuristic_requires_multiple_field_lines() -> None:
    fields = "business_stars: 4.5\nattributes.WiFi: free"
    title_and_body = (
        "Headline: Local cafe opens\nThe rest of this article is ordinary prose."
    )
    assert _looks_like_structured_source(fields) is True
    assert _looks_like_structured_source(title_and_body) is False
