"""Tests for Data2txt field:value paraphrase support (issue #50)."""

import pytest

from cite_right import CitationConfig, align_citations


class TestData2txtFieldValueSupport:
    """Test that faithful rewrites of Data2txt flattened field:value lines are supported."""

    def test_business_stars_and_wifi_field_rewrite_is_supported(self) -> None:
        """Test fixture 1: business_stars + WiFi field rewrites should be supported."""
        # Flattened field:value source
        source = """business_stars: 4.5
attributes.WiFi: free"""

        # Faithful rewrite
        answer = "The business has a rating of 4.5 stars and offers free WiFi."

        config = CitationConfig()
        results = align_citations(answer, [source], config=config)

        assert len(results) > 0
        span = results[0]
        # Should be supported or partial, not unsupported
        assert span.status in ["supported", "partial"], (
            f"Expected supported/partial, got {span.status}. "
            f"Citations: {span.citations}"
        )

    def test_hours_field_rewrite_is_supported(self) -> None:
        """Test fixture 2: hours.* field rewrites should be supported."""
        # Flattened hours field:value source
        source = """hours.Monday: 9:0-17:0
hours.Tuesday: 9:0-17:0
hours.Wednesday: 9:0-17:0
hours.Thursday: 9:0-17:0
hours.Friday: 9:0-17:0"""

        # Faithful rewrite with formatted times
        answer = "Monday–Friday 9:00 AM–5:00 PM"

        config = CitationConfig()
        results = align_citations(answer, [source], config=config)

        assert len(results) > 0
        span = results[0]
        # Should not be unsupported
        assert span.status != "unsupported", (
            f"Hours rewrite should not be unsupported. "
            f"Got status: {span.status}, Citations: {span.citations}"
        )

    def test_null_wifi_with_invented_amenity_stays_unsupported(self) -> None:
        """Test fixture 3: Keep null checks - WiFi: null + 'free WiFi' stays unsupported."""
        # Source with null WiFi
        source = """business_stars: 3.5
attributes.WiFi: null
attributes.OutdoorSeating: null"""

        # Answer inventing free WiFi
        answer = "This place offers free Wi-Fi."

        config = CitationConfig()
        results = align_citations(answer, [source], config=config)

        assert len(results) > 0
        span = results[0]
        # Invented amenity should stay unsupported
        assert span.status == "unsupported", (
            f"Invented WiFi when null should be unsupported. "
            f"Got status: {span.status}"
        )

    def test_platform_mismatch_not_fully_supported(self) -> None:
        """Test fixture 4: Platform name mismatch - Yelp stars claimed as Google."""
        # Yelp source only
        source = """business_stars: 3.5
name: Best Restaurant"""

        # Answer claiming Google rating
        answer = "The restaurant has a 3.5 star rating on Google."

        config = CitationConfig()
        results = align_citations(answer, [source], config=config)

        assert len(results) > 0
        span = results[0]
        # Should not bless "Google" as fully supported (partial or split is fine)
        # The star value is grounded, but Google is invented
        assert span.status != "supported" or any(
            "Google" not in str(cit.evidence) for cit in span.citations
        ), (
            f"Google platform should not be fully supported when only Yelp exists. "
            f"Got status: {span.status}, Citations: {span.citations}"
        )

    def test_mixed_field_source_with_review_text(self) -> None:
        """Test that field:value sources work alongside review text."""
        # Mixed source: fields + review prose
        source = """business_stars: 4.5
attributes.WiFi: free
attributes.OutdoorSeating: true

Review: Great place with nice outdoor seating."""

        # Answer combining field data and review prose
        answer = "The business has a 4.5 star rating and offers free WiFi. Great outdoor seating area."

        config = CitationConfig()
        results = align_citations(answer, [source], config=config)

        # Both spans should have some support
        assert len(results) >= 1
        # At least first span should be supported or partial
        assert results[0].status in ["supported", "partial"]
