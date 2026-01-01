"""Tests for multilingual content handling - English answers with non-English sources."""

import pytest

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig, CitationWeights

from .conftest import requires_pysbd


class TestGermanSourcesEnglishAnswer:
    """Test citation alignment with German sources and English answers."""

    @pytest.fixture
    def multilingual_config(self) -> CitationConfig:
        """Config for multilingual citation tests."""
        return CitationConfig(
            top_k=3,
            min_alignment_score=1,
            min_answer_coverage=0.3,
            supported_answer_coverage=0.6,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

    def test_exact_match_german_source_english_answer(
        self, multilingual_config: CitationConfig
    ) -> None:
        """Test that shared terms (proper nouns, numbers) enable cross-lingual matching."""
        # German source with proper nouns and numbers that appear in English answer
        # Use more shared vocabulary to ensure lexical matching works
        german_source = (
            "Berlin hat 3.6 Millionen Einwohner. "
            "Berlin ist die Hauptstadt von Deutschland."
        )

        # English answer using the same proper noun "Berlin" and number "3.6"
        english_answer = "Berlin has 3.6 million inhabitants. Berlin is the capital."

        sources = [SourceDocument(id="german_doc", text=german_source)]

        results = align_citations(english_answer, sources, config=multilingual_config)

        # Should find matches based on shared terms: "Berlin", "3.6"
        assert len(results) >= 1
        # At least one span should have citations due to shared vocabulary
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

    def test_german_umlauts_character_offsets(
        self, multilingual_config: CitationConfig
    ) -> None:
        """Test that character offsets are accurate with German umlauts (ä, ö, ü, ß)."""
        # German text with umlauts
        german_source = (
            "Die größte Stadt Österreichs ist Wien. "
            "München ist die drittgrößte Stadt Deutschlands. "
            "Zürich liegt in der Schweiz."
        )

        # Answer referencing proper nouns from the source
        english_answer = "Wien is the largest city. München is the third largest city."

        sources = [SourceDocument(id="cities", text=german_source)]
        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(english_answer, sources, config=config)

        # Verify character offsets are accurate for any found citations
        for span_result in results:
            for citation in span_result.citations:
                # Critical: verify the evidence text matches the source at given offsets
                extracted = german_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence, (
                    f"Offset mismatch: extracted '{extracted}' but expected "
                    f"'{citation.evidence}' at [{citation.char_start}:{citation.char_end}]"
                )

    def test_german_eszett_handling(self, multilingual_config: CitationConfig) -> None:
        """Test character offset accuracy with German ß (eszett/sharp s)."""
        german_source = "Die Straße ist 500 Meter lang. Der Fußball-Club gewann 3-0."

        # Answer with shared terms
        english_answer = "The street is 500 meters long."

        sources = [SourceDocument(id="street", text=german_source)]

        results = align_citations(english_answer, sources, config=multilingual_config)

        # Verify offsets are correct when source contains ß
        for span_result in results:
            for citation in span_result.citations:
                extracted = german_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence

    def test_mixed_german_english_source(
        self, multilingual_config: CitationConfig
    ) -> None:
        """Test with a source containing both German and English text."""
        mixed_source = (
            "Die Firma Apple Inc. wurde 1976 gegründet. "
            "Steve Jobs was the co-founder. "
            "Der Hauptsitz ist in Cupertino, California."
        )

        english_answer = "Apple Inc. was founded in 1976. Steve Jobs was the co-founder."

        sources = [SourceDocument(id="apple", text=mixed_source)]

        results = align_citations(english_answer, sources, config=multilingual_config)

        assert len(results) >= 1
        # Should find strong matches for the English portions
        has_citation = any(r.citations for r in results)
        assert has_citation, "Should find citations in mixed German/English source"

        # Verify all citations have correct offsets
        for span_result in results:
            for citation in span_result.citations:
                extracted = mixed_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence

    def test_german_numbers_and_dates(self) -> None:
        """Test citation matching on numbers and dates in German context."""
        # Use years and shared proper nouns to create sufficient lexical overlap
        german_source = (
            "Die Berlin Mauer fiel 1989 am 9. November. "
            "Die Wiedervereinigung von Berlin erfolgte 1990."
        )

        # Include same numbers and "Berlin" in English answer for lexical matching
        english_answer = (
            "The Berlin Wall fell in 1989. "
            "Berlin reunification happened in 1990."
        )

        sources = [SourceDocument(id="history", text=german_source)]

        # Use more permissive config for cross-language matching
        config = CitationConfig(
            top_k=3,
            min_alignment_score=1,
            min_answer_coverage=0.15,  # Lower threshold for cross-lingual
            supported_answer_coverage=0.4,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(english_answer, sources, config=config)

        # Numbers and proper nouns should provide matching anchors
        assert len(results) >= 1
        # Verify dates/numbers/proper nouns create citation links
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1, "Numbers and proper nouns should enable cross-lingual matching"

    def test_multiple_german_sources_correct_attribution(self) -> None:
        """Test that citations correctly identify the right German source."""
        source_berlin = SourceDocument(
            id="berlin",
            text="Berlin hat 3.6 Millionen Einwohner und ist die Hauptstadt.",
        )
        source_munich = SourceDocument(
            id="munich",
            text="München hat 1.5 Millionen Einwohner und liegt in Bayern.",
        )
        source_hamburg = SourceDocument(
            id="hamburg",
            text="Hamburg hat 1.9 Millionen Einwohner und ist eine Hafenstadt.",
        )

        english_answer = (
            "Berlin has 3.6 million inhabitants. "
            "Munich has 1.5 million inhabitants. "
            "Hamburg has 1.9 million inhabitants."
        )

        sources = [source_berlin, source_munich, source_hamburg]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.3,
            supported_answer_coverage=0.6,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(english_answer, sources, config=config)

        # Check source attribution is correct based on numbers
        for span_result in results:
            text = span_result.answer_span.text
            for citation in span_result.citations:
                if "3.6" in text:
                    assert citation.source_id == "berlin"
                elif "1.5" in text:
                    assert citation.source_id == "munich"
                elif "1.9" in text:
                    assert citation.source_id == "hamburg"

    def test_german_compound_words(self, multilingual_config: CitationConfig) -> None:
        """Test handling of German compound words which are common in technical text."""
        german_source = (
            "Die Geschwindigkeitsbegrenzung auf der Autobahn beträgt 130 km/h. "
            "Das Bundesverfassungsgericht entschied im Jahr 2023."
        )

        english_answer = (
            "The speed limit on the Autobahn is 130 km/h. "
            "The decision was made in 2023."
        )

        sources = [SourceDocument(id="traffic", text=german_source)]

        results = align_citations(english_answer, sources, config=multilingual_config)

        # Verify we can find matches despite German compound words
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offset accuracy
        for span_result in results:
            for citation in span_result.citations:
                extracted = german_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence


@requires_pysbd
class TestPySBDGermanSegmentation:
    """Test German sentence segmentation using pySBD."""

    def test_pysbd_german_segmenter_with_citations(self) -> None:
        """Test that pySBD German segmenter handles abbreviations correctly."""
        from cite_right.text.segmenter_pysbd import PySBDSegmenter

        german_segmenter = PySBDSegmenter(language="de")

        german_source = (
            "Dr. Müller ist ein bekannter Wissenschaftler. "
            "Er arbeitet an der Universität Berlin. "
            "Seine Forschung konzentriert sich auf KI."
        )

        # Test that abbreviation "Dr." doesn't cause incorrect splits
        segments = german_segmenter.segment(german_source)

        # Should have 3 sentences (Dr. should not cause extra split)
        assert len(segments) == 3
        # First sentence should include "Dr." as part of the sentence
        assert "Dr. Müller" in segments[0].text

    def test_german_pysbd_in_citation_pipeline(self) -> None:
        """Test full citation pipeline with German pySBD segmenter."""
        from cite_right.text.segmenter_pysbd import PySBDSegmenter

        german_segmenter = PySBDSegmenter(language="de")

        german_source = (
            "Die Firma wurde am 15. Januar 2020 gegründet. "
            "Der CEO heißt Dr. Schmidt. "
            "Der Umsatz betrug 50 Millionen Euro."
        )

        english_answer = (
            "The company was founded on January 15, 2020. "
            "The CEO is named Dr. Schmidt. "
            "Revenue was 50 million euros."
        )

        sources = [SourceDocument(id="company", text=german_source)]

        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(
            english_answer, sources, config=config, source_segmenter=german_segmenter
        )

        # Verify we get results and offsets are correct
        assert len(results) >= 1

        for span_result in results:
            for citation in span_result.citations:
                extracted = german_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence


class TestUnicodeNormalization:
    """Test Unicode normalization for cross-language citation matching."""

    def test_unicode_apostrophe_variants(self) -> None:
        """Test that different Unicode apostrophe forms are normalized."""
        # Using curly apostrophe (U+2019) in source
        source_curly = "The company's revenue grew by 20 percent."
        # Using straight apostrophe (U+0027) in answer
        answer_straight = "The company's revenue grew by 20 percent."

        sources = [SourceDocument(id="revenue", text=source_curly)]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.5,
            supported_answer_coverage=0.7,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(answer_straight, sources, config=config)

        # Should match despite different apostrophe characters
        assert len(results) == 1
        assert results[0].citations

    def test_german_quotation_marks(self) -> None:
        """Test handling of German quotation marks („ and ")."""
        german_source = '„Wir werden investieren", sagte der CEO. Der Betrag ist 100 Millionen.'

        english_answer = "The CEO said they will invest. The amount is 100 million."

        sources = [SourceDocument(id="quote", text=german_source)]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(english_answer, sources, config=config)

        # Verify character offsets work with special quotation marks
        for span_result in results:
            for citation in span_result.citations:
                extracted = german_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence


class TestCrossLingualFactExtraction:
    """Test fact extraction across language boundaries."""

    def test_scientific_terminology_shared(self) -> None:
        """Test that scientific/technical terms enable cross-lingual matching."""
        german_source = (
            "Die DNA-Sequenzierung wurde 1977 von Frederick Sanger entwickelt. "
            "Die Methode verwendet Didesoxynukleotide."
        )

        english_answer = (
            "DNA sequencing was developed by Frederick Sanger in 1977. "
            "The method uses dideoxynucleotides."
        )

        sources = [SourceDocument(id="science", text=german_source)]

        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(english_answer, sources, config=config)

        # Scientific terms like "DNA", "Frederick Sanger", "1977" should match
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

    def test_url_and_email_in_german_context(self) -> None:
        """Test that URLs and emails are handled correctly in German sources."""
        german_source = (
            "Weitere Informationen finden Sie unter https://example.com/info. "
            "Kontakt: info@example.com. Telefon: +49 30 12345678."
        )

        english_answer = (
            "More information at https://example.com/info. "
            "Contact: info@example.com."
        )

        sources = [SourceDocument(id="contact", text=german_source)]

        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.3,
            supported_answer_coverage=0.6,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(english_answer, sources, config=config)

        assert len(results) >= 1
        # URLs and emails should enable matching
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offsets
        for span_result in results:
            for citation in span_result.citations:
                extracted = german_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence
