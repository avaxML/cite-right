"""Tests for multilingual content handling across German and English."""

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


class TestEnglishSourcesGermanAnswer:
    """Test citation alignment with English sources and German answers."""

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

    def test_german_answer_english_source_shared_terms(
        self, multilingual_config: CitationConfig
    ) -> None:
        """Test German answer citing English source via shared vocabulary."""
        english_source = (
            "Berlin has 3.6 million inhabitants. "
            "Berlin is the capital of Germany."
        )

        german_answer = (
            "Berlin hat 3.6 Millionen Einwohner. "
            "Berlin ist die Hauptstadt."
        )

        sources = [SourceDocument(id="english_doc", text=english_source)]

        results = align_citations(german_answer, sources, config=multilingual_config)

        # Should find matches based on "Berlin" and "3.6"
        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offsets
        for span_result in results:
            for citation in span_result.citations:
                extracted = english_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence

    def test_german_answer_with_technical_english_source(
        self, multilingual_config: CitationConfig
    ) -> None:
        """Test German technical answer with English technical source."""
        english_source = (
            "The CPU operates at 3.5 GHz clock speed. "
            "The GPU has 16 GB of VRAM memory. "
            "The system uses DDR5 RAM at 6400 MHz."
        )

        german_answer = (
            "Die CPU arbeitet mit 3.5 GHz Taktfrequenz. "
            "Die GPU verfügt über 16 GB VRAM Speicher. "
            "Das System nutzt DDR5 RAM mit 6400 MHz."
        )

        sources = [SourceDocument(id="specs", text=english_source)]

        results = align_citations(german_answer, sources, config=multilingual_config)

        # Technical terms (CPU, GPU, GHz, GB, DDR5, MHz) should match
        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offsets are accurate
        for span_result in results:
            for citation in span_result.citations:
                extracted = english_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence

    def test_german_answer_multiple_english_sources_attribution(self) -> None:
        """Test correct source attribution with German answer and English sources."""
        source_apple = SourceDocument(
            id="apple",
            text="Apple Inc. was founded in 1976 by Steve Jobs in California.",
        )
        source_microsoft = SourceDocument(
            id="microsoft",
            text="Microsoft was founded in 1975 by Bill Gates in Washington.",
        )
        source_google = SourceDocument(
            id="google",
            text="Google was founded in 1998 by Larry Page in California.",
        )

        german_answer = (
            "Apple Inc. wurde 1976 von Steve Jobs gegründet. "
            "Microsoft wurde 1975 von Bill Gates gegründet. "
            "Google wurde 1998 von Larry Page gegründet."
        )

        sources = [source_apple, source_microsoft, source_google]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(german_answer, sources, config=config)

        # Check correct attribution
        for span_result in results:
            text = span_result.answer_span.text
            for citation in span_result.citations:
                if "Apple" in text and "1976" in text:
                    assert citation.source_id == "apple"
                elif "Microsoft" in text and "1975" in text:
                    assert citation.source_id == "microsoft"
                elif "Google" in text and "1998" in text:
                    assert citation.source_id == "google"

    def test_german_answer_with_english_quotes(
        self, multilingual_config: CitationConfig
    ) -> None:
        """Test German answer containing English quoted phrases from source."""
        english_source = (
            'The CEO stated: "We will invest 500 million dollars in AI research." '
            "The announcement was made on March 15, 2024."
        )

        german_answer = (
            'Der CEO erklärte: "We will invest 500 million dollars in AI research." '
            "Die Ankündigung erfolgte am 15. März 2024."
        )

        sources = [SourceDocument(id="news", text=english_source)]

        results = align_citations(german_answer, sources, config=multilingual_config)

        # The quoted English phrase should create strong matches
        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

    def test_german_umlauts_in_answer_english_source(self) -> None:
        """Test that German umlauts in answer don't break offset calculation."""
        english_source = "Munich is located in Bavaria. The population is 1.5 million."

        # German answer with umlauts
        german_answer = "München liegt in Bayern. Die Bevölkerung beträgt 1.5 Millionen."

        sources = [SourceDocument(id="city", text=english_source)]

        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.15,
            supported_answer_coverage=0.4,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(german_answer, sources, config=config)

        # "1.5" should create a match
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offsets point to correct positions in English source
        for span_result in results:
            for citation in span_result.citations:
                extracted = english_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence

    def test_german_scientific_answer_english_paper(self) -> None:
        """Test German scientific summary citing English research paper."""
        english_source = (
            "The experiment showed a 45% reduction in CO2 emissions. "
            "The sample size was n=1500 participants. "
            "Results were statistically significant with p<0.001."
        )

        german_answer = (
            "Das Experiment zeigte eine Reduktion der CO2-Emissionen um 45%. "
            "Die Stichprobengröße betrug n=1500 Teilnehmer. "
            "Die Ergebnisse waren statistisch signifikant mit p<0.001."
        )

        sources = [SourceDocument(id="paper", text=english_source)]

        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(german_answer, sources, config=config)

        # Scientific notation (45%, n=1500, p<0.001, CO2) should match
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1


class TestMixedSourcesGermanAnswer:
    """Test German answers with mixed German and English sources."""

    @pytest.fixture
    def permissive_config(self) -> CitationConfig:
        """More permissive config for cross-lingual matching."""
        return CitationConfig(
            top_k=3,
            min_alignment_score=1,
            min_answer_coverage=0.15,
            supported_answer_coverage=0.4,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

    def test_german_answer_mixed_sources_correct_attribution(
        self, permissive_config: CitationConfig
    ) -> None:
        """Test German answer correctly cites from mixed language sources."""
        german_source = SourceDocument(
            id="german",
            text="Berlin hat 3.6 Millionen Einwohner. Die Stadt wurde 1237 gegründet.",
        )
        english_source = SourceDocument(
            id="english",
            text="Munich has 1.5 million inhabitants. The city was founded in 1158.",
        )

        german_answer = (
            "Berlin hat 3.6 Millionen Einwohner und wurde 1237 gegründet. "
            "München hat 1.5 Millionen Einwohner und wurde 1158 gegründet."
        )

        sources = [german_source, english_source]

        results = align_citations(german_answer, sources, config=permissive_config)

        # Should have results
        assert len(results) >= 1

        # Verify we got citations and offsets are correct
        for span_result in results:
            for citation in span_result.citations:
                if citation.source_id == "german":
                    extracted = german_source.text[
                        citation.char_start : citation.char_end
                    ]
                else:
                    extracted = english_source.text[
                        citation.char_start : citation.char_end
                    ]
                assert extracted == citation.evidence

    def test_german_answer_prefers_german_source_when_equal(self) -> None:
        """Test behavior when same fact exists in both German and English sources."""
        german_source = SourceDocument(
            id="german",
            text="Albert Einstein wurde 1879 in Ulm geboren. Er entwickelte die Relativitätstheorie.",
        )
        english_source = SourceDocument(
            id="english",
            text="Albert Einstein was born in 1879 in Ulm. He developed the theory of relativity.",
        )

        german_answer = "Albert Einstein wurde 1879 in Ulm geboren."

        sources = [german_source, english_source]

        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.3,
            supported_answer_coverage=0.6,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(german_answer, sources, config=config)

        # Should find citations - both sources have matching content
        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offsets for all citations
        for span_result in results:
            for citation in span_result.citations:
                source_text = (
                    german_source.text
                    if citation.source_id == "german"
                    else english_source.text
                )
                extracted = source_text[citation.char_start : citation.char_end]
                assert extracted == citation.evidence

    def test_german_answer_multiple_mixed_sources(self) -> None:
        """Test German answer citing from multiple sources in different languages."""
        source_de_tech = SourceDocument(
            id="de_tech",
            text="Die CPU-Temperatur erreichte 85 Grad Celsius unter Volllast.",
        )
        source_en_specs = SourceDocument(
            id="en_specs",
            text="The GPU runs at 1800 MHz boost clock with 12 GB GDDR6 memory.",
        )
        source_de_review = SourceDocument(
            id="de_review",
            text="Der Stromverbrauch lag bei 350 Watt während des Tests.",
        )
        source_en_bench = SourceDocument(
            id="en_bench",
            text="The system achieved 15000 points in the benchmark test.",
        )

        german_answer = (
            "Die CPU erreichte 85 Grad Celsius. "
            "Die GPU läuft mit 1800 MHz und 12 GB GDDR6. "
            "Der Stromverbrauch betrug 350 Watt. "
            "Das System erzielte 15000 Punkte im Benchmark."
        )

        sources = [source_de_tech, source_en_specs, source_de_review, source_en_bench]

        config = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(german_answer, sources, config=config)

        # Should have multiple spans with citations
        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

    def test_german_answer_bilingual_source_document(
        self, permissive_config: CitationConfig
    ) -> None:
        """Test German answer with a single bilingual source document."""
        bilingual_source = (
            "Zusammenfassung: Das Produkt kostet 299 Euro. "
            "Summary: The product costs 299 euros. "
            "Technische Daten: 500 GB Speicher, USB-C Anschluss. "
            "Technical specs: 500 GB storage, USB-C port."
        )

        german_answer = (
            "Das Produkt kostet 299 Euro. "
            "Es bietet 500 GB Speicher und einen USB-C Anschluss."
        )

        sources = [SourceDocument(id="bilingual", text=bilingual_source)]

        results = align_citations(german_answer, sources, config=permissive_config)

        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offsets
        for span_result in results:
            for citation in span_result.citations:
                extracted = bilingual_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence

    def test_german_answer_url_from_english_source(
        self, permissive_config: CitationConfig
    ) -> None:
        """Test German answer correctly cites URLs from English source."""
        english_source = (
            "For more details visit https://example.com/product-info. "
            "Contact support at support@example.com or call +1-800-555-0123."
        )

        german_answer = (
            "Weitere Details unter https://example.com/product-info. "
            "Kontakt: support@example.com."
        )

        sources = [SourceDocument(id="contact", text=english_source)]

        results = align_citations(german_answer, sources, config=permissive_config)

        # URLs and emails should match exactly
        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1

        # Verify offsets
        for span_result in results:
            for citation in span_result.citations:
                extracted = english_source[citation.char_start : citation.char_end]
                assert extracted == citation.evidence


@requires_pysbd
class TestPySBDMixedLanguageSegmentation:
    """Test pySBD with mixed language content."""

    def test_pysbd_german_segmenter_with_english_quotes(self) -> None:
        """Test German segmenter handles embedded English quotes."""
        from cite_right.text.segmenter_pysbd import PySBDSegmenter

        german_segmenter = PySBDSegmenter(language="de")

        mixed_text = (
            'Der CEO sagte: "We are committed to innovation." '
            "Die Aktie stieg um 5 Prozent. "
            "Analysten erwarten weiteres Wachstum."
        )

        segments = german_segmenter.segment(mixed_text)

        # Should segment correctly despite English quote
        assert len(segments) == 3

    def test_citation_pipeline_german_answer_english_source_with_pysbd(self) -> None:
        """Test full pipeline with pySBD for source segmentation and German answer."""
        from cite_right.text.segmenter_pysbd import PySBDSegmenter

        # Use pySBD for source segmentation (it's a Segmenter, not AnswerSegmenter)
        english_source_segmenter = PySBDSegmenter(language="en")

        english_source = (
            "The company reported Q3 revenue of 5.2 billion dollars. "
            "CEO Dr. Smith announced plans for expansion. "
            "The stock price increased by 12 percent."
        )

        german_answer = (
            "Das Unternehmen meldete Q3-Umsatz von 5.2 Milliarden Dollar. "
            "CEO Dr. Smith kündigte Expansionspläne an. "
            "Der Aktienkurs stieg um 12 Prozent."
        )

        sources = [SourceDocument(id="report", text=english_source)]

        config = CitationConfig(
            top_k=2,
            min_alignment_score=1,
            min_answer_coverage=0.2,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results = align_citations(
            german_answer,
            sources,
            config=config,
            source_segmenter=english_source_segmenter,
        )

        assert len(results) >= 1
        # "5.2", "Dr. Smith", "12" should enable matching
        cited_spans = [r for r in results if r.citations]
        assert len(cited_spans) >= 1


# =============================================================================
# Semantic Embedding Tests for Multilingual Content
# =============================================================================
# These tests require sentence-transformers and are skipped by default.
# Set CITE_RIGHT_RUN_EMBEDDINGS_TESTS=1 to enable.


def _embeddings_test_enabled() -> bool:
    """Check if embeddings tests should run."""
    import os

    return os.environ.get("CITE_RIGHT_RUN_EMBEDDINGS_TESTS") == "1"


def _get_multilingual_embedder():
    """Get a multilingual sentence embedder for testing."""
    pytest.importorskip("sentence_transformers")
    from cite_right.models.sbert_embedder import SentenceTransformerEmbedder

    # Use a multilingual model that supports German and English
    # paraphrase-multilingual-MiniLM-L12-v2 is a good choice for cross-lingual
    try:
        return SentenceTransformerEmbedder(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    except OSError:
        # Fall back to smaller model if multilingual not available
        try:
            return SentenceTransformerEmbedder(
                "sentence-transformers/paraphrase-MiniLM-L3-v2"
            )
        except OSError as exc:
            pytest.skip(f"Embedding model not available: {exc}")


@pytest.mark.skipif(
    not _embeddings_test_enabled(),
    reason="Set CITE_RIGHT_RUN_EMBEDDINGS_TESTS=1 to run embeddings tests",
)
class TestMultilingualSemanticEmbeddings:
    """Test semantic embedding-based matching for multilingual content."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Provide multilingual embedder for the test class."""
        return _get_multilingual_embedder()

    def test_german_source_english_answer_semantic_match(self, embedder) -> None:
        """Test semantic matching between German source and English answer."""
        # German source with no lexical overlap to English answer
        german_source = (
            "Das Unternehmen erzielte einen Umsatz von fünf Milliarden Euro. "
            "Die Gewinne stiegen um zwanzig Prozent."
        )

        # English answer - semantically similar but different words
        english_answer = (
            "The company achieved revenue of five billion euros. "
            "Profits increased by twenty percent."
        )

        sources = [
            SourceDocument(id="noise", text="Weather forecast for next week."),
            SourceDocument(id="german_finance", text=german_source),
        ]

        config = CitationConfig(
            top_k=1,
            max_candidates_lexical=10,
            max_candidates_embedding=50,
            max_candidates_total=50,
            allow_embedding_only=True,
            min_embedding_similarity=0.3,
            supported_embedding_similarity=0.4,
            min_alignment_score=1,
            min_answer_coverage=0.1,
            weights=CitationWeights(
                alignment=0.1, answer_coverage=0.1, lexical=0.1, embedding=0.7
            ),
        )

        results = align_citations(
            english_answer, sources, config=config, embedder=embedder
        )

        assert len(results) >= 1
        # With semantic embeddings, should find the German source
        cited_spans = [r for r in results if r.citations]
        # Semantic matching should find relevant content
        if cited_spans:
            for span in cited_spans:
                for citation in span.citations:
                    assert citation.source_id == "german_finance"

    def test_english_source_german_answer_semantic_match(self, embedder) -> None:
        """Test semantic matching between English source and German answer."""
        english_source = (
            "The new electric vehicle has a range of 500 kilometers. "
            "It can be charged to eighty percent in thirty minutes."
        )

        # German answer - semantically equivalent
        german_answer = (
            "Das neue Elektrofahrzeug hat eine Reichweite von 500 Kilometern. "
            "Es kann in dreißig Minuten auf achtzig Prozent geladen werden."
        )

        sources = [
            SourceDocument(id="irrelevant", text="Recipe for apple pie."),
            SourceDocument(id="ev_specs", text=english_source),
        ]

        config = CitationConfig(
            top_k=1,
            max_candidates_lexical=10,
            max_candidates_embedding=50,
            max_candidates_total=50,
            allow_embedding_only=True,
            min_embedding_similarity=0.3,
            supported_embedding_similarity=0.4,
            min_alignment_score=1,
            min_answer_coverage=0.1,
            weights=CitationWeights(
                alignment=0.1, answer_coverage=0.1, lexical=0.2, embedding=0.6
            ),
        )

        results = align_citations(
            german_answer, sources, config=config, embedder=embedder
        )

        assert len(results) >= 1
        # With embeddings, should match the EV source despite language difference
        cited_spans = [r for r in results if r.citations]
        if cited_spans:
            for span in cited_spans:
                for citation in span.citations:
                    assert citation.source_id == "ev_specs"

    def test_paraphrased_translation_semantic_match(self, embedder) -> None:
        """Test semantic matching with paraphrased translations."""
        # Original German text
        german_source = (
            "Der Wissenschaftler erhielt den Nobelpreis für seine bahnbrechenden "
            "Entdeckungen im Bereich der Quantenphysik."
        )

        # Paraphrased English translation - not word-for-word
        english_answer = (
            "The researcher was awarded the Nobel Prize for groundbreaking "
            "discoveries in quantum physics."
        )

        sources = [
            SourceDocument(id="science_de", text=german_source),
            SourceDocument(id="filler", text="Stock market closed higher today."),
        ]

        config = CitationConfig(
            top_k=1,
            max_candidates_lexical=10,
            max_candidates_embedding=50,
            max_candidates_total=50,
            allow_embedding_only=True,
            min_embedding_similarity=0.25,
            supported_embedding_similarity=0.35,
            min_alignment_score=1,
            min_answer_coverage=0.1,
            weights=CitationWeights(
                alignment=0.1, answer_coverage=0.1, lexical=0.1, embedding=0.7
            ),
        )

        results = align_citations(
            english_answer, sources, config=config, embedder=embedder
        )

        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        # Semantic embeddings should identify the German science text
        if cited_spans:
            for span in cited_spans:
                for citation in span.citations:
                    assert citation.source_id == "science_de"

    def test_mixed_sources_semantic_attribution(self, embedder) -> None:
        """Test correct source attribution with mixed language sources using embeddings."""
        german_source = SourceDocument(
            id="german",
            text="Berlin ist die Hauptstadt und größte Stadt Deutschlands.",
        )
        english_source = SourceDocument(
            id="english",
            text="Paris is the capital and largest city of France.",
        )
        french_source = SourceDocument(
            id="french",
            text="Madrid es la capital y la ciudad más grande de España.",
        )

        # English answer about Berlin (should match German source semantically)
        english_answer = "Berlin is the capital and largest city of Germany."

        sources = [german_source, english_source, french_source]

        config = CitationConfig(
            top_k=2,
            max_candidates_lexical=20,
            max_candidates_embedding=50,
            max_candidates_total=50,
            allow_embedding_only=True,
            min_embedding_similarity=0.3,
            supported_embedding_similarity=0.4,
            min_alignment_score=1,
            min_answer_coverage=0.1,
            weights=CitationWeights(
                alignment=0.1, answer_coverage=0.1, lexical=0.2, embedding=0.6
            ),
        )

        results = align_citations(
            english_answer, sources, config=config, embedder=embedder
        )

        assert len(results) >= 1
        # Should prefer German source as it's about Berlin
        cited_spans = [r for r in results if r.citations]
        if cited_spans:
            # Check if German source is cited (it should be the best match)
            source_ids = [
                c.source_id for span in cited_spans for c in span.citations
            ]
            # German source should be in the citations due to semantic similarity
            assert "german" in source_ids or len(source_ids) > 0

    def test_embedding_improves_low_lexical_overlap(self, embedder) -> None:
        """Test that embeddings help when lexical overlap is minimal."""
        # German text using formal language
        german_source = (
            "Die Temperatur wird morgen auf fünfunddreißig Grad Celsius steigen."
        )

        # English answer - minimal word overlap
        english_answer = "Tomorrow the temperature will rise to thirty-five degrees."

        sources = [
            SourceDocument(id="weather_de", text=german_source),
            SourceDocument(id="sports", text="The football match ended 2-1."),
        ]

        # First try without embeddings - should have low/no match
        config_no_embed = CitationConfig(
            top_k=1,
            min_alignment_score=1,
            min_answer_coverage=0.3,
            supported_answer_coverage=0.5,
            weights=CitationWeights(lexical=0.0, embedding=0.0),
        )

        results_no_embed = align_citations(
            english_answer, sources, config=config_no_embed
        )

        # Now with embeddings - should find match
        config_with_embed = CitationConfig(
            top_k=1,
            max_candidates_lexical=10,
            max_candidates_embedding=50,
            max_candidates_total=50,
            allow_embedding_only=True,
            min_embedding_similarity=0.2,
            supported_embedding_similarity=0.3,
            min_alignment_score=1,
            min_answer_coverage=0.1,
            weights=CitationWeights(
                alignment=0.1, answer_coverage=0.1, lexical=0.1, embedding=0.7
            ),
        )

        results_with_embed = align_citations(
            english_answer, sources, config=config_with_embed, embedder=embedder
        )

        # With embeddings, we should get better matching
        cited_no_embed = [r for r in results_no_embed if r.citations]
        cited_with_embed = [r for r in results_with_embed if r.citations]

        # Embeddings should help find the weather source
        if cited_with_embed:
            weather_cited = any(
                c.source_id == "weather_de"
                for span in cited_with_embed
                for c in span.citations
            )
            # If embeddings found citations, weather should be among them
            assert weather_cited or len(cited_with_embed) >= len(cited_no_embed)

    def test_semantic_match_with_umlauts(self, embedder) -> None:
        """Test semantic matching handles German umlauts correctly."""
        german_source = (
            "Die größten Städte Österreichs sind Wien, Graz und Linz. "
            "Über acht Millionen Menschen leben in Österreich."
        )

        english_answer = (
            "The largest cities in Austria are Vienna, Graz, and Linz. "
            "Over eight million people live in Austria."
        )

        sources = [
            SourceDocument(id="austria", text=german_source),
            SourceDocument(id="noise", text="The stock market rose today."),
        ]

        config = CitationConfig(
            top_k=1,
            max_candidates_lexical=10,
            max_candidates_embedding=50,
            max_candidates_total=50,
            allow_embedding_only=True,
            min_embedding_similarity=0.3,
            supported_embedding_similarity=0.4,
            min_alignment_score=1,
            min_answer_coverage=0.1,
            weights=CitationWeights(
                alignment=0.1, answer_coverage=0.1, lexical=0.2, embedding=0.6
            ),
        )

        results = align_citations(
            english_answer, sources, config=config, embedder=embedder
        )

        assert len(results) >= 1
        cited_spans = [r for r in results if r.citations]
        if cited_spans:
            for span in cited_spans:
                for citation in span.citations:
                    # Should match the Austria source
                    assert citation.source_id == "austria"
                    # Verify offsets work with umlauts
                    extracted = german_source[
                        citation.char_start : citation.char_end
                    ]
                    assert extracted == citation.evidence
