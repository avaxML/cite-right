"""Tests for text segmenters (Simple, SpaCy, PySBD)."""

from cite_right.text.segmenter_pysbd import PySBDSegmenter
from cite_right.text.segmenter_simple import SimpleSegmenter
from cite_right.text.segmenter_spacy import SpacySegmenter

from .conftest import requires_pysbd, requires_spacy_model

# =============================================================================
# SimpleSegmenter Tests
# =============================================================================


def test_simple_segmenter_offsets() -> None:
    """Verify SimpleSegmenter produces correct character offsets."""
    text = "One. Two!"
    segmenter = SimpleSegmenter()
    segments = segmenter.segment(text)

    assert [segment.text for segment in segments] == ["One.", "Two!"]
    assert segments[0].doc_char_start == 0
    assert segments[0].doc_char_end == 4
    assert segments[1].doc_char_start == 5
    assert segments[1].doc_char_end == 9


# =============================================================================
# SpaCy Segmenter Tests
# =============================================================================


@requires_spacy_model
def test_spacy_segmenter_clauses() -> None:
    """Verify SpaCy segmenter splits clauses correctly."""
    segmenter = SpacySegmenter()
    text = "Apple revenue is up and stocks are down."
    segments = segmenter.segment(text)
    assert len(segments) == 2, "Expected two clauses"
    assert segments[0].text == "Apple revenue is up"
    assert segments[1].text == "stocks are down."

    # Should not split comma-separated lists
    list_text = "Apples, oranges, and pears are tasty."
    list_segments = segmenter.segment(list_text)
    assert len(list_segments) == 1, "Should not split list into multiple segments"
    assert list_segments[0].text == list_text


# =============================================================================
# PySBD Segmenter Tests
# =============================================================================


@requires_pysbd
def test_pysbd_segmenter_basic() -> None:
    """Verify PySBD performs basic sentence segmentation."""
    segmenter = PySBDSegmenter()
    text = "Hello world. How are you?"
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "Hello world."
    assert segments[1].text == "How are you?"


@requires_pysbd
def test_pysbd_segmenter_offsets() -> None:
    """Verify PySBD produces correct character offsets."""
    segmenter = PySBDSegmenter()
    text = "One. Two!"
    segments = segmenter.segment(text)

    assert [segment.text for segment in segments] == ["One.", "Two!"]
    assert segments[0].doc_char_start == 0
    assert segments[0].doc_char_end == 4
    assert segments[1].doc_char_start == 5
    assert segments[1].doc_char_end == 9


@requires_pysbd
def test_pysbd_segmenter_abbreviations() -> None:
    """Verify PySBD handles abbreviations correctly."""
    segmenter = PySBDSegmenter()
    text = "Dr. Smith went to the store. He bought milk."
    segments = segmenter.segment(text)

    assert len(segments) == 2, "Should not split on 'Dr.'"
    assert segments[0].text == "Dr. Smith went to the store."
    assert segments[1].text == "He bought milk."


@requires_pysbd
def test_pysbd_segmenter_multiple_punctuation() -> None:
    """Verify PySBD handles multiple punctuation marks."""
    segmenter = PySBDSegmenter()
    text = "Really?! Yes, really."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "Really?!"
    assert segments[1].text == "Yes, really."


@requires_pysbd
def test_pysbd_segmenter_whitespace_handling() -> None:
    """Verify PySBD handles leading/trailing whitespace correctly."""
    segmenter = PySBDSegmenter()
    text = "  First sentence.   Second sentence.  "
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "First sentence."
    assert segments[1].text == "Second sentence."
    # Verify offsets point to trimmed content
    assert (
        text[segments[0].doc_char_start : segments[0].doc_char_end] == "First sentence."
    ), "First segment offset mismatch"
    assert (
        text[segments[1].doc_char_start : segments[1].doc_char_end]
        == "Second sentence."
    ), "Second segment offset mismatch"


@requires_pysbd
def test_pysbd_segmenter_empty_text() -> None:
    """Verify PySBD returns empty list for empty input."""
    segmenter = PySBDSegmenter()
    segments = segmenter.segment("")

    assert len(segments) == 0


@requires_pysbd
def test_pysbd_segmenter_single_sentence() -> None:
    """Verify PySBD handles single sentence correctly."""
    segmenter = PySBDSegmenter()
    text = "Just one sentence here."
    segments = segmenter.segment(text)

    assert len(segments) == 1
    assert segments[0].text == "Just one sentence here."
    assert segments[0].doc_char_start == 0
    assert segments[0].doc_char_end == len(text)


@requires_pysbd
def test_pysbd_segmenter_no_punctuation() -> None:
    """Verify PySBD handles text without sentence-ending punctuation."""
    segmenter = PySBDSegmenter()
    text = "No punctuation here"
    segments = segmenter.segment(text)

    assert len(segments) == 1
    assert segments[0].text == text


@requires_pysbd
def test_pysbd_segmenter_newlines() -> None:
    """Verify PySBD handles newlines correctly."""
    segmenter = PySBDSegmenter()
    text = "First sentence.\nSecond sentence."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "First sentence."
    assert segments[1].text == "Second sentence."


@requires_pysbd
def test_pysbd_segmenter_urls() -> None:
    """Verify PySBD doesn't split on dots in URLs."""
    segmenter = PySBDSegmenter()
    text = "Visit https://example.com for more info. Thank you."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert "https://example.com" in segments[0].text


@requires_pysbd
def test_pysbd_segmenter_numbers() -> None:
    """Verify PySBD doesn't split on dots in numbers."""
    segmenter = PySBDSegmenter()
    text = "The price is $19.99 per item. That's affordable."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert "$19.99" in segments[0].text


@requires_pysbd
def test_pysbd_segmenter_conforms_to_protocol() -> None:
    """Verify PySBDSegmenter conforms to Segmenter protocol."""
    from cite_right.core.interfaces import Segmenter

    segmenter = PySBDSegmenter()
    assert isinstance(segmenter, Segmenter)


@requires_pysbd
def test_pysbd_segmenter_offset_integrity() -> None:
    """Verify that all segment offsets correctly map back to original text."""
    segmenter = PySBDSegmenter()
    text = "The quick brown fox. Jumps over the lazy dog. And runs away!"
    segments = segmenter.segment(text)

    for segment in segments:
        extracted = text[segment.doc_char_start : segment.doc_char_end]
        assert extracted == segment.text, (
            f"Offset mismatch: expected '{segment.text}' but got '{extracted}'"
        )


@requires_pysbd
def test_pysbd_segmenter_consecutive_offsets() -> None:
    """Verify segments don't overlap and cover the text properly."""
    segmenter = PySBDSegmenter()
    text = "First. Second. Third."
    segments = segmenter.segment(text)

    assert len(segments) == 3

    # Verify no overlapping offsets
    for i in range(len(segments) - 1):
        assert segments[i].doc_char_end <= segments[i + 1].doc_char_start, (
            f"Segments {i} and {i + 1} overlap"
        )
