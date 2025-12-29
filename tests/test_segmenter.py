import pytest

from cite_right.text.segmenter_pysbd import PySBDSegmenter
from cite_right.text.segmenter_simple import SimpleSegmenter
from cite_right.text.segmenter_spacy import SpacySegmenter


def test_simple_segmenter_offsets() -> None:
    text = "One. Two!"
    segmenter = SimpleSegmenter()
    segments = segmenter.segment(text)

    assert [segment.text for segment in segments] == ["One.", "Two!"]
    assert segments[0].doc_char_start == 0
    assert segments[0].doc_char_end == 4
    assert segments[1].doc_char_start == 5
    assert segments[1].doc_char_end == 9


def test_spacy_segmenter_clauses() -> None:
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model not installed")

    segmenter = SpacySegmenter()
    text = "Apple revenue is up and stocks are down."
    segments = segmenter.segment(text)
    assert len(segments) == 2
    assert segments[0].text == "Apple revenue is up"
    assert segments[1].text == "stocks are down."

    list_text = "Apples, oranges, and pears are tasty."
    list_segments = segmenter.segment(list_text)
    assert len(list_segments) == 1
    assert list_segments[0].text == list_text


def test_pysbd_segmenter_basic() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "Hello world. How are you?"
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "Hello world."
    assert segments[1].text == "How are you?"


def test_pysbd_segmenter_offsets() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "One. Two!"
    segments = segmenter.segment(text)

    assert [segment.text for segment in segments] == ["One.", "Two!"]
    assert segments[0].doc_char_start == 0
    assert segments[0].doc_char_end == 4
    assert segments[1].doc_char_start == 5
    assert segments[1].doc_char_end == 9


def test_pysbd_segmenter_abbreviations() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    # pySBD should handle abbreviations correctly
    text = "Dr. Smith went to the store. He bought milk."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "Dr. Smith went to the store."
    assert segments[1].text == "He bought milk."


def test_pysbd_segmenter_multiple_punctuation() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "Really?! Yes, really."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "Really?!"
    assert segments[1].text == "Yes, really."


def test_pysbd_segmenter_whitespace_handling() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "  First sentence.   Second sentence.  "
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "First sentence."
    assert segments[1].text == "Second sentence."
    # Verify offsets point to trimmed content
    assert text[segments[0].doc_char_start : segments[0].doc_char_end] == "First sentence."
    assert text[segments[1].doc_char_start : segments[1].doc_char_end] == "Second sentence."


def test_pysbd_segmenter_empty_text() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    segments = segmenter.segment("")

    assert len(segments) == 0


def test_pysbd_segmenter_single_sentence() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "Just one sentence here."
    segments = segmenter.segment(text)

    assert len(segments) == 1
    assert segments[0].text == "Just one sentence here."
    assert segments[0].doc_char_start == 0
    assert segments[0].doc_char_end == len(text)


def test_pysbd_segmenter_no_punctuation() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "No punctuation here"
    segments = segmenter.segment(text)

    assert len(segments) == 1
    assert segments[0].text == text


def test_pysbd_segmenter_newlines() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "First sentence.\nSecond sentence."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert segments[0].text == "First sentence."
    assert segments[1].text == "Second sentence."


def test_pysbd_segmenter_urls() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    # pySBD should not split on dots in URLs
    text = "Visit https://example.com for more info. Thank you."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert "https://example.com" in segments[0].text


def test_pysbd_segmenter_numbers() -> None:
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    # pySBD should not split on dots in numbers
    text = "The price is $19.99 per item. That's affordable."
    segments = segmenter.segment(text)

    assert len(segments) == 2
    assert "$19.99" in segments[0].text


def test_pysbd_segmenter_conforms_to_protocol() -> None:
    pytest.importorskip("pysbd")

    from cite_right.core.interfaces import Segmenter

    segmenter = PySBDSegmenter()
    assert isinstance(segmenter, Segmenter)


def test_pysbd_segmenter_offset_integrity() -> None:
    """Verify that all segment offsets correctly map back to original text."""
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "The quick brown fox. Jumps over the lazy dog. And runs away!"
    segments = segmenter.segment(text)

    for segment in segments:
        # Verify the text at the given offsets matches the segment text
        extracted = text[segment.doc_char_start:segment.doc_char_end]
        assert extracted == segment.text, (
            f"Offset mismatch: expected '{segment.text}' but got '{extracted}'"
        )


def test_pysbd_segmenter_consecutive_offsets() -> None:
    """Verify segments don't overlap and cover the text properly."""
    pytest.importorskip("pysbd")

    segmenter = PySBDSegmenter()
    text = "First. Second. Third."
    segments = segmenter.segment(text)

    assert len(segments) == 3

    # Verify no overlapping offsets
    for i in range(len(segments) - 1):
        assert segments[i].doc_char_end <= segments[i + 1].doc_char_start, (
            f"Segments {i} and {i+1} overlap"
        )
