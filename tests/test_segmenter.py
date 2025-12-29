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
