import pytest

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
