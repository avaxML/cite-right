"""Tests for passage window generation."""

from cite_right.text.passage import generate_passages
from cite_right.text.segmenter_simple import SimpleSegmenter


def test_generate_passages_exposes_text_from_source_offsets() -> None:
    text = "First sentence. Second sentence. Third sentence."

    passages = generate_passages(
        text,
        segmenter=SimpleSegmenter(),
        window_size_sentences=2,
        window_stride_sentences=1,
    )

    assert [passage.text for passage in passages] == [
        "First sentence. Second sentence.",
        "Second sentence. Third sentence.",
    ]
    assert [
        text[passage.doc_char_start : passage.doc_char_end] for passage in passages
    ] == [passage.text for passage in passages]
