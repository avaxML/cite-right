import importlib.util

import pytest


def test_spacy_segmenter_import_guard_message_when_missing() -> None:
    if importlib.util.find_spec("spacy") is not None:
        pytest.skip("spaCy is installed")

    from cite_right.text.segmenter_spacy import SpacySegmenter

    with pytest.raises(RuntimeError, match="spaCy is not installed"):
        SpacySegmenter()


def test_spacy_answer_segmenter_import_guard_message_when_missing() -> None:
    if importlib.util.find_spec("spacy") is not None:
        pytest.skip("spaCy is installed")

    from cite_right.text.answer_segmenter_spacy import SpacyAnswerSegmenter

    with pytest.raises(RuntimeError, match="spaCy is not installed"):
        SpacyAnswerSegmenter()


def test_sentence_transformer_embedder_import_guard_message_when_missing() -> None:
    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers is installed")

    from cite_right.models.sbert_embedder import SentenceTransformerEmbedder

    with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
        SentenceTransformerEmbedder()
