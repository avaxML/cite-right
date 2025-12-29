"""Shared pytest fixtures and configuration for cite-right tests."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

from cite_right.core.citation_config import CitationConfig, CitationWeights

if TYPE_CHECKING:
    from types import ModuleType

    import spacy


# =============================================================================
# Pytest Markers Registration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "rust: requires Rust extension")
    config.addinivalue_line("markers", "spacy: requires spaCy and en_core_web_sm model")
    config.addinivalue_line("markers", "embeddings: requires sentence-transformers")
    config.addinivalue_line("markers", "tiktoken: requires tiktoken")
    config.addinivalue_line("markers", "huggingface: requires transformers/tokenizers")
    config.addinivalue_line("markers", "pysbd: requires pysbd")
    config.addinivalue_line("markers", "slow: marks tests as slow")


# =============================================================================
# Rust Extension Fixtures
# =============================================================================


def _rust_available() -> bool:
    """Check if Rust extension is available."""
    try:
        from cite_right import _core  # noqa: F401

        return True
    except ImportError:
        return False


def _rust_has_blocks_details() -> bool:
    """Check if Rust extension has align_pair_blocks_details."""
    if not _rust_available():
        return False
    from cite_right import _core

    return hasattr(_core, "align_pair_blocks_details")


@pytest.fixture
def rust_core() -> ModuleType:
    """Provide Rust extension module, skipping if not available."""
    try:
        from cite_right import _core

        return _core
    except ImportError:
        pytest.skip("Rust extension not built")


@pytest.fixture
def rust_core_with_blocks() -> ModuleType:
    """Provide Rust extension with align_pair_blocks_details, skipping if not available."""
    try:
        from cite_right import _core
    except ImportError:
        pytest.skip("Rust extension not built")

    if not hasattr(_core, "align_pair_blocks_details"):
        pytest.skip(
            "Rust extension is missing align_pair_blocks_details (rebuild required)"
        )
    return _core


# Skip decorators for Rust tests
requires_rust = pytest.mark.skipif(
    not _rust_available(),
    reason="Rust extension not built",
)

requires_rust_blocks = pytest.mark.skipif(
    not _rust_has_blocks_details(),
    reason="Rust extension missing align_pair_blocks_details",
)


# =============================================================================
# SpaCy Fixtures
# =============================================================================


def _spacy_available() -> bool:
    """Check if spaCy is available."""
    return importlib.util.find_spec("spacy") is not None


def _spacy_model_available() -> bool:
    """Check if spaCy and en_core_web_sm model are available."""
    if not _spacy_available():
        return False
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except OSError:
        return False


@pytest.fixture
def spacy_nlp() -> spacy.Language:
    """Provide spaCy nlp object with en_core_web_sm, skipping if not available."""
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not installed")


requires_spacy = pytest.mark.skipif(
    not _spacy_available(),
    reason="spaCy is not installed",
)

requires_spacy_model = pytest.mark.skipif(
    not _spacy_model_available(),
    reason="spaCy model en_core_web_sm not installed",
)


# =============================================================================
# Embeddings Fixtures
# =============================================================================


def _embeddings_available() -> bool:
    """Check if sentence-transformers is available."""
    return importlib.util.find_spec("sentence_transformers") is not None


requires_embeddings = pytest.mark.skipif(
    not _embeddings_available(),
    reason="sentence-transformers is not installed",
)


# =============================================================================
# Tiktoken Fixtures
# =============================================================================


def _tiktoken_available() -> bool:
    """Check if tiktoken is available."""
    return importlib.util.find_spec("tiktoken") is not None


requires_tiktoken = pytest.mark.skipif(
    not _tiktoken_available(),
    reason="tiktoken is not installed",
)


# =============================================================================
# HuggingFace Fixtures
# =============================================================================


def _huggingface_available() -> bool:
    """Check if transformers and tokenizers are available."""
    return (
        importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("tokenizers") is not None
    )


requires_huggingface = pytest.mark.skipif(
    not _huggingface_available(),
    reason="transformers/tokenizers not installed",
)


# =============================================================================
# PySBD Fixtures
# =============================================================================


def _pysbd_available() -> bool:
    """Check if pysbd is available."""
    return importlib.util.find_spec("pysbd") is not None


requires_pysbd = pytest.mark.skipif(
    not _pysbd_available(),
    reason="pysbd is not installed",
)


# =============================================================================
# Citation Config Fixtures
# =============================================================================


@pytest.fixture
def basic_citation_config() -> CitationConfig:
    """Provide a basic citation config for testing."""
    return CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.5,
        supported_answer_coverage=0.9,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
    )


@pytest.fixture
def multi_span_config() -> CitationConfig:
    """Provide a config enabling multi-span evidence for deterministic tests."""
    return CitationConfig(
        top_k=1,
        min_alignment_score=1,
        min_answer_coverage=0.8,
        supported_answer_coverage=0.8,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
        multi_span_evidence=True,
        multi_span_merge_gap_chars=0,
        multi_span_max_spans=5,
    )


@pytest.fixture
def paper_scenario_config() -> CitationConfig:
    """Provide a config tuned for deterministic, paper-style scenarios."""
    return CitationConfig(
        top_k=1,
        min_alignment_score=6,
        min_answer_coverage=0.6,
        supported_answer_coverage=0.75,
        weights=CitationWeights(lexical=0.0, embedding=0.0),
        window_size_sentences=2,
        window_stride_sentences=1,
        multi_span_evidence=True,
        multi_span_merge_gap_chars=0,
    )


# =============================================================================
# DSPy Paper Test Data
# =============================================================================


DSPY_MODEL = (
    "DSPy abstracts LM pipelines as text transformation graphs, where LMs are invoked "
    "through declarative modules."
)

DSPY_COMPILER = (
    "Compiling relies on a teleprompter, which is an optimizer for DSPy programs. "
    "The compiler first finds all unique Predict modules in a program."
)

DSPY_ASSERTIONS = (
    "We propose LM Assertions, expressed as boolean conditions, and integrate them into "
    "DSPy. We propose two types of LM Assertions: hard Assertions and soft Suggestions."
)

IRRELEVANT_SOURCES = [
    "A teleprompter helps speakers read a script on stage.",
    "Graphs can represent pipelines in data engineering.",
    "Assertions in unit tests are boolean checks.",
    "Soft suggestions can improve writing quality.",
    "Predict functions map inputs to outputs in statistics.",
    "Declarative modules describe configuration rather than execution.",
    "Text transformation can refer to editing operations in documents.",
    "LM is short for language model in NLP.",
    "Pipelines often include several modules and stages.",
    "Programs can be optimized to run faster.",
    "Unique values are deduplicated in data processing.",
    "Weather report: storms are likely this weekend.",
]
