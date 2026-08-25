"""Tests for contradiction detection in citations."""

from cite_right import CitationConfig, align_citations
from cite_right.contradiction import check_contradiction


def test_negation_mismatch_marked_unsupported() -> None:
    """Test that negated claims vs affirmative sources are not marked as supported."""
    sources = ["The vaccine is safe and effective."]
    answer = "The vaccine is not safe."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    # Should not be fully supported due to negation mismatch
    # The system should detect lexical overlap but mark as partial (not supported)
    assert results[0].status == "partial"
    # Should still have citations showing the conflicting evidence
    assert len(results[0].citations) > 0


def test_affirmative_match_is_supported() -> None:
    """Test that matching affirmative statements are supported."""
    sources = ["The vaccine is safe and effective."]
    answer = "The vaccine is safe."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    assert results[0].status == "supported"


def test_number_mismatch_not_supported() -> None:
    """Test that claims with different numbers are not marked as supported."""
    sources = ["Revenue grew by 15% in Q4."]
    answer = "Revenue grew by 20% in Q4."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    # Should not be supported due to number mismatch
    # Should be partial if there's lexical overlap
    assert results[0].status == "partial"


def test_matching_numbers_are_supported() -> None:
    """Test that claims with matching numbers are supported."""
    sources = ["Revenue grew by 15% in Q4 2024."]
    answer = "Revenue grew by 15% in Q4."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    assert results[0].status == "supported"


def test_entity_mismatch_not_supported() -> None:
    """Test that claims with different entities are not marked as supported."""
    sources = ["Apple released a new iPhone model."]
    answer = "Google released a new iPhone model."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    assert results[0].status == "partial"
    assert results[0].citations


# Issue #48: Leftover n-gram / truncated evidence fixtures
def test_issue48_number_leftover_rebounds() -> None:
    """Issue #48 fixture 1: leftover '10' blesses '10 rebounds'.

    Source says '10 of which came in the first half' but answer says '10 rebounds'.
    Smith-Waterman truncates evidence at '18 points, 10'. Status must be partial.
    """
    sources = ["Jahlil Okafor had 18 points, 10 of which came in the first half"]
    answer = "Okafor had 18 points and 10 rebounds."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    evidence = results[0].citations[0].evidence
    # Alignment leftover: shared '10' without the true slot ('of which came...')
    assert "10" in evidence
    assert "rebounds" not in evidence.lower()
    assert results[0].status == "partial"


def test_issue48_entity_swap_india_france() -> None:
    """Issue #48 fixture 2: entity swap - India/France, American/Indian.

    Shared content words (opposed, involvement, Vietnam War) bless the entity flip.
    """
    sources = ["India strongly opposed American involvement in the Vietnam War"]
    answer = "France opposed Indian involvement in the Vietnam War"

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    assert results[0].status == "partial"
    assert results[0].citations


def test_issue48_temporal_polarity_bc_vs_ago() -> None:
    """Issue #48 fixture 3: temporal polarity hidden by truncated evidence.

    Source: over 300 years BC (evidence historically cut before 'BC').
    Answer: around 300 years ago.
    Contradiction check must see leftover passage tokens, not only the SW span.
    """
    sources = ["The structure was built over 300 years BC"]
    answer = "The structure was built around 300 years ago"

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    evidence = results[0].citations[0].evidence
    # Truncated SW span does not include the contradicting era marker.
    assert "300" in evidence
    assert "bc" not in evidence.lower()
    assert results[0].status == "partial"


def test_issue48_polarity_flip_oppose_vs_urged() -> None:
    """Issue #48 fixture 4: polarity flip - oppose vs urged.

    Leftover 'laws' + 'prohibit' must not bless the flip from 'oppose' to 'urged'.
    """
    sources = [
        "The organization continues to oppose laws that require or prohibit certain actions"
    ]
    answer = "The organization urged laws restricting such actions"

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    assert results[0].status == "partial"
    assert results[0].citations


def test_issue48_extractive_near_copy_stays_supported() -> None:
    """Extractive near-copies (how-to / zipper ledes) must stay supported."""
    source = "Click on the Start menu and scroll down to Web browser."
    answer = "Click on the Start menu and scroll down to Web browser."

    results = align_citations(answer, [source], config=CitationConfig(top_k=1))

    assert len(results) == 1
    assert results[0].status == "supported"


def test_issue48_extractive_subset_stays_supported() -> None:
    """A faithful subset of source text is still extractive support."""
    sources = ["Jahlil Okafor had 18 points, 10 of which came in the first half"]
    answer = "Okafor had 18 points."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    assert results[0].status == "supported"


def test_check_contradiction_uses_passage_not_truncated_span() -> None:
    """Direct check: leftover passage tokens (BC) contradict 'ago'."""
    truncated = "The structure was built over 300 years"
    passage = "The structure was built over 300 years BC"
    answer = "The structure was built around 300 years ago"

    # Truncated SW evidence hides BC; the candidate passage does not.
    assert check_contradiction(answer, passage)
    assert not check_contradiction("The structure was built over 300 years BC", passage)
    # Number-slot leftover still flags truncated evidence via 'ago'.
    assert check_contradiction(answer, truncated)
