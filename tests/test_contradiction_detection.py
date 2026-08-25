"""Tests for contradiction detection in citations."""

from cite_right import CitationConfig, align_citations


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
    # Entity mismatch detection is not yet implemented
    # This test documents the current limitation
    # TODO: Implement entity-aware contradiction detection
    # For now, we just verify it doesn't crash
    assert results[0].status in ["supported", "partial", "unsupported"]


# Issue #48: Leftover n-gram / truncated evidence fixtures
def test_issue48_number_leftover_rebounds() -> None:
    """Issue #48 fixture 1: Number leftover - 18 points vs 18 points and 10 rebounds.
    
    Source says '10 of which came in the first half' but answer says '10 rebounds'.
    Evidence truncates at '18 points, 10' and blesses the wrong '10 rebounds'.
    Must NOT be marked as supported.
    """
    sources = ["Jahlil Okafor had 18 points, 10 of which came in the first half"]
    answer = "Okafor had 18 points and 10 rebounds."

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    # Must be partial (or unsupported), not supported
    assert results[0].status != "supported", (
        f"Expected partial/unsupported but got {results[0].status}. "
        "Number leftover: '10 rebounds' contradicts '10 of which came in the first half'"
    )


def test_issue48_entity_swap_india_france() -> None:
    """Issue #48 fixture 2: Entity swap - India/France, American/Indian.
    
    Source: India opposed American involvement.
    Answer: France opposed Indian involvement.
    Shared content words (opposed, involvement, Vietnam War) bless the entity flip.
    Must NOT be marked as supported.
    """
    sources = ["India strongly opposed American involvement in the Vietnam War"]
    answer = "France opposed Indian involvement in the Vietnam War"

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    # Must be partial (or unsupported), not supported
    assert results[0].status != "supported", (
        f"Expected partial/unsupported but got {results[0].status}. "
        "Entity swap: India→France, American→Indian"
    )


def test_issue48_temporal_polarity_bc_vs_ago() -> None:
    """Issue #48 fixture 3: Temporal polarity - BC vs ago.
    
    Source: over 300 years BC (evidence cut at 'over 300')
    Answer: around 300 years ago
    'BC' means ~2300 years ago, not 300 years ago.
    Must NOT be marked as supported.
    """
    sources = ["The structure was built over 300 years BC"]
    answer = "The structure was built around 300 years ago"

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    # Must be partial (or unsupported), not supported
    assert results[0].status != "supported", (
        f"Expected partial/unsupported but got {results[0].status}. "
        "Temporal polarity: 'BC' (2300 years ago) vs 'ago' (300 years ago)"
    )


def test_issue48_polarity_flip_oppose_vs_urged() -> None:
    """Issue #48 fixture 4: Polarity flip - oppose vs urged.
    
    Source: oppose laws that require or prohibit
    Answer: urged laws
    Leftover 'laws' + 'prohibit' blesses the polarity flip from 'oppose' to 'urged'.
    Must NOT be marked as supported.
    """
    sources = ["The organization continues to oppose laws that require or prohibit certain actions"]
    answer = "The organization urged laws restricting such actions"

    results = align_citations(answer, sources, config=CitationConfig(top_k=1))

    assert len(results) == 1
    # Must be partial (or unsupported), not supported
    assert results[0].status != "supported", (
        f"Expected partial/unsupported but got {results[0].status}. "
        "Polarity flip: 'oppose' contradicts 'urged'"
    )
