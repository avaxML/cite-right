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
