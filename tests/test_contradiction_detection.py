"""Tests for contradiction detection in citations."""

import pytest

from cite_right import CitationConfig, align_citations
from cite_right.contradiction import check_contradiction, has_entity_swap


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


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        (
            "Open Settings, choose Privacy, then press Save.",
            "In the Settings menu select the privacy tab and click Apply to keep your changes.",
        ),
        ("Export Options", "Download Options"),
        ("Report Layout.", "Archive Layout."),
        ("Summary: Review the report.", "Overview: Review the report."),
        ("The toolbar shows Export.", "The toolbar shows Download."),
        (
            "Plan • “Is Loading” appears while the plan details are displayed.",
            "Plan • “Is Refreshing” appears while the plan details are displayed.",
        ),
        ("- First find the source.", "- Next find the source."),
        ("– First find the source.", "– Next find the source."),
        ("Summary only; review the report.", "Overview only; review the report."),
        (
            "With the “EXPORT” toolbar visible, continue editing the document.",
            "With the “Download” toolbar visible, continue editing the document.",
        ),
        ("1. First find the source.", "1. Next find the source."),
        (
            "Use the Search field to find a document.",
            "Use the Filter field to find a document.",
        ),
        (
            "Maria approved the london budget.",
            "maria approved the London budget.",
        ),
        ("Step 2: Select Export", "Step 2: Choose Download"),
        ("Menu: Open Details.", "Menu: View Summary."),
        ("• Review Queue", "• Inspect Inbox"),
        ("Use 5 EUR for the allowance.", "Use 5 KG for the allowance."),
        (
            "Click Settings, then select advanced.",
            "Click settings, then select Advanced.",
        ),
    ],
)
def test_entity_swap_ignores_interface_and_layout(answer: str, source: str) -> None:
    assert not has_entity_swap(answer, source)


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("Revenue rose in Paris.", "Revenue rose in London."),
        ("Maria Lopez approved the budget.", "James Chen approved the budget."),
        ("The board picked Maria.", "The board picked James."),
        ("The board chose Maria.", "The board chose James."),
        ("The reviewer was Maria.", "The reviewer was James."),
        (
            "The owner has been Maria since the start of the project.",
            "The owner is James and has been since the start of the project.",
        ),
        (
            "Reviewer: Maria approved the budget.",
            "Reviewer: James approved the budget.",
        ),
        ("Reviewer: Maria", "Reviewer: James"),
        ("The product is Photoshop.", "The product is Lightroom."),
        ('The product is "Photoshop".', 'The product is "Lightroom".'),
        ("Upgrade to the Pro plan.", "Upgrade to the Team plan."),
        ("Select the Pro Plan.", "Select the Team Plan."),
        ('Upgrade to the "Pro" plan.', 'Upgrade to the "Team" plan.'),
    ],
)
def test_entity_swap_keeps_role_changes(answer: str, source: str) -> None:
    assert has_entity_swap(answer, source)


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        (
            "Also, sealed samples may never be stored in cupboards.",
            "Sealed samples may never be stored in cupboards.",
        ),
        (
            "Luneston to Pineford: Up to 600 kg in 5 working days.",
            "Luneston to Pineford: up to 600 kg for 385.00 EUR in 5 working days.",
        ),
        (
            "Plan • “Is Loading”: review the Pro plan details and payment options when the preview opens.",
            "Plan • “Is Refreshing”: review the Pro plan details and payment options when the preview opens.",
        ),
    ],
)
def test_issue69_restatements_are_supported(answer: str, source: str) -> None:
    [span] = align_citations(answer, [source], config=CitationConfig())
    assert span.status == "supported"


@pytest.mark.parametrize(
    ("answer", "passage"),
    [
        (
            "The board awarded Maria the maintenance contract for the third quarter.",
            "The board awarded James the maintenance contract for the third quarter.",
        ),
        ("ALICE SIGNED THE CONTRACT.", "BOB SIGNED THE CONTRACT."),
    ],
    ids=["recipient-before-determiner", "uppercase-statement"],
)
def test_entity_swap_keeps_actor_roles_in_complete_statements(
    answer: str, passage: str
) -> None:
    assert has_entity_swap(answer, passage)
    [span] = align_citations(answer, [passage], config=CitationConfig())
    assert span.status == "partial"


def test_entity_swap_ignores_uppercase_layout_heading() -> None:
    assert not has_entity_swap("EXPORT DEVICE SETTINGS.", "IMPORT DEVICE SETTINGS.")
