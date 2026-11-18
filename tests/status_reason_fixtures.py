"""Synthetic restatements and controls for citation diagnostics."""

from cite_right import SourceDocument

CERAMIC_RESTATEMENT = "Flush the ceramic filter with fresh water prior to the shift."

SOURCES = (
    SourceDocument(id="safety", text="Check whether the cartridge latch is clean."),
    SourceDocument(
        id="clamp",
        text="Use a spanner to loosen the clamp before removing the cartridge.",
    ),
    SourceDocument(
        id="lid",
        text="Turn the dial until the amber lamp glows, then lift off the lid.",
    ),
    SourceDocument(
        id="plate",
        text="Reattach the plate and push the lever until the catch engages.",
    ),
    SourceDocument(
        id="cable", text="Disconnect the cable before servicing the chassis."
    ),
    SourceDocument(
        id="privacy",
        text="In the Setup menu, choose advanced Privacy Options and then press Apply.",
    ),
    SourceDocument(
        id="mode",
        text="To configure the filter, select Mode, then select Confirm to save the panel.",
    ),
    SourceDocument(
        id="ceramic",
        text="Before the shift begins, rinse the ceramic filter with clean water.",
    ),
)

RESTATEMENTS = (
    (
        "Loosen the clamp with a wrench before taking out the cartridge.",
        "clamp",
        "contradiction",
    ),
    (
        "Rotate the dial until the amber lamp shines, then remove the lid.",
        "lid",
        "contradiction",
    ),
    (
        "Push the lever to engage the catch after reinstalling the plate.",
        "plate",
        "contradiction",
    ),
    (
        "Unplug the cable before doing maintenance on the chassis.",
        "cable",
        "contradiction",
    ),
    (
        "From the Setup menu, select Privacy Options before pressing Apply.",
        "privacy",
        "below_coverage",
    ),
    ("To set up the filter, choose Mode and press Confirm.", "mode", "below_coverage"),
    (CERAMIC_RESTATEMENT, "ceramic", "below_coverage"),
    (
        "Before extracting the cartridge, slacken the clamp using a wrench.",
        "clamp",
        "no_candidate",
    ),
)

GENERIC_SPANS = (
    "Here is a quick overview.",
    "Cable maintenance",
    "Revision metadata follows.",
    "Always check whether the warranty includes shipping.",
)

QUANTITY_PASSAGE = "The ceramic filter needs 20 liters of water."
QUANTITY_TWINS = (
    (QUANTITY_PASSAGE, "supported", ()),
    (
        "The ceramic filter needs 18 liters of water.",
        "contradiction",
        ("has_number_mismatch",),
    ),
    (
        "The ceramic filter never needs 20 liters of water.",
        "contradiction",
        ("has_negation_mismatch", "has_number_context_mismatch"),
    ),
)
