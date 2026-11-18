"""Synthetic restatements and changed-detail controls adapted from #88."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Restatement:
    name: str
    passage: str
    positive: str
    twins: tuple[str, ...]
    aliases: tuple[tuple[str, ...], ...] = ()


PASSAGES = (
    "Use a spanner to loosen the clamp before removing the cartridge.",
    "Turn the dial until the amber lamp glows, then lift off the lid.",
    "Reattach the plate and push the lever until the catch engages.",
    "Polish the casing with a cloth.",
    "Set the finished housing on the stand.",
    "Disconnect the cable before servicing the chassis.",
    "In the Setup menu, choose advanced Privacy Options and then press Apply.",
    "To configure the filter, select Mode, then select Confirm to save the panel.",
)
ANSWERS = (
    "A wrench can slacken the bracket prior to extracting the cartridge.",
    "Lift off the lid when yellow illumination starts after rotating the knob.",
    "Put the cover back and push the handle till the latch clicks.",
    "Buff the casing using a soft rag and before leaving place the clean outer shell on the stand.",
    "Make certain that you unplug the cable ahead of work on the frame.",
    "Open the Setup menu, choose privacy options before hitting Save now.",
    "Select the display mode and choose Commit to save the panel.",
    "Never disconnect the cable before servicing the chassis.",
    "Remove the cartridge using a socket from the tool pouch beside the workstation.",
    "Watch the lamp while measuring vibration with a sensor clipped into the holder.",
    "Push a release pin using pliers when the indicator brightness drops.",
    "Store the cloth after wrapping it around both rubber feet on the platform.",
    "Here is a quick overview.",
    "Routine equipment maintenance.",
    "Wear insulated gloves whenever feasible.",
    "Always check whether the warranty includes shipping.",
    "Revision metadata follows.",
    "See also troubleshooting procedures.",
)


ALIASES = (
    ("spanner", "wrench"),
    ("loosen", "slacken"),
    ("clamp", "bracket"),
    ("removing", "extracting"),
    ("dial", "knob"),
    ("turn", "rotate"),
    ("catch", "latch"),
    ("polish", "buff"),
    ("set", "place"),
    ("housing", "shell"),
    ("disconnect", "unplug"),
    ("cord", "cable"),
    ("maintaining", "servicing"),
    ("press", "tap"),
    ("select", "choose"),
    ("tighten", "fasten"),
    ("drying", "airing"),
    ("asked", "requested"),
    ("transfer", "handoff"),
    ("inspection", "check"),
)

RESTATEMENTS = (
    Restatement(
        "tool-and-action",
        PASSAGES[0],
        "slacken bracket",
        ("tighten bracket", "slacken cover", "slacken brass bracket"),
    ),
    Restatement(
        "dial",
        PASSAGES[1],
        "rotate the knob",
        ("rotate the shaft",),
    ),
    Restatement(
        "engagement",
        PASSAGES[2],
        "the latch engages",
        ("the latch clicks",),
    ),
    Restatement(
        "polishing",
        PASSAGES[3],
        "buff casing",
        ("advice to buff casing", "buff soft casing"),
    ),
    Restatement(
        "placement",
        PASSAGES[4],
        "place finished shell",
        ("place finished sleeve",),
    ),
    Restatement(
        "polarity",
        PASSAGES[5],
        "unplug cable",
        ("never unplug cable",),
    ),
    Restatement(
        "polarity-full-context",
        PASSAGES[5],
        "unplug cord before maintaining chassis",
        ("never unplug cord before maintaining chassis",),
    ),
    Restatement(
        "apply-label",
        PASSAGES[6],
        "tap Apply",
        ("tap Save",),
    ),
    Restatement(
        "confirm-label",
        PASSAGES[7],
        "choose Confirm",
        ("choose Commit",),
    ),
    Restatement(
        "quantity",
        "Tighten the clamp after drying the latch for 30 days.",
        "fasten bracket after airing latch for 30 days",
        ("fasten bracket after airing latch for 14 days",),
    ),
    Restatement(
        "actor",
        "James asked Maria to approve the transfer after the inspection.",
        "James requested Maria approve the handoff after check",
        ("Owen requested Maria approve the handoff after check",),
    ),
)

REALISTIC_RESTATEMENTS = (
    Restatement(
        "cartridge-sentence",
        PASSAGES[0],
        "A wrench can slacken the bracket prior to extracting the cartridge.",
        (
            "A wrench can slacken the brass bracket prior to extracting the cartridge.",
            "A wrench can tighten the bracket prior to extracting the cartridge.",
            "A wrench can slacken the cover prior to extracting the cartridge.",
        ),
        (
            ("bracket", "clamp"),
            ("spanner", "wrench"),
            ("loosen", "slacken"),
            ("extracting", "removing"),
            ("before", "prior"),
        ),
    ),
    Restatement(
        "polishing-sentence",
        "Polish the casing with a cloth before storage.",
        "Buff the casing with a cloth before storage.",
        ("In general, buff the casing with a cloth before storage.",),
        (("polish", "buff"),),
    ),
    Restatement(
        "polarity-sentence",
        "Disconnect the cable before servicing the chassis or replacing the fuse.",
        "Unplug the cable before servicing the chassis or replacing the fuse.",
        ("Never unplug the cable before servicing the chassis or replacing the fuse.",),
        (("disconnect", "unplug"),),
    ),
    Restatement(
        "quantity-sentence",
        "Tighten the clamp after drying the latch for 30 days.",
        "Fasten the clamp after drying the latch for 30 days.",
        ("Fasten the clamp after drying the latch for 14 days.",),
        (("tighten", "fasten"),),
    ),
    Restatement(
        "apply-label-sentence",
        PASSAGES[6],
        "In the Setup menu, choose advanced Privacy Options and then tap Apply.",
        ("In the Setup menu, choose advanced Privacy Options and then tap Save.",),
        (("press", "tap"),),
    ),
)

LONG_POLISH_TWIN = "buff casing with a soft cloth"

BASE_POSITIVE_STATUSES = {
    "tool-and-action": "unsupported",
    "dial": "unsupported",
    "engagement": "partial",
    "polishing": "partial",
    "placement": "unsupported",
    "polarity": "partial",
    "polarity-full-context": "unsupported",
    "apply-label": "partial",
    "confirm-label": "partial",
    "quantity": "partial",
    "actor": "partial",
    "cartridge-sentence": "unsupported",
    "polishing-sentence": "partial",
    "polarity-sentence": "partial",
    "quantity-sentence": "partial",
    "apply-label-sentence": "supported",
}


# Closed PR #88 controls, replayed at the deployment threshold.
DEPLOYMENT_GUARDS = (
    (
        "quantity",
        "Tighten the clamp after drying the latch for 30 days.",
        ("After drying the latch for 14 days, tighten the clamp.",),
    ),
    (
        "polarity",
        "Tighten the clamp after drying the latch.",
        ("After drying the latch, never tighten the clamp.",),
    ),
    (
        "role",
        "James asked Maria to approve the transfer after the inspection.",
        ("After the inspection, Maria asked James to approve the transfer.",),
    ),
    (
        "object",
        "Tighten the cover after drying the latch and checking the pressure.",
        ("After drying the latch and checking the pressure, tighten the clamp.",),
    ),
    (
        "action",
        "Loosen the clamp after carefully drying the latch.",
        ("After carefully drying the latch, tighten the clamp.",),
    ),
    (
        "wrong-source",
        "Tighten the clamp after drying the latch.",
        (
            "Tighten the clamp before drying the latch.",
            "After drying the latch, tighten the clamp.",
        ),
    ),
    (
        "partial-claim",
        "Tighten the brass clamp after drying the latch.",
        ("After drying the latch, tighten the clamp.",),
    ),
    (
        "generic-advice",
        "Advice to tighten the clamp after drying the latch.",
        ("After drying the latch, tighten the clamp.",),
    ),
)
