"""Synthetic manual text and claims shared by citation regressions."""

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
EXPECTED_PASSAGES = (0, 1, 2, (3, 4), 5, 6, 7, 5, 0, 1, 2, 3)
