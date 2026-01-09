"""Static content for the perplexity-style citation demo.

The paragraphs below are excerpt-style summaries of DeepSeek's latest mHC
paper. They provide enough lexical overlap for the citation aligner to locate
evidence spans while still reading like real prose.
"""

from __future__ import annotations

from cite_right import SourceDocument

QUESTION = (
    "How does DeepSeek describe the mHC benchmark and what lifted its mHC scores"
    " for the DeepSeekMath models?"
)

ANSWER = (
    "The mHC benchmark is a 10,000 human-curated pool of olympiad-style math"
    " hard cases that require multi-step reasoning, and every problem is cleaned"
    " through dual annotation and rule-based normalization to keep statements"
    " unambiguous. "
    "On exact-match grading, DeepSeekMath-7B reaches 75.8% accuracy while the"
    " 32B variant scores 82.1%, and those gains arrive after a verifier-guided"
    " rejection pipeline where symbolic calculators catch arithmetic slips and a"
    " lightweight verifier filters inconsistent chain-of-thought traces."
)

SOURCES = [
    SourceDocument(
        id="mhc_overview",
        text=(
            "mHC contains 10,000 human-curated math hard cases sampled from"
            " olympiad-style algebra, geometry, combinatorics, and number theory."
            " Each item requires multi-step reasoning and was cleaned through"
            " dual-annotation and rule-based normalization to remove ambiguous"
            " statements."
        ),
    ),
    SourceDocument(
        id="mhc_results",
        text=(
            "DeepSeekMath-7B and DeepSeekMath-32B are evaluated on mHC with"
            " exact-match grading. After reinforcement-style refinement, the 7B"
            " model reaches 75.8% accuracy and the 32B variant scores 82.1%,"
            " surpassing previously open baselines by 8-15 points."
        ),
    ),
    SourceDocument(
        id="mhc_pipeline",
        text=(
            "The paper uses a verifier-guided rejection pipeline: candidate"
            " chain-of-thought traces are generated, symbolic calculators catch"
            " arithmetic slips, and a lightweight verifier filters inconsistent"
            " steps before distillation. These process rewards stabilize long"
            " reasoning traces and reduce hallucinated algebra."
        ),
    ),
]
