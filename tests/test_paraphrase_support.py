"""Tests for grounded how-to/news paraphrases (issue #49).

These fixtures follow the RAGTruth repros. Sequential Smith-Waterman match
count on a one-sentence window was marking them unsupported even though the
same content words are in the source. Status may be partial.
"""

from cite_right import SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig
from cite_right.text.content_coverage import content_token_coverage, stopword_token_ids
from cite_right.text.tokenizer import SimpleTokenizer

from .conftest import requires_rust

_BNP_ANSWER = (
    "The normal B-type Natriuretic Peptide (BNP) refers to a low amount of the "
    "BNP hormone found in the blood, produced by the heart as a measure of how "
    "well it is working."
)

_BNP_SOURCES = [
    SourceDocument(
        id="bnp_p1",
        text=(
            "A brain natriuretic peptide (BNP) test measures the amount of the "
            "BNP hormone in the blood. BNP values tend to increase with age and "
            "are higher in women than men."
        ),
    ),
    SourceDocument(
        id="bnp_p2",
        text=(
            "The B-type Natriuretic Peptide (BNP) Test is a blood test for heart "
            "failure. If your B-type natriuretic peptide level is high, you "
            "probably have heart failure."
        ),
    ),
    SourceDocument(
        id="bnp_p3",
        text=(
            "BNP is made by your heart and shows how well your heart is working. "
            "Normally, only a low amount of BNP is found in your blood."
        ),
    ),
]

_FARAGE_ANSWER = (
    "Farage has declined the challenge, saying he does not want to engage in a "
    "physical altercation and prefers to focus on the upcoming elections."
)

_FARAGE_SOURCE = SourceDocument(
    id="farage_cnn",
    text=(
        "Farage, who is on the campaign trail ahead of Britain's general "
        "elections, said he did not intend to cross swords with the prince. "
        "I'm not intending to accept the offer, a spokesman quoted him as saying."
    ),
)

_ZIPPER_SOURCE = SourceDocument(
    id="zipper_p2",
    text=(
        "Make a zipper sandwich. Place 1 piece of lining fabric face up, then "
        "the zipper and then the outer fabric right side facing down. Line up "
        "the three edges. With a zipper foot sew across the top between the "
        "edge of your zipper sandwich and the zipper teeth."
    ),
)


def _lexical_config() -> CitationConfig:
    return CitationConfig(top_k=3)


def _align(answer: str, sources: list[SourceDocument], *, backend: str = "auto"):
    return align_citations(
        answer,
        sources,
        config=_lexical_config(),
        backend=backend,  # type: ignore[arg-type]
    )


def test_definition_paraphrase_bnp_not_unsupported() -> None:
    """14760-style definition restatement across three RAG passages."""
    results = _align(_BNP_ANSWER, _BNP_SOURCES)
    assert len(results) == 1
    assert results[0].status in {"supported", "partial"}
    assert results[0].citations, "Smith-Waterman must still localize evidence"


def test_negation_paraphrase_farage_not_unsupported() -> None:
    """466-style refusal paraphrase: declined vs did not intend / not intending."""
    results = _align(_FARAGE_ANSWER, [_FARAGE_SOURCE])
    assert len(results) == 1
    assert results[0].status in {"supported", "partial"}
    assert results[0].citations


def test_extractive_zipper_near_copy_stays_supported() -> None:
    """12840-style extractive how-to copy stays supported."""
    answer = (
        "Place 1 piece of lining fabric face up, then the zipper and then the "
        "outer fabric right side facing down. Line up the three edges."
    )
    results = _align(answer, [_ZIPPER_SOURCE])
    assert results
    assert all(result.status == "supported" for result in results)


def test_conversational_wrapper_stays_unsupported() -> None:
    """Wrappers with no source content stay unsupported."""
    results = _align("Sure!", _BNP_SOURCES)
    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []


def test_heres_the_summary_wrapper_stays_unsupported() -> None:
    results = _align("Here's the summary.", _BNP_SOURCES)
    assert len(results) == 1
    assert results[0].status == "unsupported"


def test_contradiction_stays_partial_not_unsupported() -> None:
    """Issue #49: contradiction remains partial, not unsupported."""
    sources = [SourceDocument(id="vax", text="The vaccine is safe and effective.")]
    results = _align("The vaccine is not safe.", sources)
    assert len(results) == 1
    assert results[0].status == "partial"
    assert results[0].citations


@requires_rust
def test_paraphrase_python_and_rust_agree() -> None:
    python = _align(_BNP_ANSWER, _BNP_SOURCES, backend="python")
    rust = _align(_BNP_ANSWER, _BNP_SOURCES, backend="rust")
    assert python[0].status == rust[0].status
    assert python[0].status in {"supported", "partial"}


def test_content_token_coverage_ignores_stopwords() -> None:
    tokenizer = SimpleTokenizer()
    answer = tokenizer.tokenize("Farage declined the challenge")
    passage = tokenizer.tokenize("Farage said he did not intend to cross swords")
    vocab = tokenizer._vocab
    coverage = content_token_coverage(
        answer.token_ids,
        passage.token_ids,
        stopword_token_ids(vocab),
    )
    assert coverage > 0.0
    # "the" must not be enough to count as coverage by itself
    wrapper = tokenizer.tokenize("Here's the summary")
    empty = content_token_coverage(
        wrapper.token_ids,
        passage.token_ids,
        stopword_token_ids(vocab),
    )
    assert empty == 0.0
