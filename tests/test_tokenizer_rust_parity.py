"""Rust prepare must normalize source tokens exactly like the Python tokenizer."""

from __future__ import annotations

import itertools

import pytest

from cite_right import CitationConfig, SimpleTokenizer, SourceDocument, align_citations
from cite_right.core.prepared_corpus import PreparedCitationCorpus
from cite_right.text.tokenizer import TokenizerConfig

from .conftest import requires_rust

CORPUS = [
    "Closing balance: 2,410.12 EUR.",
    "Fee 1,204.50 €",
    "$1,200 and £3,000",
    "34% vs 34 ％",
    "２,４１０ ＄",
    "２，４１０．１２ ＄",
    "٢,٤١٠",
    "￡5 ﹩6 ﹪",
    "don’t re‑enter – 455.00",
]

FLAG_COMBOS = [
    {
        "normalize_numbers": numbers,
        "normalize_percent": percent,
        "normalize_currency": currency,
    }
    for numbers, percent, currency in itertools.product((True, False), repeat=3)
]


def _combo_id(flags: dict[str, bool]) -> str:
    return "".join("1" if on else "0" for on in flags.values())


class PythonPrepare:
    """Not a SimpleTokenizer, so prepare takes the Python path."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self._t = SimpleTokenizer(config)

    def tokenize(self, text: str):  # noqa: ANN201
        return self._t.tokenize(text)


@requires_rust
@pytest.mark.parametrize("flags", FLAG_COMBOS, ids=_combo_id)
@pytest.mark.parametrize("text", CORPUS)
def test_rust_vocab_matches_python_tokenizer_vocab(
    text: str, flags: dict[str, bool]
) -> None:
    from cite_right._core import rust_tokenize_and_prepare

    rust_vocab = {
        key for key, _ in rust_tokenize_and_prepare([text], 3, 1, **flags).get_vocab()
    }
    python_tokenizer = SimpleTokenizer(TokenizerConfig(**flags))
    python_tokenizer.tokenize(text)

    assert rust_vocab == set(python_tokenizer._vocab)


@requires_rust
@pytest.mark.parametrize("flags", FLAG_COMBOS, ids=_combo_id)
@pytest.mark.parametrize("text", CORPUS)
def test_rust_prepared_source_ids_match_answer_tokenizer_ids(
    text: str, flags: dict[str, bool]
) -> None:
    tokenizer = SimpleTokenizer(TokenizerConfig(**flags))
    corpus = PreparedCitationCorpus.from_sources(
        [text], tokenizer=tokenizer, use_rust=True
    )

    assert corpus.rust_corpus is not None
    [rust_ids] = corpus.rust_corpus.get_candidate_tokens([0])
    assert rust_ids == corpus.tokenizer.tokenize(text).token_ids


def _align(source: str, answer: str, tokenizer: object):  # noqa: ANN202
    [span] = align_citations(
        answer,
        [SourceDocument(id="d", text=source)],
        config=CitationConfig(),
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )
    return span, span.citations[0]


@requires_rust
@pytest.mark.parametrize(
    ("source", "answer"),
    [
        ("Closing balance: 2,410.12 EUR.", "The closing balance was 2,410.12."),
        ("Fee: 1,204.50 EUR for the stay.", "Fee 1,204.50 EUR"),
    ],
)
def test_default_rust_prepare_matches_python_prepare(source: str, answer: str) -> None:
    rust_span, rust_citation = _align(source, answer, SimpleTokenizer())
    py_span, py_citation = _align(source, answer, PythonPrepare())

    assert rust_span.status == py_span.status
    assert rust_citation.components["matches"] == py_citation.components["matches"]
    assert rust_citation.evidence == py_citation.evidence


PREPARE_BACKENDS = [pytest.param("rust", marks=requires_rust), "python"]


@pytest.mark.parametrize("backend", PREPARE_BACKENDS)
def test_grouped_number_restatement_is_supported(backend: str) -> None:
    use_rust = backend == "rust"
    corpus = PreparedCitationCorpus.from_sources(
        ["Closing balance: 2,410.12 EUR."], use_rust=use_rust
    )
    assert (corpus.rust_corpus is not None) == use_rust

    [span] = corpus.align("The closing balance was 2410.12.", backend=backend)

    assert span.status == "supported"
    [citation] = span.citations
    assert citation.components["matches"] == 3.0
    assert citation.evidence == "Closing balance: 2,410.12"


@pytest.mark.parametrize("backend", PREPARE_BACKENDS)
@pytest.mark.parametrize(
    ("source", "answer"),
    [
        ("Copper yield: 25﹪.", "The copper yield was 25 percent."),
        ("Copper credit: 10€.", "The copper credit was 10 euro."),
        ("Copper credit: € 10.", "Copper credit: €10."),
        ("Copper credit: USD 10.", "Copper credit: $10."),
        ("Copper credit: CAD $10.", "Copper credit: $10 CAD."),
        ("Copper credit: CAD $10.", "Copper credit: $10."),
        ("Copper alloy price: $10.", "Copper alloy cost: $10."),
        ("Sample range: 30 - 40.", "Sample range: 30–40."),
    ],
)
def test_equivalent_quantity_restatements_stay_supported(
    source: str, answer: str, backend: str
) -> None:
    corpus = PreparedCitationCorpus.from_sources([source], use_rust=backend == "rust")
    assert (corpus.rust_corpus is not None) == (backend == "rust")

    [span] = corpus.align(answer, backend=backend)

    assert span.status == "supported"


@pytest.mark.parametrize("backend", PREPARE_BACKENDS)
@pytest.mark.parametrize(
    ("source", "answer"),
    [
        ("Closing balance: 2,410.12 EUR.", "The closing balance was 2470.12."),
        ("Closing balance: 2,410.12 EUR.", "The closing balance was −2410.12."),
        ("Closing balance: -2,410.12 EUR.", "The closing balance was 2410.12."),
        ("Closing balance: 2,410.12 EUR.", "The closing balance was − 2410.12."),
        ("Closing balance: − 2,410.12 EUR.", "The closing balance was 2410.12."),
        ("Closing balance: 2,41.", "The closing balance was 241."),
        ("Copper credit: $1,100.", "Copper credit: €1100."),
        ("Copper credit: $ 1,100.", "Copper credit: € 1100."),
        ("Copper credit: $1,100 CAD.", "Copper credit: $1100 USD."),
        ("Closing balance: 2,410.12%.", "The closing balance was 2410.12."),
        ("Copper yield: 2,410.12%.", "Copper yield: 2410.12 percentage points."),
        (
            "Copper score: 1,100 and silver score: 50.",
            "Copper score was 1100 and silver score was 80.",
        ),
        ("Copper yield: 25﹪.", "Copper losses: 25 percent."),
        ("Copper yield: 25﹪ decrease.", "Copper yield was 25 percent increase."),
        ("Copper credit: £10.", "Copper debt: £10."),
        ("Copper credit: CAD $10.", "Copper debt: $10."),
        ("Copper credit: 1,100.", "Copper debit: 1100."),
    ],
    ids=[
        "amount",
        "negative-answer",
        "negative-source",
        "spaced-negative-answer",
        "spaced-negative-source",
        "decimal-comma",
        "currency",
        "spaced-currency",
        "explicit-dollar-currency",
        "percent",
        "percentage-points",
        "multiple-amounts",
        "percent-label",
        "percent-direction",
        "currency-label",
        "unqualified-currency-label",
        "grouped-number-label",
    ],
)
def test_changed_quantity_is_never_supported(
    source: str, answer: str, backend: str
) -> None:
    use_rust = backend == "rust"
    corpus = PreparedCitationCorpus.from_sources([source], use_rust=use_rust)
    assert (corpus.rust_corpus is not None) == use_rust

    [span] = corpus.align(answer, backend=backend)

    assert span.status == "partial"


@requires_rust
def test_fee_restatement_matches_on_both_prepare_paths() -> None:
    _, rust_citation = _align(
        "Fee: 1,204.50 EUR for the stay.", "Fee 1,204.50 EUR", SimpleTokenizer()
    )
    _, py_citation = _align(
        "Fee: 1,204.50 EUR for the stay.", "Fee 1,204.50 EUR", PythonPrepare()
    )

    assert rust_citation.components["matches"] == 3.0
    assert py_citation.components["matches"] == 3.0
