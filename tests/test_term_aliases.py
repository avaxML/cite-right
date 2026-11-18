"""Public config and prepared-corpus edge cases for term aliases."""

import re
from unittest.mock import patch

import pytest

from cite_right import CitationConfig, PreparedCitationCorpus
from cite_right.contradiction import has_negation_mismatch
from cite_right.text.tokenizer import SimpleTokenizer

from .conftest import requires_rust


@pytest.mark.parametrize(
    "groups,offending_term",
    [
        ((("", "clamp"),), "''"),
        ((("blue clamp", "bracket"),), "blue clamp"),
        ((("Clamp",),), "Clamp"),
        ((("Clamp", "ＣＬＡＭＰ"),), "ＣＬＡＭＰ"),
        ((("clamp", "bracket"), ("ＢＲＡＣＫＥＴ", "fastener")), "ＢＲＡＣＫＥＴ"),
        ((("１２", "clamp"),), "１２"),
        ((("Ⅷ", "clamp"),), "Ⅷ"),
        ((("never", "clamp"),), "never"),
        ((("supports", "clamp"),), "supports"),
        ((("DON’T", "clamp"),), "DON’T"),
        ((("Before", "AFTER"),), "cannot equate 'before' and 'after'"),
    ],
)
def test_invalid_alias_names_offending_term(
    groups: tuple[tuple[str, ...], ...], offending_term: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(offending_term)):
        CitationConfig.strict().model_copy(update={"term_aliases": groups})


@pytest.mark.parametrize(
    "use_rust,backend",
    [
        (False, "python"),
        pytest.param(False, "rust", marks=requires_rust),
        pytest.param(True, "rust", marks=requires_rust),
    ],
)
@pytest.mark.parametrize(
    "answer,passage,group",
    [
        (
            "Die ＳＴＯＲＥＨＯＵＳＥ öffnet.",
            "Die Straße öffnet.",
            ("Straße", "storehouse"),
        ),
        ("Call Sullivan.", "Call O’Neill.", ("O'Neill", "Sullivan")),
        ("bracket", "Clamp- then store it.", ("bracket", "clamp")),
        ("plus wires", "and wires", ("plus", "and")),
        ("and wires", "plus wires", ("plus", "and")),
    ],
    ids=[
        "unicode",
        "entity-punctuation",
        "trailing-dash",
        "stopword-forward",
        "stopword-reverse",
    ],
)
def test_alias_normalization_coverage_and_corpus_reuse(
    use_rust: bool,
    backend: str,
    answer: str,
    passage: str,
    group: tuple[str, str],
) -> None:
    tokenizer = SimpleTokenizer()
    config = CitationConfig.strict().model_copy(
        update={"term_aliases": (group, ("tray", "pan"))}
    )
    corpus = PreparedCitationCorpus.from_sources(
        [passage], config=config, tokenizer=tokenizer, use_rust=use_rust
    )
    before = dict(tokenizer.vocabulary())
    [span] = corpus.align(answer, backend=backend)
    assert span.status == "supported"
    [citation] = span.citations
    assert citation.components["answer_coverage"] == 1
    assert citation.components["content_coverage"] == 1
    assert citation.evidence == passage[citation.char_start : citation.char_end]
    assert dict(tokenizer.vocabulary()) == before

    reordered = config.model_copy(
        update={"term_aliases": (("pan", "tray"), group[::-1])}
    )
    assert reordered.model_dump_json() == config.model_dump_json()
    [replayed] = corpus.model_copy(update={"config": reordered}).align(
        answer, backend=backend
    )
    assert replayed.model_dump_json() == span.model_dump_json()


def test_unused_alias_does_not_displace_literal_candidate() -> None:
    config = CitationConfig(top_k=1, max_candidates_lexical=1, max_candidates_total=1)
    corpus = PreparedCitationCorpus.from_sources(
        ["use", "bracket", "use"], config=config, use_rust=False
    )
    [control] = corpus.align("use bracket", backend="python")
    assert control.citations[0].source_index == 1
    aliased_config = config.model_copy(update={"term_aliases": (("bracket", "clamp"),)})
    assert corpus.model_copy(update={"config": aliased_config}).align(
        "use bracket", backend="python"
    ) == [control]


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_alias_resolution_preserves_curly_apostrophe_negation(backend: str) -> None:
    config = CitationConfig(
        term_aliases=(("polish", "buff"),), supported_answer_coverage=0.75
    )
    corpus = PreparedCitationCorpus.from_sources(
        ["Polish the casing with a cloth."], config=config
    )
    with patch(
        "cite_right.contradiction.has_negation_mismatch", wraps=has_negation_mismatch
    ) as check_negation:
        [span] = corpus.align("don’t buff casing with a cloth.", backend=backend)
    check_negation.assert_called_once()
    assert has_negation_mismatch(*check_negation.call_args.args)
    [citation] = span.citations
    assert citation.components["answer_coverage"] == pytest.approx(5 / 6)
    assert span.status == "partial"


def test_alias_resolution_preserves_non_negating_compatibility_glyphs() -> None:
    config = CitationConfig(
        term_aliases=(("bracket", "clamp"),), supported_answer_coverage=0.75
    )
    corpus = PreparedCitationCorpus.from_sources(
        ["The label uses a symbol to identify each clamp."],
        config=config,
        use_rust=False,
    )
    with patch(
        "cite_right.contradiction.has_negation_mismatch", wraps=has_negation_mismatch
    ) as check_negation:
        corpus.align("The label uses nº to identify each bracket.", backend="python")
    check_negation.assert_called_once()
    assert not has_negation_mismatch(*check_negation.call_args.args)


def test_config_exports_field_schema_and_omits_empty_aliases_from_payload() -> None:
    config = CitationConfig()
    assert "term_aliases" not in config.model_dump(mode="json")
    properties = CitationConfig.model_json_schema(mode="serialization")["properties"]
    assert properties["term_aliases"]["default"] == []
    assert "weights" in properties
