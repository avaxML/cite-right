# Document-attribution suite

188 answer sentences from a document-QA agent, labeled by which of eight
fixture documents (invoices, a bank statement, a freight rate card, and
expense, travel, retention, and security policies) each sentence may cite.
It measures what a citation UI shows: the set of documents cited per sentence.

Unlike the v1 dataset, labels are document-level, not character spans, so the
suite lives beside the sealed v1 lifecycle instead of inside it.

## Cases

| Group | Cases | Source |
|---|---|---|
| `live` | 44 | LLM agent answers to four product questions |
| `synthetic:*` | 120 | LLM agent answers to 100 generated questions, 15 per question type |
| `paraphrase` | 8 | hand-written paraphrases of stated facts |
| `adversarial` | 16 | hand-written swapped numbers, swapped entities, contradictions, absent facts |

Answers are plain text: markdown and model citation markers were stripped
before alignment, as the integrating product does.

Labels:

- `must_cite`: the sentence restates a fact from the listed documents. No
  citation is a recall miss.
- `may_cite`: inference, meta, or mixed sentence. Citing a listed document is
  fine; not citing is not penalised.
- `must_not_cite`: no document supports the sentence.

Citing any document outside a case's `docs` is `wrong` under every label. Only
spans with `status == "supported"` count as citations, because `partial`
includes contradiction downgrades.

Each case has a `split`, stratified by group with seed 7513. Tune on `train`
and report `dev`.

## Run

```bash
uv run python -m evaluation.doc_attribution --config default
uv run python -m evaluation.doc_attribution --config strict --split dev --output /tmp/strict-dev.jsonl
```

`--embedder` adds `SentenceTransformerEmbedder` (requires the `embeddings`
extra). `--fail-on-wrong` exits 1 when any case cites a wrong document.

## Baseline

cite-right 0.4.0 (`219ff2f`), no embedder:

| Config | Split | Wrong | must_cite recall | must_not_cite clean |
|---|---|---|---|---|
| `default` | train | 10 | 4/46 | 12/12 |
| `default` | dev | 10 | 5/45 | 13/13 |
| `strict` | train | 0 | 7/46 | 12/12 |
| `strict` | dev | 1 | 4/45 | 13/13 |

Most `default` wrong citations are secondary citations (`top_k=3`) attached to
a span whose status comes from its best citation.
