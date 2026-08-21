# Strict-Attribution Evaluation and Optimization Design

## Status

Approved design for a leakage-resistant evaluation dataset, evaluator, and
hill-climbing workflow for Cite-Right.

## Goal

Build a reproducible evaluation system that measures Cite-Right's defining
promise: selecting the right source and returning defensible character-accurate
evidence. Use that system to improve exact-citation recall and runtime without
regressing citation precision, offset validity, contradiction safety, or memory.

Semantic retrieval is useful, but it is not an exact citation. Retrieval support
must therefore be evaluated separately and must never upgrade a claim to an
exact-citation success.

## Decisions

- Strict attribution is the primary objective.
- The initial dataset targets approximately 750 cases.
- Dataset provenance is hybrid: redistributable authored documents provide
  deterministic adversarial coverage, while a smaller public-domain or
  permissively licensed slice tests realism.
- Splits are grouped by document and transformation family, not by individual
  row.
- Every development and holdout annotation is manually reviewed. Training data
  is deterministically generated and automatically validated, with a sampled
  manual audit.
- Optimization is lexicographic: protect correctness and resource gates first,
  then maximize recall.
- The sealed holdout is evaluated only after a candidate is frozen. A failed
  holdout starts a new dataset version; it does not become additional tuning
  data for the current version.

## Non-Goals

- Treating embedding similarity as proof of an exact citation.
- Claiming general semantic entailment without localized source evidence.
- Optimizing a single weighted score that can hide precision or offset
  regressions.
- Reusing test-generated outputs as independent ground truth.
- Publishing a generalized zero-false-positive claim from a small tuned set.

## System Architecture

The evaluation system has four isolated layers.

### Corpus and Case Builder

The builder produces versioned cases from authored source documents,
deterministic adversarial transformations, and a smaller real-world slice. It
owns case identifiers, provenance, transformation metadata, and grouped split
assignment. It must generate canonical output deterministically for a fixed
seed and dataset version.

### Dataset Validator

The validator enforces schema, source/span consistency, exact offsets, unique
identifiers, label completeness, provenance, review metadata, and split
isolation. It detects duplicated or related document families across splits. It
also emits a manifest containing canonical file hashes, counts, family
distributions, review status, and the dataset version.

### Evaluator

The evaluator consumes a frozen dataset and a Cite-Right configuration. It
scores exact citations, support status, retrieval support, performance, and
Python/Rust parity as separate dimensions. It emits both aggregate metrics and
case-level failure records suitable for diagnosis.

### Hill Climber

The hill climber evaluates configuration and implementation candidates against
train and development data. It rejects candidates that violate correctness or
resource gates, then ranks survivors by exact-citation recall. It records every
experiment with the dataset manifest hash, git revision, environment, candidate
parameters, metrics, and decision.

The optimizer runs from a tuning bundle that contains only train and development
artifacts. The repository stores the holdout as an encrypted, content-addressed
artifact; its decryption key is managed outside the repository and is unavailable
to tuning processes. A separate release-gate command decrypts the holdout into an
ephemeral directory, evaluates one frozen revision and configuration, emits a
signed aggregate report, and removes the plaintext artifact. It never returns
case-level holdout labels or failures to the optimizer.

## Dataset Composition

The initial target is approximately 750 cases:

- About 450 training cases for hill climbing.
- About 150 development cases for candidate selection.
- About 150 sealed holdout cases for final verification.

Counts may vary slightly to preserve whole document and transformation
families. Family isolation takes precedence over exact split percentages.

The authored portion supplies controlled source facts and deterministic label
generation. The real-world portion uses public-domain or clearly
redistributable sources and records source title, origin, license, retrieval
date, and local snapshot hash. Network access is never required to run the
evaluation.

## Case Schema

Each case contains:

- `case_id`: stable identifier within a dataset version.
- `dataset_version`: version governing schema and labels.
- `split`: `train`, `dev`, or `holdout`.
- `document_family_id`: groups related source documents and variants.
- `transformation_family_id`: groups transformations that share a generation
  rule or template.
- `provenance`: `authored`, `public_domain`, or `permissive_license`, plus source
  metadata where applicable.
- `sources`: immutable source IDs and text, with optional chunk metadata.
- `answer`: answer text under evaluation.
- `evaluation_units`: expected answer-segmentation units containing atomic claim
  annotations and a normative expected status.
- `difficulty_tags`: coverage and diagnostic categories.
- `generation`: deterministic recipe name, parameters, and seed, when generated.
- `review`: validation state, reviewer identifier, notes, and timestamp for dev
  and holdout labels.

Each evaluation unit contains:

- Answer character offsets and exact unit text.
- One or more atomic claims wholly contained in that unit.
- `expected_status`, derived by the normative status mapping below.

Each atomic claim annotation contains:

- Answer character offsets and exact claim text.
- A support label: `entailed`, `contradicted`, or `not_in_sources`.
- Zero or more conjunctive citation requirements.
- Zero or more acceptable retrieval source IDs, scored separately.
- Whether non-contiguous evidence is required.

An `entailed` claim must contain at least one citation requirement. A
`contradicted` or `not_in_sources` claim must contain no citation requirements.
This invariant makes the exact-citation recall denominator explicit.

Citation requirements are conjunctive: every requirement must be satisfied for
the claim to be fully attributed. Each requirement contains one or more
alternative exact-citation targets. Any one alternative satisfies that
requirement. Each target contains a source ID and one or more exact character
spans; multiple spans on one target express non-contiguous evidence. Claims that
require two sources therefore contain two requirements, while alternative
sources or alternative defensible spans appear as alternatives within one
requirement. Target evidence text is derived from the source at validation time,
not stored as an independent authority.

The evaluator uses maximum one-to-one matching between emitted citations and
requirements so one emitted citation cannot satisfy multiple requirements unless
the annotation explicitly combines those requirements into one target.

## Coverage Families

The suite covers:

- Exact copying, faithful compression, and faithful paraphrase.
- Negation and polarity changes.
- Numeric, percentage, currency, unit, date, duration, and magnitude changes.
- Entity, subject/object, relation, location, and attribution swaps.
- Modality and certainty changes such as `may` to `will`.
- Partially supported claims and unsupported appended clauses.
- Multi-source answers and claims requiring multiple sources.
- Contiguous and multi-span evidence.
- Duplicate evidence, near-duplicate evidence, and plausible distractors.
- Long documents, overlapping passage windows, and source chunks with rebased
  absolute offsets.
- Unicode normalization, full-width forms, combining characters, punctuation
  variants, and multilingual text.
- Empty inputs, boundary offsets, repeated text, deterministic tie cases, and
  malformed cases that should fail validation.

Generated transformations must identify the exact fact changed and include a
metamorphic assertion proving unrelated facts and source text remain unchanged.

## Split and Leakage Policy

Splits are assigned at the connected-component level. Cases are connected when
they share a document family, transformation family, source snapshot, normalized
source fingerprint, or generated template lineage. A connected component may
belong to only one split.

The validator reports exact and near-duplicate overlap across splits. Holdout
case IDs and aggregate category counts may be visible, but source text, answers,
labels, per-case outputs, and failure details are unavailable to hill-climbing
code.

The public holdout manifest exposes only version, ciphertext hash, case count,
category counts, schema version, total claim count, reviewed claim count, and a
signed review-completion attestation. The sealing command can issue that
attestation only after validating that every holdout claim is reviewed. Tuning
readiness requires a valid signature and equality between total and reviewed
claim counts. Holdout plaintext, decryption keys, labels, reviewer notes, and
per-case metadata are absent from the tuning checkout and tuning runtime. Any
holdout content or label change requires a new dataset version, ciphertext hash,
review attestation, and baseline report.

## Annotation and Review

Authored training cases derive labels from deterministic construction and pass
all automatic invariants. A stratified sample from every training family is
manually audited.

Every development and holdout claim is manually reviewed for:

- Atomic claim boundaries.
- Support label correctness.
- Source identity.
- Minimal defensible evidence spans.
- Multi-span necessity.
- Alternative acceptable evidence.
- Retrieval-only source acceptability.

Review tooling presents the claim beside all sources with highlighted evidence
and supports explicit approval or correction. Review corrections modify the
dataset source annotation, after which validation and manifest generation run
again.

## Correctness Metrics

Primary metrics are:

- Exact-citation precision.
- Source-selection accuracy.
- Character-offset validity.
- Citation recall at exact span match, 0.9 IoU, and 0.5 IoU.
- Supported/partial/unsupported macro-F1.
- Multi-span precision and recall.
- False-citation rate on contradicted claims.

The normative mapping from claim truth to an evaluation unit's expected runtime
status is:

- `supported` when every claim in the unit is `entailed`. The schema invariant
  guarantees each such claim has at least one citation requirement with a valid
  alternative target.
- `partial` when the unit contains at least one `entailed` claim and at least one
  `contradicted` or `not_in_sources` claim.
- `unsupported` when the unit contains no `entailed` claims.

Runtime answer spans are matched to evaluation units by answer offsets. A runtime
span crossing multiple units is scored against their union using the same
mapping. A runtime span that cannot be mapped unambiguously is an evaluator
error, not a silently skipped sample.

An emitted citation is a true positive only when it selects an acceptable source
and matches an alternative for an unsatisfied citation requirement at the
reported threshold. Requirement recall is the fraction of conjunctive
requirements satisfied; claim-level exact-citation recall counts a claim only
when all its requirements are satisfied. Invalid offsets are always errors, even
if the evidence string appears similar.

Retrieval metrics are secondary and separate:

- Retrieval source recall.
- Mean reciprocal rank.
- Recall at configured retrieval-support limits.

Retrieval support cannot count toward exact-citation precision, recall, support
status, or contradiction safety.

Metric reports include raw counts and uncertainty intervals. Reports must not
use `100% precision` as a generalized claim; they state the observed numerator,
denominator, dataset version, and interval.

## Performance Metrics

Performance evaluation records:

- Corpus preparation latency.
- Per-answer latency for one-shot and prepared-corpus paths.
- Median and p95 latency after warm-up.
- Cases and source characters processed per second.
- Peak resident memory and retained cache size.
- Scaling across candidate count, source length, answer length, and repeated
  answers against a fixed corpus.
- Python and Rust backend results and parity.

Runs use fixed seeds, pinned inputs, recorded environment metadata, repeated
trials, and a stable warm-up protocol. Relative performance claims compare the
same dataset, backend, dependency set, and hardware.

## Hill-Climbing Policy

Candidate evaluation is lexicographic:

1. Reject any candidate with an offset-validity failure.
2. Reject candidates that regress exact-citation precision or contradiction
   safety beyond a predeclared confidence-aware tolerance.
3. Reject candidates that exceed frozen p95 latency or peak-memory budgets.
4. Among survivors, maximize development-set exact-citation recall.
5. Break ties by status macro-F1, retrieval quality, then lower latency.

The first implementation freezes budgets from measured default, strict,
permissive, Python, and Rust baselines rather than inventing thresholds in
advance. Every experiment records why it was accepted or rejected.

Search may include citation configuration, candidate-selection limits, scoring
weights, alignment implementation choices, caching, batching, and data-structure
changes. Configuration searches and code changes use the same evaluator and
gates.

After selection, the winning candidate is frozen by git revision and complete
configuration. A release operator provides the external holdout key to the
release-gate process, which runs in an environment without optimizer write-back
or experiment-search APIs. The sealed holdout runs once. If it fails a
correctness or resource gate, the candidate is not released and the current
holdout is not used for further tuning. A new iteration requires a newly
versioned holdout.

## Failure Handling

Dataset construction and validation fail closed on:

- Evidence that does not exactly match its source offsets.
- `not_in_sources` or `contradicted` claims with positive exact-citation
  requirements.
- Cross-split family or source leakage.
- Duplicate or unstable identifiers.
- Missing provenance or mandatory review metadata.
- Non-deterministic canonical regeneration.
- Holdout changes without a dataset-version change.

Evaluator failures retain the case ID, metric stage, exception type, and bounded
diagnostic context. A failed case is not silently dropped from metric
denominators.

## Verification Strategy

Verification includes:

- Unit tests for schema validation, span scoring, split grouping, duplicate
  detection, and leakage detection.
- Metamorphic tests showing each adversarial transformation changes only its
  intended fact.
- Golden evaluator tests with hand-calculated counts and metrics.
- Canonical regeneration and manifest-hash tests.
- Tests proving the optimizer cannot access holdout content or case failures.
- Integration tests proving tuning commands reject holdout paths, the tuning
  bundle contains no holdout plaintext, and only the release-gate command accepts
  an external decryption key.
- Baseline runs for default, strict, permissive, Python, and Rust modes.
- Repeated performance trials using the frozen protocol.
- Python/Rust output parity checks on all compatible cases.
- Manual review completion checks for every dev and holdout claim.

## Deliverables

The implementation will commit:

- Dataset schema and versioning rules.
- Authored and real-world source snapshots with provenance.
- Deterministic case generators and transformation families.
- Canonical train and dev artifacts plus an encrypted, content-addressed sealed
  holdout artifact and public redacted manifest.
- Manual review records for development plus holdout review records embedded
  inside the encrypted holdout artifact.
- Dataset validator, leakage detector, and manifest generator.
- Accuracy and performance evaluator.
- Experiment record format and hill-climbing runner.
- Frozen baseline and final holdout reports.
- Documentation for reviewing labels, reproducing results, and creating a new
  dataset version without contaminating the holdout.

## Acceptance Criteria

The evaluation system is ready for optimization when:

- The dataset contains approximately 750 valid cases with the intended grouped
  split proportions.
- All development claims are manually reviewed, and the public holdout manifest
  contains a valid signed attestation that all holdout claims are reviewed.
- Automatic validation, leakage checks, deterministic regeneration, and manifest
  verification pass.
- Golden evaluator tests prove metric calculations.
- Baselines exist for the required configurations and backends.
- Performance results include median, p95, throughput, and peak memory.
- The hill climber can improve train/dev candidates while being technically
  unable to decrypt or inspect holdout content, labels, or failures.

The objective is complete only after at least one measured hill-climbing cycle
produces a frozen candidate, the full verification suite passes, and the sealed
holdout report demonstrates no correctness or resource-gate regression relative
to the frozen baseline.
