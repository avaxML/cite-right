# Strict-Attribution Evaluation and Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-resistant approximately 750-case strict-attribution dataset, a trustworthy accuracy/performance evaluator, and a constrained hill-climbing workflow that produces and verifies a faster, higher-recall Cite-Right candidate without precision or offset regressions.

**Architecture:** Keep evaluation tooling outside the distributable library in a top-level `evaluation` package. Build and validate canonical train/dev artifacts, seal holdout content with an external key, score correctness and performance through separate evaluators, and let a coordinate/grid hill climber consume only train/dev bundles. Freeze one candidate before a separate release-gate process decrypts and evaluates the holdout.

**Tech Stack:** Python 3.11+, Pydantic 2, NumPy, pytest, standard-library JSON/HTML/statistics/resource tooling, `cryptography` for authenticated holdout encryption and signed attestations, optional sentence-transformers and Rust backend.

**Reference spec:** `docs/superpowers/specs/2026-07-17-strict-attribution-evaluation-design.md`

---

## File Structure

Create a non-wheel evaluation package with one responsibility per module:

```text
evaluation/
  __init__.py                 # package marker and dataset version
  schema.py                   # canonical Pydantic annotation models
  canonical.py                # canonical JSON and hashing
  builders/
    authored_sources.py       # controlled source-family catalog
    transformations.py        # deterministic adversarial transformations
    real_sources.py           # redistributable source snapshot catalog
    cases.py                  # case expansion and expected labels
  splitting.py                # connected-family grouped split assignment
  leakage.py                  # exact/near-duplicate cross-split detection
  tuning_bundle.py            # train/dev-only optimizer artifact and isolation
  validation.py               # fail-closed dataset invariants
  manifest.py                 # public and private manifests
  sealing.py                  # AES-GCM holdout sealing and Ed25519 attestation
  review.py                   # review queues, ledgers, and completion checks
  matching.py                 # output-to-unit/requirement matching
  metrics.py                  # hand-auditable correctness metrics
  runner.py                   # run Cite-Right and emit case-level records
  performance.py              # repeatable latency/throughput/memory trials
  worker.py                   # subprocess entrypoint for isolated bundle runs
  baselines.py                # default/strict/permissive/backend matrix
  experiments.py              # experiment record schema and persistence
  hill_climb.py               # gated train/dev candidate search
  release_gate.py             # isolated sealed-holdout execution
  cli.py                      # explicit build/validate/evaluate/tune/release commands
  data/v1/
    sources/authored.json
    sources/real.json
    provenance.json
    train.json
    dev.json
    dev_reviews.json
    tuning/
      train.json
      dev.json
      manifest.json
    holdout.aesgcm
    holdout.public.json
    holdout_public_key.pem
    manifest.json
  reports/v1/
    baseline.json
    candidate.json
    holdout.json
  search_spaces/v1.json       # bounded configuration neighborhood
  experiments/v1/            # canonical train/dev experiment records
  README.md
tests/evaluation/
  test_schema.py
  test_canonical.py
  test_transformations.py
  test_splitting.py
  test_leakage.py
  test_validation.py
  test_sealing.py
  test_review.py
  test_matching.py
  test_metrics.py
  test_runner.py
  test_performance.py
  test_baselines.py
  test_experiments.py
  test_hill_climb.py
  test_release_gate.py
  test_dataset_v1.py
  fixtures/
    tuning/
      train.json
      dev.json
      manifest.json
    three-candidates.json
```

Do not make `evaluation` part of `src/cite_right` or export it from the public API. The committed evaluation data is a development asset, not wheel payload.

---

### Task 1: Establish the Evaluation Package and Dependency Boundary

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `evaluation/__init__.py`
- Create: `evaluation/README.md`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_schema.py`

- [ ] **Step 1: Write the package-boundary test**

```python
def test_evaluation_package_is_not_public_library_api() -> None:
    import cite_right
    import evaluation

    assert evaluation.DATASET_VERSION == "1.0.0"
    assert not hasattr(cite_right, "evaluation")
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/evaluation/test_schema.py -q`

Expected: collection fails because `evaluation` does not exist.

- [ ] **Step 3: Add the minimal package and development dependency**

Add `cryptography>=45` to `[dependency-groups].dev`, add `evaluation` to
`[tool.pyright].include`, run `uv lock`, and create:

```python
# evaluation/__init__.py
DATASET_VERSION = "1.0.0"
```

Document in `evaluation/README.md` that the package is development-only and that holdout private keys must never enter the repository.

- [ ] **Step 4: Verify GREEN and dependency integrity**

Run:

```bash
uv run pytest tests/evaluation/test_schema.py -q
uv lock --check
uv run pyright evaluation
```

Expected: test passes; lock file is current.

- [ ] **Step 5: Commit this boundary**

Stage only the task files. Commit with intent `Make evaluation tooling reproducible without expanding the library API` and trailers `Constraint: Holdout sealing requires authenticated encryption`, `Confidence: high`, `Scope-risk: narrow`, and the exact tests run.

---

### Task 2: Define Canonical Annotation Models

**Files:**
- Create: `evaluation/schema.py`
- Modify: `tests/evaluation/test_schema.py`

- [ ] **Step 1: Write failing schema-contract tests**

Cover six named tests: `test_entailed_claim_requires_citation_requirement`,
`test_negative_claim_forbids_citation_requirements`,
`test_target_spans_are_ordered_non_overlapping_and_in_bounds`,
`test_evaluation_unit_status_is_derived_from_claim_labels`,
`test_multi_source_claim_requires_all_requirements`, and
`test_case_offsets_slice_exact_answer_and_source_text`.

Construct a two-source claim whose two conjunctive requirements each contain alternative targets. Assert that `expected_status` is computed, not freely supplied.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/evaluation/test_schema.py -q`

Expected: imports or constructors fail because schema models do not exist.

- [ ] **Step 3: Implement the schema minimally**

Define frozen Pydantic models:

```python
SupportLabel = Literal["entailed", "contradicted", "not_in_sources"]
Split = Literal["train", "dev", "holdout"]

class CharSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: int
    end: int

class CitationTarget(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_id: str
    spans: tuple[CharSpan, ...]

class CitationRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)
    requirement_id: str
    alternatives: tuple[CitationTarget, ...]

class ClaimAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)
    claim_id: str
    answer_span: CharSpan
    label: SupportLabel
    citation_requirements: tuple[CitationRequirement, ...] = ()
    acceptable_retrieval_source_ids: tuple[str, ...] = ()

class EvaluationUnit(BaseModel):
    model_config = ConfigDict(frozen=True)
    unit_id: str
    answer_span: CharSpan
    claims: tuple[ClaimAnnotation, ...]

    @computed_field
    @property
    def expected_status(self) -> Literal["supported", "partial", "unsupported"]:
        labels = {claim.label for claim in self.claims}
        if labels == {"entailed"}:
            return "supported"
        if "entailed" in labels:
            return "partial"
        return "unsupported"
```

Add `Source`, `Provenance`, `GenerationRecipe`, `ReviewRecord`, and `EvaluationCase`. Put all cross-field validation in model validators with stable error messages.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/evaluation/test_schema.py -q`

Expected: all schema tests pass.

- [ ] **Step 5: Commit the schema contract**

Commit with intent `Make attribution ground truth explicit enough for unambiguous scoring`, `Scope-risk: moderate`, and tested commands.

---

### Task 3: Canonical Serialization, IDs, and Dataset Hashes

**Files:**
- Create: `evaluation/canonical.py`
- Create: `tests/evaluation/test_canonical.py`

- [ ] **Step 1: Write failing canonicalization tests**

Test that dictionary insertion order does not affect bytes, floats use stable JSON representation, Unicode remains UTF-8, a case ID changes when authoritative content changes, and it does not depend on split/review metadata.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_canonical.py -q`

- [ ] **Step 3: Implement canonical helpers**

```python
def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def authoritative_case_id(
    case_without_operational_metadata: Mapping[str, object],
) -> str:
    return f"case-{sha256_hex(canonical_json_bytes(case_without_operational_metadata))[:20]}"
```

Use full SHA-256 hashes in manifests and a collision-resistant readable prefix in case IDs.

- [ ] **Step 4: Verify GREEN and deterministic subprocess output**

Run: `uv run pytest tests/evaluation/test_canonical.py -q`

Expected: unit tests, including the subprocess determinism test, pass and both
processes emit the same canonical hash.

- [ ] **Step 5: Commit canonical identity**

Commit with intent `Make dataset identity stable across machines and regeneration` and exact verification trailers.

---

### Task 4: Build Authored Source Families and Deterministic Transformations

**Files:**
- Create: `evaluation/builders/__init__.py`
- Create: `evaluation/builders/authored_sources.py`
- Create: `evaluation/builders/transformations.py`
- Create: `evaluation/builders/cases.py`
- Create: `tests/evaluation/test_transformations.py`

- [ ] **Step 1: Write metamorphic transformation tests first**

For negation, number, unit, date, entity, relation, modality, unsupported-clause, Unicode, duplicate-distractor, multi-span, and multi-source transformations, assert:

- deterministic output for a fixed seed;
- only the declared fact changes;
- positive source spans remain exact slices;
- negative claims have no citation requirements;
- source and transformation family IDs remain stable;
- generated siblings retain shared lineage.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_transformations.py -q`

- [ ] **Step 3: Implement a declarative authored fact catalog**

Use structured records rather than generating prose from the library:

```python
class FactTemplate(BaseModel):
    family_id: str
    domain: Literal["science", "finance", "policy", "technology", "health", "history"]
    source_text: str
    facts: tuple[Fact, ...]

class Transformation(Protocol):
    name: str
    def generate(
        self, template: FactTemplate, seed: int
    ) -> tuple[EvaluationCase, ...]:
        raise NotImplementedError
```

Start with at least 60 authored document families, balanced across six domains. Each family must yield positive and adversarial siblings without calling `align_citations()` to create labels.

- [ ] **Step 4: Implement transformations one family at a time**

After each transformation, run
`uv run pytest tests/evaluation/test_transformations.py -q`. Prefer explicit
substitutions tied to fact slots; do not use an LLM or fuzzy search to infer gold
offsets.

- [ ] **Step 5: Verify the authored generator**

Run twice:

```bash
uv run pytest tests/evaluation/test_transformations.py -q
uv run pytest tests/evaluation/test_transformations.py -q
```

Expected: both runs pass; the shuffled-order fixture asserts the same frozen
canonical hash.

- [ ] **Step 6: Commit deterministic authored generation**

Commit with intent `Create attribution cases whose labels do not depend on Cite-Right outputs`, `Directive: Never call align_citations while generating gold labels`, and verification trailers.

---

### Task 5: Add Redistributable Real-World Source Snapshots

**Files:**
- Create: `evaluation/builders/real_sources.py`
- Create: `evaluation/data/v1/sources/real.json`
- Create: `evaluation/data/v1/provenance.json`
- Modify: `tests/evaluation/test_transformations.py`

- [ ] **Step 1: Write provenance validation tests**

Require origin URL, title, publisher, license/public-domain basis, retrieval date, local snapshot SHA-256, and exact source text. Reject network-only references and missing license evidence.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_transformations.py -k provenance -q`

Expected: missing metadata fixtures fail validation while complete fixtures pass.

- [ ] **Step 3: Curate at least 15 real-world document families**

Use stable public-domain United States government or clearly permissively licensed sources. Store short, self-contained snapshots locally. Use source diversity across science, public health, economics, technology, environment, and history. Record provenance without relying on live network access at evaluation time.

- [ ] **Step 4: Annotate real cases independently**

Create positive, partial, contradicted, and distractor cases by reading the snapshots and writing exact targets. Do not bootstrap expected citations from current library output.

- [ ] **Step 5: Verify snapshot hashes and offline operation**

Run: `uv run pytest tests/evaluation/test_transformations.py -k 'provenance or real_source_offline' -q`

Expected: every snapshot hash and target span validates without a network call.

- [ ] **Step 6: Commit the real-world slice**

Commit with intent `Add realistic attribution cases with auditable redistribution rights`, one `Constraint:` trailer per provenance limitation, and verification evidence.

---

### Task 6: Grouped Splitting and Leakage Detection

**Files:**
- Create: `evaluation/splitting.py`
- Create: `evaluation/leakage.py`
- Create: `tests/evaluation/test_splitting.py`
- Create: `tests/evaluation/test_leakage.py`

- [ ] **Step 1: Write failing split and leakage tests**

Cover connected components through document family, transformation family,
snapshot hash, normalized source fingerprint, and template lineage. Include a
transitive A-B-C connection. Assert no connected component crosses splits.

Test exact duplicates, Unicode-normalized duplicates, whitespace/punctuation
near-duplicates, and high token-shingle Jaccard overlap across splits.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_splitting.py tests/evaluation/test_leakage.py -q`

- [ ] **Step 3: Implement deterministic connected-component splitting**

Use union-find over all lineage edges. Sort components by stable hash, then assign
whole components toward 60/20/20 targets while balancing support labels,
difficulty families, domains, and provenance. Accept slight count deviations.

- [ ] **Step 4: Implement leakage reports**

Emit machine-readable findings with severity, case IDs, shared fingerprints, and
similarity. Exact/lineage leakage is fatal; near-duplicate thresholds are fatal
above the frozen threshold and advisory below it.

- [ ] **Step 5: Verify determinism and adversarial fixtures**

Run:

```bash
uv run pytest tests/evaluation/test_splitting.py -k 'deterministic or connected_component' -q
uv run pytest tests/evaluation/test_leakage.py -q
```

Expected: the frozen seed produces identical assignments under all input orders,
and each leakage fixture produces its expected finding code.

- [ ] **Step 6: Commit split isolation**

Commit with intent `Prevent generated siblings and source variants from leaking across evaluation splits` and tests.

---

### Task 7: Dataset Validation and Manifests

**Files:**
- Create: `evaluation/validation.py`
- Create: `evaluation/manifest.py`
- Create: `tests/evaluation/test_validation.py`

- [ ] **Step 1: Write fail-closed validator tests**

Test invalid answer/source slices, missing requirements, forbidden negative
requirements, ambiguous runtime units, duplicate IDs, incomplete provenance,
review gaps, split leakage, and non-deterministic canonical order. Assert every
invalid case remains in the denominator/report rather than disappearing.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_validation.py -q`

- [ ] **Step 3: Implement structured validation**

```python
class ValidationFinding(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    case_id: str | None
    path: str
    message: str

def validate_dataset(bundle: DatasetBundle) -> ValidationReport:
    findings = [
        *validate_cases(bundle),
        *detect_leakage(bundle),
        *validate_reviews(bundle),
        *validate_manifests(bundle),
    ]
    return ValidationReport(findings=tuple(findings))
```

No broad exception handling or silent drops. `assert_valid()` raises one summary
exception after collecting all bounded findings.

- [ ] **Step 4: Implement private and redacted manifests**

Private manifests include canonical file hashes, counts, label/domain/family
distributions, and review state. Public holdout manifests expose only the fields
approved by the spec.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest tests/evaluation/test_validation.py -q`

Expected: all validator tests and the hand-audited manifest snapshot pass.

- [ ] **Step 6: Commit validation and manifests**

Commit with intent `Fail closed when evaluation ground truth cannot support trustworthy claims` and evidence.

---

### Task 8: Manual Review Queue and Completion Ledger

**Files:**
- Create: `evaluation/review.py`
- Create: `tests/evaluation/test_review.py`
- Create: `evaluation/data/v1/dev_reviews.json`

- [ ] **Step 1: Write failing review-workflow tests**

Test deterministic queue shards, escaped HTML rendering, evidence highlighting,
approve/correct/reject ledger entries, reviewer identity, source-hash binding,
stale-review invalidation, and completion counts.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_review.py -q`

- [ ] **Step 3: Implement review artifacts**

Generate self-contained HTML queues with answer units, atomic claims, all source
texts, highlighted alternatives, transformation metadata, and provenance. Store
review decisions in canonical JSON keyed by case and source hash.

- [ ] **Step 4: Implement completion gates**

Dev validation requires every claim reviewed. Holdout sealing requires every
claim reviewed before plaintext is encrypted. A changed source, answer, label, or
target invalidates the prior review.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run pytest tests/evaluation/test_review.py -q
uv run python -m evaluation.review render-fixture --output /tmp/cite-right-review-fixture.html
```

Inspect `/tmp/cite-right-review-fixture.html` for correct escaping and span
highlighting.

- [ ] **Step 6: Commit review tooling**

Commit with intent `Make dev and holdout labels auditable before they influence optimization` and verification evidence.

---

### Task 9: Authenticated Holdout Sealing and Attestation

**Files:**
- Create: `evaluation/sealing.py`
- Create: `tests/evaluation/test_sealing.py`
- Create: `evaluation/data/v1/holdout_public_key.pem`

- [ ] **Step 1: Write cryptographic boundary tests first**

Test AES-256-GCM round-trip, random nonce uniqueness, ciphertext tamper failure,
wrong-key failure, Ed25519 review-attestation verification, redacted manifest
fields, key-file permission checks, and absence of plaintext labels in committed
artifacts.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_sealing.py -q`

- [ ] **Step 3: Implement key and envelope formats**

Use `cryptography.hazmat.primitives.ciphers.aead.AESGCM` with a 32-byte external
key and authenticated associated data containing dataset/schema versions. Use an
external Ed25519 private key to sign the public review-completion attestation;
commit only the public key.

Never accept raw keys on the command line. Accept paths via
`CITE_RIGHT_HOLDOUT_KEY_FILE` and `CITE_RIGHT_ATTESTATION_KEY_FILE`, reject
group/world-readable key files on POSIX, and zero mutable key buffers after use
where practical.

- [ ] **Step 4: Implement `seal` and `verify-public-manifest` commands**

`seal` validates plaintext and review completion before encryption. Public
verification requires no secret and checks ciphertext hash, signature, claim
counts, and schema version.

- [ ] **Step 5: Verify GREEN and secret scanning**

Run:

```bash
uv run pytest tests/evaluation/test_sealing.py -q
git grep -nE 'BEGIN (PRIVATE|ENCRYPTED PRIVATE) KEY|CITE_RIGHT_(HOLDOUT|ATTESTATION)_KEY' -- ':!docs/superpowers/plans/*'
```

Expected: tests pass and grep finds no committed private key or secret value.

- [ ] **Step 6: Commit sealing primitives**

Commit with intent `Keep holdout labels unavailable to tuning while preserving verifiable readiness`, `Directive: Never commit holdout private keys`, and tests.

---

### Task 10: Explicit CLI and Train/Dev-Only Tuning Bundle

**Files:**
- Create: `evaluation/tuning_bundle.py`
- Create: `evaluation/cli.py`
- Create: `evaluation/worker.py`
- Create: `tests/evaluation/test_tuning_bundle.py`
- Create: `tests/evaluation/test_cli.py`

- [ ] **Step 1: Write failing tuning-bundle isolation tests**

Build a fixture bundle from train, dev, encrypted holdout, private review, and
manifest inputs. Assert the produced tuning directory contains only canonical
train/dev cases, redacted dataset metadata, and hashes. Assert no filename or
serialized value contains holdout ciphertext paths, holdout case IDs, labels,
review notes, or per-case metadata. Assert `load_tuning_bundle()` rejects bundles
whose split is not train/dev or whose hash does not match.

- [ ] **Step 2: Write failing CLI contract tests**

Use `evaluation.cli.main(argv)` and assert exact exit codes and JSON output for:

```text
build --output DIR --seed INT
validate --bundle DIR
seal --plaintext FILE --output FILE --public-manifest FILE
verify-public-manifest --bundle DIR
build-tuning-bundle --dataset DIR --output DIR
promote --staging DIR --dataset DIR
```

Assert unknown commands and missing required arguments exit with code 2. Assert
validation/sealing failures exit with code 1 and structured errors on stderr.

- [ ] **Step 3: Verify RED**

Run:

```bash
uv run pytest tests/evaluation/test_tuning_bundle.py tests/evaluation/test_cli.py -q
```

Expected: imports fail because the bundle and CLI modules do not exist.

- [ ] **Step 4: Implement the tuning-bundle boundary**

Create the bundle in a new directory with mode 0700. Copy canonical train/dev
bytes and a redacted manifest rather than storing paths back into the full
dataset tree. Refuse an output directory nested inside the dataset's holdout or
review directories. Benchmark and optimizer subprocesses run with this bundle as
their working directory. Launch them as `sys.executable -m evaluation.worker`
with `PYTHONPATH` set to the immutable repository root solely so the top-level
development package can import; pass no full dataset-root argument and remove
holdout-key environment variables. The worker loads cases only through
`load_tuning_bundle(Path.cwd())`. Tests launch this exact subprocess contract
from a temporary bundle directory and prove imports work while external case
paths are rejected.

- [ ] **Step 5: Implement the foundational CLI**

Use `argparse` without a new CLI dependency. Route commands to builder,
validator, sealing, manifest, tuning-bundle, and atomic promotion functions.
`promote` accepts a validated staging directory and atomically installs only
source snapshots, provenance, train/dev, completed dev reviews, encrypted
holdout, redacted holdout manifest/public key, and non-holdout manifest. It
rejects holdout plaintext. Keep `main(argv) -> int` testable and call it from
`if __name__ == "__main__"`.

- [ ] **Step 6: Verify GREEN and process isolation**

Run:

```bash
uv run pytest tests/evaluation/test_tuning_bundle.py tests/evaluation/test_cli.py -q
uv run python -m evaluation.cli --help
```

Expected: tests pass and help lists the six foundational commands exactly.

- [ ] **Step 7: Commit the operational boundary**

Commit with intent `Give tuning code a train-dev artifact with no holdout
content`, `Directive: Optimizers accept only load_tuning_bundle output`, and
exact tests.

---

### Task 11: Matching and Hand-Auditable Accuracy Metrics

**Files:**
- Create: `evaluation/matching.py`
- Create: `evaluation/metrics.py`
- Create: `tests/evaluation/test_matching.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write golden matching tests**

Cover exact and 0.9/0.5 IoU thresholds, wrong source, invalid offset, alternative
targets, multi-span targets, conjunctive multi-source requirements, duplicate
emissions, and maximum one-to-one matching. Include hand-calculated cases where
greedy matching is wrong.

- [ ] **Step 2: Write golden metric tests**

Hand-calculate raw TP/FP/FN, requirement recall, fully-attributed claim recall,
source accuracy, contradiction false-citation rate, status confusion matrix and
macro-F1, retrieval recall/MRR, and Wilson confidence intervals.

- [ ] **Step 3: Verify RED**

Run:

```bash
uv run pytest tests/evaluation/test_matching.py tests/evaluation/test_metrics.py -q
```

Expected: imports fail because matching and metric modules do not exist.

- [ ] **Step 4: Implement deterministic matching**

Implement maximum bipartite matching without a new dependency because candidate
sets are small. Sort all nodes and ties canonically. Return explicit matches and
unmatched outputs/requirements.

- [ ] **Step 5: Implement metrics from raw counts**

Use immutable count models. Never average per-case precision. Emit numerator,
denominator, point estimate, and confidence interval for rate metrics. Preserve
failed cases as evaluator errors and denominator entries according to the spec.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
uv run pytest tests/evaluation/test_matching.py tests/evaluation/test_metrics.py -q
```

Expected: all golden and permutation-invariance tests pass.

- [ ] **Step 7: Commit evaluator mathematics**

Commit with intent `Make citation quality claims reproducible from raw counts` and exact golden tests.

---

### Task 12: Execute Cite-Right Against Cases

**Files:**
- Create: `evaluation/runner.py`
- Create: `tests/evaluation/test_runner.py`

- [ ] **Step 1: Write failing runner tests**

Test answer-unit mapping by offsets, runtime spans crossing units, unmappable
spans, exact citations versus retrieval support separation, exceptions, timeouts,
deterministic configuration serialization, and Python/Rust parity records.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_runner.py -q`

- [ ] **Step 3: Implement case execution records**

```python
class CaseRun(BaseModel):
    case_id: str
    backend: Literal["python", "rust"]
    config: dict[str, object]
    outputs: tuple[SpanCitations, ...] = ()
    duration_ns: int
    error: RunError | None = None
```

Use `perf_counter_ns()`. Do not swallow failures. Make the evaluator consume
`CaseRun` rather than call library functions directly.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/evaluation/test_runner.py tests/evaluation/test_matching.py tests/evaluation/test_metrics.py -q
```

Expected: all execution and scoring tests pass.

- [ ] **Step 5: Commit the execution boundary**

Commit with intent `Separate library execution from scoring so failures remain observable` and tests.

---

### Task 13: Reproducible Performance Harness

**Files:**
- Create: `evaluation/performance.py`
- Create: `tests/evaluation/test_performance.py`

- [ ] **Step 1: Write performance-protocol tests**

Test warm-up exclusion, sample count, median and nearest-rank p95, throughput,
prepared-corpus build versus answer timing, retained cache reporting, peak memory,
environment metadata, and deterministic workload selection. Use a fake clock for
metric mathematics and a real smoke test for process isolation.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_performance.py -q`

- [ ] **Step 3: Implement isolated benchmark trials**

Run each backend/config/workload in a subprocess with a JSON request/response.
Record Python, OS, CPU, dependency versions, git revision, dataset hash, warm-up,
trial count, and raw nanosecond samples. Measure peak RSS with platform-specific
standard-library support and clearly mark unsupported fields.

- [ ] **Step 4: Freeze workload strata**

Include one-shot and prepared paths across small/medium/large candidate counts,
short/long sources, single/multi-sentence answers, embedding off/on, and Python/
Rust where supported.

- [ ] **Step 5: Verify GREEN and repeatability**

Register `performance-smoke` in `evaluation.cli`, then run:

```bash
uv run pytest tests/evaluation/test_performance.py tests/evaluation/test_cli.py -q
uv run python -m evaluation.cli performance-smoke --output /tmp/cite-right-perf-1.json
uv run python -m evaluation.cli performance-smoke --output /tmp/cite-right-perf-2.json
uv run python -m evaluation.performance compare-smoke /tmp/cite-right-perf-1.json /tmp/cite-right-perf-2.json
```

Expected: tests pass; correctness hashes match; comparison reports raw timing
variance and exits zero.

- [ ] **Step 6: Commit performance measurement**

Commit with intent `Measure speed without conflating warm-up, preparation, and repeated-answer costs` and evidence.

---

### Task 14: Generate, Review, Validate, and Seal Dataset v1

**Files:**
- Create: `evaluation/data/v1/sources/authored.json`
- Create: `evaluation/data/v1/train.json`
- Create: `evaluation/data/v1/dev.json`
- Create: `evaluation/data/v1/dev_reviews.json`
- Create: `evaluation/data/v1/tuning/train.json`
- Create: `evaluation/data/v1/tuning/dev.json`
- Create: `evaluation/data/v1/tuning/manifest.json`
- Create: `evaluation/data/v1/holdout.aesgcm`
- Create: `evaluation/data/v1/holdout.public.json`
- Create: `evaluation/data/v1/manifest.json`
- Create: `tests/evaluation/test_dataset_v1.py`

- [ ] **Step 1: Write dataset acceptance tests before generation**

Assert total cases are within 725-775, grouped split proportions are within five
percentage points of 60/20/20 unless component constraints are documented,
minimum domain/family/provenance coverage is met, dev review is complete, public
holdout attestation is valid, no cross-split leakage exists, and regeneration of
train/dev plus ciphertext-independent private holdout canonical bytes is stable.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_dataset_v1.py -q`

Expected: missing dataset artifacts.

- [ ] **Step 3: Generate candidate cases and grouped splits**

Run:

```bash
uv run python -m evaluation.cli build --output /tmp/cite-right-evaluation-v1 --seed 20260717
```

Expected: the command writes canonical private split candidates and build
metadata. Do not hand-edit canonical generated files; correct catalogs or
transformations and regenerate.

- [ ] **Step 4: Review every development claim**

Shard the deterministic HTML review queue by family. Review source identity,
support label, minimal spans, alternatives, and status. Record corrections in
source annotations, regenerate, invalidate stale reviews, and repeat until the
dev completion gate passes.

- [ ] **Step 5: Review every holdout claim without exposing it to tuning code**

Use a separate review workspace and external key material. Complete the ledger,
run private validation/leakage checks, seal the canonical holdout, sign the
redacted readiness attestation, and remove plaintext from the repository
workspace.

- [ ] **Step 6: Promote reviewed staging artifacts atomically**

Run:

```bash
uv run python -m evaluation.cli promote --staging /tmp/cite-right-evaluation-v1 --dataset evaluation/data/v1
uv run python -m evaluation.cli build-tuning-bundle --dataset evaluation/data/v1 --output evaluation/data/v1/tuning
```

Expected: the repository dataset tree receives reviewed canonical train/dev,
completed dev reviews, encrypted holdout, redacted holdout attestation, source
snapshots, provenance, and manifests in one atomic replacement. No plaintext
holdout file is copied.

- [ ] **Step 7: Run full dataset validation**

Run:

```bash
uv run python -m evaluation.cli validate --bundle evaluation/data/v1
uv run python -m evaluation.cli verify-public-manifest --bundle evaluation/data/v1
uv run pytest tests/evaluation/test_dataset_v1.py -q
```

Expected: zero errors, no leakage, complete dev reviews, valid holdout attestation,
and target case-count/coverage gates.

- [ ] **Step 8: Commit dataset v1**

Commit with intent `Establish independent strict-attribution ground truth for optimization`, `Constraint: Holdout plaintext and keys remain external`, `Scope-risk: broad`, and full validation evidence.

---

### Task 15: Baseline Matrix and Frozen Resource Gates

**Files:**
- Create: `evaluation/baselines.py`
- Create: `tests/evaluation/test_baselines.py`
- Create: `evaluation/reports/v1/baseline.json`

- [ ] **Step 1: Write baseline-matrix tests**

Require default, strict, permissive configurations; Python and Rust when
available; embeddings off and the pinned small embedding model when available;
train/dev accuracy; and all performance strata. Assert reports include raw
counts, dataset hash, git revision, environment, and no holdout metrics.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_baselines.py -q`

Expected: failure because baseline orchestration/report does not exist.

- [ ] **Step 3: Implement baseline orchestration**

Resolve configurations to complete serialized models. Run accuracy before
performance and fail if repeated performance runs change outputs. Select the
strict-attribution baseline used for gates by declared policy, not by peeking at
holdout.

- [ ] **Step 4: Freeze gates from evidence**

Set offset validity and contradiction safety to zero tolerated failures. Freeze
precision confidence tolerance, p95 latency, and peak-memory budgets from the
chosen baseline with documented noise margins based on repeated trials.

- [ ] **Step 5: Generate and verify baseline report**

Register `baseline` in `evaluation.cli`, then run:

```bash
uv run pytest tests/evaluation/test_baselines.py tests/evaluation/test_cli.py -q
uv run python -m evaluation.cli baseline --tuning-bundle evaluation/data/v1/tuning --output /tmp/cite-right-baseline-1.json
uv run python -m evaluation.cli baseline --tuning-bundle evaluation/data/v1/tuning --output /tmp/cite-right-baseline-2.json
uv run python -m evaluation.baselines compare /tmp/cite-right-baseline-1.json /tmp/cite-right-baseline-2.json
```

Expected: reports agree on all correctness counts and measured performance stays
within the declared variance envelope.

- [ ] **Step 6: Commit baselines**

Commit with intent `Freeze honest accuracy and resource baselines before optimization begins` and the exact report hash.

---

### Task 16: Experiment Records and Constrained Hill Climber

**Files:**
- Create: `evaluation/experiments.py`
- Create: `evaluation/hill_climb.py`
- Create: `tests/evaluation/test_experiments.py`
- Create: `tests/evaluation/test_hill_climb.py`
- Create: `evaluation/search_spaces/v1.json`
- Create: `tests/evaluation/fixtures/tuning/train.json`
- Create: `tests/evaluation/fixtures/tuning/dev.json`
- Create: `tests/evaluation/fixtures/tuning/manifest.json`
- Create: `tests/evaluation/fixtures/three-candidates.json`

- [ ] **Step 1: Write experiment-record tests**

Require dataset/baseline hashes, git revision, complete configuration, code-path
candidate ID, train/dev metrics, resource metrics, gate decisions, parent
experiment, and deterministic ordering. Reject holdout fields and holdout paths.

- [ ] **Step 2: Write hill-climber policy tests**

Use synthetic candidates to prove lexicographic behavior: offset failures always
lose; precision/contradiction regressions lose; resource-budget violations lose;
recall wins among survivors; status F1, retrieval, and latency break ties in that
order. Test resume and duplicate-candidate suppression.

- [ ] **Step 3: Verify RED**

Run:

```bash
uv run pytest tests/evaluation/test_experiments.py tests/evaluation/test_hill_climb.py -q
```

Expected: imports fail because experiment and hill-climb modules do not exist.

- [ ] **Step 4: Implement candidate generation**

Start with bounded coordinate/grid neighborhoods over public `CitationConfig`
fields, candidate limits, weights, windowing, and alignment scores. Represent
code variants as explicit named flags only after a tested implementation exists;
do not generate arbitrary source patches.

- [ ] **Step 5: Implement gated selection and persistence**

Evaluate candidates on train, discard clear failures early, then evaluate
survivors on dev. Persist every decision atomically. The package must reject
`split == "holdout"`, encrypted holdout paths, and release-gate result files as
optimizer input.

- [ ] **Step 6: Verify GREEN**

Register `tune` in `evaluation.cli`, then run:

```bash
uv run pytest tests/evaluation/test_experiments.py tests/evaluation/test_hill_climb.py tests/evaluation/test_cli.py -q
uv run python -m evaluation.cli tune --tuning-bundle tests/evaluation/fixtures/tuning --search-space tests/evaluation/fixtures/three-candidates.json --output /tmp/cite-right-tune-smoke.json
```

Expected: all tests pass and the smoke search selects the known gated winner.

- [ ] **Step 7: Commit optimization machinery**

Commit with intent `Improve recall only after correctness and resource gates pass`, `Directive: Never add holdout feedback to experiment records`, and tests.

---

### Task 17: Run the First Hill-Climbing Cycle and Implement Winning Improvements

**Files:**
- Modify as evidence requires: `src/cite_right/citations.py`
- Modify as evidence requires: `src/cite_right/core/prepared_corpus.py`
- Modify as evidence requires: `src/cite_right/core/aligner_py.py`
- Modify as evidence requires: `rust_core/src/smith_waterman.rs`
- Modify: targeted existing tests under `tests/`
- Create: `evaluation/reports/v1/candidate.json`

- [ ] **Step 1: Run configuration-only search**

Search the bounded configuration neighborhood. Preserve all experiment records.
Freeze the best config-only candidate that passes every gate.

Run:

```bash
uv run python -m evaluation.cli tune --tuning-bundle evaluation/data/v1/tuning --search-space evaluation/search_spaces/v1.json --output evaluation/experiments/v1
```

Expected: the experiment store contains every evaluated configuration and one
frozen best config-only candidate.

- [ ] **Step 2: Diagnose remaining train/dev failures**

Group failures by dataset family and root cause. Select only fixes supported by
multiple cases or a clear invariant. Do not add case-specific rules, source text,
case IDs, or transformation names to production code.

- [ ] **Step 3: For each implementation change, follow TDD**

Add the smallest general regression test to the existing library test suite,
observe RED on the current implementation, implement the general fix, observe
GREEN, then rerun the complete evaluation gates. Use
`superpowers:test-driven-development` for every behavior change.

- [ ] **Step 4: Re-run bounded search after each accepted implementation**

Keep only changes that pass precision, contradiction, offset, latency, and memory
gates and improve dev recall or a higher-priority tie-breaker. Revert rejected
experimental changes without touching unrelated user work.

- [ ] **Step 5: Freeze one candidate**

Record exact git revision, complete configuration, train/dev raw counts,
confidence intervals, resource samples, baseline deltas, and why it won in
`candidate.json`.

- [ ] **Step 6: Commit winning improvements and candidate report**

Use one Lore commit per independent production change and a final report commit.
Every commit records rejected alternatives, scope risk, tests, and known gaps.

---

### Task 18: Sealed Holdout Release Gate

**Files:**
- Create: `evaluation/release_gate.py`
- Create: `tests/evaluation/test_release_gate.py`
- Create: `evaluation/reports/v1/holdout.json`

- [ ] **Step 1: Write release-gate isolation tests**

Test frozen-revision/config verification, external-key requirement, ephemeral
decryption, public-manifest verification, aggregate-only output, failure cleanup,
no experiment-store access, one-run receipt, and refusal to run a non-frozen
candidate.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_release_gate.py -q`

- [ ] **Step 3: Implement isolated release execution**

Decrypt into a mode-0700 temporary directory outside the repository. Verify
canonical hash and private manifest, run correctness and performance gates once,
write an aggregate signed report and receipt, then delete plaintext in `finally`.
Do not serialize case-level labels, outputs, or failures to the report.

- [ ] **Step 4: Verify GREEN on fixture artifacts**

Register `release-gate` in `evaluation.cli`, then run:

```bash
uv run pytest tests/evaluation/test_release_gate.py tests/evaluation/test_cli.py -q
```

Expected: all normal, injected-failure, and interrupted-cleanup fixtures pass.

- [ ] **Step 5: Run the real sealed holdout once**

Provide external keys, verify the frozen candidate, execute the release gate,
and record pass/fail relative to the frozen baseline. Do not tune after reading
the aggregate result.

Run:

```bash
CITE_RIGHT_HOLDOUT_KEY_FILE="$CITE_RIGHT_HOLDOUT_KEY_FILE" CITE_RIGHT_ATTESTATION_KEY_FILE="$CITE_RIGHT_ATTESTATION_KEY_FILE" uv run python -m evaluation.cli release-gate --candidate evaluation/reports/v1/candidate.json --sealed-holdout evaluation/data/v1/holdout.aesgcm --public-manifest evaluation/data/v1/holdout.public.json --output evaluation/reports/v1/holdout.json
```

Expected: one aggregate signed report and one run receipt; no plaintext holdout
or case-level output remains in the repository or experiment store.

- [ ] **Step 6: Commit the aggregate holdout report**

Commit with intent `Verify the frozen candidate on attribution cases unavailable during tuning`, include report and manifest hashes, and state any failed gate honestly.

---

### Task 19: Documentation, Removal of Circular Benchmarks, and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/advanced/performance-tuning.md`
- Create: `docs/advanced/evaluation.md`
- Modify: `mkdocs.yml`
- Create: `tests/evaluation/test_documentation.py`
- Remove after migration: `benchmark_precision.py`
- Remove after migration: `optimize_precision.py`
- Remove after migration: `evaluation_dataset.json`
- Remove after migration: `gold_dataset.json`
- Remove after migration: `gold_dataset_candidate.json`
- Remove after migration: `precision_benchmark_dataset.json`
- Remove after migration: `extract_dataset.py`
- Remove after migration: `analyze_dataset.py`
- Remove after migration: `curate_and_enrich_gold_dataset.py`
- Remove after migration: `verify_gold_dataset.py`

- [ ] **Step 1: Write documentation assertions first**

Add tests proving README claims include observed counts, dataset version, and
confidence intervals; no generalized zero-false-positive language remains; and
the old circular benchmark artifacts are no longer referenced.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/evaluation/test_documentation.py -q`

Expected: failure while generalized claims and obsolete references remain.

- [ ] **Step 3: Document the evaluation contract and results**

Explain strict attribution, retrieval separation, dataset composition, leakage
policy, review process, metric math, performance protocol, hill-climbing gates,
sealed holdout lifecycle, baseline, and final aggregate holdout result. State
limitations and optional-dependency coverage.

- [ ] **Step 4: Remove obsolete circular artifacts only after migration**

First prove the new evaluator covers every useful scenario family from the old
files. Then remove the untracked/obsolete scripts and datasets listed above. Do
not delete unrelated user artifacts.

- [ ] **Step 5: Run completion verification**

Run:

```bash
uv run pytest -q
uv run ruff check src tests evaluation
uv run pyright
uv run bandit -q -c pyproject.toml -r src evaluation
uv run radon cc src evaluation -s -n C
cargo test --manifest-path rust_core/Cargo.toml
cargo clippy --manifest-path rust_core/Cargo.toml --all-targets -- -D warnings
uv run --group docs mkdocs build --strict
uv run python -m evaluation.cli validate --bundle evaluation/data/v1
uv run python -m evaluation.cli verify-public-manifest --bundle evaluation/data/v1
git diff --check
```

Expected: all tests and static checks pass; optional-dependency skips are listed;
dataset and public holdout verification pass; docs build strictly; no whitespace
errors remain.

- [ ] **Step 6: Conduct final independent reviews**

Run `code-review`, security review of sealing/key handling, and verification of
all spec acceptance criteria. Fix and repeat until no blocking findings remain.

- [ ] **Step 7: Commit documentation and migration**

Commit with intent `Replace circular accuracy claims with reproducible strict-attribution evidence`, list removed artifacts, report hashes, verification, and remaining risks.

---

## Completion Gate

Do not declare the objective complete until all of the following are proven by
current artifacts and command output:

- Dataset v1 contains 725-775 valid cases with grouped train/dev/holdout splits.
- Every dev claim is reviewed and the signed public holdout attestation proves
  every holdout claim is reviewed.
- Dataset validation, leakage detection, canonical regeneration, and manifest
  verification pass.
- Golden tests prove matching and metric calculations.
- Baseline reports cover required configurations, backends, and performance
  strata.
- At least one full train/dev hill-climbing cycle produces a frozen candidate.
- The candidate passes precision, contradiction, offset, p95 latency, and memory
  gates while improving recall or a higher-priority approved tie-breaker.
- The sealed holdout is executed once for that frozen candidate and the aggregate
  report shows no correctness or resource-gate regression.
- Full Python/Rust tests, lint, types, security, complexity, docs, and independent
  reviews are clean.
