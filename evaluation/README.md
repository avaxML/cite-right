# Evaluation package

`evaluation` is development-only tooling for local benchmarking and dataset handling.
It is excluded from Cite-Right's public API and must not be treated as part of the
wheel-supported library surface.

Holdout private keys are sensitive material for sealed evaluation data. They must
never be committed to this repository.

## Dataset lifecycle

`evaluation.cli build` creates private staging candidates with empty review state.
Reviewers approve the development and holdout claims outside tuning code, then
`seal` encrypts the holdout and signs its redacted public manifest. `promote`
publishes the reviewed train/dev files, encrypted holdout, public attestation, and
source snapshots; `build-tuning-bundle` derives the review-free train/dev inputs.

The promoted `manifest.json` intentionally covers only the reproducible plaintext
train and development files. Holdout counts, ciphertext identity, review-completion
counts, and signature live in `holdout.public.json`. This separation lets the
repository verify one 750-case dataset without committing holdout plaintext or
private key material.
