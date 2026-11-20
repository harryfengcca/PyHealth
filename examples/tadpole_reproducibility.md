# TADPOLE cohort and experiment provenance

Run an aggregate cohort audit without training or fitting feature processors:

```bash
python examples/tadpole_ad_progression_prediction_visit_transformer.py \
  --root /authorized/local/tadpole --audit-only --output-dir output/tadpole-audit
```

Omit `--root` for a synthetic audit. The JSON report counts parsed subjects and
visits, D1/D2 membership, exclusions, sequence types, history groups, target
diagnoses, and transitions. Its disjoint visit ledger accounts for every parsed
visit. CSV rows rejected by the dataset loader are outside this population.
Unknown diagnosis strings are counted separately from missing diagnoses.

The current protocol uses every subject in the supplied file. D1/D2 counts are
descriptive: the audit does not silently change cohort selection, diagnosis
mapping, or the sequence rules. Run this audit on authorized data to investigate
the proposal's 1,548 sequences versus the paper's 1,627. Synthetic counts cannot
resolve that discrepancy.

## Resuming experiments

Each experiment directory contains:

- `manifest.json`: source-file SHA-256, Git revision, actual Python/YAML source
  fingerprints, effective training/split settings, Python and dependency versions.
- `cohort_default.json` and, when used, `cohort_keep_reverters.json`: aggregate
  accounting for each sequence protocol.
- `runs.jsonl`: configuration, fold, aggregate metrics, partition sizes and a
  hashed partition identifier; each record has an experiment and run fingerprint.
- `summary.csv` and `summary.md`: results and explicit completed/missing-fold
  coverage. Partial summaries are not complete cross-validation results.

A changed source, executable code, Git revision, package version, seed or training
setting requires a new `--output-dir`. Existing results are never overwritten to
resolve a provenance conflict. Legacy results without manifests are rejected;
retain them separately as unverified historical records.

`--only`, `--fold`, output locations and cache locations do not change the
experiment identity. This supports resuming subsets and separate fold jobs with
identical effective settings. Writers serialize result updates under a lock and
atomically replace complete artifacts, avoiding partial records and duplicate
folds. `--summarize-only` verifies stored fingerprints and requires no source data.
Checksums identify runs; they are not cryptographic authentication of metrics.
Commit reviewed implementation code before real experiments: source fingerprints
identify uncommitted edits but do not archive a recoverable patch.

Synthetic source directories are separated by code fingerprint, seed and cohort
size. Processed caches are namespaced by source/code fingerprints, preventing
stale data or preprocessing reuse after changes. `--cache-dir` selects the parent
of these cache directories; the default is `output/tadpole/private_cache`.

## Restricted data and interpretation

Reports and run records contain aggregate counts and hashed split identifiers,
not patient IDs or clinical measurements. Dataset/task caches contain restricted
patient-level data: keep caches and authorized source files in controlled private
storage and out of the public PR. Review even aggregate artifacts before sharing.
Do not publish a whole output directory.

The historical numerical tables in the example docstring have no accompanying
run artifacts and remain unverified. This change establishes traceability; it does
not reproduce their numbers or establish paper-model equivalence. Early stopping
still monitors PyHealth's `balanced_accuracy` (macro recall), while final metrics
include the paper's sensitivity/specificity-based BCA; the manifest records this
choice explicitly.

## Sequence policy version 2

The task, cohort audit, and monthly baselines share `prepare_patient_visits`.
The original source rows are unchanged. Explicit destination-lower-than-origin
transition strings trigger reverter exclusion even when observed stages do not
show that reversal. Reversion checks cover the full resolved history before
first-conversion truncation. `drop_reverters=False` retains these subjects but
labels still describe the selected observed destination diagnoses.

Identical selected visit fields collapse. Complementary measurements merge only
for matching timestamps and nonempty visit codes with agreeing overlapping
values. Diagnosis comparison uses canonical stages; numeric comparison uses the
existing parser, including its censored-value bound substitution. Transition
flags from every source row are preserved before merging. No values are averaged
and no later visit supplies measurements for a different visit.

Different visit codes at one timestamp have no reliable temporal order and
exclude the subject as `ambiguous_same_day_visits`. Unidentified nonidentical
records receive the same exclusion. Contradictory records with a matching visit
identity exclude the subject as `conflicting_duplicate_visit`. Manual source
review can resolve such cases separately; CSV row order is never used to choose
a diagnosis. This duplicate policy is a documented team deviation because the
paper does not specify duplicate resolution.

Preparation diagnostics may overlap. The disjoint visit ledger counts all rows
of excluded subjects as excluded; collapsed duplicates are a separate ledger
bucket only for retained subjects. Thus no row is counted twice. The task's
sequence-policy cache version changes automatically, and experiment code
fingerprints reject resumes from the previous policy. Use a fresh output
directory for the new audit and experiments, preserving earlier reports.
