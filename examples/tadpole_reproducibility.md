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
- `cohort_balanced.json`: aggregate before/after counts and removals by sequence
  type and visit-history group, with the balancing policy version and seed.
- `runs.jsonl`: configuration, fold, aggregate metrics, partition sizes and a
  hashed partition identifier; each record has an experiment and run fingerprint.
- `summary.csv` and `summary.md`: results, per-metric finite-fold counts, and
  completed/missing-run coverage. Partial metric coverage is reported even
  when all runs have finished; see the metric conventions below.

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

## Copy-last-diagnosis baseline

`reproduce`, `paper_setup`, and `quick` include `raw/copy_last` and
`balanced/copy_last`; `all` runs each baseline once. They use exactly the
patient splits and sequence policy of the corresponding learned comparisons.
Run only these configurations without training:

```bash
python examples/tadpole_ad_progression_prediction_visit_transformer.py \
  --root /authorized/local/tadpole --experiment reproduce --only copy_last \
  --output-dir output/tadpole-copy-last
```

Omit `--root` to check the workflow with synthetic data. A fresh output directory
is required after code changes, as for other experiments.

The prediction is the last **historical** diagnosis; the held-out diagnosis is
used only for evaluation. No features, visit dates, fitted preprocessing, model
filling, or training/validation labels are used to make predictions. Partition
sizes describe the comparison split; `epochs`, `n_params`, and `train_seconds`
are zero. Scores are one-hot decisions rather than calibrated probabilities.
The existing metrics handle ties in these hard scores; AUC should therefore be
interpreted as discrimination of the hard rule, not learned risk ranking.

With the default progression-only protocol, stable accuracy should be 1 and
converter accuracy 0, while overall accuracy is the stable fraction. This is a
sanity check, not evidence of clinical predictive performance. The baseline is
an additional team comparison, not one of the paper's learned models. It does
not establish that the other evaluation metrics have been fully validated.

## Model implementation assumptions and fixes

`reproduce` uses the default **encoder-only team variant** (width 128, 8 heads,
4 layers and masked mean pooling). `paper_setup` and M3a use the reported paper
dimensions (width 256, 4 heads, 4 encoder and 8 decoder layers), but their
`paper_transformer` / `PAPER_ARCH` names are configuration identifiers, not a
claim of exact author-code equivalence.

The learned decoder query and its readout, learned absolute positions, pre-norm
blocks, final encoder/decoder layer norms, feed-forward width 512 and dropout
0.3 are team assumptions. With the default task encoding (22 measurements,
3 diagnosis columns, one time column), 3 output classes and `max_visits=64`,
the paper-oriented model has **8,460,035 parameters**. The paper reports
**6,603,267**. This discrepancy remains unresolved; matching the headline layer
counts does not resolve the other architectural choices. Weighted CE and focal
loss are team ablations.

The team retains the current model architectures. Matching a reported parameter
total alone cannot determine the authors' decoder inputs, readout, normalization
or feed-forward width. The settings established by the paper and the remaining
team choices above define the scope of this reconstruction.

The current implementation independently initializes encoder/decoder matrices
with Xavier uniform weights, zero biases and unit layer-normalization scales.
The decoder query uses a normal distribution with standard deviation 0.02;
input projection, learned positions and the classifier keep PyTorch defaults.
The initializers are reproducible under the caller's Torch seed but have not
been verified against the authors' initialization scheme.

TST is an adaptation of the cited baseline. Its batch normalization now uses
only valid visits across patients and time; padding is excluded from moments,
outputs and input gradients. Other patients' real visits can still affect a
training prediction. A batch with one valid visit uses stored running statistics
(initially zero mean and unit variance), skips moment/counter updates and retains
gradients. Masked normalization and this singleton fallback are team adaptations,
not established choices in the Alzheimer's paper. Empty histories retain the
existing synthetic zero-token fallback for finite predictions.

Dimensions and layer counts must be positive integers. Dropout must be finite
and in `[0, 1]`, and focal gamma must be finite and nonnegative. Explicit class
weights must be a finite, strictly positive vector in label-processor order;
they also weight `ce` when supplied. Automatic weights use only the dataset
passed to the model, which should be the training split. Weighted CE normalizes
by target-weight sum; focal loss averages weighted terms over the batch size.
Focal loss also keeps gradients finite at saturated correct predictions for
fractional gamma.

Both `forward` and `encode` fill NaN measurements with zero. After masking and
per-patient history truncation, infinite retained measurements raise `ValueError`
before any projection, attention or batch-normalization updates. Discarded visits
cannot change predictions. Regression tests cover positive/negative infinity,
masked and truncated invalid visits, and NaN-imputed batches with finite gradients
for all three architectures.

The tensor processor's zero imputation preserves positive and negative infinity;
it replaces only NaN, so invalid held-out measurements reach the model's input
validation instead of becoming extreme finite values. Normalization fitting
rejects infinite training measurements and non-finite fitted statistics. The
task's finite-value cache version rejects older processed caches that concealed
infinity. Processor-to-dataloader-to-model regressions cover all three
normalization choices and model architectures, including finite predictions,
losses and gradients for NaN-imputed batches. Normalization learns from the
training patients only; invalid training measurements fail before fitting.

These fixes change initialization and TST training behavior. Start experiments
in a fresh output directory; historical tables do not describe these updated
models. The API documentation contains the detailed assumption list:
`docs/api/models/pyhealth.models.VisitTransformer.rst`.

M1 compares full recurrent training protocols: input representation, imputation,
dropout, learning rate and stopping rules change along with the objective.
It cannot isolate a causal effect of the objective alone. Early stopping still
uses macro recall rather than the final sensitivity/specificity-based BCA, as
recorded above. Synthetic regression checks validate software behavior and do
not establish real-data reproduction results.

## Balancing and fold policy version 2

Balancing uses the smaller stable/converter count in each exact visit-history
group. The larger type is randomly downsampled with the experiment seed. Groups
missing either type are omitted, and no patient is duplicated. Where stable
sequences are sufficient, all converters remain and only stable sequences are
discarded, as described in the paper. Downsampling converters in scarce-stable
groups and omitting unpaired groups are explicit team rules to guarantee equal
counts instead of silently labeling an imbalanced cohort as balanced.

`cohort_balanced.json` records aggregate before/after counts and removed counts.
It contains no patient identifiers. This balancing applies to the selected
cohort; individual train/validation/test partitions are approximately stratified
and need not each contain exactly equal type counts.

Test patients are assigned by seeded round-robin across sequence-type/history
strata, continuing the allocation across stratum boundaries. Total test-fold
sizes and each stratum's test-fold counts differ by at most one patient. Groups
above 6 are still pooled for stratification. Every selected patient occurs in
exactly one test fold, and train, validation and test patients are disjoint.

Within each fold, the global validation target is the rounded non-test count
times `--val-ratio`, clamped to leave at least one training and one validation
patient. Largest-remainder allocation distributes that count proportionally
across the available strata, using seeded tie-breaking. This avoids rounding
every small stratum's validation allocation to zero. Unlike the paper's stated
temporal split, this implementation uses randomized patient-level folds and a
team validation holdout; it does not establish an exact paper split reproduction.

If the selected cohort cannot support the requested number of folds with three
nonempty partitions, planning raises an explicit error suggesting fewer folds
or more patients. Fold counts are never silently reduced, and training does not
fall back to using test patients as validation. Learned runs also reject empty
processed datasets before constructing a model. The known clinical output space
is fixed to CN/MCI/AD, so a missing training class does not remove its output or
make held-out labels unprocessable; feature statistics remain training-only.

The manifest records separate fold and balancing policy versions. These changes
alter patient selections, splits and label indices. Use a fresh output directory;
results from earlier policies must not be merged with the updated runs.

The previously failing small synthetic run can be checked without ADNI data:

```bash
python examples/tadpole_ad_progression_prediction_visit_transformer.py \
  --experiment quick --synthetic-patients 40 --device cpu \
  --output-dir /tmp/tadpole-small-check
```

## Metric and summary conventions

The source for the metric definitions is the paper's **Section 2.5, Training
and Evaluation**, not Section 4.3:
[Moghaddami et al.](https://proceedings.mlr.press/v287/moghaddami25a.html).
The published description does not settle every averaging or undefined-value
convention. The following rules describe this implementation and are covered
by numerical reference tests; they do not establish author-code equivalence.

- Sensitivity and specificity use one-vs-rest confusion counts and macro
  averages over the fixed CN/MCI/AD classes. A zero sensitivity denominator
  (no positives) or specificity denominator (no negatives) contributes 0,
  by team convention. BCA is half the sum of these two macro averages.
  Perfect predictions on an MCI/AD-only converter subset therefore have
  BCA 5/6 under this convention. This is not a claim that the paper used the
  same zero-denominator rule.
- `f1_weighted` averages class F1 values by true-label support; `f1_macro`
  averages them equally. Both include only classes present in the true labels,
  with zero-division terms set to 0. A predicted class absent from the true
  labels is not an additional macro-F1 term, but its errors still contribute
  to the observed classes' confusion counts. The overview table explicitly
  labels its F1 column as weighted. The paper does not establish this choice.
- Hand-Till mAUC averages both directional rank AUCs for each observed class
  pair, using the original class scores without pairwise renormalization.
  A tied positive/negative pair receives half credit. Absent-class pairs are
  omitted by team convention; with fewer than two observed classes, mAUC is
  undefined (`NaN`). An observed-pair subset AUC does not establish a full
  three-class AUC.
- T4 ranks the summed probability of stages more severe than the last
  historical diagnosis, excluding subjects already at AD. Detection AUROC
  requires both converter and nonconverter subjects among those at risk.
  Sensitivity at 90% specificity is the largest empirical ROC sensitivity
  with false-positive rate at most 0.10; no interpolation is used, and tied
  scores move together. Undefined detection metrics are omitted.

`summary.csv` keeps `folds` as the number of completed runs and adds
`expected_folds` and a JSON list of `missing_folds`. Each recorded metric has
`<metric>_mean`, `<metric>_std` and `<metric>_n_valid` columns. All overview
metrics receive these columns even when absent from every run. Missing metrics,
`NaN`, and positive/negative infinity are excluded independently per metric.
The expected denominator is the configuration's requested fold count; a
completed run with an undefined metric is not a valid observation for that
metric. Means are blank when no finite values exist. Standard deviations are
blank with fewer than two finite values; for two or more, the existing
population-SD convention (`ddof=0`) is retained.

The Markdown overview shows `n=valid/expected` in every metric cell. Its
incomplete-coverage table also covers CSV-only metrics and distinguishes
undefined/missing metrics in completed folds from whole runs that have not
finished. An example is converter mAUC with one valid and one undefined fold:
the mean has coverage 1/2 and SD is unavailable, rather than appearing as
`mean ± 0` for a two-fold estimate.

Welch comparisons require at least two finite values in each configuration.
The table reports both effective fold counts; if either the statistic or its
p-value is non-finite, both are shown as `NA`. These comparisons preserve the
existing unpaired test and do not establish that overlapping CV training folds
are independent.

Manifests record `metric_policy_version` and `summary_format_version`.
Synthetic validation outputs are labeled accordingly. Store new validation
runs in a fresh directory; they are separate evidence from existing teammate
results. Reference tests use hand-calculated confusion matrices, asymmetric
pair rankings, ties, absent classes and incomplete metric coverage.

## Current synthetic model validation

Fresh quick and model-ablation results, reproduction commands, and test outcomes
are documented in [tadpole_model_validation.md](tadpole_model_validation.md).
Its companion `tadpole_validation/` files contain aggregate synthetic runs,
manifests and CSV summaries suitable for inclusion with the contribution. They
are independent of the historical teammate tables, which remain unchanged.
