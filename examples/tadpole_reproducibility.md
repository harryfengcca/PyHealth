# TADPOLE experiments: protocol notes

Companion to `tadpole_ad_progression_prediction_visit_transformer.py`. The
results and findings are in that script's docstring; this file records the
protocol choices the paper leaves open and how to run and resume experiments.

## Cohort

The task applies the paper's sequence rules (see
`pyhealth.tasks.ADProgressionPrediction`). Two cases the paper does not cover
are handled as follows:

- **Repeated records of one exam date** are merged when they agree; the
  subject is dropped when they conflict or carry different visit codes, since
  their order is unknown (3 of 1,737 subjects).
- **Transition strings** such as `"MCI to NL"` count as reversions even when
  the visit diagnoses alone do not show one (3 more subjects dropped).

On TADPOLE D1/D2 this gives 1,543 sequences (1,159 stable, 384 converter);
the paper reports 1,627. Write the cohort counts without training:

```bash
python examples/tadpole_ad_progression_prediction_visit_transformer.py \
  --root /path/to/tadpole --audit-only --output-dir output/tadpole-audit
```

The report holds aggregate counts only (exclusion reasons, sequence types,
history groups, transitions), no patient IDs.

## Balanced data and folds

- **Balanced data**: within each history group (number of input visits), the
  paper drops stable sequences until they match the converters. Where a group
  has fewer stable sequences than converters, we drop converters instead, and
  groups with only one type are left out (one group of 2 sequences).
- **Folds**: patient-level folds, stratified by sequence type and history
  group (groups above 6 pooled). 15% of the training patients are held out
  for early stopping. The paper uses a temporal split instead.
- **Early stopping** monitors PyHealth's `balanced_accuracy` (macro recall);
  the reported BCA is the paper's sensitivity/specificity-based one.

## Metrics

The paper (Section 2.5) defines mAUC (Hand and Till), F1, accuracy,
sensitivity, specificity and BCA, but not their edge cases. We use:

- sensitivity and specificity one-vs-rest, averaged over CN, MCI and AD; a
  class with no positives (or no negatives) in a subset counts as 0. This
  reproduces the paper's converter sensitivity of 0.430 = (0 + 0.645 +
  0.645) / 3. BCA = (sensitivity + specificity) / 2;
- F1 weighted by class support (`summary.csv` also has macro F1);
- mAUC over the class pairs present in the subset;
- T4 converter-detection AUROC on the summed probability of the stages more
  severe than the last input diagnosis, for subjects not yet at AD.

Summaries count the valid folds of every metric (`n=valid/expected`); a metric
that is undefined in a fold is left out of its mean.

## Copy-last baseline

`copy_last` predicts the diagnosis of the last input visit, with no training.
Its stable accuracy is 1 and its converter accuracy 0 by construction, so it
shows how much of each metric comes from stable sequences alone.

## Running, resuming and outputs

Each `--output-dir/<experiment>/` holds `manifest.json` (code, data and
settings fingerprints), `cohort_*.json` (aggregate counts), `runs.jsonl` (one
record per configuration and fold), `summary.csv` and `summary.md`.

- Finished runs are skipped on restart. Folds can run as separate processes
  (`--fold k`) that write to the same directory; writes are locked.
- A run with different code, data or settings is refused in an existing
  directory: commit the code first and use a new `--output-dir` after any
  change.
- `--summarize-only` rebuilds the summaries from `runs.jsonl`.

The processed caches under `--cache-dir` contain patient-level ADNI data. Keep
them and the source file private; share only the summaries.
