# Current synthetic model validation

These fresh runs validate the current team implementation using generated data.
The historical teammate results in the example remain unchanged. These runs
provide functional and ablation evidence; they do not estimate real-data
performance or establish equivalence to the paper.

## Setup

- Source: 40 subjects generated with seed 42; 35 sequences remain after the
  default exclusion policy. The balanced cohort contains 18 sequences.
- CPU, one Torch/OpenMP thread, seed 42, two patient-level folds, and two epochs
  for learned models and monthly imputers. Copy-last requires no training.
- Validation fraction 0.15, batch size 32, Adam learning rate 1e-4 and weight
  decay 1e-5 for next-visit models; M3f changes the learning rate to 1e-3.
  Validation macro recall selects the checkpoint. It is distinct from final BCA.
- Monthly baselines use `MonthlyConfig` (two layers of 128 units, batch 128,
  learning rate 1.290666e-3), with the epoch budget reduced to two.
- Normalization and model filling learn only from training patients.
- Quick: 7 configurations / 14 runs. Model ablation: 9 / 18.
  Paper setup: 12 / 24. Every configuration completed both folds.

The chosen encoder, encoder-decoder and TST structures are retained. The
default encoder has 805,379 parameters with 26 input columns and 3 classes.
The encoder-decoder configuration has 8,460,035 parameters, compared with
6,603,267 reported by the paper. Matching a parameter total cannot identify
the missing architectural details; this document makes no exact-author-model
claim. See [implementation assumptions](tadpole_reproducibility.md#model-implementation-assumptions-and-fixes).

## Model ablation comparison

Only model or optimizer settings change across M0/M2/M3; the task and patient
splits are fixed. M3a changes architecture, width, heads and decoder depth
together, so it measures a configuration change rather than one isolated factor.
M2 weighted CE and focal loss are team extensions. The paper already explores
architecture hyperparameters; M3 is a sensitivity check, not a novel extension.

Values are mean ± population SD (`ddof=0`). Every metric in this table has
two finite values out of two expected folds. Per-metric counts for all metrics
are included in the CSV and complete summaries below.

| Configuration | Parameters | Overall mAUC | Overall BCA | Converter accuracy | T4 AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 default (encoder, d128, ce) | 805,379 | 0.437 ± 0.068 | 0.577 ± 0.014 | 0.167 ± 0.167 | 0.325 ± 0.025 |
| M2a weighted cross-entropy | 805,379 | 0.447 ± 0.046 | 0.562 ± 0.001 | 0.167 ± 0.167 | 0.300 ± 0.000 |
| M2b focal loss | 805,379 | 0.434 ± 0.071 | 0.559 ± 0.004 | 0.167 ± 0.167 | 0.317 ± 0.033 |
| M3a paper encoder-decoder (d256, h4, 4+8 layers) | 8,460,035 | 0.693 ± 0.028 | 0.551 ± 0.020 | 0.417 ± 0.417 | 0.279 ± 0.154 |
| M3b embedding 64 | 338,179 | 0.596 ± 0.122 | 0.500 ± 0.000 | 0.917 ± 0.083 | 0.392 ± 0.192 |
| M3c embedding 256 | 2,132,995 | 0.558 ± 0.044 | 0.544 ± 0.020 | 0.517 ± 0.317 | 0.292 ± 0.142 |
| M3d 2 layers | 408,835 | 0.542 ± 0.043 | 0.482 ± 0.029 | 0.383 ± 0.217 | 0.404 ± 0.029 |
| M3e dropout 0.1 | 805,379 | 0.461 ± 0.075 | 0.596 ± 0.002 | 0.417 ± 0.417 | 0.333 ± 0.017 |
| M3f learning rate 1e-3 | 805,379 | 0.797 ± 0.041 | 0.651 ± 0.006 | 0.517 ± 0.317 | 0.683 ± 0.183 |

## Observations and limits

- M3f reaches overall BCA 0.651 ± 0.006 versus 0.577 ± 0.014 for M0 in
  this two-epoch check. It shows that optimizer settings affect this short
  run; it does not establish a preferred learning rate for real data.
- Weighted CE and focal loss execute successfully but do not improve overall
  BCA over M0 in this check (0.562 and 0.559 versus 0.577).
- M3b has converter accuracy 0.917 but stable accuracy 0.250 and BCA 0.500.
  Converter accuracy alone is insufficient to assess the model.
- Converter mAUC has only one finite fold in the raw-cohort configurations:
  the other converter subset has fewer than two observed classes. Its SD is
  unavailable. Complete run coverage does not imply complete metric coverage.
- All reported rankings are descriptive. Two folds and two epochs are too
  small a budget to infer clinical performance or model superiority.

## Reproduce

Run from the repository root in a Python environment with PyHealth installed.
Each output directory must be fresh for the current code and environment.

```bash
OMP_NUM_THREADS=1 python examples/tadpole_ad_progression_prediction_visit_transformer.py \
  --experiment quick --synthetic-patients 40 --device cpu --seed 42 \
  --epochs 2 --monthly-epochs 2 --output-dir output/tadpole-model-validation

OMP_NUM_THREADS=1 python examples/tadpole_ad_progression_prediction_visit_transformer.py \
  --experiment model_ablation --synthetic-patients 40 --device cpu --seed 42 \
  --folds 2 --ablation-folds 2 --epochs 2 --monthly-epochs 2 \
  --output-dir output/tadpole-model-validation

OMP_NUM_THREADS=1 python examples/tadpole_ad_progression_prediction_visit_transformer.py \
  --experiment paper_setup --synthetic-patients 40 --device cpu --seed 42 \
  --folds 2 --ablation-folds 2 --epochs 2 --monthly-epochs 2 \
  --output-dir output/tadpole-model-validation
```

## Regression and documentation checks

- 223 related tests passed in 2.184 seconds, including
  34 model tests and 8 preprocessing and processor-to-model regressions.
  The expanded suite also covers normalization, sample building, schemas,
  processor transfer, partial processors and disk-backed data loading.
- 44 executable docstring examples passed (11 model, 16 task, 5 tensor
  processor and 12 shared processor utilities).
- Ruff, PEP8 whitespace/88-character checks, Black for the model and model
  tests, and the model/task/processor API documentation build passed.

Both `forward` and `encode` accept NaN as missing and fill it with zero.
Positive or negative infinity in retained visits raises `ValueError` before
projection, attention or normalization. Masked or truncated visits are ignored.
This input validation leaves the selected model architecture unchanged.

Zero imputation in `TensorProcessor` replaces only NaN and preserves signed
infinity. Infinite held-out measurements therefore reach model validation
instead of being silently converted to float32 extremes. Normalization
fitting rejects infinite training measurements and non-finite fitted centers
or scales. The task cache version prevents reuse of processed TADPOLE data
from the earlier imputation behavior.

End-to-end regressions exercise no normalization, z-score and min-max
normalization with all three model architectures. They verify rejection of
infinite retained values, finite outputs/losses/gradients for NaN-imputed
batches, exclusion of masked/truncated invalid visits, and reuse of training
statistics without fitting on held-out subjects.

Single-pass training iterators are also covered: the first valid sample
must contribute to the same statistics as a reusable collection, and an
infinite first sample must be rejected before normalization fitting.

Model regressions can be run independently:

```bash
python -m unittest discover -s tests/core -p test_visit_transformer.py -v
```

Run the processor-to-model regressions:

```bash
python -m unittest discover -s tests/core -p test_tensor_processor_nonfinite.py -v
```

## Aggregate evidence

- [Quick summary](tadpole_validation/quick/summary.md)
- [Model ablation summary](tadpole_validation/model_ablation/summary.md)
- [Paper-setup summary](tadpole_validation/paper_setup/summary.md)
- [Validation record and artifact checksums](tadpole_validation/validation.json)

Each experiment directory also contains `summary.csv`, `runs.jsonl`,
`manifest.json`, and aggregate cohort reports. The manifests record the
actual source fingerprint, dependency versions, settings and Git revision at
run time. These runs used committed implementation code; the source digest
identifies its executable contents. The evidence is committed separately
after the recorded implementation revision.
Implementation revision used for these runs: `ba5eac64c4c24dae4a45f1c28b235712ff580c61`.
No patient-level source data, processed caches or checkpoints are bundled.

Executable source SHA-256: `5b760223e890511a091e8c61c16fe922d782daee1fd3282b61e4c5e1cea36c90`.
