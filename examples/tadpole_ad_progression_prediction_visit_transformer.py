"""TADPOLE next-visit Alzheimer's stage prediction with VisitTransformer.

Contributors:
    Michael Mendes (NetID: mmendes3), Yiqun Ma (NetID: yiqunma3),
    Zhihao Feng (NetID: zhihaof2)

Paper:
    Mahdi Moghaddami, Clayton Schubring, Mohammad-Reza Siadat.
    "Transformer Model for Alzheimer's Disease Progression Prediction Using
    Longitudinal Visit Sequences." Proceedings of the Sixth Conference on
    Health, Inference, and Learning (CHIL 2025), PMLR 287:804-816.
    https://proceedings.mlr.press/v287/moghaddami25a.html

Description:
    Full pipeline, reproduction experiments and ablations.

Reproduction of Moghaddami, Schubring and Siadat, "Transformer Model for
Alzheimer's Disease Progression Prediction Using Longitudinal Visit
Sequences", CHIL 2025 (https://proceedings.mlr.press/v287/moghaddami25a.html),
plus ablations that are not in the paper.

Pipeline
--------
``TADPOLEDataset`` (data loading) -> ``ADProgressionPrediction`` (one stable
or converter sequence per subject) -> ``VisitTransformer``, PyHealth's
``RNN`` (LSTM / GRU) or the paper's month-by-month RNN baselines
(``tadpole_monthly_rnn.py``) -> evaluation with the paper's metrics.

Every run uses patient-level cross-validation, stratified by sequence type
and group (number of input visits). Each fold calls
``dataset.set_task(task, split=...)`` so feature normalization and
imputation statistics are fitted on the training patients only.

Experiments (``--experiment``)
------------------------------
``reproduce``
    VisitTransformer, LSTM and GRU on the raw and the balanced dataset
    (stable sequences dropped within each group until they match the
    converters), 10 folds. LSTM and GRU are PyHealth's ``RNN`` trained with
    the same next-visit objective as the Transformer.
``paper_setup`` (10 folds)
    The paper's own setup, on the same folds: its encoder-decoder
    Transformer and the TST (Zerveas et al., 2021), both on inputs imputed
    by "model filling" with a minimalRNN trained on the training patients
    of the fold; and its minimalRNN, LSTM and GRU baselines trained month by
    month as in Nguyen et al. Compare with the paper's Tables 2-5.
``task_ablation`` (new, not in the paper; 5 folds)
    Varies the task, with the model fixed. Every task configuration is run
    with PyHealth's existing ``RNN`` (LSTM) and with ``VisitTransformer``, so
    the effect of a task change can be read off an existing PyHealth model.
    T1 missing data: forward fill vs. no forward fill vs. forward fill with
    missing-value indicators. T2 time feature: none vs. months to target
    (paper) vs. months since the previous visit. T3 label definition: keep
    reverter subjects instead of dropping them.
``model_ablation`` (new, not in the paper; 5 folds)
    Varies the model, with the task fixed at the default. M2 class
    imbalance: cross-entropy vs. class-weighted cross-entropy vs. focal loss.
    M3 architecture and hyperparameters: the paper's encoder-decoder
    configuration, embedding size, depth, dropout, learning rate.
``all``
    The four above. ``quick`` runs a two-fold, two-epoch smoke test.

Every run also reports T4, converter detection: among subjects not yet at
AD, the score "probability of a more severe stage than now" is ranked
against the true converters (AUROC, and sensitivity at 90% specificity).
This tests whether a model's converter accuracy reflects better ranking or
only a shifted decision threshold.

Metrics follow the paper (Section 4.3): Hand-Till mAUC, F1, accuracy, and
per-subset sensitivity, specificity and balanced class accuracy (BCA). The
paper does not say how F1 is averaged; the table reports weighted F1 and
``summary.csv`` also has macro F1. Sensitivity and specificity are averaged
over all three classes, with a class that has no positives in the subset
counted as 0 (this reproduces, e.g., the paper's converter-subset
sensitivity of 0.430 = (0 + 0.645 + 0.645) / 3), and
BCA = (sensitivity + specificity) / 2.

Ablation M1 (new, not in the paper) compares the two ways of training the
recurrent baselines on the same folds: next-visit classification
(``reproduce``) and the paper's month-by-month forecasting
(``paper_setup``). It separates the effect of the training objective from
the effect of the architecture.

Usage
-----
Synthetic data (no ADNI access needed)::

    python examples/tadpole_ad_progression_prediction_visit_transformer.py \\
        --experiment quick

Real TADPOLE data on a GPU::

    python examples/tadpole_ad_progression_prediction_visit_transformer.py \\
        --root /path/to/tadpole --experiment reproduce --device cuda

Split folds across jobs with ``--fold k`` and aggregate afterwards with
``--summarize-only``. Finished runs are skipped on restart.

Outputs, in ``--output-dir/<experiment>/``: ``runs.jsonl`` (one line per
config and fold), ``summary.csv`` (mean and std per config) and
``summary.md`` (a readable table plus Welch t-tests against the
Transformer).

Results
-------
Setup: TADPOLE D1/D2 (1,548 sequences after the paper's rules: 1,163 stable,
385 converter), patient-level folds stratified by sequence type and group,
15% of the training patients held out for early stopping on validation BCA,
batch size 32, Adam lr 1e-4, weight decay 1e-5, up to 200 epochs with
patience 20, seed 42, one GH200 GPU (20-50 s per training run). Tables show
the mean over folds; standard deviations are in ``summary.csv``.
"Conv." is the converter subset; "T4 AUROC" is converter detection.

Reproduction (10 folds). Paper rows are from its Tables 2-5::

    Model                     mAUC    F1   Acc   BCA  Stable  Conv.  Conv.  T4
                                                       Acc    Acc    BCA  AUROC
    Raw, VisitTransformer     .957  .836  .840  .884  .920   .598   .559  .862
    Raw, LSTM (PyHealth)      .943  .826  .833  .879  .906   .611   .563  .835
    Raw, GRU (PyHealth)       .944  .832  .835  .881  .921   .577   .550  .840
    Raw, Transformer (paper)  .920  .824  .824  .874  .883   .648   .648    -
    Raw, LSTM (paper)         .857  .786  .789  .829  .973   .217   .476    -
    Raw, GRU (paper)          .846  .786  .792  .830  .984   .184   .464    -
    Bal., VisitTransformer    .917  .801  .812  .844  .838   .787   .657  .853
    Bal., LSTM (PyHealth)     .885  .779  .794  .831  .814   .774   .647  .828
    Bal., GRU (PyHealth)      .896  .787  .796  .821  .782   .810   .685  .844
    Bal., Transformer (paper) .883  .746  .802  .839  .814   .792   .713    -
    Bal., LSTM (paper)        .857  .786  .684  .770  .926   .439   .566    -
    Bal., GRU (paper)         .846  .786  .648  .758  .974   .323   .512    -

Findings. (1) VisitTransformer reaches the paper's overall numbers (raw BCA
.884 vs .874, mAUC .957 vs .920) and its balanced converter accuracy (.787
vs .792); on the raw data its converter accuracy is lower (.598 vs .648).
(2) The paper's large converter gap between the Transformer and the RNNs
(.648 vs at most .217) does not reproduce: PyHealth's LSTM and GRU, trained
with the same next-visit objective, reach .58-.61 (ablation M1). The gap in
the paper therefore appears to come from how its RNN baselines were
trained rather than from recurrence itself. (3) Welch's t-test across the
10 folds: the Transformer has significantly higher mAUC than LSTM (raw
p = .019, balanced p = .005) and GRU (raw p = .033), and ranks converters
better (T4 AUROC .862 vs .835-.840), but its BCA and converter accuracy
are not significantly different (p > .28); on the balanced data GRU has a
higher converter BCA (p = .037). (4) The converter sensitivity averaged over
three classes is lower than the paper's (.270 vs .430): on the raw data our
Transformer recognizes MCI-to-AD conversions well (sensitivity .739) but
CN-to-MCI conversions rarely (.072).

Task ablation (5 folds; the same splits for both models)::

                                      LSTM (PyHealth)     VisitTransformer
    Task configuration               BCA  Conv. T4 AUC   BCA  Conv. T4 AUC
                                          Acc                 Acc
    T0  paper task (default)         .884  .603  .826    .890  .592  .858
    T1a no forward fill              .879  .577  .821    .884  .563  .855
    T1b ffill + missing indicators   .881  .610  .826    .885  .615  .859
    T2a no time feature              .875  .564  .825    .882  .581  .853
    T2b months since prev. visit     .880  .592  .828    .889  .633  .858
    T3  keep reverters               .842  .578  .823    .850  .650  .851

Findings. Both models move in the same direction. Forward filling helps
converters (accuracy -.03 without it); explicit missing-value indicators
do not improve BCA. Dropping the time feature costs about .01 BCA. Months since
the previous visit, which does not reveal how far ahead the target visit
is, performs as well as the paper's months-to-target feature (BCA .889 vs
.890) and gives the Transformer its best converter accuracy among T0-T2
(.633). Keeping reverters lowers BCA by about .04 for both models: the
models almost never predict an improving diagnosis (reverter accuracy .07
for LSTM, .18 for the Transformer). Apart from T3, the differences are at
most .01 BCA, comparable to the fold-to-fold standard deviation
(.003-.011).

Model ablation (5 folds, VisitTransformer, task T0)::

    Configuration                 Params   BCA  Stable  Conv.  Conv.   T4
                                                 Acc    Acc    Sens  AUROC
    M0  encoder, d128, CE (def.)   0.81M  .890   .932   .592   .272   .858
    M2a class-weighted CE          0.81M  .887   .923   .600   .272   .852
    M2b focal loss                 0.81M  .883   .911   .620   .272   .857
    M3a paper enc.-dec. d256 4+8   8.46M  .888   .929   .589   .264   .864
    M3b embedding 64               0.34M  .886   .934   .555   .236   .853
    M3c embedding 256              2.13M  .886   .922   .602   .279   .858
    M3d 2 encoder layers           0.41M  .885   .918   .602   .269   .861
    M3e dropout 0.1                0.81M  .885   .926   .587   .301   .859
    M3f learning rate 1e-3         0.81M  .881   .908   .623   .317   .855

Findings. The model is insensitive to its configuration: every variant is
within .01 BCA of the default. The paper's encoder-decoder (8.46M
parameters here; the paper reports 6.6M with an unreported feed-forward
width) is no better than a 0.81M-parameter encoder on 1.5k sequences.
Class-weighted and focal losses, and a larger learning rate, trade stable
accuracy for converter accuracy without improving BCA. T0 and M0 are the
same configuration and give identical numbers, which confirms that runs
are deterministic for a given seed and split.

Synthetic smoke test (``--experiment quick``): the pipeline runs end to end
(data loading -> task -> training -> evaluation) and produces every metric.
"""

import argparse
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from pyhealth.datasets import SampleDataset, Split, TADPOLEDataset, get_dataloader
from pyhealth.datasets.tadpole import generate_synthetic_tadpole
from pyhealth.models import RNN, BaseModel, VisitTransformer
from pyhealth.tasks import ADProgressionPrediction
from pyhealth.tasks.ad_progression import DX_CLASSES
from pyhealth.trainer import Trainer
from tadpole_monthly_rnn import (
    MonthlyConfig,
    ModelFiller,
    MonthlyRNN,
    fit_standardizer,
    model_fill_table,
    predict_monthly_rnn,
    to_monthly,
    train_monthly_rnn,
)

logger = logging.getLogger("tadpole_example")

# --------------------------------------------------------------------------
# Experiment configurations
# --------------------------------------------------------------------------

PAPER_ARCH = {
    "architecture": "encoder_decoder",
    "embedding_dim": 256,
    "num_heads": 4,
    "num_layers": 4,
    "num_decoder_layers": 8,
}
# TST with the defaults of Zerveas et al. (2021). Every input is padded to
# ``max_visits``, which must cover the longest input sequence (12 visits).
TST_PARAMS = {
    "architecture": "tst",
    "embedding_dim": 64,
    "num_heads": 8,
    "num_layers": 3,
    "feedforward_dim": 256,
    "dropout": 0.1,
    "max_visits": 16,
}
MONTHLY_CELLS = {
    "minrnn_monthly": "minrnn",
    "lstm_monthly": "lstm",
    "gru_monthly": "gru",
}


@dataclass
class RunConfig:
    """One model + task + data configuration, run over several folds."""

    name: str
    model: str = "visit_transformer"
    model_params: Dict[str, Any] = field(default_factory=dict)
    task_params: Dict[str, Any] = field(default_factory=dict)
    data: str = "raw"
    lr: Optional[float] = None
    group: str = ""


def experiment_configs(experiment: str) -> List[RunConfig]:
    """Returns the configurations of one experiment.

    Args:
        experiment: ``"reproduce"``, ``"paper_setup"``, ``"task_ablation"``,
            ``"model_ablation"``, ``"quick"`` or ``"all"``.

    Returns:
        The run configurations, in the order they are run.
    """
    # Reproduction: the paper's three models on the raw and balanced data.
    reproduce = [
        RunConfig(f"{data}/{model}", model=model, data=data, group="reproduce")
        for data in ("raw", "balanced")
        for model in ("visit_transformer", "lstm", "gru")
    ]
    # The paper's own setup. "Model filling" replaces forward fill: missing
    # input values are predicted by a minimalRNN trained month by month on
    # the training patients of the fold. The RNN baselines are trained the
    # same month-by-month way (Nguyen et al.), not as next-visit classifiers.
    paper_fill = {"forward_fill": False, "model_fill": True}
    paper_setup = []
    for data in ("raw", "balanced"):
        paper_setup += [
            RunConfig(
                f"{data}/paper_transformer",
                model_params=PAPER_ARCH,
                task_params=paper_fill,
                data=data,
                group="paper_setup",
            ),
            RunConfig(
                f"{data}/tst",
                model_params=TST_PARAMS,
                task_params=paper_fill,
                data=data,
                group="paper_setup",
            ),
        ] + [
            RunConfig(
                f"{data}/{cell}_monthly",
                model=f"{cell}_monthly",
                data=data,
                group="paper_setup",
            )
            for cell in ("minrnn", "lstm", "gru")
        ]
    # Task ablation: one row per task variant; only ``task_params`` change.
    # T0 is the paper's task. T1 asks how much forward filling the missing
    # measurements matters (and whether explicit "was missing" flags help).
    # T2 asks whether the time feature carries signal and whether the
    # paper's "months to the target visit" (which reveals the prediction
    # horizon) beats a feature that only uses past visits. T3 changes the
    # label definition by keeping subjects whose diagnosis improves.
    task_variants = [
        ("T0 default (ffill, to_target, drop reverters)", {}, "T"),
        ("T1a no forward fill", {"forward_fill": False}, "T1"),
        (
            "T1b forward fill + missing indicators",
            {"missing_indicators": True},
            "T1",
        ),
        ("T2a no time feature", {"time_feature": "none"}, "T2"),
        (
            "T2b months since previous visit",
            {"time_feature": "since_previous"},
            "T2",
        ),
        ("T3 keep reverters", {"drop_reverters": False}, "T3"),
    ]
    # Each variant is trained with PyHealth's existing RNN (LSTM) and with
    # VisitTransformer, so a task effect is not tied to the new model.
    task_ablation = [
        RunConfig(f"{name} [{label}]", model=model, task_params=params, group=group)
        for model, label in (("lstm", "LSTM"), ("visit_transformer", "Transformer"))
        for name, params, group in task_variants
    ]
    # Model ablation: task fixed at T0; only the model or optimizer changes.
    # M2 targets the class imbalance (few converters, few AD targets).
    # M3 checks the paper's larger encoder-decoder and the sensitivity to
    # width, depth, dropout and learning rate.
    model_ablation = [
        RunConfig("M0 default (encoder, d128, ce)", group="M"),
        RunConfig(
            "M2a weighted cross-entropy",
            model_params={"loss": "weighted_ce"},
            group="M2",
        ),
        RunConfig("M2b focal loss", model_params={"loss": "focal"}, group="M2"),
        RunConfig(
            "M3a paper encoder-decoder (d256, h4, 4+8 layers)",
            model_params=PAPER_ARCH,
            group="M3",
        ),
        RunConfig("M3b embedding 64", model_params={"embedding_dim": 64}, group="M3"),
        RunConfig("M3c embedding 256", model_params={"embedding_dim": 256}, group="M3"),
        RunConfig("M3d 2 layers", model_params={"num_layers": 2}, group="M3"),
        RunConfig("M3e dropout 0.1", model_params={"dropout": 0.1}, group="M3"),
        RunConfig("M3f learning rate 1e-3", lr=1e-3, group="M3"),
    ]
    quick = [
        RunConfig("raw/visit_transformer", group="quick"),
        RunConfig("raw/lstm", model="lstm", group="quick"),
        RunConfig("balanced/visit_transformer", data="balanced", group="quick"),
        RunConfig("raw/minrnn_monthly", model="minrnn_monthly", group="quick"),
        RunConfig(
            "raw/tst",
            model_params=TST_PARAMS,
            task_params={"forward_fill": False, "model_fill": True},
            group="quick",
        ),
    ]
    table = {
        "reproduce": reproduce,
        "paper_setup": paper_setup,
        "task_ablation": task_ablation,
        "model_ablation": model_ablation,
        "quick": quick,
    }
    if experiment == "all":
        return reproduce + paper_setup + task_ablation + model_ablation
    return table[experiment]


# --------------------------------------------------------------------------
# Patient-level stratified cross-validation as a PyHealth Split
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientListSplit(Split):
    """Train / validation / test parts given as explicit patient ID lists.

    Patients outside all three lists (e.g. stable sequences dropped by the
    balanced dataset) are left out.
    """

    train: Tuple[str, ...]
    val: Tuple[str, ...]
    test: Tuple[str, ...]

    def split_indices(self, patient_to_index: Dict[str, List[int]]) -> List[np.ndarray]:
        """Maps the three patient lists to sample indices.

        Args:
            patient_to_index: Patient ID to the indices of its samples.

        Returns:
            Sorted sample indices for train, validation and test.
        """
        parts = []
        for pids in (self.train, self.val, self.test):
            idx = [i for pid in pids for i in patient_to_index.get(pid, [])]
            parts.append(np.array(sorted(idx), dtype=np.int64))
        return parts

    def to_dict(self) -> Dict[str, str]:
        """Returns a short, hashable description used in cache keys."""
        digest = hashlib.sha1(
            json.dumps([self.train, self.val, self.test]).encode()
        ).hexdigest()
        return {"kind": "patient_lists", "sha1": digest}


def plan_folds(
    meta: List[Dict[str, Any]],
    n_folds: int,
    val_ratio: float,
    seed: int,
    keep: Optional[Set[str]] = None,
) -> List[PatientListSplit]:
    """Assigns patients to folds, stratified by sequence type and group.

    Groups above 6 are pooled into one stratum because they are small.

    Args:
        meta: One dict per sequence with ``pid``, ``seq_type`` and ``group``.
        n_folds: Number of folds.
        val_ratio: Share of each fold's training patients used for
            validation (early stopping), per stratum.
        seed: Random seed.
        keep: If given, only these patients are used.

    Returns:
        One :class:`PatientListSplit` per fold.
    """
    rng = random.Random(seed)
    strata: Dict[str, List[str]] = {}
    for m in meta:
        if keep is not None and m["pid"] not in keep:
            continue
        key = f"{m['seq_type']}-{min(m['group'], 6)}"
        strata.setdefault(key, []).append(m["pid"])
    fold_of: Dict[str, int] = {}
    for key in sorted(strata):
        pids = sorted(strata[key])
        rng.shuffle(pids)
        offset = rng.randrange(n_folds)
        for i, pid in enumerate(pids):
            fold_of[pid] = (i + offset) % n_folds
    splits = []
    for k in range(n_folds):
        test = sorted(p for p, f in fold_of.items() if f == k)
        val: List[str] = []
        train: List[str] = []
        for key in sorted(strata):
            rest = sorted(p for p in strata[key] if p in fold_of and fold_of[p] != k)
            random.Random(f"{seed}-{k}-{key}").shuffle(rest)
            n_val = int(round(len(rest) * val_ratio))
            val += rest[:n_val]
            train += rest[n_val:]
        splits.append(
            PatientListSplit(tuple(sorted(train)), tuple(sorted(val)), tuple(test))
        )
    return splits


def balanced_patients(meta: List[Dict[str, Any]], seed: int) -> Set[str]:
    """The paper's balanced dataset: per group, as many stable as converters.

    Args:
        meta: One dict per sequence with ``pid``, ``seq_type`` and ``group``.
        seed: Random seed for choosing which stable sequences to keep.

    Returns:
        The patient IDs kept.
    """
    rng = random.Random(seed)
    keep: Set[str] = set()
    for group in sorted({m["group"] for m in meta}):
        conv = sorted(
            m["pid"]
            for m in meta
            if m["group"] == group and m["seq_type"] == "converter"
        )
        stable = sorted(
            m["pid"] for m in meta if m["group"] == group and m["seq_type"] == "stable"
        )
        rng.shuffle(stable)
        keep.update(conv)
        keep.update(stable[: len(conv)])
    return keep


# --------------------------------------------------------------------------
# Metrics (the paper's definitions)
# --------------------------------------------------------------------------


def hand_till_auc(y: np.ndarray, prob: np.ndarray, classes: Sequence[int]) -> float:
    """Multiclass AUC of Hand and Till (2001) over the classes present.

    Args:
        y: True class indices ``[N]``.
        prob: Class probabilities ``[N, C]``.
        classes: Class indices to consider.

    Returns:
        The mean pairwise AUC, or NaN if fewer than two classes are present.
    """
    present = [c for c in classes if np.any(y == c)]
    if len(present) < 2:
        return float("nan")
    pair_aucs = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            mask = (y == a) | (y == b)
            ya = (y[mask] == a).astype(int)
            auc_ab = roc_auc_score(ya, prob[mask, a])
            auc_ba = roc_auc_score(1 - ya, prob[mask, b])
            pair_aucs.append((auc_ab + auc_ba) / 2)
    return float(np.mean(pair_aucs))


def subset_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """The paper's metrics on one subset of the test sequences.

    Sensitivity and specificity are computed one-vs-rest for each of the
    three classes and averaged over all three; a class with no positives
    (or no negatives) in the subset contributes 0, as in the paper. BCA is
    ``(sensitivity + specificity) / 2``.

    Args:
        y: True class indices ``[N]`` in ``DX_CLASSES`` order.
        prob: Class probabilities ``[N, 3]`` in ``DX_CLASSES`` order.

    Returns:
        ``n``, ``accuracy``, ``sens``, ``spec``, ``bca``, ``mauc``,
        ``f1_weighted``, ``f1_macro`` and per-class ``sens_*`` / ``spec_*``.
    """
    pred = prob.argmax(axis=1)
    out: Dict[str, float] = {"n": int(len(y))}
    if len(y) == 0:
        return out
    labels = list(range(len(DX_CLASSES)))
    out["accuracy"] = float(np.mean(pred == y))
    sens_all, spec_all = [], []
    for c in labels:
        pos, neg = y == c, y != c
        sens = float(np.mean(pred[pos] == c)) if pos.any() else 0.0
        spec = float(np.mean(pred[neg] != c)) if neg.any() else 0.0
        out[f"sens_{DX_CLASSES[c]}"] = sens
        out[f"spec_{DX_CLASSES[c]}"] = spec
        sens_all.append(sens)
        spec_all.append(spec)
    out["sens"] = float(np.mean(sens_all))
    out["spec"] = float(np.mean(spec_all))
    out["bca"] = (out["sens"] + out["spec"]) / 2
    out["mauc"] = hand_till_auc(y, prob, labels)
    present = [c for c in labels if np.any(y == c)]
    out["f1_weighted"] = float(
        f1_score(y, pred, labels=present, average="weighted", zero_division=0)
    )
    out["f1_macro"] = float(
        f1_score(y, pred, labels=present, average="macro", zero_division=0)
    )
    return out


def converter_detection(
    y: np.ndarray, prob: np.ndarray, last_dx: np.ndarray, seq_type: np.ndarray
) -> Dict[str, float]:
    """T4: ranks P(more severe stage than now) against true converters.

    Args:
        y: True class indices ``[N]``.
        prob: Class probabilities ``[N, 3]`` in ``DX_CLASSES`` order.
        last_dx: Class index of each sequence's last input visit ``[N]``.
        seq_type: ``"stable"``, ``"converter"`` or ``"reverter"`` ``[N]``.

    Returns:
        ``auroc`` and ``sens_at_spec90``, or an empty dict if undefined.
    """
    at_risk = last_dx < len(DX_CLASSES) - 1
    if not at_risk.any():
        return {}
    score = np.array([prob[i, last_dx[i] + 1 :].sum() for i in range(len(y))])[at_risk]
    target = (seq_type[at_risk] == "converter").astype(int)
    if target.min() == target.max():
        return {}
    fpr, tpr, _ = roc_curve(target, score)
    return {
        "auroc": float(roc_auc_score(target, score)),
        "sens_at_spec90": float(tpr[fpr <= 0.10].max()),
    }


def evaluate(pred: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Metrics on all, stable-only and converter-only sequences.

    Args:
        pred: Output of :func:`predict`.

    Returns:
        A dict of metric dicts keyed by subset, plus ``conv_detect`` (T4).
    """
    y, prob = pred["y"], pred["prob"]
    seq_type = pred["seq_type"]
    out = {"overall": subset_metrics(y, prob)}
    for kind in ("stable", "converter", "reverter"):
        mask = seq_type == kind
        if mask.any():
            out[kind] = subset_metrics(y[mask], prob[mask])
    out["conv_detect"] = converter_detection(y, prob, pred["last_dx"], seq_type)
    return out


# --------------------------------------------------------------------------
# Training and prediction
# --------------------------------------------------------------------------


def build_model(cfg: RunConfig, dataset: SampleDataset) -> BaseModel:
    """Creates the model named in ``cfg`` for ``dataset``.

    Args:
        cfg: The run configuration.
        dataset: The training split, used to size the model.

    Returns:
        A ``VisitTransformer`` or a PyHealth ``RNN`` (LSTM / GRU).
    """
    if cfg.model == "visit_transformer":
        return VisitTransformer(dataset=dataset, **cfg.model_params)
    if cfg.model in ("lstm", "gru"):
        params = {
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.3,
        }
        params.update(cfg.model_params)
        return RNN(dataset=dataset, rnn_type=cfg.model.upper(), **params)
    raise ValueError(f"Unknown model {cfg.model}")


def predict(
    model: BaseModel, dataset: SampleDataset, batch_size: int
) -> Dict[str, np.ndarray]:
    """Runs the model on ``dataset`` and returns labels in DX_CLASSES order.

    Args:
        model: A trained model.
        dataset: The split to predict on.
        batch_size: Batch size.

    Returns:
        ``y`` ``[N]``, ``prob`` ``[N, 3]``, ``seq_type`` ``[N]`` and
        ``last_dx`` ``[N]``, all in ``DX_CLASSES`` order.
    """
    vocab = dataset.output_processors["label"].label_vocab
    order = [vocab[c] for c in DX_CLASSES]
    to_canonical = {vocab[c]: i for i, c in enumerate(DX_CLASSES)}
    rank = {c: i for i, c in enumerate(DX_CLASSES)}
    ys, probs, seq_types, last_dx = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in get_dataloader(dataset, batch_size=batch_size, shuffle=False):
            out = model(**batch)
            probs.append(out["y_prob"].cpu().numpy()[:, order])
            ys += [to_canonical[int(v)] for v in out["y_true"].cpu().numpy()]
            seq_types += list(batch["seq_type"])
            last_dx += [rank[str(v)] for v in batch["last_dx"]]
    return {
        "y": np.array(ys),
        "prob": np.concatenate(probs),
        "seq_type": np.array(seq_types),
        "last_dx": np.array(last_dx),
    }


class MonthlyContext:
    """Raw sequences and month-by-month models shared by the runs of a fold.

    One minimalRNN per fold and data set is both the paper's minimalRNN
    baseline and the model that fills missing values for the Transformer and
    the TST, so it is trained once and reused.

    Args:
        dataset: The loaded TADPOLE dataset.
        args: Command-line arguments (device, seed, ``monthly_epochs``).
    """

    def __init__(self, dataset: TADPOLEDataset, args: argparse.Namespace) -> None:
        self.dataset = dataset
        self.args = args
        self._raw: Dict[bool, Dict[str, Dict[str, Any]]] = {}
        self._models: Dict[Tuple[str, str], Tuple[MonthlyRNN, Any, Any, float]] = {}

    def raw(self, drop_reverters: bool = True) -> Dict[str, Dict[str, Any]]:
        """Patient ID to raw sequence, from ``ADProgressionPrediction``."""
        if drop_reverters not in self._raw:
            task = ADProgressionPrediction(drop_reverters=drop_reverters)
            table = {}
            for patient in self.dataset.iter_patients():
                seq = task.raw_sequence(patient)
                if seq is not None:
                    table[str(seq["patient_id"])] = seq
            self._raw[drop_reverters] = table
        return self._raw[drop_reverters]

    def model(
        self, cell: str, split: PatientListSplit
    ) -> Tuple[MonthlyRNN, Any, Any, float]:
        """Trains (once) a month-by-month RNN on the training patients.

        Returns:
            The model, the feature mean and standard deviation it was trained
            with, and its training time in seconds.
        """
        key = (cell, split.to_dict()["sha1"])
        if key not in self._models:
            raw = self.raw()
            train_raw = [raw[p] for p in split.train if p in raw]
            start = time.time()
            mean, std = fit_standardizer(train_raw)
            config = MonthlyConfig(cell=cell, epochs=self.args.monthly_epochs)
            model = train_monthly_rnn(
                to_monthly(train_raw, mean, std),
                config,
                self.args.device,
                seed=self.args.seed,
            )
            self._models[key] = (model, mean, std, time.time() - start)
        return self._models[key]

    def filler(self, split: PatientListSplit) -> ModelFiller:
        """Model filling for every patient of the split (paper, Section 2.2)."""
        model, mean, std, _ = self.model("minrnn", split)
        raw = self.raw()
        pids = list(split.train) + list(split.val) + list(split.test)
        seqs = to_monthly([raw[p] for p in pids if p in raw], mean, std)
        return ModelFiller(model_fill_table(model, seqs, mean, std, self.args.device))


def run_monthly_fold(
    cfg: RunConfig, split: PatientListSplit, ctx: MonthlyContext
) -> Dict[str, Any]:
    """Trains and evaluates one of the paper's month-by-month RNN baselines.

    The model is trained on the training patients for a fixed number of
    epochs, as in the reference implementation (the validation patients are
    not used), then rolled forward from each test sequence's input visits to
    the month of its target visit.

    Args:
        cfg: The run configuration (``model`` is ``"<cell>_monthly"``).
        split: The fold's patient split.
        ctx: Shared raw sequences and trained models.

    Returns:
        The fold's metrics, training time and split sizes.
    """
    model, mean, std, seconds = ctx.model(MONTHLY_CELLS[cfg.model], split)
    raw = ctx.raw()
    test_raw = [raw[p] for p in split.test if p in raw]
    prob = predict_monthly_rnn(model, to_monthly(test_raw, mean, std), ctx.args.device)
    rank = {c: i for i, c in enumerate(DX_CLASSES)}
    metrics = evaluate(
        {
            "y": np.array([rank[s["diagnoses"][s["group"]]] for s in test_raw]),
            "prob": prob,
            "seq_type": np.array([s["seq_type"] for s in test_raw]),
            "last_dx": np.array(
                [rank[s["diagnoses"][s["group"] - 1]] for s in test_raw]
            ),
        }
    )
    metrics["train_seconds"] = round(seconds, 1)
    metrics["n_params"] = int(sum(p.numel() for p in model.parameters()))
    metrics["n_train"] = sum(p in raw for p in split.train)
    metrics["n_val"] = sum(p in raw for p in split.val)
    metrics["n_test"] = len(test_raw)
    return metrics


def run_fold(
    dataset: TADPOLEDataset,
    cfg: RunConfig,
    split: PatientListSplit,
    args: argparse.Namespace,
    ctx: Optional[MonthlyContext] = None,
) -> Dict[str, Any]:
    """Trains one configuration on one fold and evaluates it.

    Args:
        dataset: The loaded TADPOLE dataset.
        cfg: The run configuration.
        split: The fold's patient split.
        args: Command-line arguments (training settings).
        ctx: Shared state for the month-by-month models and model filling;
            created on demand if ``None``.

    Returns:
        The fold's metrics, training time and split sizes.
    """
    ctx = ctx or MonthlyContext(dataset, args)
    if cfg.model in MONTHLY_CELLS:
        return run_monthly_fold(cfg, split, ctx)
    task_params = dict(cfg.task_params)
    if task_params.pop("model_fill", False):
        task_params["imputer"] = ctx.filler(split)
    task = ADProgressionPrediction(**task_params)
    train_ds, val_ds, test_ds = dataset.set_task(task, split=split)
    torch.manual_seed(args.seed)
    model = build_model(cfg, train_ds)
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "balanced_accuracy"],
        device=args.device,
        enable_logging=False,
    )
    start = time.time()
    trainer.train(
        train_dataloader=get_dataloader(
            train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed
        ),
        val_dataloader=get_dataloader(val_ds, batch_size=args.batch_size),
        epochs=args.epochs,
        optimizer_params={"lr": cfg.lr or args.lr},
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        monitor="balanced_accuracy",
        patience=args.patience,
    )
    metrics = evaluate(predict(model, test_ds, args.batch_size))
    metrics["train_seconds"] = round(time.time() - start, 1)
    metrics["n_params"] = int(sum(p.numel() for p in model.parameters()))
    metrics["n_train"], metrics["n_val"], metrics["n_test"] = (
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return metrics


def sequence_metadata(
    dataset: TADPOLEDataset, task_params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Reads patient ID, sequence type and group of every sample.

    Args:
        dataset: The loaded TADPOLE dataset.
        task_params: Arguments for :class:`ADProgressionPrediction`.

    Returns:
        One dict per sample with ``pid``, ``seq_type`` and ``group``.
    """
    samples = dataset.set_task(ADProgressionPrediction(**task_params))
    meta = [
        {
            "pid": str(s["patient_id"]),
            "seq_type": s["seq_type"],
            "group": int(s["group"]),
        }
        for s in samples
    ]
    return meta


# --------------------------------------------------------------------------
# Results: resume, aggregate, report
# --------------------------------------------------------------------------


def load_runs(path: Path) -> List[Dict[str, Any]]:
    """Reads finished runs from a ``runs.jsonl`` file (empty if missing)."""
    if not path.exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flattens nested metric dicts into ``{"subset.metric": value}``."""
    flat = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


# Columns of summary.md, matching the paper's Tables 2-4.
KEY_METRICS = [
    ("overall.mauc", "mAUC"),
    ("overall.f1_weighted", "F1"),
    ("overall.accuracy", "Acc"),
    ("overall.bca", "BCA"),
    ("stable.accuracy", "Stable Acc"),
    ("stable.bca", "Stable BCA"),
    ("converter.accuracy", "Conv. Acc"),
    ("converter.bca", "Conv. BCA"),
    ("converter.sens", "Conv. Sens"),
    ("conv_detect.auroc", "Conv. AUROC (T4)"),
    ("conv_detect.sens_at_spec90", "Sens@Spec90 (T4)"),
]
# Metrics compared with Welch's t-test (the paper's significance test).
TEST_METRICS = ["overall.mauc", "overall.bca", "converter.accuracy", "converter.bca"]
# Welch's t-tests: each reference model against the models listed with it.
WELCH_PAIRS = [
    ("visit_transformer", ["lstm", "gru"]),
    ("paper_transformer", ["tst", "minrnn_monthly", "lstm_monthly", "gru_monthly"]),
]


def summarize(runs: List[Dict[str, Any]], out_dir: Path) -> str:
    """Writes summary.csv and summary.md; returns the markdown.

    Args:
        runs: Finished runs, as read by :func:`load_runs`.
        out_dir: Directory to write to.

    Returns:
        The markdown table, followed by Welch's t-tests of the Transformer
        against the other models of the same experiment (``WELCH_PAIRS``).
    """
    by_name: Dict[str, List[Dict[str, float]]] = {}
    for run in runs:
        by_name.setdefault(run["config"]["name"], []).append(flatten(run["metrics"]))
    keys = sorted({k for rows in by_name.values() for row in rows for k in row})
    with open(out_dir / "summary.csv", "w") as handle:
        handle.write(
            "config,folds," + ",".join(f"{k}_mean,{k}_std" for k in keys) + "\n"
        )
        for name, rows in by_name.items():
            cells = []
            for k in keys:
                vals = [r[k] for r in rows if k in r and not math.isnan(r[k])]
                cells += [
                    f"{np.mean(vals):.4f}" if vals else "",
                    f"{np.std(vals):.4f}" if vals else "",
                ]
            handle.write(f'"{name}",{len(rows)},' + ",".join(cells) + "\n")
    lines = [
        "| Config | Folds | " + " | ".join(label for _, label in KEY_METRICS) + " |"
    ]
    lines.append("|" + " --- |" * (len(KEY_METRICS) + 2))
    for name, rows in by_name.items():
        cells = []
        for key, _ in KEY_METRICS:
            vals = [r[key] for r in rows if key in r and not math.isnan(r[key])]
            cells.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if vals else "–")
        lines.append(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")
    tests = []
    for data in ("raw", "balanced"):
        for ref_name, others in WELCH_PAIRS:
            ref = by_name.get(f"{data}/{ref_name}")
            for other in others:
                rows = by_name.get(f"{data}/{other}")
                if not ref or not rows or len(ref) < 2 or len(rows) < 2:
                    continue
                for key, label in KEY_METRICS:
                    if key not in TEST_METRICS:
                        continue
                    a = [r[key] for r in ref if key in r and not math.isnan(r[key])]
                    b = [r[key] for r in rows if key in r and not math.isnan(r[key])]
                    if len(a) > 1 and len(b) > 1:
                        t, p = stats.ttest_ind(a, b, equal_var=False)
                        tests.append(
                            f"| {data} | {ref_name} vs {other} | {label} "
                            f"| {t:.2f} | {p:.4f} |"
                        )
    if tests:
        lines += [
            "",
            "Welch's t-test across folds:",
            "",
            "| Data | Comparison | Metric | t | p |",
            "| --- | --- | --- | --- | --- |",
        ] + tests
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    return text


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments; ``quick`` shrinks the run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root",
        default=None,
        help="Directory with TADPOLE_D1_D2.csv. Omit to use synthetic data.",
    )
    parser.add_argument(
        "--experiment",
        default="quick",
        choices=[
            "quick",
            "reproduce",
            "paper_setup",
            "task_ablation",
            "model_ablation",
            "all",
        ],
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only configs whose name contains one of these strings.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Folds for 'reproduce' and 'paper_setup'.",
    )
    parser.add_argument(
        "--ablation-folds", type=int, default=5, help="Folds for the ablations."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run a single fold index (for job arrays).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--monthly-epochs",
        type=int,
        default=100,
        help="Epochs of the month-by-month RNNs (reference: 100, no early stop).",
    )
    parser.add_argument(
        "--device", default=None, help="cuda, cpu or mps. Default: cuda if available."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/tadpole")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--synthetic-patients", type=int, default=600)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.experiment == "quick":
        args.folds = args.ablation_folds = 2
        args.epochs = min(args.epochs, 2)
        args.monthly_epochs = min(args.monthly_epochs, 2)
        args.synthetic_patients = min(args.synthetic_patients, 200)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Runs the chosen experiment and prints the summary table."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_dir = Path(args.output_dir) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    if args.summarize_only:
        print(summarize(load_runs(runs_path), out_dir))
        return

    root = args.root
    if root is None:
        root = str(Path(args.output_dir) / "synthetic_tadpole")
        generate_synthetic_tadpole(
            root, num_patients=args.synthetic_patients, seed=args.seed
        )
        logger.info("Using synthetic data in %s", root)
    dataset = TADPOLEDataset(
        root=root, cache_dir=args.cache_dir, num_workers=args.num_workers
    )

    configs = experiment_configs(args.experiment)
    if args.only:
        configs = [c for c in configs if any(s in c.name for s in args.only)]
    done = {(r["config"]["name"], r["fold"]) for r in load_runs(runs_path)}
    meta_cache: Dict[str, List[Dict[str, Any]]] = {}
    ctx = MonthlyContext(dataset, args)

    for cfg in configs:
        n_folds = (
            args.folds
            if cfg.group in ("reproduce", "paper_setup", "quick")
            else args.ablation_folds
        )
        meta_key = json.dumps(
            {"drop_reverters": cfg.task_params.get("drop_reverters", True)}
        )
        if meta_key not in meta_cache:
            meta_cache[meta_key] = sequence_metadata(dataset, json.loads(meta_key))
        meta = meta_cache[meta_key]
        keep = balanced_patients(meta, args.seed) if cfg.data == "balanced" else None
        splits = plan_folds(meta, n_folds, args.val_ratio, args.seed, keep)
        folds = [args.fold] if args.fold is not None else range(n_folds)
        for k in folds:
            if (cfg.name, k) in done:
                logger.info("Skipping %s fold %d (done)", cfg.name, k)
                continue
            logger.info("Running %s fold %d/%d", cfg.name, k + 1, n_folds)
            metrics = run_fold(dataset, cfg, splits[k], args, ctx)
            record = {
                "config": asdict(cfg),
                "fold": k,
                "metrics": metrics,
                "seed": args.seed,
                "epochs": (
                    args.monthly_epochs if cfg.model in MONTHLY_CELLS else args.epochs
                ),
            }
            with open(runs_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            o = metrics["overall"]
            logger.info(
                "%s fold %d: BCA %.3f mAUC %.3f conv acc %s",
                cfg.name,
                k,
                o["bca"],
                o["mauc"],
                f"{metrics.get('converter', {}).get('accuracy', float('nan')):.3f}",
            )
    print(summarize(load_runs(runs_path), out_dir))


if __name__ == "__main__":
    main()
