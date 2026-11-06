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
or converter sequence per subject) -> ``VisitTransformer`` or PyHealth's
``RNN`` (LSTM / GRU baselines) -> evaluation with the paper's metrics.

Every run uses patient-level cross-validation, stratified by sequence type
and group (number of input visits). Each fold calls
``dataset.set_task(task, split=...)`` so feature normalization and
imputation statistics are fitted on the training patients only.

Experiments (``--experiment``)
------------------------------
``reproduce``
    VisitTransformer, LSTM and GRU on the raw and the balanced dataset
    (stable sequences dropped within each group until they match the
    converters), 10 folds. Compare with the paper's Tables 2-4.
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
    The three above. ``quick`` runs a two-fold, two-epoch smoke test.

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

LSTM and GRU here are PyHealth's ``RNN`` trained with the same next-visit
objective as the Transformer, unlike the paper's monthly roll-out RNNs
(minimalRNN). Comparing them with the paper's RNN rows is ablation M1
(new, not in the paper): it separates the effect of the training objective
from the effect of the architecture.

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
patience 20, seed 42. Values are mean +- std over folds.

TODO(results): after the real-data runs, replace this paragraph with the
three tables from ``summary.md`` (reproduce, task_ablation,
model_ablation), each followed by two or three sentences of findings.

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
        experiment: ``"reproduce"``, ``"task_ablation"``, ``"model_ablation"``,
            ``"quick"`` or ``"all"``.

    Returns:
        The run configurations, in the order they are run.
    """
    # Reproduction: the paper's three models on the raw and balanced data.
    reproduce = [
        RunConfig(f"{data}/{model}", model=model, data=data, group="reproduce")
        for data in ("raw", "balanced")
        for model in ("visit_transformer", "lstm", "gru")
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
    ]
    table = {
        "reproduce": reproduce,
        "task_ablation": task_ablation,
        "model_ablation": model_ablation,
        "quick": quick,
    }
    if experiment == "all":
        return reproduce + task_ablation + model_ablation
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


def run_fold(
    dataset: TADPOLEDataset,
    cfg: RunConfig,
    split: Split,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Trains one configuration on one fold and evaluates it.

    Args:
        dataset: The loaded TADPOLE dataset.
        cfg: The run configuration.
        split: The fold's patient split.
        args: Command-line arguments (training settings).

    Returns:
        The fold's metrics, training time and split sizes.
    """
    task = ADProgressionPrediction(**cfg.task_params)
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


def summarize(runs: List[Dict[str, Any]], out_dir: Path) -> str:
    """Writes summary.csv and summary.md; returns the markdown.

    Args:
        runs: Finished runs, as read by :func:`load_runs`.
        out_dir: Directory to write to.

    Returns:
        The markdown table, followed by Welch's t-tests of the Transformer
        against LSTM and GRU when the reproduction runs are present.
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
        ref = by_name.get(f"{data}/visit_transformer")
        for other in ("lstm", "gru"):
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
                        f"| {data} | Transformer vs {other.upper()} | {label} "
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
        choices=["quick", "reproduce", "task_ablation", "model_ablation", "all"],
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only configs whose name contains one of these strings.",
    )
    parser.add_argument("--folds", type=int, default=10, help="Folds for 'reproduce'.")
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

    for cfg in configs:
        n_folds = (
            args.folds if cfg.group in ("reproduce", "quick") else args.ablation_folds
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
            metrics = run_fold(dataset, cfg, splits[k], args)
            record = {
                "config": asdict(cfg),
                "fold": k,
                "metrics": metrics,
                "seed": args.seed,
                "epochs": args.epochs,
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
