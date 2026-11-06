"""Month-by-month recurrent baselines and model filling for TADPOLE.

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
    The paper's RNN baselines (minimalRNN, LSTM, GRU) and its "model
    filling" imputation, used by
    ``tadpole_ad_progression_prediction_visit_transformer.py``.

The paper trains its RNN baselines "following Nguyen et al." (Section 2.4):
each sequence is laid on a monthly grid that starts at the first visit. At
every month the model predicts the next month's diagnosis and features; where
the next month has a visit, the prediction is scored against it and the
observed values are fed in, and where it has none, the prediction itself is
fed in. To predict a target visit, the model is given the input visits and
rolled forward month by month to the target month. The paper also fills
missing feature values of the Transformer's input visits with a minimalRNN
trained this way ("model filling").

This module ports the reference implementation of Nguyen et al., "Predicting
Alzheimer's disease progression using deep recurrent neural networks"
(NeuroImage 2020; ThomasYeoLab/CBIG, ``Nguyen2020_RNNAD``, MIT license),
written for Python 2.7 and PyTorch 1.0, to Python 3 and current PyTorch.
Kept from the reference: the minimalRNN cell, the residual feature head
(``next = previous + Linear(h)``), filling missing inputs with the model's
own predictions, filling the first month's missing features with the
sequence's first observed value (or the training mean), input and recurrent
dropout masks, the loss (summed MAE over observed features plus cross-entropy
over observed diagnoses, divided by the batch size; the cross-entropy is
applied to softmax outputs, as in the reference), and the hyperparameters of
its replication script (2 layers of 128 units, input dropout 0.1, recurrent
dropout 0.4, Adam with learning rate 1.29e-3 and weight decay 1e-5, batch
128, 100 epochs). The reference has no GRU; ours uses ``nn.GRUCell`` in the
same loop.

Shapes are time-major, ``[T, B, ...]``, as in the reference.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.tasks.ad_progression import DX_CLASSES, FEATURE_COLUMNS

CELLS = ("minrnn", "lstm", "gru")
NUM_CLASSES = len(DX_CLASSES)
NUM_FEATURES = len(FEATURE_COLUMNS)
_DX_INDEX = {dx: i for i, dx in enumerate(DX_CLASSES)}


@dataclass
class MonthlyConfig:
    """Hyperparameters of the reference replication script.

    Attributes:
        cell: ``"minrnn"``, ``"lstm"`` or ``"gru"``.
        hidden_size: Units per recurrent layer.
        num_layers: Recurrent layers.
        input_dropout: Probability of dropping an input feature.
        hidden_dropout: Probability of dropping a recurrent unit.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size: Sequences per batch.
        epochs: Training epochs (fixed, no early stopping).
        w_ent: Weight of the diagnosis loss relative to the feature loss.
    """

    cell: str = "minrnn"
    hidden_size: int = 128
    num_layers: int = 2
    input_dropout: float = 0.1
    hidden_dropout: float = 0.4
    lr: float = 1.290666e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 100
    w_ent: float = 1.0


class MinimalRNNCell(nn.Module):
    """MinimalRNN cell (Chen, 2018), as in the reference implementation.

    ``z = tanh(x @ W + b)``, ``u = sigmoid(h @ U_h + z @ U_z + b_u)``,
    ``h' = u * h + (1 - u) * z``.

    Args:
        input_size: Input width.
        hidden_size: Hidden width.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.weight_uh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_uz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        stdv = 1.0 / math.sqrt(hidden_size)
        nn.init.uniform_(self.weight_uh, -stdv, stdv)
        nn.init.uniform_(self.weight_uz, -stdv, stdv)
        nn.init.constant_(self.bias_hh, 1.0)  # forget-gate bias of 1

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One step: ``x`` ``[B, input]``, ``h`` ``[B, hidden]`` -> ``[B, hidden]``."""
        z = torch.tanh(self.W(x))
        u = torch.sigmoid(self.bias_hh + h @ self.weight_uh + z @ self.weight_uz)
        return u * h + (1 - u) * z


class MonthlyRNN(nn.Module):
    """Recurrent model that steps one month at a time and fills its own gaps.

    Args:
        config: Hyperparameters; ``config.cell`` picks the recurrent cell.
        num_features: Number of continuous features per month.
    """

    def __init__(self, config: MonthlyConfig, num_features: int = NUM_FEATURES):
        super().__init__()
        if config.cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}, got {config.cell!r}")
        self.config = config
        h = config.hidden_size
        cell_class = {"minrnn": MinimalRNNCell, "lstm": nn.LSTMCell, "gru": nn.GRUCell}
        sizes = [NUM_CLASSES + num_features] + [h] * (config.num_layers - 1)
        self.cells = nn.ModuleList(cell_class[config.cell](n, h) for n in sizes)
        if config.cell == "lstm":
            for cell in self.cells:  # forget-gate bias of 1, as in the reference
                nn.init.constant_(cell.bias_hh[h : 2 * h], 1.0)
        self.hid2category = nn.Linear(h, NUM_CLASSES)
        self.hid2measures = nn.Linear(h, num_features)

    def _initial_state(self, batch: int, device: torch.device) -> List[Any]:
        h = self.config.hidden_size
        zeros = [torch.zeros(batch, h, device=device) for _ in self.cells]
        if self.config.cell == "lstm":
            return [(z, torch.zeros_like(z)) for z in zeros]
        return zeros

    def _dropout_masks(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Masks fixed for a whole sequence; unscaled, as in the reference."""
        i_mask = torch.ones(batch, self.hid2measures.out_features, device=device)
        r_masks = [torch.ones(batch, self.config.hidden_size, device=device)]
        r_masks = r_masks * len(self.cells)
        if self.training:
            i_mask = i_mask.bernoulli(1.0 - self.config.input_dropout)
            r_masks = [m.bernoulli(1.0 - self.config.hidden_dropout) for m in r_masks]
        return i_mask, r_masks

    def _step(
        self,
        cat: torch.Tensor,
        val: torch.Tensor,
        state: List[Any],
        masks: Tuple[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Any]]:
        i_mask, r_masks = masks
        x = torch.cat([cat, val * i_mask], dim=-1)
        new_state = []
        for cell, prev, mask in zip(self.cells, state, r_masks):
            if self.config.cell == "lstm":
                x, c = cell(x, (prev[0] * mask, prev[1]))
                new_state.append((x, c))
            else:
                x = cell(x, prev * mask)
                new_state.append(x)
        prob = F.softmax(self.hid2category(x), dim=-1)
        return prob, self.hid2measures(x) + val, new_state

    def forward(
        self, cat: torch.Tensor, val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the model month by month.

        Args:
            cat: Diagnosis one-hot ``[T, B, 3]``, ``nan`` where unobserved.
            val: Standardized features ``[T, B, F]``, ``nan`` where
                unobserved. The first month must have no ``nan``.

        Returns:
            Predicted diagnosis probabilities ``[T - 1, B, 3]`` and features
            ``[T - 1, B, F]`` for months 1 to ``T - 1``, and the features
            after filling ``[T, B, F]``.
        """
        cat, val = cat.clone(), val.clone()
        state = self._initial_state(cat.size(1), cat.device)
        masks = self._dropout_masks(cat.size(1), cat.device)
        probs, vals = [], []
        for t in range(cat.size(0) - 1):
            prob, pred, state = self._step(cat[t], val[t], state, masks)
            probs.append(prob)
            vals.append(pred)
            # Fill the next month's missing values with this prediction.
            val[t + 1] = torch.where(torch.isnan(val[t + 1]), pred.detach(), val[t + 1])
            cat[t + 1] = torch.where(torch.isnan(cat[t + 1]), prob.detach(), cat[t + 1])
        if not probs:
            empty = cat.new_zeros(0, cat.size(1), NUM_CLASSES)
            return empty, val.new_zeros(0, *val.shape[1:]), val
        return torch.stack(probs), torch.stack(vals), val


# --------------------------------------------------------------------------
# Data: raw sequences -> monthly grids
# --------------------------------------------------------------------------


@dataclass
class MonthlySequence:
    """One subject's sequence on integer months.

    Attributes:
        patient_id: Subject ID.
        months: Month of each visit since the first input visit, strictly
            increasing; the last entry is the target visit.
        features: Standardized features of each visit ``[V, F]`` (``nan``
            where missing).
        dx: Diagnosis index of each visit ``[V]``.
        n_inputs: Number of input visits (``V - 1``).
        seq_type: ``"stable"``, ``"converter"`` or ``"reverter"``.
    """

    patient_id: str
    months: np.ndarray
    features: np.ndarray
    dx: np.ndarray
    n_inputs: int
    seq_type: str


def fit_standardizer(raw: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and standard deviation over every visit of ``raw``.

    Args:
        raw: Training sequences from ``ADProgressionPrediction.raw_sequence``.

    Returns:
        ``mean`` and ``std``, each ``[F]``; ``std`` is 1 where undefined.
    """
    rows = np.array([row for seq in raw for row in seq["features"]], dtype=float)
    mean = np.nanmean(rows, axis=0)
    std = np.nanstd(rows, axis=0, ddof=1)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std) | (std == 0), 1.0, std)
    return mean, std


def to_monthly(
    raw: Sequence[Dict[str, Any]], mean: np.ndarray, std: np.ndarray
) -> List[MonthlySequence]:
    """Rounds visit times to months and standardizes features.

    Two visits that round to the same month are kept one month apart.

    Args:
        raw: Sequences from ``ADProgressionPrediction.raw_sequence``.
        mean: Feature means from :func:`fit_standardizer`.
        std: Feature standard deviations from :func:`fit_standardizer`.

    Returns:
        One :class:`MonthlySequence` per input sequence.
    """
    out = []
    for seq in raw:
        months = []
        for m in seq["months"]:
            r = int(round(m))
            months.append(max(r, months[-1] + 1) if months else 0)
        feats = (np.array(seq["features"], dtype=float) - mean) / std
        out.append(
            MonthlySequence(
                patient_id=str(seq["patient_id"]),
                months=np.array(months, dtype=int),
                features=feats,
                dx=np.array([_DX_INDEX[d] for d in seq["diagnoses"]], dtype=int),
                n_inputs=int(seq["group"]),
                seq_type=seq["seq_type"],
            )
        )
    return out


def _grid(
    seq: MonthlySequence, n_visits: int, length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Places the first ``n_visits`` visits on a monthly grid of ``length``.

    The first month's missing features take the first observed value among
    those visits, or 0 (the training mean), as the reference's
    ``bl_fill``; every other unobserved value stays ``nan``.
    """
    cat = np.full((length, NUM_CLASSES), np.nan)
    val = np.full((length, seq.features.shape[1]), np.nan)
    for k in range(n_visits):
        m = seq.months[k]
        cat[m] = np.eye(NUM_CLASSES)[seq.dx[k]]
        val[m] = seq.features[k]
    first = np.zeros(seq.features.shape[1])
    for j in range(seq.features.shape[1]):
        seen = seq.features[:n_visits, j]
        seen = seen[~np.isnan(seen)]
        if seen.size:
            first[j] = seen[0]
    val[0] = np.where(np.isnan(val[0]), first, val[0])
    return cat, val


def _batch(
    grids: List[Tuple[np.ndarray, np.ndarray]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stacks grids to ``[T, B, ...]``, padding the end with zeros."""
    length = max(c.shape[0] for c, _ in grids)
    cats = np.zeros((length, len(grids), NUM_CLASSES))
    vals = np.zeros((length, len(grids), grids[0][1].shape[1]))
    for b, (c, v) in enumerate(grids):
        cats[: len(c), b] = c
        vals[: len(v), b] = v
    as_tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa
    return as_tensor(cats), as_tensor(vals)


# --------------------------------------------------------------------------
# Training, prediction and model filling
# --------------------------------------------------------------------------


def train_monthly_rnn(
    sequences: Sequence[MonthlySequence],
    config: MonthlyConfig,
    device: str,
    seed: int = 0,
) -> MonthlyRNN:
    """Trains a :class:`MonthlyRNN` on full sequences (inputs and target).

    Args:
        sequences: Training sequences.
        config: Hyperparameters.
        device: Torch device name.
        seed: Random seed for initialization, dropout and batch order.

    Returns:
        The trained model, in eval mode.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    dev = torch.device(device)
    model = MonthlyRNN(config).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    seqs = [s for s in sequences if len(s.months) > 1]
    grids = [_grid(s, len(s.months), int(s.months[-1]) + 1) for s in seqs]
    order = np.arange(len(seqs))
    for _ in range(config.epochs):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            cat, val = _batch([grids[i] for i in idx], dev)
            probs, preds, _ = model(cat, val)
            true_cat, true_val = cat[1:], val[1:]
            observed_dx = ~torch.isnan(true_cat).any(-1) & (true_cat.sum(-1) > 0)
            observed_val = ~torch.isnan(true_val)
            # Months without a visit hold model-filled inputs, not truth.
            has_visit = torch.zeros_like(observed_dx)
            for b, i in enumerate(idx):
                has_visit[seqs[i].months[1:] - 1, b] = True
            observed_dx &= has_visit
            observed_val &= has_visit.unsqueeze(-1)
            observed_val &= ~torch.isnan(_raw_values(seqs, idx, val.shape, dev))
            target = true_cat.nan_to_num(0).argmax(-1)
            # As in the reference, cross-entropy is applied to the softmax
            # outputs, and both losses are sums divided by the batch size.
            ent = F.cross_entropy(
                probs[observed_dx], target[observed_dx], reduction="sum"
            )
            mae = F.l1_loss(
                preds[observed_val], true_val[observed_val], reduction="sum"
            )
            loss = (mae + config.w_ent * ent) / len(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def _raw_values(
    seqs: Sequence[MonthlySequence],
    idx: Sequence[int],
    shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    """Observed features on the grid of months 1..T-1 (``nan`` elsewhere)."""
    out = np.full((shape[0] - 1, shape[1], shape[2]), np.nan)
    for b, i in enumerate(idx):
        s = seqs[i]
        for k in range(1, len(s.months)):
            out[s.months[k] - 1, b] = s.features[k]
    return torch.tensor(out, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_monthly_rnn(
    model: MonthlyRNN,
    sequences: Sequence[MonthlySequence],
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Diagnosis probabilities at each sequence's target month.

    The model sees the input visits only and rolls forward, month by month,
    to the month of the target visit.

    Args:
        model: A trained model.
        sequences: Sequences to predict.
        device: Torch device name.
        batch_size: Sequences per batch.

    Returns:
        Probabilities ``[N, 3]`` in ``DX_CLASSES`` order.
    """
    dev = torch.device(device)
    model.eval()
    out = np.zeros((len(sequences), NUM_CLASSES))
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        grids = [_grid(s, s.n_inputs, int(s.months[s.n_inputs]) + 1) for s in chunk]
        cat, val = _batch(grids, dev)
        probs, _, _ = model(cat, val)
        for b, s in enumerate(chunk):
            out[start + b] = probs[s.months[s.n_inputs] - 1, b].cpu().numpy()
    return out


@torch.no_grad()
def model_fill_table(
    model: MonthlyRNN,
    sequences: Sequence[MonthlySequence],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> Dict[str, List[List[float]]]:
    """Input-visit features after model filling, in raw units.

    Each sequence's input visits are run through the model; a missing value
    at a visit is replaced by the model's prediction for that month (and at
    the first visit by the first observed value, or the training mean).

    Args:
        model: A minimalRNN trained with :func:`train_monthly_rnn`.
        sequences: Sequences whose input visits to fill.
        mean: Feature means used for standardization.
        std: Feature standard deviations used for standardization.
        device: Torch device name.
        batch_size: Sequences per batch.

    Returns:
        Patient ID to the filled rows of its input visits.
    """
    dev = torch.device(device)
    model.eval()
    table: Dict[str, List[List[float]]] = {}
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        grids = [_grid(s, s.n_inputs, int(s.months[s.n_inputs - 1]) + 1) for s in chunk]
        cat, val = _batch(grids, dev)
        _, _, filled = model(cat, val)
        filled = filled.cpu().numpy()
        for b, s in enumerate(chunk):
            rows = filled[s.months[: s.n_inputs], b] * std + mean
            table[s.patient_id] = rows.tolist()
    return table


class ModelFiller:
    """Imputer for ``ADProgressionPrediction(imputer=...)``.

    Keeps observed values and fills missing ones from a table made by
    :func:`model_fill_table`. Its ``repr`` is a digest of the table, so
    PyHealth caches each fitted filler separately.

    Args:
        table: Patient ID to filled input-visit rows.
    """

    def __init__(self, table: Dict[str, List[List[float]]]) -> None:
        self.table = table
        blob = json.dumps(
            {k: np.round(v, 4).tolist() for k, v in sorted(table.items())}
        )
        self.digest = hashlib.sha1(blob.encode()).hexdigest()[:16]

    def __call__(
        self, patient_id: str, rows: List[List[float]], months: List[float]
    ) -> List[List[float]]:
        """Fills ``nan`` entries of ``rows`` for ``patient_id``."""
        filled = self.table.get(str(patient_id))
        if filled is None or len(filled) != len(rows):
            return rows
        return [
            [f if math.isnan(v) else v for v, f in zip(row, frow)]
            for row, frow in zip(rows, filled)
        ]

    def __repr__(self) -> str:
        return f"ModelFiller({self.digest})"


def fit_model_filler(
    train_raw: Sequence[Dict[str, Any]],
    all_raw: Sequence[Dict[str, Any]],
    device: str,
    seed: int = 0,
    epochs: Optional[int] = None,
) -> ModelFiller:
    """Trains a minimalRNN on ``train_raw`` and fills ``all_raw``.

    Args:
        train_raw: Training sequences (raw).
        all_raw: Every sequence whose input visits should be filled.
        device: Torch device name.
        seed: Random seed.
        epochs: Training epochs; ``None`` keeps the reference's 100.

    Returns:
        A :class:`ModelFiller` for the task.
    """
    config = MonthlyConfig(cell="minrnn")
    if epochs is not None:
        config.epochs = epochs
    mean, std = fit_standardizer(train_raw)
    model = train_monthly_rnn(to_monthly(train_raw, mean, std), config, device, seed)
    table = model_fill_table(model, to_monthly(all_raw, mean, std), mean, std, device)
    return ModelFiller(table)
