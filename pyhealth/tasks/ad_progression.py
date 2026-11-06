"""Next-visit Alzheimer's disease stage prediction from longitudinal visits.

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
    Next-visit Alzheimer's disease stage prediction task with the paper's
    stable / converter sequence construction.

Implements the task of Moghaddami, Schubring and Siadat, "Transformer Model
for Alzheimer's Disease Progression Prediction Using Longitudinal Visit
Sequences" (CHIL 2025, https://proceedings.mlr.press/v287/moghaddami25a.html):
given a subject's first ``n`` visits, predict the clinical diagnosis
(CN, MCI or AD) at visit ``n + 1``.

Each subject yields at most one sample, built with the paper's rules:

1. Subjects with a single visit are dropped.
2. Subjects whose diagnosis ever moves to a milder stage ("reverters",
   MCI to CN or AD to MCI) are dropped.
3. Visits after a subject's first conversion are discarded.
4. Visits without a diagnosis are discarded.

The longest remaining sequence is used. It is a *converter* sequence if the
diagnosis at its last visit is more severe than at the visit before, and a
*stable* sequence otherwise.
"""

import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .base_task import BaseTask

#: The 22 numeric features per visit, in model input order.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "cdrsb",
    "adas11",
    "adas13",
    "mmse",
    "ravlt_immediate",
    "ravlt_learning",
    "ravlt_forgetting",
    "ravlt_perc_forgetting",
    "moca",
    "faq",
    "entorhinal",
    "fusiform",
    "hippocampus",
    "icv",
    "midtemp",
    "ventricles",
    "wholebrain",
    "av45",
    "fdg",
    "abeta_upennbiomk9_04_19_17",
    "tau_upennbiomk9_04_19_17",
    "ptau_upennbiomk9_04_19_17",
)

#: Diagnosis classes from least to most severe.
DX_CLASSES: Tuple[str, ...] = ("CN", "MCI", "AD")

TIME_FEATURES: Tuple[str, ...] = ("none", "to_target", "since_previous")

#: Signature of an imputer: ``(patient_id, rows, months) -> rows``. ``rows``
#: are the input visits' raw feature lists (``nan`` where missing) and
#: ``months`` their times in months since the first input visit.
Imputer = Callable[[str, List[List[float]], List[float]], List[List[float]]]

_DX_RANK: Dict[str, int] = {dx: rank for rank, dx in enumerate(DX_CLASSES)}
_DAYS_PER_MONTH = 30.4375


def map_diagnosis(raw: Optional[str]) -> Optional[str]:
    """Maps a raw TADPOLE/ADNI diagnosis string to ``CN``, ``MCI`` or ``AD``.

    Transition strings such as ``"MCI to Dementia"`` map to the stage after
    the transition.

    Args:
        raw: A diagnosis string such as ``"NL"``, ``"MCI"``, ``"Dementia"``,
            ``"NL to MCI"`` or ``"CN"``. ``None`` or empty means missing.

    Returns:
        ``"CN"``, ``"MCI"``, ``"AD"``, or ``None`` if missing or unknown.

    Examples:
        >>> map_diagnosis("MCI to Dementia")
        'AD'
        >>> map_diagnosis("NL") is None
        False
        >>> map_diagnosis("") is None
        True
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("nan", "none", "<na>"):
        return None
    stage = text.split(" to ")[-1].strip().lower()
    if stage in ("nl", "cn", "normal"):
        return "CN"
    if stage in ("mci", "emci", "lmci"):
        return "MCI"
    if stage in ("dementia", "ad"):
        return "AD"
    return None


def parse_numeric(value: Any) -> float:
    """Parses a TADPOLE value to float, returning ``nan`` when missing.

    Censoring markers are ignored, so ``"<8"`` parses as ``8.0``.

    Args:
        value: A number, a numeric string, or a missing marker.

    Returns:
        The value as a float, or ``nan``.

    Examples:
        >>> parse_numeric("<8")
        8.0
        >>> math.isnan(parse_numeric(""))
        True
    """
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lstrip("<>").strip()
    try:
        return float(text)
    except ValueError:
        return math.nan


def build_sequence(
    diagnoses: Sequence[Optional[str]], drop_reverters: bool = True
) -> Optional[Tuple[List[int], int, str]]:
    """Applies the paper's four rules to one subject's visit diagnoses.

    Args:
        diagnoses: Mapped diagnosis (``CN``, ``MCI``, ``AD`` or ``None``) of
            each visit, in time order.
        drop_reverters: If ``True`` (the paper's rule), subjects whose
            diagnosis ever moves to a milder stage are dropped. If ``False``,
            they are kept and their sequence ends at the first change of
            diagnosis in either direction.

    Returns:
        ``None`` if the subject is dropped. Otherwise a tuple of the indices
        of the input visits, the index of the target visit, and the sequence
        type: ``"stable"``, ``"converter"`` or ``"reverter"``.

    Examples:
        >>> build_sequence(["MCI", "MCI", "AD", "AD"])
        ([0, 1], 2, 'converter')
        >>> build_sequence(["CN", None, "CN"])
        ([0], 2, 'stable')
        >>> build_sequence(["MCI", "CN"]) is None
        True
    """
    if len(diagnoses) < 2:
        return None
    observed = [i for i, dx in enumerate(diagnoses) if dx is not None]
    ranks = [_DX_RANK[diagnoses[i]] for i in observed]
    reverts = any(later < earlier for earlier, later in zip(ranks, ranks[1:]))
    if reverts and drop_reverters:
        return None
    if len(observed) < 2:
        return None
    for k in range(1, len(observed)):
        if ranks[k] != ranks[k - 1]:
            kind = "converter" if ranks[k] > ranks[k - 1] else "reverter"
            return observed[:k], observed[k], kind
    return observed[:-1], observed[-1], "stable"


def forward_fill(rows: List[List[float]]) -> List[List[float]]:
    """Carries each feature's last observed value forward in time.

    Values never observed before a visit stay ``nan``.

    Args:
        rows: Per-visit feature lists, in time order.

    Returns:
        New lists with missing values filled from earlier visits.
    """
    filled: List[List[float]] = []
    last: List[float] = [math.nan] * (len(rows[0]) if rows else 0)
    for row in rows:
        new_row = [last[j] if math.isnan(v) else v for j, v in enumerate(row)]
        last = new_row
        filled.append(new_row)
    return filled


def _months_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 86400.0 / _DAYS_PER_MONTH


class ADProgressionPrediction(BaseTask):
    """Predicts the diagnosis at a subject's next visit from earlier visits.

    Reproduces the task of Moghaddami et al. (CHIL 2025) on
    :class:`~pyhealth.datasets.TADPOLEDataset`. Each eligible subject gives
    one sample:

    * ``visits``: an ``n x F`` matrix, one row per input visit. A row holds
      the 22 features in :data:`FEATURE_COLUMNS` (``nan`` where missing),
      the diagnosis one-hot in :data:`DX_CLASSES` order, and an optional time
      feature in months. By default ``F = 26``.
    * ``label``: the diagnosis at the target visit, ``CN``, ``MCI`` or ``AD``.
    * ``seq_type``: ``"stable"``, ``"converter"`` or ``"reverter"``.
    * ``group``: the number of input visits ``n`` (the paper's group).
    * ``last_dx``: the diagnosis at the last input visit.

    Missing values are handled by the ``tensor`` processor, whose statistics
    are fitted on the training split when the dataset is split with
    ``set_task(task, split=...)``: features are z-scored, optional
    missing-value indicators are appended, and remaining ``nan`` values are
    set to 0 (the training mean).

    Args:
        forward_fill: Carry each feature's last observed value forward within
            a subject before imputation. Default ``True``.
        missing_indicators: Append one missing-value indicator per input
            column. Default ``False``.
        time_feature: ``"to_target"`` (months from the visit to the target
            visit, the paper's feature), ``"since_previous"`` (months since
            the previous input visit) or ``"none"``.
        drop_reverters: Drop subjects whose diagnosis reverts (the paper's
            rule). Default ``True``.
        normalize: Z-score features with training statistics. Default
            ``True``.
        imputer: Optional callable that fills missing feature values of the
            input visits before forward filling, e.g. the paper's "model
            filling" with a recurrent model trained on the training split.
            Called as ``imputer(patient_id, rows, months)`` (see
            :data:`Imputer`). Its ``repr`` becomes part of the task's cache
            key, so it should identify the fitted imputer. Default ``None``.

    Examples:
        >>> from pyhealth.datasets import TADPOLEDataset
        >>> from pyhealth.tasks import ADProgressionPrediction
        >>> dataset = TADPOLEDataset(root="/path/to/tadpole")  # doctest: +SKIP
        >>> task = ADProgressionPrediction(time_feature="to_target")
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> samples[0]["visits"].shape  # doctest: +SKIP
        torch.Size([3, 26])
    """

    task_name: str = "ADProgressionPrediction"
    input_schema: Dict[str, Any] = {
        "visits": (
            "tensor",
            {"normalization": "zscore", "impute_strategy": "zero"},
        )
    }
    output_schema: Dict[str, Any] = {"label": "multiclass"}

    def __init__(
        self,
        forward_fill: bool = True,
        missing_indicators: bool = False,
        time_feature: str = "to_target",
        drop_reverters: bool = True,
        normalize: bool = True,
        imputer: Optional[Imputer] = None,
    ) -> None:
        if time_feature not in TIME_FEATURES:
            raise ValueError(
                f"time_feature must be one of {TIME_FEATURES}, got {time_feature!r}"
            )
        self.forward_fill = bool(forward_fill)
        self.missing_indicators = bool(missing_indicators)
        self.time_feature = time_feature
        self.drop_reverters = bool(drop_reverters)
        self.normalize = bool(normalize)
        self.imputer = imputer
        self.input_schema = {
            "visits": (
                "tensor",
                {
                    "normalization": "zscore" if self.normalize else None,
                    "missing_indicators": self.missing_indicators,
                    "impute_strategy": "zero",
                },
            )
        }
        self.output_schema = {"label": "multiclass"}
        super().__init__()

    @property
    def num_features(self) -> int:
        """Number of columns per visit before missing-value indicators."""
        extra = 0 if self.time_feature == "none" else 1
        return len(FEATURE_COLUMNS) + len(DX_CLASSES) + extra

    def raw_sequence(self, patient: Any) -> Optional[Dict[str, Any]]:
        """Returns a subject's sequence before any feature processing.

        Used by :meth:`__call__`, and by models that need the raw visits,
        such as the paper's month-by-month recurrent baselines.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with ``tadpole``
                events.

        Returns:
            ``None`` if the subject is dropped by the preprocessing rules.
            Otherwise a dict with ``patient_id``; ``months``, the time of
            each input visit and then of the target visit, in months since
            the first input visit; ``features``, the matching rows of the 22
            raw features in :data:`FEATURE_COLUMNS` (``nan`` where missing);
            ``diagnoses``, the matching diagnoses; ``seq_type``; and
            ``group``, the number of input visits.
        """
        events = sorted(
            patient.get_events(event_type="tadpole"), key=lambda e: e.timestamp
        )
        diagnoses = [map_diagnosis(getattr(e, "dx", None)) for e in events]
        sequence = build_sequence(diagnoses, drop_reverters=self.drop_reverters)
        if sequence is None:
            return None
        input_idx, target_idx, seq_type = sequence
        order = input_idx + [target_idx]
        start = events[input_idx[0]].timestamp
        return {
            "patient_id": patient.patient_id,
            "months": [_months_between(start, events[i].timestamp) for i in order],
            "features": [
                [
                    parse_numeric(getattr(events[i], col, None))
                    for col in FEATURE_COLUMNS
                ]
                for i in order
            ],
            "diagnoses": [diagnoses[i] for i in order],
            "seq_type": seq_type,
            "group": len(input_idx),
        }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Builds the sample of one subject.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with ``tadpole``
                events.

        Returns:
            A list with one sample, or an empty list if the subject is
            dropped by the preprocessing rules.
        """
        seq = self.raw_sequence(patient)
        if seq is None:
            return []
        n = seq["group"]
        months = seq["months"]
        rows = [list(row) for row in seq["features"][:n]]
        if self.imputer is not None:
            rows = self.imputer(seq["patient_id"], rows, months[:n])
        if self.forward_fill:
            rows = forward_fill(rows)

        visits: List[List[float]] = []
        for k, row in enumerate(rows):
            dx = seq["diagnoses"][k]
            onehot = [1.0 if dx == c else 0.0 for c in DX_CLASSES]
            if self.time_feature == "to_target":
                time = [months[n] - months[k]]
            elif self.time_feature == "since_previous":
                time = [0.0 if k == 0 else months[k] - months[k - 1]]
            else:
                time = []
            visits.append(list(row) + onehot + time)

        return [
            {
                "patient_id": seq["patient_id"],
                "visits": visits,
                "label": seq["diagnoses"][n],
                "seq_type": seq["seq_type"],
                "group": n,
                "last_dx": seq["diagnoses"][n - 1],
            }
        ]
