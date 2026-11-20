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

Visits first pass through :func:`prepare_patient_visits`: matching visit
identities are deduplicated or compatibly merged; conflicts and ambiguous
same-timestamp visits exclude the subject. This duplicate policy is a team
processing decision. Explicit transition strings and observed stage changes
are both checked for reversions before first-conversion truncation.

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
import re
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import pairwise
from typing import Any, ClassVar

import torch

from .base_task import BaseTask

#: The 22 numeric features per visit, in model input order.
FEATURE_COLUMNS: tuple[str, ...] = (
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
DX_CLASSES: tuple[str, ...] = ("CN", "MCI", "AD")

#: Allowed values of ``ADProgressionPrediction(time_feature=...)``.
TIME_FEATURES: tuple[str, ...] = ("none", "to_target", "since_previous")

#: Signature of an imputer: ``(patient_id, rows, months) -> rows``. ``rows``
#: are the input visits' raw feature lists (``nan`` where missing) and
#: ``months`` their times in months since the first input visit.
Imputer = Callable[[str, list[list[float]], list[float]], list[list[float]]]

_DX_RANK: dict[str, int] = {dx: rank for rank, dx in enumerate(DX_CLASSES)}
_DAYS_PER_MONTH = 30.4375


def map_diagnosis(raw: str | None) -> str | None:
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
    return _diagnosis_transition(raw)[1]


def _diagnosis_transition(raw: Any) -> tuple[str | None, str | None]:
    """Preserves both stages of recognized transition strings."""
    aliases = {
        "nl": "CN", "cn": "CN", "normal": "CN",
        "mci": "MCI", "emci": "MCI", "lmci": "MCI",
        "dementia": "AD", "ad": "AD",
    }
    parts = re.split(r"\s+to\s+", str(raw).strip().lower())
    if len(parts) == 1:
        return None, aliases.get(parts[0])
    if len(parts) == 2:
        return aliases.get(parts[0]), aliases.get(parts[1])
    return None, None


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
    diagnoses: Sequence[str | None], drop_reverters: bool = True
) -> tuple[list[int], int, str] | None:
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
    reverts = any(later < earlier for earlier, later in pairwise(ranks))
    if reverts and drop_reverters:
        return None
    if len(observed) < 2:
        return None
    for k in range(1, len(observed)):
        if ranks[k] != ranks[k - 1]:
            kind = "converter" if ranks[k] > ranks[k - 1] else "reverter"
            return observed[:k], observed[k], kind
    return observed[:-1], observed[-1], "stable"


SEQUENCE_POLICY_VERSION = 2


@dataclass
class _PreparedVisit:
    """One resolved visit; source rows remain untouched."""

    timestamp: Any
    dx: str | None
    features: tuple[float, ...]


@dataclass
class _PreparedPatient:
    """Shared selection result and aggregate-safe diagnostics."""

    source_events: list[Any]
    visits: list[_PreparedVisit]
    sequence: tuple[list[int], int, str] | None
    exclusion_reason: str | None
    diagnostics: dict[str, int]


def prepare_patient_visits(
    patient: Any, drop_reverters: bool = True
) -> _PreparedPatient:
    """Resolves visits and applies the task's shared sequence policy.

    Exact duplicates in selected fields collapse to one visit. Complementary
    records merge only when timestamp and nonempty visit code agree and all
    overlapping selected measurements/diagnoses agree. Different visit codes
    at the same timestamp, unidentified nonidentical visits, and conflicting
    records exclude the subject. This conservative duplicate policy is a team
    decision, not a duplicate-resolution method specified by the paper.

    Explicit reversions are detected from every source row before merging.
    Both explicit and observed reversions exclude the subject by default,
    including reversions after a first conversion. With ``drop_reverters=False``
    those subjects are kept and labels follow observed destination diagnoses.

    Args:
        patient: Patient exposing timestamped ``tadpole`` events.
        drop_reverters: Whether to exclude explicit or observed reversions.

    Returns:
        Resolved visits, selection or exclusion reason, original events, and
        aggregate diagnostics. No original event is modified. Diagnostic
        counts can overlap; the audit's visit ledger accounts for rows once.

    Examples:
        >>> from types import SimpleNamespace
        >>> patient = SimpleNamespace(get_events=lambda **kwargs: [])
        >>> prepare_patient_visits(patient).exclusion_reason
        'fewer_than_two_total_visits'
    """
    source = sorted(patient.get_events(event_type="tadpole"), key=lambda e: e.timestamp)
    transitions = [_diagnosis_transition(getattr(e, "dx", None)) for e in source]
    explicit = any(
        a is not None and b is not None and _DX_RANK[b] < _DX_RANK[a]
        for a, b in transitions
    )
    groups: dict[Any, list[Any]] = defaultdict(list)
    for event in source:
        groups[event.timestamp].append(event)
    diagnostics = {
        "explicit_reversion_subjects": int(explicit),
        "observed_reversion_subjects": 0,
        "additional_explicit_reversion_exclusions": 0,
        "same_timestamp_groups": 0,
        "exact_duplicate_groups": 0,
        "complementary_merge_groups": 0,
        "collapsed_source_rows": 0,
        "ambiguous_same_day_groups": 0,
        "conflicting_duplicate_groups": 0,
    }
    visits = []
    for timestamp, group in groups.items():
        diagnoses = [map_diagnosis(getattr(e, "dx", None)) for e in group]
        values = [tuple(parse_numeric(getattr(e, c, None)) for c in FEATURE_COLUMNS)
                  for e in group]
        if len(group) == 1:
            visits.append(_PreparedVisit(timestamp, diagnoses[0], values[0]))
            continue
        diagnostics["same_timestamp_groups"] += 1
        codes = {str(getattr(e, "viscode", "") or "").strip().lower() for e in group}
        # Distinct codes establish different visit identities, not their order.
        if len(codes) != 1:
            diagnostics["ambiguous_same_day_groups"] += 1
            continue
        raw_dx = [str(getattr(e, "dx", None)).strip().lower() for e in group]
        exact = len(set(raw_dx)) == 1 and all(
            all(a == b or (math.isnan(a) and math.isnan(b))
                for a, b in zip(values[0], row)) for row in values[1:]
        )
        code = next(iter(codes))
        if code in ("", "nan", "none", "<na>") and not exact:
            diagnostics["ambiguous_same_day_groups"] += 1
            continue
        known_dx = {dx for dx in diagnoses if dx is not None}
        columns = [{row[j] for row in values if not math.isnan(row[j])}
                   for j in range(len(FEATURE_COLUMNS))]
        unknown_dx = any(
            dx is None and raw not in ("", "nan", "none", "<na>")
            for dx, raw in zip(diagnoses, raw_dx)
        )
        if len(known_dx) > 1 or any(len(c) > 1 for c in columns) or (
            unknown_dx and not exact
        ):
            diagnostics["conflicting_duplicate_groups"] += 1
            continue
        merged = tuple(next(iter(c)) if c else math.nan for c in columns)
        visits.append(_PreparedVisit(timestamp, next(iter(known_dx), None), merged))
        key = "exact_duplicate_groups" if exact else "complementary_merge_groups"
        diagnostics[key] += 1
        diagnostics["collapsed_source_rows"] += len(group) - 1
    diagnoses = [v.dx for v in visits]
    ranks = [_DX_RANK[dx] for dx in diagnoses if dx is not None]
    unresolved = (diagnostics["conflicting_duplicate_groups"]
                  or diagnostics["ambiguous_same_day_groups"])
    observed = not unresolved and any(b < a for a, b in pairwise(ranks))
    diagnostics["observed_reversion_subjects"] = int(observed)
    reason = None
    if len(source) < 2:
        reason = "fewer_than_two_total_visits"
    elif diagnostics["conflicting_duplicate_groups"]:
        reason = "conflicting_duplicate_visit"
    elif diagnostics["ambiguous_same_day_groups"]:
        reason = "ambiguous_same_day_visits"
    elif len(visits) < 2:
        reason = "fewer_than_two_distinct_visits"
    elif drop_reverters and (explicit or observed):
        reason = "reverter"
        diagnostics["additional_explicit_reversion_exclusions"] = int(
            explicit and not observed
        )
    elif len(ranks) < 2:
        reason = "fewer_than_two_diagnosed_visits"
    sequence = None if reason else build_sequence(diagnoses, drop_reverters)
    return _PreparedPatient(source, visits, sequence, reason, diagnostics)


def forward_fill(rows: list[list[float]]) -> list[list[float]]:
    """Carries each feature's last observed value forward in time.

    Values never observed before a visit stay ``nan``.

    Args:
        rows: Per-visit feature lists, in time order.

    Returns:
        New lists with missing values filled from earlier visits.

    Examples:
        >>> nan = float("nan")
        >>> forward_fill([[1.0, nan], [nan, 2.0], [nan, nan]])
        [[1.0, nan], [1.0, 2.0], [1.0, 2.0]]
    """
    filled: list[list[float]] = []
    last: list[float] = [math.nan] * (len(rows[0]) if rows else 0)
    for row in rows:
        new_row = [last[j] if math.isnan(v) else v for j, v in enumerate(row)]
        last = new_row
        filled.append(new_row)
    return filled


def _months_between(start: datetime, end: datetime) -> float:
    """Months (of 30.4375 days) from ``start`` to ``end``."""
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
    * ``visits_mask``: a boolean tensor marking every input visit as real.
      The dataloader pads it with False, preserving real visits even if all
      their features become zero after normalization and imputation.
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
        drop_reverters: Drop subjects with explicit transition-string or
            observed-stage reversions anywhere in their history. Default True.
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
    input_schema: ClassVar[dict[str, Any]] = {
        "visits": (
            "tensor",
            {"normalization": "zscore", "impute_strategy": "zero"},
        )
    }
    output_schema: ClassVar[dict[str, Any]] = {"label": "multiclass"}

    def __init__(
        self,
        forward_fill: bool = True,
        missing_indicators: bool = False,
        time_feature: str = "to_target",
        drop_reverters: bool = True,
        normalize: bool = True,
        imputer: Imputer | None = None,
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
        # Include the sample-layout version in BaseDataset's task cache key.
        self._visit_mask_version = 1
        self._sequence_policy_version = SEQUENCE_POLICY_VERSION
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

    def raw_sequence(self, patient: Any) -> dict[str, Any] | None:
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
        prepared = prepare_patient_visits(patient, self.drop_reverters)
        events = prepared.visits
        diagnoses = [event.dx for event in events]
        sequence = prepared.sequence
        if sequence is None:
            return None
        input_idx, target_idx, seq_type = sequence
        order = input_idx + [target_idx]
        start = events[input_idx[0]].timestamp
        return {
            "patient_id": patient.patient_id,
            "months": [_months_between(start, events[i].timestamp) for i in order],
            "features": [list(events[i].features) for i in order],
            "diagnoses": [diagnoses[i] for i in order],
            "seq_type": seq_type,
            "group": len(input_idx),
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
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

        visits: list[list[float]] = []
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
                "visits_mask": torch.ones(len(visits), dtype=torch.bool),
                "label": seq["diagnoses"][n],
                "seq_type": seq["seq_type"],
                "group": n,
                "last_dx": seq["diagnoses"][n - 1],
            }
        ]
