"""TADPOLE dataset for longitudinal Alzheimer's disease progression modeling.

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
    TADPOLE (ADNI) dataset loader with input validation and synthetic data
    generators for tests and examples.

The TADPOLE challenge (Marinescu et al., 2018) packages longitudinal data
from the Alzheimer's Disease Neuroimaging Initiative (ADNI) into one table,
``TADPOLE_D1_D2.csv``, with one row per clinical visit. Each row holds the
clinical diagnosis and cognitive, MRI, PET and CSF measurements.

This module provides:

* :class:`TADPOLEDataset`, which loads ``TADPOLE_D1_D2.csv`` as one event
  per visit, keyed by the ADNI subject ID (``RID``).
* :func:`write_tadpole_fixture`, a five-subject synthetic file used by the
  unit tests.
* :func:`generate_synthetic_tadpole`, a larger synthetic cohort with
  realistic disease trajectories, used to run the example pipeline without
  access to ADNI data.

The data is distributed by the ADNI LONI Image and Data Archive
(https://ida.loni.usc.edu) under the ADNI Data Use Agreement and cannot be
redistributed. The synthetic helpers contain no ADNI data.
"""

import csv
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import narwhals as nw
import numpy as np

from .base_dataset import BaseDataset

if TYPE_CHECKING:
    from ..tasks.ad_progression import ADProgressionPrediction

logger = logging.getLogger(__name__)

TADPOLE_FILE_NAME = "TADPOLE_D1_D2.csv"

#: The 22 numeric TADPOLE features used by Moghaddami et al. (2025), lowercase.
TADPOLE_FEATURE_COLUMNS: Tuple[str, ...] = (
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

#: CSF columns whose out-of-range values are stored as strings like ``"<8"``.
TADPOLE_CENSORED_COLUMNS: Tuple[str, ...] = (
    "abeta_upennbiomk9_04_19_17",
    "tau_upennbiomk9_04_19_17",
    "ptau_upennbiomk9_04_19_17",
)

#: Columns that must exist in the source file (compared case-insensitively).
TADPOLE_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "rid",
    "examdate",
    "viscode",
    "dx",
    "d1",
    "d2",
) + TADPOLE_FEATURE_COLUMNS

# Original TADPOLE spelling of each column, used by the synthetic writers.
_SOURCE_NAMES: Dict[str, str] = {
    "rid": "RID",
    "examdate": "EXAMDATE",
    "viscode": "VISCODE",
    "dx": "DX",
    "d1": "D1",
    "d2": "D2",
    "cdrsb": "CDRSB",
    "adas11": "ADAS11",
    "adas13": "ADAS13",
    "mmse": "MMSE",
    "ravlt_immediate": "RAVLT_immediate",
    "ravlt_learning": "RAVLT_learning",
    "ravlt_forgetting": "RAVLT_forgetting",
    "ravlt_perc_forgetting": "RAVLT_perc_forgetting",
    "moca": "MOCA",
    "faq": "FAQ",
    "entorhinal": "Entorhinal",
    "fusiform": "Fusiform",
    "hippocampus": "Hippocampus",
    "icv": "ICV",
    "midtemp": "MidTemp",
    "ventricles": "Ventricles",
    "wholebrain": "WholeBrain",
    "av45": "AV45",
    "fdg": "FDG",
    "abeta_upennbiomk9_04_19_17": "ABETA_UPENNBIOMK9_04_19_17",
    "tau_upennbiomk9_04_19_17": "TAU_UPENNBIOMK9_04_19_17",
    "ptau_upennbiomk9_04_19_17": "PTAU_UPENNBIOMK9_04_19_17",
}
_HEADER: List[str] = [_SOURCE_NAMES[c] for c in TADPOLE_REQUIRED_COLUMNS]


class TADPOLEDataset(BaseDataset):
    """TADPOLE / ADNI longitudinal dataset for Alzheimer's disease progression.

    Each row of ``TADPOLE_D1_D2.csv`` becomes one event of type ``"tadpole"``
    on the subject identified by ``RID``, timestamped by ``EXAMDATE``. Every
    event carries the visit code (``viscode``), the raw TADPOLE diagnosis
    string (``dx``, e.g. ``"MCI to Dementia"``), the D1/D2 membership flags
    and the 22 numeric features listed in :data:`TADPOLE_FEATURE_COLUMNS`.

    Values are kept as strings, as in the source file, so tasks decide how to
    parse them. Censored CSF values such as ``"<8"`` or ``">1300"`` are
    cleaned to plain numbers (``"8"``, ``"1300"``) while loading.

    The dataset requires access to ADNI through the LONI Image and Data
    Archive (https://ida.loni.usc.edu). Download the TADPOLE challenge package
    and place ``TADPOLE_D1_D2.csv`` in ``root``.

    Args:
        root: Directory containing ``TADPOLE_D1_D2.csv``.
        dataset_name: Name of the dataset. Defaults to ``"tadpole"``.
        config_path: Path to a YAML config. Defaults to the bundled
            ``configs/tadpole.yaml``.
        cache_dir: Directory for PyHealth's processed cache. Defaults to the
            PyHealth cache directory.
        num_workers: Number of worker processes used by ``set_task``.
        dev: If ``True``, loads a small subset of patients for development.

    Raises:
        FileNotFoundError: If ``root`` or ``TADPOLE_D1_D2.csv`` does not exist.
        ValueError: If the file lacks any column in
            :data:`TADPOLE_REQUIRED_COLUMNS`.

    Examples:
        >>> from pyhealth.datasets import TADPOLEDataset
        >>> from pyhealth.tasks import ADProgressionPrediction
        >>> dataset = TADPOLEDataset(root="/path/to/tadpole")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task(ADProgressionPrediction())  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "tadpole.yaml")
        self.validate_source(root)
        super().__init__(
            root=root,
            tables=["tadpole"],
            dataset_name=dataset_name or "tadpole",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def validate_source(root: str) -> Path:
        """Checks that ``root`` holds a readable TADPOLE file.

        Args:
            root: Directory expected to contain ``TADPOLE_D1_D2.csv``.

        Returns:
            The path of the TADPOLE file.

        Raises:
            FileNotFoundError: If the directory or the file is missing.
            ValueError: If required columns are missing from the header.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"TADPOLE root directory not found: {root}")
        source = root_path / TADPOLE_FILE_NAME
        if not source.is_file():
            raise FileNotFoundError(
                f"{TADPOLE_FILE_NAME} not found in {root}. Download the TADPOLE "
                "challenge package from the ADNI LONI archive "
                "(https://ida.loni.usc.edu) and place the file in this directory."
            )
        with open(source, newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        present = {name.strip().lower() for name in header}
        missing = [col for col in TADPOLE_REQUIRED_COLUMNS if col not in present]
        if missing:
            raise ValueError(
                f"{TADPOLE_FILE_NAME} is missing required columns: {missing}"
            )
        return source

    def preprocess_tadpole(self, df: nw.LazyFrame) -> nw.LazyFrame:
        """Strips the ``<`` and ``>`` markers from censored CSF values.

        Called automatically by :class:`BaseDataset` when the ``tadpole``
        table is loaded. Column names are already lowercase at this point.

        Args:
            df: The raw ``tadpole`` table, every column a string.

        Returns:
            The table with ``"<8"`` stored as ``"8"`` and ``">1300"`` as
            ``"1300"`` in the CSF columns.
        """
        return df.with_columns(
            [
                nw.col(col)
                .str.replace_all("<", "", literal=True)
                .str.replace_all(">", "", literal=True)
                .str.strip_chars()
                for col in TADPOLE_CENSORED_COLUMNS
            ]
        )

    @property
    def default_task(self) -> "ADProgressionPrediction":
        """Returns the default task, ``ADProgressionPrediction()``."""
        from ..tasks import ADProgressionPrediction

        return ADProgressionPrediction()


def _write_rows(root: Union[str, Path], rows: Sequence[Dict[str, object]]) -> Path:
    """Writes TADPOLE-format rows (lowercase keys) to ``root``."""
    os.makedirs(root, exist_ok=True)
    path = Path(root) / TADPOLE_FILE_NAME
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        for row in rows:
            writer.writerow(
                ["" if row.get(c) is None else row[c] for c in TADPOLE_REQUIRED_COLUMNS]
            )
    return path


def write_tadpole_fixture(root: Union[str, Path]) -> Path:
    """Writes a five-subject synthetic ``TADPOLE_D1_D2.csv`` for unit tests.

    The subjects cover every preprocessing rule of the paper:

    * ``"1"``: cognitively normal at all four visits (a stable sequence).
    * ``"2"``: MCI that converts to AD at the third visit (a converter).
    * ``"3"``: MCI that reverts to normal (a reverter, dropped by default).
    * ``"4"``: a single visit (dropped).
    * ``"5"``: normal, a visit with no diagnosis, then MCI and later AD, with
      censored CSF values (a CN to MCI converter).

    Args:
        root: Directory to write the file to. It is created if needed.

    Returns:
        The path of the written file.
    """

    def visit(
        rid: str,
        day: str,
        viscode: str,
        dx: Optional[str],
        mmse: Optional[float],
        adas13: Optional[float],
        hippo: Optional[float],
        csf: Optional[Tuple[str, str, str]] = None,
    ) -> Dict[str, object]:
        """Builds one visit row; unspecified features are left missing."""
        row: Dict[str, object] = {c: None for c in TADPOLE_REQUIRED_COLUMNS}
        row.update(
            rid=rid,
            examdate=day,
            viscode=viscode,
            dx=dx,
            d1=1,
            d2=0,
            mmse=mmse,
            adas13=adas13,
            adas11=None if adas13 is None else round(adas13 * 0.7, 2),
            cdrsb=0.5,
            hippocampus=hippo,
            icv=1500000,
        )
        if csf is not None:
            row.update(
                abeta_upennbiomk9_04_19_17=csf[0],
                tau_upennbiomk9_04_19_17=csf[1],
                ptau_upennbiomk9_04_19_17=csf[2],
            )
        return row

    rows = [
        visit("1", "2010-01-15", "bl", "NL", 29, 8.0, 7300, ("1200", "210", "19")),
        visit("1", "2010-07-15", "m06", "NL", 29, 8.5, 7290),
        visit("1", "2011-01-15", "m12", "NL", 30, 7.0, None),
        visit("1", "2012-01-15", "m24", "NL", 29, 8.0, 7250),
        visit("2", "2010-01-15", "bl", "MCI", 27, 18.0, 6400),
        visit("2", "2010-07-15", "m06", "MCI", 26, 20.5, None),
        visit("2", "2011-01-15", "m12", "MCI to Dementia", 23, 27.0, 6100),
        visit("2", "2012-01-15", "m24", "Dementia", 21, 31.0, 5900),
        visit("3", "2010-01-15", "bl", "MCI", 28, 12.0, 6900),
        visit("3", "2011-01-15", "m12", "MCI to NL", 29, 9.0, 6950),
        visit("3", "2012-01-15", "m24", "NL", 29, 9.0, 6940),
        visit("4", "2010-01-15", "bl", "NL", 30, 6.0, 7400),
        visit("5", "2010-01-15", "bl", "NL", 29, 9.0, 7100, ("1500", "<80", "<8")),
        visit("5", "2010-07-15", "m06", None, None, None, 7080),
        visit("5", "2011-01-15", "m12", "NL", 28, 11.0, 7020),
        visit("5", "2012-01-15", "m24", "NL to MCI", 27, 14.0, 6900),
        visit("5", "2013-01-15", "m36", "MCI", 26, 16.0, 6800),
        visit("5", "2014-01-15", "m48", "MCI to Dementia", 22, 28.0, 6300),
    ]
    return _write_rows(root, rows)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _features(
    severity: float, icv: float, rng: np.random.Generator
) -> Dict[str, float]:
    """Draws one visit's features from a latent severity (0 normal, 2 AD)."""
    s = severity
    noise = rng.normal
    adas13 = _clip(8 + 6 * s + 3 * s * s + noise(0, 3), 0, 85)
    return {
        "mmse": round(_clip(29.3 - 1.0 * s - 1.6 * s * s + noise(0, 1.2), 0, 30)),
        "adas13": round(adas13, 2),
        "adas11": round(_clip(adas13 * 0.68 + noise(0, 1), 0, 70), 2),
        "cdrsb": round(_clip(0.1 + 0.9 * s + 0.9 * s * s + noise(0, 0.6), 0, 18) * 2)
        / 2,
        "faq": round(_clip(0.2 + 1.5 * s + 3 * s * s + noise(0, 2), 0, 30)),
        "moca": round(_clip(26 - 1.5 * s - 1.6 * s * s + noise(0, 1.5), 0, 30)),
        "ravlt_immediate": round(_clip(45 - 7 * s - 2 * s * s + noise(0, 6), 0, 75)),
        "ravlt_learning": round(_clip(5.5 - 1.6 * s + noise(0, 1.8), -5, 14)),
        "ravlt_forgetting": round(_clip(4 + 0.4 * s + noise(0, 2), -5, 15)),
        "ravlt_perc_forgetting": round(_clip(40 + 25 * s + noise(0, 15), -100, 100), 2),
        "hippocampus": round(7300 - 500 * s - 200 * s * s + noise(0, 500)),
        "entorhinal": round(3800 - 450 * s + noise(0, 400)),
        "fusiform": round(17800 - 900 * s + noise(0, 1500)),
        "midtemp": round(19800 - 1100 * s + noise(0, 1700)),
        "ventricles": round(_clip(33000 + 9000 * s + noise(0, 9000), 5000, 2e5)),
        "wholebrain": round(1.04e6 - 40000 * s + noise(0, 60000)),
        "icv": round(icv),
        "fdg": round(1.30 - 0.12 * s + noise(0, 0.08), 4),
        "av45": round(1.05 + 0.15 * s + noise(0, 0.1), 4),
        "abeta_upennbiomk9_04_19_17": round(1300 - 330 * s + noise(0, 250), 1),
        "tau_upennbiomk9_04_19_17": round(220 + 60 * s + noise(0, 60), 1),
        "ptau_upennbiomk9_04_19_17": round(20 + 7 * s + noise(0, 6), 2),
    }


_MISSING_RATE: Dict[str, float] = {
    **{c: 0.05 for c in TADPOLE_FEATURE_COLUMNS[:10]},
    "moca": 0.35,
    **{c: 0.35 for c in TADPOLE_FEATURE_COLUMNS[10:17]},
    "fdg": 0.6,
    "av45": 0.75,
    **{c: 0.8 for c in TADPOLE_CENSORED_COLUMNS},
}


def generate_synthetic_tadpole(
    root: Union[str, Path], num_patients: int = 400, seed: int = 0
) -> Path:
    """Writes a synthetic TADPOLE cohort with realistic progression patterns.

    Subjects follow one of seven trajectories (stable CN, MCI or AD; CN to
    MCI; MCI to AD; a reverter; a single visit). Features are drawn from a
    latent severity that rises before a conversion, so a model can learn to
    anticipate converters. Missingness, censored CSF strings and irregular
    visit dates mimic the real file. No ADNI data is used.

    Args:
        root: Directory to write ``TADPOLE_D1_D2.csv`` to.
        num_patients: Number of synthetic subjects.
        seed: Random seed.

    Returns:
        The path of the written file.
    """
    rng = np.random.default_rng(seed)
    kinds = ["cn", "mci", "ad", "cn_mci", "mci_ad", "reverter", "single"]
    weights = [0.24, 0.24, 0.16, 0.08, 0.18, 0.05, 0.05]
    schedule = [0, 6, 12, 18, 24, 36, 48, 60, 72, 84]
    rows: List[Dict[str, object]] = []
    for rid in range(1, num_patients + 1):
        kind = kinds[int(rng.choice(len(kinds), p=weights))]
        n_visits = 1 if kind == "single" else int(rng.integers(3, 9))
        months = schedule[:n_visits]
        change = int(rng.integers(1, n_visits)) if n_visits > 1 else 0
        start = date(2005, 9, 1) + timedelta(days=int(rng.integers(0, 2500)))
        icv = rng.normal(1.53e6, 1.5e5)
        prev_dx = None
        for k, month in enumerate(months):
            if kind in ("cn", "single"):
                dx, sev = "CN", rng.normal(0.1, 0.1)
            elif kind == "mci":
                dx, sev = "MCI", rng.normal(0.9, 0.15) + 0.01 * month / 12
            elif kind == "ad":
                dx, sev = "AD", 1.9 + 0.08 * month / 12 + rng.normal(0, 0.1)
            elif kind == "cn_mci":
                dx = "CN" if k < change else "MCI"
                sev = 0.15 + 0.75 * (k + 1) / (change + 1) + rng.normal(0, 0.08)
            elif kind == "mci_ad":
                dx = "MCI" if k < change else "AD"
                sev = 1.0 + 0.85 * (k + 1) / (change + 1) + rng.normal(0, 0.1)
            else:
                dx = "MCI" if k < change else "CN"
                sev = 0.6 + rng.normal(0, 0.1)
            dx_text = {"CN": "NL", "MCI": "MCI", "AD": "Dementia"}[dx]
            if prev_dx is not None and prev_dx != dx_text:
                dx_text = f"{prev_dx} to {dx_text}"
            row: Dict[str, object] = dict(_features(max(sev, 0.0), icv, rng))
            for col, rate in _MISSING_RATE.items():
                if rng.random() < (rate * 0.5 if k == 0 else rate):
                    row[col] = None
            ptau = row["ptau_upennbiomk9_04_19_17"]
            if isinstance(ptau, float) and ptau < 8:
                row["ptau_upennbiomk9_04_19_17"] = "<8"
            tau = row["tau_upennbiomk9_04_19_17"]
            if isinstance(tau, float) and tau < 80:
                row["tau_upennbiomk9_04_19_17"] = "<80"
            abeta = row["abeta_upennbiomk9_04_19_17"]
            if isinstance(abeta, float) and abeta > 1700:
                row["abeta_upennbiomk9_04_19_17"] = ">1700"
            missing_dx = k > 0 and rng.random() < 0.08
            jitter = int(rng.integers(-10, 11))
            row.update(
                rid=str(rid),
                examdate=(
                    start + timedelta(days=int(month * 30.44) + jitter)
                ).isoformat(),
                viscode="bl" if month == 0 else f"m{month:02d}",
                dx=None if missing_dx else dx_text,
                d1=1,
                d2=int(rng.random() < 0.5),
            )
            if missing_dx:
                for col in TADPOLE_FEATURE_COLUMNS[:10]:
                    row[col] = None
            else:
                prev_dx = dx_text.split(" to ")[-1]
            rows.append(row)
    return _write_rows(root, rows)
