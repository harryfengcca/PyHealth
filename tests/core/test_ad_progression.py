"""Unit tests for the ADProgressionPrediction task.

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
    Unit tests for ADProgressionPrediction on synthetic data.

The rule functions are tested on plain lists. The task itself runs on
in-memory :class:`~pyhealth.data.Patient` objects built from the five-subject
synthetic TADPOLE fixture, in the same event format the dataset produces, so
these tests need no dataset build and run in milliseconds. Loading the
fixture through ``TADPOLEDataset`` is tested in ``test_tadpole.py``. No ADNI
data is used.
"""

import math
import shutil
import tempfile
import unittest
from typing import Dict, Optional

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.tadpole import TADPOLE_FEATURE_COLUMNS, write_tadpole_fixture
from pyhealth.tasks import ADProgressionPrediction
from pyhealth.tasks.ad_progression import (
    DX_CLASSES,
    FEATURE_COLUMNS,
    build_sequence,
    forward_fill,
    map_diagnosis,
    parse_numeric,
)

HIPPO = FEATURE_COLUMNS.index("hippocampus")
PTAU = FEATURE_COLUMNS.index("ptau_upennbiomk9_04_19_17")


def fixture_patients(root: str) -> Dict[str, Patient]:
    """Builds the fixture's patients in memory, as TADPOLEDataset would.

    Every column is read as a string and missing cells stay null, matching
    the dataset's event table (``tadpole/<column>`` attributes).
    """
    path = write_tadpole_fixture(root)
    raw = pl.read_csv(path, infer_schema_length=0)
    raw = raw.rename({c: c.lower() for c in raw.columns})
    attributes = ["viscode", "dx", "d1", "d2", *TADPOLE_FEATURE_COLUMNS]
    events = raw.select(
        pl.col("rid").alias("patient_id"),
        pl.lit("tadpole").alias("event_type"),
        pl.col("examdate")
        .str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
        .alias("timestamp"),
        *[pl.col(c).alias(f"tadpole/{c}") for c in attributes],
    )
    return {
        pid: Patient(patient_id=pid, data_source=group)
        for (pid,), group in events.group_by("patient_id")
    }


class TestRuleFunctions(unittest.TestCase):
    """The paper's preprocessing rules, without a dataset."""

    def test_map_diagnosis(self):
        cases = {
            "NL": "CN",
            "CN": "CN",
            "MCI": "MCI",
            "Dementia": "AD",
            "NL to MCI": "MCI",
            "MCI to Dementia": "AD",
            "MCI to NL": "CN",
            "": None,
            None: None,
            "unknown": None,
        }
        for raw, expected in cases.items():
            self.assertEqual(map_diagnosis(raw), expected, raw)

    def test_parse_numeric(self):
        self.assertEqual(parse_numeric("27"), 27.0)
        self.assertEqual(parse_numeric("<8"), 8.0)
        self.assertEqual(parse_numeric(">1300"), 1300.0)
        self.assertEqual(parse_numeric(3), 3.0)
        for missing in (None, "", " ", "abc"):
            self.assertTrue(math.isnan(parse_numeric(missing)))

    def test_stable_sequence(self):
        self.assertEqual(build_sequence(["CN", "CN", "CN"]), ([0, 1], 2, "stable"))

    def test_converter_truncated_at_first_conversion(self):
        self.assertEqual(
            build_sequence(["CN", "CN", "MCI", "MCI", "AD"]), ([0, 1], 2, "converter")
        )

    def test_missing_diagnosis_visits_are_skipped(self):
        self.assertEqual(
            build_sequence(["CN", None, "CN", "MCI"]), ([0, 2], 3, "converter")
        )

    def test_single_visit_is_dropped(self):
        self.assertIsNone(build_sequence(["MCI"]))

    def test_too_few_diagnosed_visits_is_dropped(self):
        self.assertIsNone(build_sequence(["MCI", None, None]))

    def test_reverter_dropped_or_kept(self):
        self.assertIsNone(build_sequence(["MCI", "CN", "CN"]))
        self.assertEqual(
            build_sequence(["MCI", "CN", "CN"], drop_reverters=False),
            ([0], 1, "reverter"),
        )
        # A reverter anywhere drops the subject, even after a conversion.
        self.assertIsNone(build_sequence(["CN", "MCI", "CN"]))

    def test_forward_fill(self):
        nan = math.nan
        filled = forward_fill([[nan, 1.0], [2.0, nan], [nan, nan]])
        self.assertTrue(math.isnan(filled[0][0]))
        self.assertEqual(filled[1], [2.0, 1.0])
        self.assertEqual(filled[2], [2.0, 1.0])

    def test_feature_columns_match_dataset(self):
        self.assertEqual(FEATURE_COLUMNS, TADPOLE_FEATURE_COLUMNS)

    def test_invalid_time_feature(self):
        with self.assertRaises(ValueError):
            ADProgressionPrediction(time_feature="weeks")

    def test_schema_follows_options(self):
        task = ADProgressionPrediction(missing_indicators=True, normalize=False)
        _, options = task.input_schema["visits"]
        self.assertTrue(options["missing_indicators"])
        self.assertIsNone(options["normalization"])
        self.assertEqual(task.output_schema, {"label": "multiclass"})
        self.assertEqual(ADProgressionPrediction().num_features, 26)
        self.assertEqual(ADProgressionPrediction(time_feature="none").num_features, 25)


class TestADProgressionPrediction(unittest.TestCase):
    """The task on the synthetic five-subject fixture."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp()
        cls.patients = fixture_patients(cls.tmp)
        cls.task = ADProgressionPrediction()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _raw(self, pid: str, task: Optional[ADProgressionPrediction] = None) -> list:
        return (task or self.task)(self.patients[pid])

    def test_dropped_subjects(self):
        self.assertEqual(self._raw("3"), [])  # reverter
        self.assertEqual(self._raw("4"), [])  # single visit

    def test_stable_sample(self):
        (sample,) = self._raw("1")
        self.assertEqual(sample["label"], "CN")
        self.assertEqual(sample["seq_type"], "stable")
        self.assertEqual(sample["group"], 3)
        self.assertEqual(sample["last_dx"], "CN")
        self.assertEqual(len(sample["visits"]), 3)
        self.assertEqual(len(sample["visits"][0]), 26)

    def test_converter_samples(self):
        (mci_ad,) = self._raw("2")
        self.assertEqual(
            (mci_ad["label"], mci_ad["seq_type"], mci_ad["group"]),
            ("AD", "converter", 2),
        )
        (cn_mci,) = self._raw("5")
        # The visit without a diagnosis is skipped; the target is the m24 visit.
        self.assertEqual(
            (cn_mci["label"], cn_mci["seq_type"], cn_mci["group"]),
            ("MCI", "converter", 2),
        )
        self.assertEqual(cn_mci["last_dx"], "CN")

    def test_diagnosis_one_hot_and_time_to_target(self):
        (sample,) = self._raw("2")
        first, second = sample["visits"]
        n = len(FEATURE_COLUMNS)
        self.assertEqual(first[n : n + 3], [0.0, 1.0, 0.0])  # MCI
        self.assertAlmostEqual(first[-1], 12.0, delta=0.1)
        self.assertAlmostEqual(second[-1], 6.0, delta=0.1)

    def test_forward_fill_and_missing_values(self):
        (sample,) = self._raw("1")
        self.assertEqual(sample["visits"][2][HIPPO], 7290.0)  # filled from m06
        (no_fill,) = self._raw("1", ADProgressionPrediction(forward_fill=False))
        self.assertTrue(math.isnan(no_fill["visits"][2][HIPPO]))

    def test_censored_value_parsed(self):
        (sample,) = self._raw("5")
        self.assertEqual(sample["visits"][0][PTAU], 8.0)

    def test_time_feature_options(self):
        (since,) = self._raw(
            "2", ADProgressionPrediction(time_feature="since_previous")
        )
        self.assertEqual(since["visits"][0][-1], 0.0)
        self.assertAlmostEqual(since["visits"][1][-1], 6.0, delta=0.1)
        (none,) = self._raw("2", ADProgressionPrediction(time_feature="none"))
        self.assertEqual(len(none["visits"][0]), 25)

    def test_keep_reverters(self):
        (sample,) = self._raw("3", ADProgressionPrediction(drop_reverters=False))
        self.assertEqual(
            (sample["label"], sample["seq_type"], sample["group"]),
            ("CN", "reverter", 1),
        )

    def test_raw_sequence(self):
        seq = self.task.raw_sequence(self.patients["2"])
        self.assertEqual(seq["diagnoses"], ["MCI", "MCI", "AD"])
        self.assertEqual((seq["group"], seq["seq_type"]), (2, "converter"))
        self.assertEqual(len(seq["features"]), 3)  # two inputs and the target
        self.assertEqual(seq["months"][0], 0.0)
        self.assertAlmostEqual(seq["months"][1], 6.0, delta=0.1)
        self.assertAlmostEqual(seq["months"][2], 12.0, delta=0.1)
        self.assertTrue(math.isnan(seq["features"][1][HIPPO]))  # not filled
        self.assertIsNone(self.task.raw_sequence(self.patients["3"]))  # reverter
        self.assertIsNone(self.task.raw_sequence(self.patients["4"]))  # one visit

    def test_imputer_fills_missing_inputs(self):
        calls = []

        def imputer(pid, rows, months):
            calls.append((pid, len(rows), len(months)))
            return [[99.0 if math.isnan(v) else v for v in row] for row in rows]

        task = ADProgressionPrediction(forward_fill=False, imputer=imputer)
        (sample,) = task(self.patients["1"])
        self.assertEqual(calls, [("1", 3, 3)])  # input visits only
        self.assertEqual(sample["visits"][0][HIPPO], 7300.0)  # observed kept
        self.assertEqual(sample["visits"][2][HIPPO], 99.0)  # missing filled

    def test_processed_tensors(self):
        # The task's own schema turns the samples into padded-ready tensors.
        raw = [s for pid in sorted(self.patients) for s in self._raw(pid)]
        samples = create_sample_dataset(
            samples=raw,
            input_schema=self.task.input_schema,
            output_schema=self.task.output_schema,
        )
        self.assertEqual(len(samples), 3)
        by_pid = {s["patient_id"]: s for s in samples}
        self.assertEqual(sorted(by_pid), ["1", "2", "5"])
        visits = by_pid["1"]["visits"]
        self.assertEqual(tuple(visits.shape), (3, 26))
        self.assertFalse(torch.isnan(visits).any())
        vocab = samples.output_processors["label"].label_vocab
        self.assertEqual(sorted(vocab), sorted(DX_CLASSES))
        self.assertEqual(int(by_pid["2"]["label"]), vocab["AD"])


if __name__ == "__main__":
    unittest.main()
