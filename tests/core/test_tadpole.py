"""Unit tests for TADPOLEDataset.

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
    Unit tests for TADPOLEDataset on synthetic data.

All tests use a five-subject synthetic file written to a temporary
directory; no ADNI data is used.
"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from pyhealth.datasets import TADPOLEDataset
from pyhealth.datasets.tadpole import (
    TADPOLE_FEATURE_COLUMNS,
    TADPOLE_FILE_NAME,
    TADPOLE_REQUIRED_COLUMNS,
    generate_synthetic_tadpole,
    write_tadpole_fixture,
)
from pyhealth.tasks import ADProgressionPrediction


def _is_missing(value: object) -> bool:
    """True for the ways a missing CSV cell can come back from the loader."""
    return value is None or str(value).strip() in ("", "nan", "<NA>", "None")


class TestTADPOLEDataset(unittest.TestCase):
    """Loading, parsing and validation of TADPOLE_D1_D2.csv."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        write_tadpole_fixture(cls.tmp)
        cls.dataset = TADPOLEDataset(root=cls.tmp, cache_dir=Path(cls.tmp) / "cache")
        cls.dataset.unique_patient_ids  # build the event table once, here

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _events(self, pid: str) -> list:
        patient = self.dataset.get_patient(pid)
        return sorted(
            patient.get_events(event_type="tadpole"), key=lambda e: e.timestamp
        )

    def test_dataset_name(self):
        self.assertEqual(self.dataset.dataset_name, "tadpole")

    def test_patients(self):
        self.assertEqual(
            sorted(self.dataset.unique_patient_ids), ["1", "2", "3", "4", "5"]
        )

    def test_event_counts(self):
        expected = {"1": 4, "2": 4, "3": 3, "4": 1, "5": 6}
        for pid, count in expected.items():
            self.assertEqual(len(self._events(pid)), count, pid)

    def test_timestamps_parsed_in_order(self):
        events = self._events("2")
        dates = [e.timestamp.date().isoformat() for e in events]
        self.assertEqual(
            dates, ["2010-01-15", "2010-07-15", "2011-01-15", "2012-01-15"]
        )

    def test_event_attributes(self):
        first = self._events("2")[0]
        self.assertEqual(first.dx, "MCI")
        self.assertEqual(first.viscode, "bl")
        self.assertEqual(float(first.mmse), 27.0)
        self.assertEqual(float(first.hippocampus), 6400.0)
        third = self._events("2")[2]
        self.assertEqual(third.dx, "MCI to Dementia")

    def test_censored_csf_values_are_cleaned(self):
        first = self._events("5")[0]
        self.assertEqual(float(first.ptau_upennbiomk9_04_19_17), 8.0)
        self.assertEqual(float(first.tau_upennbiomk9_04_19_17), 80.0)
        self.assertEqual(float(first.abeta_upennbiomk9_04_19_17), 1500.0)

    def test_missing_values_are_kept_missing(self):
        second = self._events("5")[1]
        self.assertTrue(_is_missing(second.dx))
        self.assertTrue(_is_missing(second.mmse))
        self.assertEqual(float(second.hippocampus), 7080.0)

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, ADProgressionPrediction)

    def test_default_task_on_parsed_patients(self):
        # Task functionality on the loaded dataset: one sample per usable
        # subject, with the next-visit label and the paper's sequence type.
        task = self.dataset.default_task
        samples = {}
        for pid in self.dataset.unique_patient_ids:
            for sample in task(self.dataset.get_patient(pid)):
                samples[pid] = sample
        self.assertEqual(sorted(samples), ["1", "2", "5"])
        self.assertEqual(
            {pid: (s["label"], s["seq_type"]) for pid, s in samples.items()},
            {
                "1": ("CN", "stable"),
                "2": ("AD", "converter"),
                "5": ("MCI", "converter"),
            },
        )
        for sample in samples.values():
            self.assertEqual(len(sample["visits"]), sample["group"])
            self.assertEqual(len(sample["visits"][0]), task.num_features)

    def test_set_task_builds_sample_dataset(self):
        samples = self.dataset.set_task()
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["visits"].shape[-1], 26)

    def test_config_lists_every_feature(self):
        config = Path(__file__).parents[2] / "pyhealth/datasets/configs/tadpole.yaml"
        with open(config) as handle:
            table = yaml.safe_load(handle)["tables"]["tadpole"]
        self.assertEqual(table["patient_id"], "rid")
        self.assertEqual(table["timestamp"], "examdate")
        for column in TADPOLE_FEATURE_COLUMNS + ("dx", "viscode"):
            self.assertIn(column, table["attributes"])


class TestTADPOLEValidation(unittest.TestCase):
    """Error handling for missing or malformed sources (no dataset build)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            TADPOLEDataset.validate_source(str(Path(self.tmp) / "absent"))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            TADPOLEDataset.validate_source(self.tmp)

    def test_missing_column_raises(self):
        header = [c.upper() for c in TADPOLE_REQUIRED_COLUMNS if c != "mmse"]
        with open(Path(self.tmp) / TADPOLE_FILE_NAME, "w", newline="") as handle:
            csv.writer(handle).writerow(header)
        with self.assertRaisesRegex(ValueError, "mmse"):
            TADPOLEDataset.validate_source(self.tmp)

    def test_fixture_passes_validation(self):
        path = write_tadpole_fixture(self.tmp)
        self.assertEqual(TADPOLEDataset.validate_source(self.tmp), path)

    def test_synthetic_cohort(self):
        path = generate_synthetic_tadpole(self.tmp, num_patients=5, seed=1)
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len({r["RID"] for r in rows}), 5)
        self.assertTrue(any(" to " in r["DX"] for r in rows))
        TADPOLEDataset.validate_source(self.tmp)


if __name__ == "__main__":
    unittest.main()
