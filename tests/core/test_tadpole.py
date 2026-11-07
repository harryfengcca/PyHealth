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
    Unit tests for TADPOLEDataset on synthetic data: data loading, patient
    and event parsing, data integrity, validation, and the default task.

All tests use a five-subject synthetic file (``write_tadpole_fixture``)
written to a temporary directory that is removed afterwards; PyHealth's cache
is kept inside that directory. No ADNI data is used.
"""

import contextlib
import csv
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

from pyhealth.data import Event
from pyhealth.datasets import TADPOLEDataset, base_dataset
from pyhealth.datasets.tadpole import (
    TADPOLE_FEATURE_COLUMNS,
    TADPOLE_FILE_NAME,
    TADPOLE_REQUIRED_COLUMNS,
    generate_synthetic_tadpole,
    write_tadpole_fixture,
)
from pyhealth.tasks import ADProgressionPrediction


def _thread_cluster() -> contextlib.AbstractContextManager:
    """Lets PyHealth's one-time event-table build use Dask threads.

    ``BaseDataset`` starts a Dask cluster of worker *processes* outside
    notebooks; on five subjects their start-up dominates the build (about
    1.5 s). Threads give the same table in about 0.7 s. Falls back to the
    normal build if PyHealth no longer has the ``in_notebook`` switch.
    """
    if hasattr(base_dataset, "in_notebook"):
        return mock.patch.object(base_dataset, "in_notebook", return_value=True)
    return contextlib.nullcontext()


def _is_missing(value: object) -> bool:
    """True for the ways a missing CSV cell can come back from the loader."""
    return value is None or str(value).strip() in ("", "nan", "<NA>", "None")


class TestTADPOLEDataset(unittest.TestCase):
    """Loading and parsing of the five-subject fixture.

    The dataset is built once for the class (under a second, see
    :func:`_thread_cluster`), and every test below runs in milliseconds.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Writes the fixture and builds the dataset in a temporary folder."""
        cls.tmp = tempfile.mkdtemp()
        cls.csv_path = write_tadpole_fixture(cls.tmp)
        cls.dataset = TADPOLEDataset(root=cls.tmp, cache_dir=Path(cls.tmp) / "cache")
        with _thread_cluster():
            _ = cls.dataset.unique_patient_ids  # build the event table once, here

    @classmethod
    def tearDownClass(cls) -> None:
        """Removes the temporary folder, cache included."""
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _events(self, pid: str) -> list[Event]:
        """The ``tadpole`` events of one subject, in time order."""
        patient = self.dataset.get_patient(pid)
        return sorted(
            patient.get_events(event_type="tadpole"), key=lambda e: e.timestamp
        )

    # Data loading -----------------------------------------------------------

    def test_dataset_name(self) -> None:
        """The dataset is named after the table it loads."""
        self.assertEqual(self.dataset.dataset_name, "tadpole")

    def test_cache_stays_in_temporary_folder(self) -> None:
        """Processed files go to the given cache folder, not the home cache."""
        self.assertTrue(any((Path(self.tmp) / "cache").iterdir()))

    def test_config_lists_every_feature(self) -> None:
        """The YAML config maps RID, EXAMDATE and every used column."""
        config = Path(__file__).parents[2] / "pyhealth/datasets/configs/tadpole.yaml"
        with open(config) as handle:
            table = yaml.safe_load(handle)["tables"]["tadpole"]
        self.assertEqual(table["file_path"], TADPOLE_FILE_NAME)
        self.assertEqual(table["patient_id"], "rid")
        self.assertEqual(table["timestamp"], "examdate")
        for column in (*TADPOLE_FEATURE_COLUMNS, "dx", "viscode"):
            self.assertIn(column, table["attributes"])

    # Patient parsing --------------------------------------------------------

    def test_patients(self) -> None:
        """Every RID becomes one patient."""
        self.assertEqual(
            sorted(self.dataset.unique_patient_ids), ["1", "2", "3", "4", "5"]
        )

    def test_event_counts(self) -> None:
        """Every visit row becomes one event of its subject."""
        expected = {"1": 4, "2": 4, "3": 3, "4": 1, "5": 6}
        for pid, count in expected.items():
            self.assertEqual(len(self._events(pid)), count, pid)

    # Event parsing ----------------------------------------------------------

    def test_timestamps_parsed_in_order(self) -> None:
        """EXAMDATE is parsed as the event timestamp."""
        dates = [e.timestamp.date().isoformat() for e in self._events("2")]
        self.assertEqual(
            dates, ["2010-01-15", "2010-07-15", "2011-01-15", "2012-01-15"]
        )

    def test_event_attributes(self) -> None:
        """Diagnosis, visit code and features are kept on each event."""
        first = self._events("2")[0]
        self.assertEqual(first.dx, "MCI")
        self.assertEqual(first.viscode, "bl")
        self.assertEqual(float(first.mmse), 27.0)
        self.assertEqual(float(first.hippocampus), 6400.0)
        self.assertEqual(self._events("2")[2].dx, "MCI to Dementia")

    def test_censored_csf_values_are_cleaned(self) -> None:
        """``"<8"`` and ``"<80"`` are loaded as plain numbers."""
        first = self._events("5")[0]
        self.assertEqual(float(first.ptau_upennbiomk9_04_19_17), 8.0)
        self.assertEqual(float(first.tau_upennbiomk9_04_19_17), 80.0)
        self.assertEqual(float(first.abeta_upennbiomk9_04_19_17), 1500.0)

    # Data integrity ---------------------------------------------------------

    def test_every_row_is_loaded_once(self) -> None:
        """The number of events equals the number of rows in the file."""
        with open(self.csv_path, newline="") as handle:
            n_rows = sum(1 for _ in csv.DictReader(handle))
        n_events = sum(
            len(self._events(pid)) for pid in self.dataset.unique_patient_ids
        )
        self.assertEqual(n_events, n_rows)

    def test_every_event_has_all_features(self) -> None:
        """Each event carries all 22 feature attributes, missing or not."""
        for event in self._events("1"):
            for column in TADPOLE_FEATURE_COLUMNS:
                self.assertIn(column, event.attr_dict)

    def test_missing_values_are_kept_missing(self) -> None:
        """Empty cells stay missing; they are not turned into zeros."""
        second = self._events("5")[1]
        self.assertTrue(_is_missing(second.dx))
        self.assertTrue(_is_missing(second.mmse))
        self.assertEqual(float(second.hippocampus), 7080.0)

    # Task functionality -----------------------------------------------------

    def test_default_task(self) -> None:
        """The default task is the paper's next-visit prediction."""
        self.assertIsInstance(self.dataset.default_task, ADProgressionPrediction)

    def test_default_task_on_parsed_patients(self) -> None:
        """One sample per usable subject, with its label and sequence type."""
        task = self.dataset.default_task
        samples = {}
        for pid in self.dataset.unique_patient_ids:
            for sample in task(self.dataset.get_patient(pid)):
                samples[pid] = sample
        self.assertEqual(sorted(samples), ["1", "2", "5"])  # 3, 4 are dropped
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

    def test_task_is_fast(self) -> None:
        """The task processes every fixture subject in well under a second."""
        task = self.dataset.default_task
        patients = [
            self.dataset.get_patient(p) for p in self.dataset.unique_patient_ids
        ]
        start = time.perf_counter()
        for patient in patients:
            task(patient)
        self.assertLess(time.perf_counter() - start, 0.5)

    def test_set_task_builds_sample_dataset(self) -> None:
        """``set_task`` turns the samples into tensors."""
        samples = self.dataset.set_task()
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["visits"].shape[-1], 26)


class TestTADPOLEValidation(unittest.TestCase):
    """Error handling for missing or malformed sources (no dataset build)."""

    def setUp(self) -> None:
        """Creates an empty temporary folder."""
        self.tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Removes the temporary folder."""
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_root_raises(self) -> None:
        """A root folder that does not exist is reported."""
        with self.assertRaises(FileNotFoundError):
            TADPOLEDataset.validate_source(str(Path(self.tmp) / "absent"))

    def test_missing_file_raises(self) -> None:
        """A root folder without TADPOLE_D1_D2.csv is reported."""
        with self.assertRaises(FileNotFoundError):
            TADPOLEDataset.validate_source(self.tmp)

    def test_constructor_validates_before_loading(self) -> None:
        """The dataset checks its source before PyHealth reads anything."""
        with self.assertRaisesRegex(FileNotFoundError, TADPOLE_FILE_NAME):
            TADPOLEDataset(root=self.tmp, cache_dir=Path(self.tmp) / "cache")

    def test_missing_column_raises(self) -> None:
        """A file without a required column names the missing column."""
        header = [c.upper() for c in TADPOLE_REQUIRED_COLUMNS if c != "mmse"]
        with open(Path(self.tmp) / TADPOLE_FILE_NAME, "w", newline="") as handle:
            csv.writer(handle).writerow(header)
        with self.assertRaisesRegex(ValueError, "mmse"):
            TADPOLEDataset.validate_source(self.tmp)

    def test_fixture_passes_validation(self) -> None:
        """The unit-test fixture has the TADPOLE layout."""
        path = write_tadpole_fixture(self.tmp)
        self.assertEqual(TADPOLEDataset.validate_source(self.tmp), path)

    def test_synthetic_cohort(self) -> None:
        """The synthetic generator writes valid files with conversions."""
        path = generate_synthetic_tadpole(self.tmp, num_patients=5, seed=1)
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len({r["RID"] for r in rows}), 5)
        self.assertTrue(any(" to " in r["DX"] for r in rows))
        TADPOLEDataset.validate_source(self.tmp)


if __name__ == "__main__":
    unittest.main()
