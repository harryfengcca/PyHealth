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
from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.datasets.tadpole import TADPOLE_FEATURE_COLUMNS, write_tadpole_fixture
from pyhealth.tasks import ADProgressionPrediction
from pyhealth.tasks.ad_progression import (
    DX_CLASSES,
    FEATURE_COLUMNS,
    build_sequence,
    forward_fill,
    map_diagnosis,
    parse_numeric,
    prepare_patient_visits,
    SEQUENCE_POLICY_VERSION,
)

HIPPO = FEATURE_COLUMNS.index("hippocampus")
PTAU = FEATURE_COLUMNS.index("ptau_upennbiomk9_04_19_17")


def fixture_patients(root: str) -> dict[str, Patient]:
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

    def test_map_diagnosis(self) -> None:
        """Raw TADPOLE diagnosis strings map to CN, MCI, AD or None."""
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

    def test_parse_numeric(self) -> None:
        """Numbers, censored strings and missing markers parse to floats."""
        self.assertEqual(parse_numeric("27"), 27.0)
        self.assertEqual(parse_numeric("<8"), 8.0)
        self.assertEqual(parse_numeric(">1300"), 1300.0)
        self.assertEqual(parse_numeric(3), 3.0)
        for missing in (None, "", " ", "abc"):
            self.assertTrue(math.isnan(parse_numeric(missing)))

    def test_stable_sequence(self) -> None:
        """An unchanged diagnosis gives a stable sequence ending at the last visit."""
        self.assertEqual(build_sequence(["CN", "CN", "CN"]), ([0, 1], 2, "stable"))

    def test_converter_truncated_at_first_conversion(self) -> None:
        """Visits after the first conversion are discarded (rule 3)."""
        self.assertEqual(
            build_sequence(["CN", "CN", "MCI", "MCI", "AD"]), ([0, 1], 2, "converter")
        )

    def test_missing_diagnosis_visits_are_skipped(self) -> None:
        """Visits without a diagnosis are skipped (rule 4)."""
        self.assertEqual(
            build_sequence(["CN", None, "CN", "MCI"]), ([0, 2], 3, "converter")
        )

    def test_single_visit_is_dropped(self) -> None:
        """Subjects with one visit are dropped (rule 1)."""
        self.assertIsNone(build_sequence(["MCI"]))

    def test_too_few_diagnosed_visits_is_dropped(self) -> None:
        """Subjects with fewer than two diagnosed visits are dropped."""
        self.assertIsNone(build_sequence(["MCI", None, None]))

    def test_reverter_dropped_or_kept(self) -> None:
        """Reverters are dropped (rule 2) unless ``drop_reverters=False``."""
        self.assertIsNone(build_sequence(["MCI", "CN", "CN"]))
        self.assertEqual(
            build_sequence(["MCI", "CN", "CN"], drop_reverters=False),
            ([0], 1, "reverter"),
        )
        # A reverter anywhere drops the subject, even after a conversion.
        self.assertIsNone(build_sequence(["CN", "MCI", "CN"]))

    def test_forward_fill(self) -> None:
        """Missing values take the last observed value; leading gaps stay nan."""
        nan = math.nan
        filled = forward_fill([[nan, 1.0], [2.0, nan], [nan, nan]])
        self.assertTrue(math.isnan(filled[0][0]))
        self.assertEqual(filled[1], [2.0, 1.0])
        self.assertEqual(filled[2], [2.0, 1.0])

    def test_feature_columns_match_dataset(self) -> None:
        """The task reads the same 22 features the dataset loads."""
        self.assertEqual(FEATURE_COLUMNS, TADPOLE_FEATURE_COLUMNS)

    def test_invalid_time_feature(self) -> None:
        """An unknown time feature is rejected."""
        with self.assertRaises(ValueError):
            ADProgressionPrediction(time_feature="weeks")

    def test_schema_follows_options(self) -> None:
        """The input schema and width follow the task options."""
        task = ADProgressionPrediction(missing_indicators=True, normalize=False)
        _, options = task.input_schema["visits"]
        self.assertTrue(options["missing_indicators"])
        self.assertIsNone(options["normalization"])
        self.assertEqual(task.output_schema, {"label": "multiclass"})
        self.assertEqual(ADProgressionPrediction().num_features, 26)
        self.assertEqual(ADProgressionPrediction(time_feature="none").num_features, 25)


class TestVisitPreparation(unittest.TestCase):
    """Shared eligibility and duplicate rules on synthetic events only."""

    @staticmethod
    def _event(month: int, dx: str | None, viscode: str | None = "bl", **values):
        return SimpleNamespace(
            timestamp=datetime(2020, 1, 1) + timedelta(days=30 * month),
            dx=dx, viscode=viscode, **values,
        )

    @staticmethod
    def _patient(events):
        return SimpleNamespace(
            patient_id="synthetic", get_events=lambda **kwargs: list(events)
        )

    def test_explicit_reversion_without_observed_decrease(self) -> None:
        patient = self._patient([
            self._event(0, "NL"), self._event(6, " MCI  TO  nl ", "m06"),
            self._event(12, "NL", "m12"),
        ])
        prepared = prepare_patient_visits(patient)
        self.assertEqual(prepared.exclusion_reason, "reverter")
        self.assertEqual(
            prepared.diagnostics["additional_explicit_reversion_exclusions"], 1
        )
        self.assertEqual(map_diagnosis(" MCI  TO  nl "), "CN")
        self.assertEqual(ADProgressionPrediction()(patient), [])
        self.assertIsNotNone(prepare_patient_visits(patient, False).sequence)

    def test_explicit_reversion_after_first_progression(self) -> None:
        patient = self._patient([
            self._event(0, "NL"), self._event(6, "MCI", "m06"),
            self._event(12, "Dementia to MCI", "m12"),
        ])
        self.assertEqual(prepare_patient_visits(patient).exclusion_reason, "reverter")

    def test_observed_reversion_without_transition_text(self) -> None:
        patient = self._patient([self._event(0, "MCI"), self._event(6, "NL", "m06")])
        prepared = prepare_patient_visits(patient)
        self.assertEqual(prepared.exclusion_reason, "reverter")
        self.assertEqual(prepared.diagnostics["explicit_reversion_subjects"], 0)
        self.assertEqual(prepared.diagnostics["observed_reversion_subjects"], 1)

    def test_identical_rows_collapse_without_changing_label(self) -> None:
        first = self._event(0, "MCI", mmse=27)
        target = self._event(6, "Dementia", "m06", mmse=20)
        task = ADProgressionPrediction(normalize=False)
        original = task(self._patient([first, target]))[0]
        for rows in ([first, first, target], [target, first, first]):
            with self.subTest(order=rows[0].timestamp):
                patient = self._patient(rows)
                prepared = prepare_patient_visits(patient)
                sample = task(patient)[0]
                self.assertEqual(prepared.diagnostics["collapsed_source_rows"], 1)
                self.assertEqual(sample["label"], original["label"])
                self.assertEqual(sample["group"], original["group"])
                self.assertEqual(len(prepared.source_events), 3)

    def test_complementary_rows_merge_and_do_not_use_target_features(self) -> None:
        left = self._event(0, None, mmse=27)
        right = self._event(0, "MCI", hippocampus=6000)
        target = self._event(6, "Dementia", "m06", mmse=20, hippocampus=5000)
        for rows in ([left, right, target], [target, right, left]):
            with self.subTest(first=rows[0].timestamp):
                patient = self._patient(rows)
                prepared = prepare_patient_visits(patient)
                self.assertEqual(prepared.diagnostics["complementary_merge_groups"], 1)
                seq = ADProgressionPrediction().raw_sequence(patient)
                self.assertEqual(seq["diagnoses"], ["MCI", "AD"])
                self.assertEqual(seq["group"], 1)
                self.assertEqual(seq["features"][0][FEATURE_COLUMNS.index("mmse")], 27)
                self.assertEqual(seq["features"][0][HIPPO], 6000)
        self.assertIsNone(left.dx)
        self.assertFalse(hasattr(left, "hippocampus"))

    def test_merging_preserves_explicit_reversion_information(self) -> None:
        patient = self._patient([
            self._event(0, "MCI to NL", mmse=29),
            self._event(0, "NL", hippocampus=7000),
            self._event(6, "NL", "m06"),
        ])
        prepared = prepare_patient_visits(patient)
        self.assertEqual(prepared.diagnostics["complementary_merge_groups"], 1)
        self.assertEqual(prepared.exclusion_reason, "reverter")

    def test_conflicting_diagnoses_or_features_exclude_subject(self) -> None:
        for other in (self._event(0, "NL", mmse=27),
                      self._event(0, "MCI", mmse=28)):
            with self.subTest(diagnosis=other.dx):
                patient = self._patient([
                    self._event(0, "MCI", mmse=27), other,
                    self._event(6, "Dementia", "m06"),
                ])
                prepared = prepare_patient_visits(patient)
                self.assertEqual(
                    prepared.exclusion_reason, "conflicting_duplicate_visit"
                )
                self.assertEqual(ADProgressionPrediction()(patient), [])

    def test_distinct_codes_at_same_timestamp_are_ambiguous(self) -> None:
        patient = self._patient([
            self._event(0, "MCI", "bl"), self._event(0, "MCI", "m06"),
            self._event(12, "Dementia", "m12"),
        ])
        prepared = prepare_patient_visits(patient)
        self.assertEqual(prepared.exclusion_reason, "ambiguous_same_day_visits")
        self.assertEqual(prepared.diagnostics["collapsed_source_rows"], 0)

    def test_missing_visit_identity_does_not_allow_complementary_merge(self) -> None:
        patient = self._patient([
            self._event(0, "MCI", None, mmse=27),
            self._event(0, "MCI", None, hippocampus=6000),
            self._event(6, "Dementia", "m06"),
        ])
        self.assertEqual(
            prepare_patient_visits(patient).exclusion_reason,
            "ambiguous_same_day_visits",
        )

    def test_duplicate_only_history_has_insufficient_distinct_visits(self) -> None:
        event = self._event(0, "MCI")
        prepared = prepare_patient_visits(self._patient([event, event]))
        self.assertEqual(prepared.exclusion_reason, "fewer_than_two_distinct_visits")

    def test_duplicate_target_is_not_reused_as_an_input_visit(self) -> None:
        target = self._event(6, "MCI", "m06", mmse=24)
        patient = self._patient([self._event(0, "MCI", mmse=27), target, target])
        sequence = ADProgressionPrediction().raw_sequence(patient)
        self.assertEqual(sequence["group"], 1)
        self.assertEqual(sequence["features"][0][FEATURE_COLUMNS.index("mmse")], 27)
        self.assertEqual(sequence["features"][1][FEATURE_COLUMNS.index("mmse")], 24)

    def test_conflict_after_first_conversion_still_excludes_subject(self) -> None:
        patient = self._patient([
            self._event(0, "NL"), self._event(6, "MCI", "m06"),
            self._event(12, "MCI", "m12", mmse=27),
            self._event(12, "MCI", "m12", mmse=26),
        ])
        self.assertEqual(
            prepare_patient_visits(patient).exclusion_reason,
            "conflicting_duplicate_visit",
        )

    def test_task_cache_contains_sequence_policy_version(self) -> None:
        self.assertEqual(
            vars(ADProgressionPrediction())["_sequence_policy_version"],
            SEQUENCE_POLICY_VERSION,
        )


class TestADProgressionPrediction(unittest.TestCase):
    """Sample processing, labels, features and edge cases on the fixture.

    Subject 1 is stable, 2 converts MCI to AD, 3 reverts, 4 has one visit and
    5 converts CN to MCI after a visit without a diagnosis.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Builds the fixture's patients in a temporary folder."""
        cls.tmp = tempfile.mkdtemp()
        cls.patients = fixture_patients(cls.tmp)
        cls.task = ADProgressionPrediction()

    @classmethod
    def tearDownClass(cls) -> None:
        """Removes the temporary folder."""
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _raw(self, pid: str, task: ADProgressionPrediction | None = None) -> list[dict]:
        """Runs ``task`` (the default task if ``None``) on one subject."""
        return (task or self.task)(self.patients[pid])

    def test_patient_without_visits(self) -> None:
        """A subject with no TADPOLE events gives no sample and no error."""
        empty = self.patients["1"].data_source.clear()
        patient = Patient(patient_id="0", data_source=empty)
        self.assertEqual(self.task(patient), [])
        self.assertIsNone(self.task.raw_sequence(patient))

    def test_dropped_subjects(self) -> None:
        """Reverters and single-visit subjects give no sample."""
        self.assertEqual(self._raw("3"), [])  # reverter
        self.assertEqual(self._raw("4"), [])  # single visit

    def test_stable_sample(self) -> None:
        """A stable subject gives a CN label from its first three visits."""
        (sample,) = self._raw("1")
        self.assertEqual(sample["label"], "CN")
        self.assertEqual(sample["seq_type"], "stable")
        self.assertEqual(sample["group"], 3)
        self.assertEqual(sample["last_dx"], "CN")
        self.assertEqual(len(sample["visits"]), 3)
        self.assertEqual(len(sample["visits"][0]), 26)

    def test_visit_masks_survive_processing_and_padding(self) -> None:
        """Visit validity is preserved through processing and batch padding."""
        raw = [s for pid in sorted(self.patients) for s in self._raw(pid)]
        lengths = {s["patient_id"]: len(s["visits"]) for s in raw}
        samples = create_sample_dataset(
            samples=raw,
            input_schema=self.task.input_schema,
            output_schema=self.task.output_schema,
        )
        batch = next(iter(get_dataloader(samples, batch_size=3)))
        self.assertEqual(batch["visits_mask"].dtype, torch.bool)
        self.assertEqual(batch["visits_mask"].shape, batch["visits"].shape[:2])
        for i, pid in enumerate(batch["patient_id"]):
            n = lengths[pid]
            self.assertTrue(batch["visits_mask"][i, :n].all())
            self.assertFalse(batch["visits_mask"][i, n:].any())

    def test_converter_samples(self) -> None:
        """Converters are labeled with the stage they convert to."""
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

    def test_diagnosis_one_hot_and_time_to_target(self) -> None:
        """Each visit row ends with the diagnosis one-hot and months to target."""
        (sample,) = self._raw("2")
        first, second = sample["visits"]
        n = len(FEATURE_COLUMNS)
        self.assertEqual(first[n : n + 3], [0.0, 1.0, 0.0])  # MCI
        self.assertAlmostEqual(first[-1], 12.0, delta=0.1)
        self.assertAlmostEqual(second[-1], 6.0, delta=0.1)

    def test_forward_fill_and_missing_values(self) -> None:
        """Forward fill fills a missing feature; without it the value stays nan."""
        (sample,) = self._raw("1")
        self.assertEqual(sample["visits"][2][HIPPO], 7290.0)  # filled from m06
        (no_fill,) = self._raw("1", ADProgressionPrediction(forward_fill=False))
        self.assertTrue(math.isnan(no_fill["visits"][2][HIPPO]))

    def test_censored_value_parsed(self) -> None:
        """A censored CSF value is used as its bound."""
        (sample,) = self._raw("5")
        self.assertEqual(sample["visits"][0][PTAU], 8.0)

    def test_time_feature_options(self) -> None:
        """Months since the previous visit, or no time feature at all."""
        (since,) = self._raw(
            "2", ADProgressionPrediction(time_feature="since_previous")
        )
        self.assertEqual(since["visits"][0][-1], 0.0)
        self.assertAlmostEqual(since["visits"][1][-1], 6.0, delta=0.1)
        (none,) = self._raw("2", ADProgressionPrediction(time_feature="none"))
        self.assertEqual(len(none["visits"][0]), 25)

    def test_keep_reverters(self) -> None:
        """With ``drop_reverters=False`` a reverter gives a sample."""
        (sample,) = self._raw("3", ADProgressionPrediction(drop_reverters=False))
        self.assertEqual(
            (sample["label"], sample["seq_type"], sample["group"]),
            ("CN", "reverter", 1),
        )

    def test_raw_sequence(self) -> None:
        """``raw_sequence`` returns the unprocessed visits, inputs and target."""
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

    def test_imputer_fills_missing_inputs(self) -> None:
        """An imputer fills only missing input values and sees only inputs."""
        calls = []

        def imputer(
            pid: str, rows: list[list[float]], months: list[float]
        ) -> list[list[float]]:
            """Records the call and fills every missing value with 99."""
            calls.append((pid, len(rows), len(months)))
            return [[99.0 if math.isnan(v) else v for v in row] for row in rows]

        task = ADProgressionPrediction(forward_fill=False, imputer=imputer)
        (sample,) = task(self.patients["1"])
        self.assertEqual(calls, [("1", 3, 3)])  # input visits only
        self.assertEqual(sample["visits"][0][HIPPO], 7300.0)  # observed kept
        self.assertEqual(sample["visits"][2][HIPPO], 99.0)  # missing filled

    def test_processed_tensors(self) -> None:
        """The task's schema turns samples into tensors without nan."""
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

    def test_missing_indicators_double_the_width(self) -> None:
        """Missing-value indicators add one column per input column."""
        task = ADProgressionPrediction(missing_indicators=True)
        raw = [s for pid in sorted(self.patients) for s in self._raw(pid, task)]
        samples = create_sample_dataset(
            samples=raw,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
        )
        visits = next(s for s in samples if s["patient_id"] == "1")["visits"]
        self.assertEqual(tuple(visits.shape), (3, 2 * 26))
        self.assertFalse(torch.isnan(visits).any())


if __name__ == "__main__":
    unittest.main()
