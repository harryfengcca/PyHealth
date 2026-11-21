"""Synthetic cohort-accounting and experiment-provenance regression tests."""

import copy
import io
import json
import tempfile
import unittest

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from examples import tadpole_ad_progression_prediction_visit_transformer as pipeline
from tests.core.test_ad_progression import fixture_patients


class TestTadpoleExperimentProvenance(unittest.TestCase):
    """Runs on five synthetic subjects without training or clinical data."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patients = fixture_patients(str(self.root))
        self.dataset = SimpleNamespace(iter_patients=lambda: iter(patients.values()))
        self.source = self.root / "TADPOLE_D1_D2.csv"
        self.args = pipeline.parse_args([
            "--root", str(self.root), "--experiment", "reproduce",
            "--output-dir", str(self.root / "output"), "--device", "cpu",
        ])
        self.identity = pipeline._experiment_identity(
            self.args, self.source,
            {"source_sha256": "test-code", "git_revision": "test"},
        )
        self.out = self.root / "reproduce"
        self.out.mkdir()

    def _record(self) -> dict:
        experiment_id = pipeline._ensure_manifest(self.out, self.identity)
        record = {
            "experiment_id": experiment_id,
            "config": {"name": "raw/visit_transformer", "model": "visit_transformer"},
            "fold": 0,
            "split": {"fingerprint": "synthetic-split", "sizes": {"test": 1}},
            "metrics": {"overall": {"bca": 0.75, "mauc": 0.8}},
        }
        record["run_id"] = pipeline._run_identity(record)
        return record

    def test_cohort_accounting_matches_fixture(self) -> None:
        meta, report = pipeline.audit_cohort(self.dataset, {})
        self.assertEqual(len(meta), 3)
        self.assertEqual(report["subjects"], 5)
        self.assertEqual(report["visits"], 18)
        self.assertEqual(report["sequences_by_type"], {"stable": 1, "converter": 2})
        self.assertEqual(report["subjects_by_membership"], {"d1": 5})
        self.assertEqual(report["excluded_subjects"], {
            "reverter": 1, "fewer_than_two_total_visits": 1,
        })
        self.assertEqual(report["visit_ledger"], {
            "excluded_subject_visits": 4,
            "selected_input_and_target_visits": 10,
            "missing_diagnosis_visits_before_target": 1,
            "visits_after_selected_target": 3,
            "collapsed_duplicate_rows_in_retained_subjects": 0,
        })
        self.assertEqual(sum(report["visit_ledger"].values()), report["visits"])
        self.assertEqual(report["transitions"], {
            "CN->CN": 1, "MCI->AD": 1, "CN->MCI": 1,
        })
        self.assertNotIn("patient_id", json.dumps(report))
        _, kept = pipeline.audit_cohort(self.dataset, {"drop_reverters": False})
        self.assertEqual(kept["sequences_by_type"]["reverter"], 1)
        self.assertEqual(sum(kept["visit_ledger"].values()), 18)

    def test_task_and_audit_share_preparation_and_count_rows_once(self) -> None:
        from tests.core.test_ad_progression import TestVisitPreparation as fixture
        event, patient = fixture._event, fixture._patient
        exact = event(0, "MCI", mmse=27)
        subjects = [
            patient([exact, exact, event(6, "Dementia", "m06")]),
            patient([event(0, "MCI", mmse=27), event(0, "MCI", mmse=28),
                     event(6, "Dementia", "m06")]),
            patient([event(0, "NL"), event(6, "MCI to NL", "m06"),
                     event(12, "NL", "m12")]),
            patient([event(0, "MCI", "bl"), event(0, "MCI", "m06"),
                     event(12, "Dementia", "m12")]),
        ]
        for i, subject in enumerate(subjects):
            subject.patient_id = f"synthetic-{i}"
        dataset = SimpleNamespace(iter_patients=lambda: iter(subjects))
        for drop in (True, False):
            with self.subTest(drop_reverters=drop):
                task = pipeline.ADProgressionPrediction(drop_reverters=drop)
                samples = [s for subject in subjects for s in task(subject)]
                raw = [task.raw_sequence(subject) for subject in subjects]
                meta, report = pipeline.audit_cohort(dataset, {"drop_reverters": drop})
                self.assertEqual(len(meta), len(samples))
                self.assertEqual(len(meta), sum(s is not None for s in raw))
                self.assertEqual(sum(report["visit_ledger"].values()), 12)
                ledger = report["visit_ledger"]
                self.assertEqual(
                    ledger["collapsed_duplicate_rows_in_retained_subjects"], 1
                )
                self.assertEqual(report["sequence_policy_version"], 2)
                self.assertEqual(
                    report["excluded_subjects"]["conflicting_duplicate_visit"], 1
                )
                self.assertEqual(
                    report["excluded_subjects"]["ambiguous_same_day_visits"], 1
                )

    def test_missing_and_unknown_diagnoses_are_counted_separately(self) -> None:
        """Malformed diagnoses are visible in aggregate exclusion accounting."""
        events = [SimpleNamespace(timestamp=0, dx=None),
                  SimpleNamespace(timestamp=1, dx="not-a-stage")]
        patient = SimpleNamespace(
            patient_id="synthetic", get_events=lambda **kwargs: events
        )
        dataset = SimpleNamespace(iter_patients=lambda: iter([patient]))
        meta, report = pipeline.audit_cohort(dataset, {})
        self.assertEqual(meta, [])
        self.assertEqual(report["missing_diagnoses"], 1)
        self.assertEqual(report["unrecognized_diagnoses"], 1)
        self.assertEqual(report["excluded_subjects"], {
            "fewer_than_two_diagnosed_visits": 1,
        })
        self.assertEqual(sum(report["visit_ledger"].values()), 2)

    def test_concurrent_fold_writers_preserve_both_records(self) -> None:
        """Independent fold jobs cannot overwrite one another's results."""
        first = self._record()
        second = copy.deepcopy(first)
        second["fold"] = 1
        second["run_id"] = pipeline._run_identity(second)
        path = self.out / "runs.jsonl"
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(pipeline._append_run, path, record)
                       for record in (first, second)]
            for future in futures:
                future.result()
        self.assertEqual(sorted(r["fold"] for r in pipeline.load_runs(path)), [0, 1])

    def test_summary_only_needs_no_source_and_marks_partial_folds(self) -> None:
        """Offline summaries verify provenance and disclose incomplete CV."""
        pipeline._append_run(self.out / "runs.jsonl", self._record())
        with (
            mock.patch.object(pipeline, "TADPOLEDataset") as dataset,
            mock.patch.object(pipeline, "_file_digest", side_effect=AssertionError),
            redirect_stdout(io.StringIO()),
        ):
            pipeline.main([
                "--output-dir", str(self.root), "--experiment", "reproduce",
                "--summarize-only", "--device", "cpu",
            ])
        dataset.assert_not_called()
        self.assertIn("1/10 folds", (self.out / "summary.md").read_text())

    def test_changed_settings_or_code_reject_resume(self) -> None:
        original = pipeline._ensure_manifest(self.out, self.identity)
        self.assertEqual(pipeline._ensure_manifest(self.out, self.identity), original)
        for section, key, value in [
            ("settings", "seed", 123), ("settings", "epochs", 3),
            ("settings", "lr", 0.01), ("settings", "folds", 5),
            ("code", "source_sha256", "different-code"),
        ]:
            changed = copy.deepcopy(self.identity)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "Incompatible experiment"
            ):
                pipeline._ensure_manifest(self.out, changed)

    def test_source_content_changes_identity(self) -> None:
        self.source.write_bytes(self.source.read_bytes() + b"\n")
        changed = pipeline._experiment_identity(
            self.args, self.source, self.identity["code"]
        )
        self.assertNotEqual(changed["source_sha256"], self.identity["source_sha256"])

    def test_fold_selectors_do_not_change_experiment_identity(self) -> None:
        self.args.fold = 3
        self.args.only = ["lstm"]
        self.args.cache_dir = "/unused/cache/location"
        changed = pipeline._experiment_identity(
            self.args, self.source, self.identity["code"]
        )
        self.assertEqual(changed, self.identity)

    def test_legacy_results_are_not_overwritten(self) -> None:
        path = self.out / "runs.jsonl"
        legacy = '{"config": {"name": "old"}, "fold": 0}\n'
        path.write_text(legacy)
        with self.assertRaisesRegex(ValueError, "Legacy results"):
            pipeline._ensure_manifest(self.out, self.identity)
        self.assertEqual(path.read_text(), legacy)

    def test_atomic_results_are_idempotent_and_reject_mixed_runs(self) -> None:
        record = self._record()
        path = self.out / "runs.jsonl"
        pipeline._append_run(path, record)
        pipeline._append_run(path, record)
        self.assertEqual(pipeline.load_runs(path), [record])
        changed = copy.deepcopy(record)
        changed["experiment_id"] = "another-experiment"
        changed["run_id"] = pipeline._run_identity(changed)
        with self.assertRaisesRegex(ValueError, "mixed experiment"):
            pipeline._append_run(path, changed)
        self.assertEqual(pipeline.load_runs(path), [record])

    def test_changed_partition_or_config_cannot_duplicate_a_fold(self) -> None:
        record = self._record()
        path = self.out / "runs.jsonl"
        pipeline._append_run(path, record)
        for key, value in [("split", {"fingerprint": "other"}),
                           ("config", {"name": record["config"]["name"], "lr": .1})]:
            changed = copy.deepcopy(record)
            changed[key] = value
            changed["run_id"] = pipeline._run_identity(changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Duplicate"):
                pipeline._append_run(path, changed)

    def test_invalid_record_or_manifest_is_rejected(self) -> None:
        record = self._record()
        changed = copy.deepcopy(record)
        changed["fold"] = 7
        with self.assertRaisesRegex(ValueError, "run fingerprint"):
            pipeline.summarize([changed], self.out)
        manifest_path = self.out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["settings"]["seed"] = 999
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
            pipeline.summarize([record], self.out)

    def test_invalid_cli_values(self) -> None:
        for argv in [["--fold", "-1"], ["--folds", "1"], ["--val-ratio", "1"],
                     ["--epochs", "0"], ["--lr", "0"]]:
            with self.subTest(argv=argv), mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    pipeline.parse_args(["--experiment", "reproduce"] + argv)

    def test_audit_only_does_not_train_or_fit(self) -> None:
        self.dataset.set_task = mock.Mock(side_effect=AssertionError("must not fit"))
        with (
            mock.patch.object(pipeline, "TADPOLEDataset", return_value=self.dataset),
            mock.patch.object(pipeline, "run_fold") as train,
            redirect_stdout(io.StringIO()),
        ):
            pipeline.main([
                "--root", str(self.root), "--audit-only", "--device", "cpu",
                "--output-dir", str(self.root / "audit"),
            ])
        train.assert_not_called()
        self.dataset.set_task.assert_not_called()
        report = json.loads((self.root / "audit/quick/cohort_default.json").read_text())
        self.assertEqual(report["retained_subjects"], 3)

    def test_main_resume_and_summary_without_training(self) -> None:
        argv = [
            "--root", str(self.root), "--experiment", "reproduce", "--device", "cpu",
            "--output-dir", str(self.root / "main"), "--folds", "2", "--fold", "0",
            "--only", "raw/visit_transformer", "--epochs", "1",
        ]
        metrics = {"overall": {"bca": .75, "mauc": .8}}
        with (
            mock.patch.object(pipeline, "TADPOLEDataset", return_value=self.dataset),
            mock.patch.object(pipeline, "run_fold", return_value=metrics) as train,
            redirect_stdout(io.StringIO()),
        ):
            pipeline.main(argv)
            pipeline.main(argv)
            train.assert_called_once()
            with self.assertRaisesRegex(ValueError, "Incompatible experiment"):
                pipeline.main(argv + ["--seed", "123"])
        path = self.root / "main/reproduce/runs.jsonl"
        runs = pipeline.load_runs(path)
        self.assertEqual(len(runs), 1)
        self.assertNotIn("patient_id", json.dumps(runs))


class TestCopyLastDiagnosis(unittest.TestCase):
    """Verify persistence predictions and integration without training."""

    @staticmethod
    def _sequence(history, target, kind):
        return {"diagnoses": history + [target], "group": len(history),
                "seq_type": kind}

    def test_canonical_predictions_and_stable_converter_accuracy(self):
        raw = [
            self._sequence(["CN", "CN"], "CN", "stable"),
            self._sequence(["MCI"], "MCI", "stable"),
            self._sequence(["MCI", "AD"], "AD", "stable"),
            self._sequence(["CN"], "MCI", "converter"),
            self._sequence(["CN", "MCI"], "AD", "converter"),
        ]
        pred = pipeline.predict_copy_last(raw)
        np.testing.assert_array_equal(pred["prob"].argmax(1), [0, 1, 2, 0, 1])
        np.testing.assert_array_equal(pred["prob"].sum(1), np.ones(5))
        metrics = pipeline.evaluate(pred)
        self.assertEqual(metrics["overall"]["accuracy"], 3 / 5)
        self.assertEqual(metrics["stable"]["accuracy"], 1)
        self.assertEqual(metrics["converter"]["accuracy"], 0)

    def test_target_changes_only_labels_not_predictions(self):
        first = self._sequence(["CN", "MCI"], "MCI", "stable")
        second = self._sequence(["CN", "MCI"], "AD", "converter")
        second["features"] = object()  # Unreadable, unused measurements.
        second["months"] = object()
        a, b = (pipeline.predict_copy_last([s]) for s in (first, second))
        np.testing.assert_array_equal(a["prob"], b["prob"])
        self.assertNotEqual(a["y"][0], b["y"][0])

    def test_empty_and_invalid_history(self):
        empty = pipeline.predict_copy_last([])
        self.assertEqual(empty["prob"].shape, (0, 3))
        for n, diagnoses in [(0, ["CN"]), (2, ["CN", "MCI"])]:
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "held-out"):
                pipeline.predict_copy_last([{
                    "group": n, "diagnoses": diagnoses, "seq_type": "stable",
                }])

    def test_fold_uses_only_requested_test_patients_without_fitting(self):
        cfg = pipeline.RunConfig("raw/copy_last", model="copy_last")
        split = pipeline.PatientListSplit(("train",), ("val",), ("test",))
        ctx = mock.Mock()
        # Train/validation values cannot supply predictions or be inspected.
        ctx.raw.return_value = {
            "train": object(), "val": object(),
            "test": self._sequence(["CN"], "MCI", "converter"),
        }
        dataset = mock.Mock()
        with (
            mock.patch.object(pipeline, "Trainer") as trainer,
            mock.patch.object(pipeline, "build_model") as model,
        ):
            metrics = pipeline.run_fold(dataset, cfg, split, mock.Mock(), ctx)
        ctx.raw.assert_called_once_with(True)
        ctx.model.assert_not_called()
        ctx.filler.assert_not_called()
        dataset.set_task.assert_not_called()
        trainer.assert_not_called()
        model.assert_not_called()
        self.assertEqual(metrics["overall"]["accuracy"], 0)
        for key in ("n_params", "train_seconds"):
            self.assertEqual(metrics[key], 0)
        for key in ("n_train", "n_val", "n_test"):
            self.assertEqual(metrics[key], 1)
        with self.assertRaisesRegex(ValueError, "missing"):
            pipeline.run_copy_last_fold(
                cfg, pipeline.PatientListSplit((), (), ("missing",)), ctx
            )
        with self.assertRaisesRegex(ValueError, "nonempty"):
            pipeline.run_copy_last_fold(
                cfg, pipeline.PatientListSplit((), (), ()), ctx
            )

    def test_real_task_sequences_share_cohort_and_baseline_expectations(self):
        with tempfile.TemporaryDirectory() as root:
            patients = fixture_patients(root)
            task = pipeline.ADProgressionPrediction()
            raw = [task.raw_sequence(p) for p in patients.values()]
        pred = pipeline.predict_copy_last([s for s in raw if s is not None])
        metrics = pipeline.evaluate(pred)
        self.assertEqual(metrics["overall"]["n"], 3)
        self.assertEqual(metrics["overall"]["accuracy"], 1 / 3)
        self.assertEqual(metrics["stable"]["accuracy"], 1)
        self.assertEqual(metrics["converter"]["accuracy"], 0)

    def test_baseline_configs_are_unique_in_each_suite(self):
        for suite in ("reproduce", "paper_setup", "quick", "all"):
            with self.subTest(suite=suite):
                configs = [c for c in pipeline.experiment_configs(suite)
                           if c.model == "copy_last"]
                self.assertEqual({c.name for c in configs},
                                 {"raw/copy_last", "balanced/copy_last"})
                self.assertEqual(len(configs), 2)

    def test_cli_records_zero_epochs_and_resumes_without_training(self):
        with tempfile.TemporaryDirectory() as root:
            patients = fixture_patients(root)
            dataset = SimpleNamespace(iter_patients=lambda: iter(patients.values()))
            argv = ["--root", root, "--experiment", "reproduce", "--device", "cpu",
                    "--output-dir", str(Path(root) / "out"), "--folds", "2",
                    "--fold", "0", "--only", "raw/copy_last"]
            # Fixture is too small for stratification: explicitly fix the
            # comparison split and exercise the real baseline/provenance path.
            task = pipeline.ADProgressionPrediction()
            ids = [str(p.patient_id) for p in patients.values()
                   if task.raw_sequence(p) is not None]
            split = pipeline.PatientListSplit((ids[0],), (ids[1],), (ids[2],))
            with (
                mock.patch.object(pipeline, "TADPOLEDataset", return_value=dataset),
                mock.patch.object(pipeline, "plan_folds", return_value=[split, split]),
                mock.patch.object(pipeline, "Trainer") as trainer,
                mock.patch.object(pipeline.MonthlyContext, "model") as monthly,
                redirect_stdout(io.StringIO()),
            ):
                pipeline.main(argv)
                pipeline.main(argv)
            trainer.assert_not_called()
            monthly.assert_not_called()
            out = Path(root) / "out/reproduce"
            runs = pipeline.load_runs(out / "runs.jsonl")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["epochs"], 0)
            self.assertEqual(runs[0]["metrics"]["n_params"], 0)
            self.assertIn("raw/copy_last", (out / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
