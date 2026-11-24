"""Fast synthetic tests for TADPOLE balancing and patient-level folds."""

import io
import json
import random
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from examples import tadpole_ad_progression_prediction_visit_transformer as pipeline
from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks import ADProgressionPrediction


def _metadata(counts: list[tuple[int, str, int]]) -> list[dict]:
    """Make metadata only; these are not patient records or clinical samples."""
    return [
        {"pid": f"synthetic-{group}-{kind}-{i}", "group": group, "seq_type": kind}
        for group, kind, count in counts
        for i in range(count)
    ]


class TestBalancedCohort(unittest.TestCase):
    """Balancing handles either majority and groups with only one type."""

    def test_matches_minority_in_each_group(self) -> None:
        meta = _metadata(
            [
                (1, "stable", 2),
                (1, "converter", 4),
                (2, "stable", 3),
                (2, "converter", 1),
                (3, "converter", 2),
                (4, "stable", 2),
                (5, "stable", 1),
                (5, "converter", 1),
                (5, "reverter", 1),
            ]
        )
        keep = pipeline.balanced_patients(meta, 42)
        counts = Counter((m["group"], m["seq_type"]) for m in meta if m["pid"] in keep)
        self.assertEqual(
            counts,
            {
                (1, "stable"): 2,
                (1, "converter"): 2,
                (2, "stable"): 1,
                (2, "converter"): 1,
                (5, "stable"): 1,
                (5, "converter"): 1,
            },
        )

    def test_preserves_paper_case_when_stable_is_majority(self) -> None:
        meta = _metadata([(1, "stable", 7), (1, "converter", 2)])
        keep = pipeline.balanced_patients(meta, 42)
        converters = {m["pid"] for m in meta if m["seq_type"] == "converter"}
        self.assertTrue(converters <= keep)
        self.assertEqual(len(keep), 4)

    def test_selection_is_seeded_order_independent_and_local(self) -> None:
        meta = _metadata(
            [
                (1, "stable", 8),
                (1, "converter", 2),
                (2, "stable", 2),
                (2, "converter", 8),
            ]
        )
        before = random.getstate()
        selected = pipeline.balanced_patients(meta, 42)
        self.assertEqual(selected, pipeline.balanced_patients(meta[::-1], 42))
        self.assertNotEqual(selected, pipeline.balanced_patients(meta, 123))
        self.assertEqual(random.getstate(), before)

    def test_no_pairs_cannot_silently_run_as_balanced(self) -> None:
        meta = _metadata([(1, "stable", 4), (2, "converter", 4)])
        keep = pipeline.balanced_patients(meta, 42)
        self.assertEqual(keep, set())
        with self.assertRaisesRegex(ValueError, "nonempty train, validation and test"):
            pipeline.plan_folds(meta, 2, 0.15, 42, keep)

    def test_report_accounts_for_removed_sequences_without_identifiers(self) -> None:
        meta = _metadata(
            [
                (1, "stable", 2),
                (1, "converter", 4),
                (2, "stable", 2),
            ]
        )
        keep = pipeline.balanced_patients(meta, 42)
        report = pipeline._balance_report(meta, keep, 42)
        self.assertEqual(report["policy_version"], pipeline.BALANCE_POLICY_VERSION)
        self.assertEqual(report["subjects_before"], 8)
        self.assertEqual(report["retained_subjects"], 4)
        self.assertEqual(report["removed_by_type"], {"stable": 2, "converter": 2})
        self.assertEqual(
            report["after"]["sequences_by_type"],
            {
                "stable": 2,
                "converter": 2,
            },
        )
        self.assertEqual(report["before"]["sequences_by_group"]["2"], {"stable": 2})
        self.assertNotIn("2", report["after"]["sequences_by_group"])
        self.assertNotIn("synthetic-", json.dumps(report))
        self.assertNotIn("pid", json.dumps(report))


class TestPatientFoldPlanning(unittest.TestCase):
    """Sparse strata still give disjoint, nonempty and reproducible parts."""

    def _assert_partition(self, splits: list, expected: set[str]) -> None:
        test_counts = Counter()
        for split in splits:
            train, val, test = map(set, (split.train, split.val, split.test))
            self.assertTrue(train and val and test)
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, expected)
            test_counts.update(test)
        self.assertEqual(test_counts, dict.fromkeys(expected, 1))

    def test_four_singleton_strata_have_no_empty_parts(self) -> None:
        meta = _metadata([(group, "stable", 1) for group in range(1, 5)])
        splits = pipeline.plan_folds(meta, 2, 0.15, 42)
        self._assert_partition(splits, {m["pid"] for m in meta})
        self.assertEqual(
            [(len(s.train), len(s.val), len(s.test)) for s in splits],
            [(1, 1, 2), (1, 1, 2)],
        )

    def test_global_validation_size_tracks_ratio_and_is_clamped(self) -> None:
        meta = _metadata([(1, "stable", 8), (2, "converter", 8)])
        for ratio, expected in ((0.001, 1), (0.15, 1), (0.4, 3), (0.99, 7)):
            with self.subTest(ratio=ratio):
                splits = pipeline.plan_folds(meta, 2, ratio, 42)
                self._assert_partition(splits, {m["pid"] for m in meta})
                self.assertEqual([len(s.val) for s in splits], [expected, expected])

    def test_sparse_strata_balance_total_and_stratum_test_sizes(self) -> None:
        meta = _metadata(
            [
                (1, "stable", 7),
                (1, "converter", 2),
                (2, "stable", 3),
                (3, "converter", 1),
                (7, "stable", 2),
                (8, "stable", 2),
            ]
        )
        splits = pipeline.plan_folds(meta, 5, 0.15, 42)
        self._assert_partition(splits, {m["pid"] for m in meta})
        sizes = [len(s.test) for s in splits]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        for kind, group in {(m["seq_type"], min(m["group"], 6)) for m in meta}:
            members = {
                m["pid"]
                for m in meta
                if m["seq_type"] == kind and min(m["group"], 6) == group
            }
            sizes = [len(set(s.test) & members) for s in splits]
            self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_seed_order_keep_filter_and_rng_state(self) -> None:
        meta = _metadata([(1, "stable", 6), (1, "converter", 6)])
        keep = {m["pid"] for m in meta if m["pid"].endswith(("0", "1"))}
        before = random.getstate()
        splits = pipeline.plan_folds(meta, 2, 0.15, 42, keep)
        self._assert_partition(splits, keep)
        self.assertEqual(splits, pipeline.plan_folds(meta[::-1], 2, 0.15, 42, keep))
        self.assertNotEqual(splits, pipeline.plan_folds(meta, 2, 0.15, 123, keep))
        self.assertEqual(random.getstate(), before)

    def test_invalid_settings_duplicate_patients_and_tiny_cohorts(self) -> None:
        meta = _metadata([(1, "stable", 4)])
        for folds in (0, 1, True, 2.5):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                pipeline.plan_folds(meta, folds, 0.15, 42)
        for ratio in (0.0, 1.0, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                pipeline.plan_folds(meta, 2, ratio, 42)
        with self.assertRaisesRegex(ValueError, "unique"):
            pipeline.plan_folds(meta + [meta[0]], 2, 0.15, 42)
        for n in range(4):
            with (
                self.subTest(n=n),
                self.assertRaisesRegex(
                    ValueError, "nonempty train, validation and test"
                ),
            ):
                pipeline.plan_folds(meta[:n], 2, 0.15, 42)
        with self.assertRaisesRegex(ValueError, "nonempty train, validation and test"):
            pipeline.plan_folds(meta, 5, 0.15, 42)

    def test_three_patients_support_three_folds(self) -> None:
        meta = _metadata([(1, "stable", 3)])
        self._assert_partition(
            pipeline.plan_folds(meta, 3, 0.15, 42), {m["pid"] for m in meta}
        )

    def test_empty_split_fails_before_fitting_or_training(self) -> None:
        for empty in ("train", "val", "test"):
            parts = {"train": ("a",), "val": ("b",), "test": ("c",)}
            parts[empty] = ()
            dataset, ctx = mock.Mock(), mock.Mock()
            with (
                self.subTest(empty=empty),
                self.assertRaisesRegex(
                    ValueError, "nonempty train, validation and test"
                ),
            ):
                pipeline.run_fold(
                    dataset,
                    pipeline.RunConfig("tiny"),
                    pipeline.PatientListSplit(**parts),
                    SimpleNamespace(),
                    ctx,
                )
            dataset.set_task.assert_not_called()
            ctx.filler.assert_not_called()

    def test_empty_processed_partition_fails_before_model_creation(self) -> None:
        split = pipeline.PatientListSplit(("a",), ("b",), ("c",))
        for empty in range(3):
            parts = [[object()], [object()], [object()]]
            parts[empty] = []
            dataset = mock.Mock()
            dataset.set_task.return_value = parts
            with (
                self.subTest(empty=empty),
                mock.patch.object(pipeline, "build_model") as build,
                self.assertRaisesRegex(
                    ValueError, "nonempty train, validation and test"
                ),
            ):
                pipeline.run_fold(
                    dataset,
                    pipeline.RunConfig("tiny"),
                    split,
                    SimpleNamespace(),
                    mock.Mock(),
                )
            build.assert_not_called()


class TestSplitIntegration(unittest.TestCase):
    """Policy provenance and CLI routing use the same selected cohort."""

    def test_cli_writes_balanced_counts_and_resumes_all_folds(self) -> None:
        meta = _metadata([(1, "stable", 4), (1, "converter", 2)])
        dataset = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "TADPOLE_D1_D2.csv").write_text("synthetic fixture")
            argv = [
                "--root",
                str(root),
                "--experiment",
                "quick",
                "--device",
                "cpu",
                "--only",
                "balanced/visit_transformer",
                "--output-dir",
                str(root / "out"),
            ]
            with (
                mock.patch.object(pipeline, "TADPOLEDataset", return_value=dataset),
                mock.patch.object(pipeline, "audit_cohort", return_value=(meta, {})),
                mock.patch.object(
                    pipeline,
                    "run_fold",
                    return_value={"overall": {"bca": 0.5, "mauc": 0.5}},
                ) as train,
                redirect_stdout(io.StringIO()),
            ):
                pipeline.main(argv)
                pipeline.main(argv)
            self.assertEqual(train.call_count, 2)
            for call in train.call_args_list:
                split = call.args[2]
                self.assertEqual(
                    (len(split.train), len(split.val), len(split.test)), (1, 1, 2)
                )
            out = root / "out/quick"
            report = json.loads((out / "cohort_balanced.json").read_text())
            self.assertEqual(
                report["after"]["sequences_by_type"],
                {
                    "stable": 2,
                    "converter": 2,
                },
            )
            self.assertNotIn("synthetic-", json.dumps(report))
            manifest = json.loads((out / "manifest.json").read_text())["identity"]
            self.assertEqual(
                manifest["fold_policy_version"], pipeline.FOLD_POLICY_VERSION
            )
            self.assertEqual(
                manifest["balance_policy_version"], pipeline.BALANCE_POLICY_VERSION
            )
            self.assertIn("2/2 folds", (out / "summary.md").read_text())
            dataset.set_task.assert_not_called()

    def test_task_keeps_all_diagnosis_outputs_when_training_class_is_missing(
        self,
    ) -> None:
        task = ADProgressionPrediction()
        dataset = create_sample_dataset(
            samples=[{"patient_id": "p0", "visits": [[1.0] * 26], "label": "AD"}],
            input_schema=task.input_schema,
            output_schema=task.output_schema,
        )
        self.assertEqual(
            dataset.output_processors["label"].label_vocab, {"CN": 0, "MCI": 1, "AD": 2}
        )


if __name__ == "__main__":
    unittest.main()
