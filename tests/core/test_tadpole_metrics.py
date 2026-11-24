"""Numerical reference cases and finite-fold coverage for TADPOLE metrics."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from examples import tadpole_ad_progression_prediction_visit_transformer as pipeline


class TestHandTillAUC(unittest.TestCase):
    """Use explicit pairwise ranking results, including asymmetric scores."""

    def test_perfect_reverse_and_tied_rankings(self) -> None:
        y = np.arange(3)
        for prob, expected in (
            (np.eye(3), 1.0),
            ((1 - np.eye(3)) / 2, 0.0),
            (np.full((3, 3), 1 / 3), 0.5),
        ):
            with self.subTest(expected=expected):
                self.assertAlmostEqual(
                    pipeline.hand_till_auc(y, prob, range(3)), expected
                )

    def test_asymmetric_pair_scores_are_not_renormalized(self) -> None:
        # Pair (CN,MCI): (1+1)/2; (CN,AD): (1+.5)/2;
        # (MCI,AD): (0+.5)/2. Their mean is 2/3.
        y = np.arange(3)
        prob = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        self.assertAlmostEqual(pipeline.hand_till_auc(y, prob, range(3)), 2 / 3)
        # With CN absent, only the MCI/AD pair remains: (0+.5)/2.
        self.assertAlmostEqual(pipeline.hand_till_auc(y[1:], prob[1:], range(3)), 0.25)

    def test_label_column_permutation_preserves_auc(self) -> None:
        prob = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        old_to_new = np.array([2, 0, 1])
        reordered = np.empty_like(prob)
        reordered[:, old_to_new] = prob
        self.assertAlmostEqual(
            pipeline.hand_till_auc(old_to_new, reordered, range(3)), 2 / 3
        )

    def test_fewer_than_two_observed_classes_is_undefined(self) -> None:
        for y, prob in (
            (np.array([], dtype=int), np.empty((0, 3))),
            (np.array([2, 2]), np.array([[0.0, 0.0, 1.0]] * 2)),
        ):
            with self.subTest(size=len(y)):
                self.assertTrue(math.isnan(pipeline.hand_till_auc(y, prob, range(3))))


class TestSubsetMetrics(unittest.TestCase):
    """Check metrics against small, hand-calculated confusion matrices."""

    def test_symmetric_confusion_matrix(self) -> None:
        # Each class: TP=1, FN=1, FP=1, TN=3.
        y = np.array([0, 0, 1, 1, 2, 2])
        prob = np.eye(3)[[0, 1, 1, 2, 2, 0]]
        out = pipeline.subset_metrics(y, prob)
        self.assertEqual(out["n"], 6)
        for name, expected in (
            ("accuracy", 0.5),
            ("sens", 0.5),
            ("spec", 0.75),
            ("bca", 0.625),
            ("f1_macro", 0.5),
            ("f1_weighted", 0.5),
            ("mauc", 0.625),
        ):
            self.assertAlmostEqual(out[name], expected, msg=name)
        for dx in pipeline.DX_CLASSES:
            self.assertAlmostEqual(out[f"sens_{dx}"], 0.5)
            self.assertAlmostEqual(out[f"spec_{dx}"], 0.75)

    def test_imbalanced_confusion_matrix_and_f1_averaging(self) -> None:
        # Rows=true, columns=predicted: [[2,1,0], [0,1,1], [0,0,1]].
        y = np.array([0, 0, 0, 1, 1, 2])
        out = pipeline.subset_metrics(y, np.eye(3)[[0, 0, 1, 1, 2, 2]])
        expected = {
            "accuracy": 2 / 3,
            "sens_CN": 2 / 3,
            "sens_MCI": 0.5,
            "sens_AD": 1.0,
            "spec_CN": 1.0,
            "spec_MCI": 3 / 4,
            "spec_AD": 4 / 5,
            "sens": 13 / 18,
            "spec": 17 / 20,
            "bca": 283 / 360,
            "f1_macro": 59 / 90,
            "f1_weighted": 61 / 90,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(out[name], value, msg=name)

    def test_perfect_converter_subset_uses_documented_zero_denominator_rule(
        self,
    ) -> None:
        out = pipeline.subset_metrics(np.array([1, 2]), np.eye(3)[[1, 2]])
        self.assertEqual(out["accuracy"], 1.0)
        self.assertEqual(out["sens_CN"], 0.0)
        self.assertAlmostEqual(out["sens"], 2 / 3)
        self.assertAlmostEqual(out["spec"], 1.0)
        self.assertAlmostEqual(out["bca"], 5 / 6)
        self.assertEqual(out["f1_macro"], 1.0)

    def test_single_observed_class_and_false_positive_absent_class(self) -> None:
        # Only AD has support: its F1 is 2/(2+0+1)=2/3.
        out = pipeline.subset_metrics(np.array([2, 2]), np.eye(3)[[2, 0]])
        self.assertTrue(math.isnan(out["mauc"]))
        self.assertAlmostEqual(out["f1_macro"], 2 / 3)
        self.assertAlmostEqual(out["f1_weighted"], 2 / 3)
        self.assertAlmostEqual(out["sens"], 1 / 6)
        self.assertAlmostEqual(out["spec"], 0.5)
        self.assertAlmostEqual(out["bca"], 1 / 3)

    def test_empty_subset_has_only_a_zero_count(self) -> None:
        self.assertEqual(
            pipeline.subset_metrics(np.array([], dtype=int), np.empty((0, 3))),
            {"n": 0},
        )

    def test_evaluation_routes_stable_converter_and_reverter_subsets(self) -> None:
        out = pipeline.evaluate(
            {
                "y": np.array([0, 2, 1, 0]),
                "prob": np.eye(3)[[0, 2, 1, 0]],
                "last_dx": np.array([0, 1, 0, 1]),
                "seq_type": np.array(["stable", "converter", "converter", "reverter"]),
            }
        )
        for name, n in (
            ("overall", 4),
            ("stable", 1),
            ("converter", 2),
            ("reverter", 1),
        ):
            self.assertEqual(out[name]["n"], n)
            self.assertEqual(out[name]["accuracy"], 1.0)
        self.assertEqual(out["conv_detect"], {"auroc": 1.0, "sens_at_spec90": 1.0})


class TestConverterDetection(unittest.TestCase):
    """Reference rankings for progression scores and the specificity cutoff."""

    def test_auc_and_sensitivity_at_ninety_percent_specificity(self) -> None:
        scores = np.array([0.9, 0.5, 0.8, 0.6, 0.4])
        prob = np.column_stack([1 - scores, scores, np.zeros(5)])
        out = pipeline.converter_detection(
            np.array([1, 1, 0, 0, 0]),
            prob,
            np.zeros(5, dtype=int),
            np.array(["converter", "converter", "stable", "stable", "stable"]),
        )
        # Positive .9 beats all 3 negatives; positive .5 beats only .4.
        self.assertAlmostEqual(out["auroc"], 4 / 6)
        # With 3 negatives, allowing <=10% FPR permits no false positives.
        self.assertAlmostEqual(out["sens_at_spec90"], 0.5)

    def test_scores_sum_more_severe_classes_and_exclude_existing_ad(self) -> None:
        out = pipeline.converter_detection(
            np.array([1, 1, 2]),
            np.array([[0.2, 0.5, 0.3], [0.1, 0.4, 0.5], [0.0, 0.0, 1.0]]),
            np.array([0, 1, 2]),
            np.array(["converter", "stable", "stable"]),
        )
        self.assertEqual(out, {"auroc": 1.0, "sens_at_spec90": 1.0})

    def test_tied_scores_have_half_auc_and_zero_sensitivity_at_high_specificity(
        self,
    ) -> None:
        out = pipeline.converter_detection(
            np.array([1, 0]),
            np.array([[0.5, 0.5, 0.0]] * 2),
            np.array([0, 0]),
            np.array(["converter", "stable"]),
        )
        self.assertEqual(out, {"auroc": 0.5, "sens_at_spec90": 0.0})

    def test_undefined_detection_returns_no_metric(self) -> None:
        for last, kinds in (
            ([2, 2], ["stable", "stable"]),
            ([0, 0], ["converter", "converter"]),
            ([0, 0], ["stable", "stable"]),
            ([], []),
        ):
            with self.subTest(last=last, kinds=kinds):
                n = len(last)
                self.assertEqual(
                    pipeline.converter_detection(
                        np.zeros(n, dtype=int),
                        np.full((n, 3), 1 / 3),
                        np.array(last, dtype=int),
                        np.array(kinds),
                    ),
                    {},
                )


class TestMetricFoldSummary(unittest.TestCase):
    """Missing/undefined folds cannot masquerade as complete metric coverage."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = {
            "settings": {"folds": 3, "ablation_folds": 2},
            "synthetic": True,
        }
        self.experiment_id = pipeline._ensure_manifest(self.root, self.identity)

    def _record(
        self,
        fold: int,
        metrics: dict,
        name: str = "raw/visit_transformer",
        group: str = "reproduce",
    ) -> dict:
        row = {
            "experiment_id": self.experiment_id,
            "config": {"name": name, "group": group},
            "fold": fold,
            "split": {"fingerprint": "synthetic"},
            "metrics": metrics,
        }
        row["run_id"] = pipeline._run_identity(row)
        return row

    def _csv_rows(self) -> dict[str, dict[str, str]]:
        with (self.root / "summary.csv").open() as handle:
            return {r["config"]: r for r in csv.DictReader(handle)}

    def test_single_finite_metric_reports_coverage_without_zero_sd(self) -> None:
        runs = [
            self._record(0, {"converter": {"mauc": float("nan")}}),
            self._record(1, {"converter": {"mauc": 1.0}}),
        ]
        text = pipeline.summarize(runs, self.root)
        row = self._csv_rows()["raw/visit_transformer"]
        self.assertEqual(row["folds"], "2")
        self.assertEqual(row["expected_folds"], "3")
        self.assertEqual(json.loads(row["missing_folds"]), [2])
        self.assertEqual(row["converter.mauc_mean"], "1.0000")
        self.assertEqual(row["converter.mauc_std"], "")
        self.assertEqual(row["converter.mauc_n_valid"], "1")
        self.assertIn("converter.mauc", text)
        self.assertIn("1/3", text)
        self.assertIn("[0]", text)
        self.assertIn("[2]", text)

    def test_all_nonfinite_or_missing_metrics_have_zero_coverage(self) -> None:
        runs = [
            self._record(0, {"overall": {"mauc": float("nan")}}),
            self._record(1, {"overall": {"mauc": float("inf")}}),
            self._record(2, {"overall": {"accuracy": 0.5}}),
        ]
        text = pipeline.summarize(runs, self.root)
        row = self._csv_rows()["raw/visit_transformer"]
        self.assertEqual(row["overall.mauc_n_valid"], "0")
        self.assertEqual(row["overall.mauc_mean"], "")
        self.assertEqual(row["overall.mauc_std"], "")
        self.assertEqual(row["conv_detect.auroc_n_valid"], "0")
        self.assertIn("n=0/3", text)
        self.assertNotIn("inf ±", text)

    def test_multiple_finite_folds_preserve_population_sd(self) -> None:
        runs = [
            self._record(0, {"overall": {"mauc": 0.5}}),
            self._record(1, {"overall": {"mauc": 1.0}}),
            self._record(2, {"overall": {"mauc": float("-inf")}}),
        ]
        text = pipeline.summarize(runs, self.root)
        row = self._csv_rows()["raw/visit_transformer"]
        self.assertEqual(row["overall.mauc_mean"], "0.7500")
        self.assertEqual(row["overall.mauc_std"], "0.2500")
        self.assertEqual(row["overall.mauc_n_valid"], "2")
        self.assertIn("0.750 ± 0.250 (n=2/3)", text)

    def test_configuration_specific_expected_folds_and_csv_quoting(self) -> None:
        name = 'M3 "query", d256'
        pipeline.summarize(
            [self._record(0, {"overall": {"mauc": 0.75}}, name=name, group="M3")],
            self.root,
        )
        row = self._csv_rows()[name]
        self.assertEqual(row["expected_folds"], "2")
        self.assertEqual(json.loads(row["missing_folds"]), [1])
        self.assertEqual(row["overall.mauc_n_valid"], "1")

    def test_welch_requires_two_finite_values_in_both_configurations(self) -> None:
        runs = [
            self._record(0, {"overall": {"mauc": 0.5}}),
            self._record(1, {"overall": {"mauc": float("nan")}}),
            self._record(0, {"overall": {"mauc": 0.5}}, name="raw/lstm"),
            self._record(1, {"overall": {"mauc": 0.7}}, name="raw/lstm"),
        ]
        with mock.patch.object(pipeline.stats, "ttest_ind") as test:
            text = pipeline.summarize(runs, self.root)
        test.assert_not_called()
        self.assertNotIn("Welch's t-test across folds:", text)

    def test_welch_reports_actual_finite_sample_sizes(self) -> None:
        runs = [
            self._record(0, {"overall": {"mauc": 0.5}}),
            self._record(1, {"overall": {"mauc": 1.0}}),
            self._record(2, {"overall": {"mauc": float("inf")}}),
            self._record(0, {"overall": {"mauc": 0.5}}, name="raw/lstm"),
            self._record(1, {"overall": {"mauc": 0.6}}, name="raw/lstm"),
            self._record(2, {"overall": {"mauc": 0.7}}, name="raw/lstm"),
        ]
        with mock.patch.object(
            pipeline.stats, "ttest_ind", return_value=(1.0, 0.3)
        ) as test:
            text = pipeline.summarize(runs, self.root)
        test.assert_called_once()
        args, kwargs = test.call_args
        self.assertEqual(args, ([0.5, 1.0], [0.5, 0.6, 0.7]))
        self.assertEqual(kwargs, {"equal_var": False})
        self.assertIn("Ref. valid folds", text)
        self.assertIn("Other valid folds", text)
        self.assertIn("| 2/3 | 3/3 |", text)

    def test_empty_summary_is_well_formed(self) -> None:
        text = pipeline.summarize([], self.root)
        with (self.root / "summary.csv").open() as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(
                reader.fieldnames,
                ["config", "folds", "expected_folds", "missing_folds"],
            )
            self.assertEqual(list(reader), [])
        self.assertIn("No completed runs", text)

    def test_undefined_welch_statistics_are_labeled_unavailable(self) -> None:
        runs = [
            self._record(fold, {"overall": {"mauc": 0.5}}, name=name)
            for name in ("raw/visit_transformer", "raw/lstm")
            for fold in (0, 1)
        ]
        for result in (
            (float("nan"), float("nan")),
            (float("inf"), 0.0),
            (0.0, float("nan")),
        ):
            with (
                self.subTest(result=result),
                mock.patch.object(pipeline.stats, "ttest_ind", return_value=result),
            ):
                text = pipeline.summarize(runs, self.root)
            self.assertIn("| NA | NA | 2/3 | 2/3 |", text)


if __name__ == "__main__":
    unittest.main()
