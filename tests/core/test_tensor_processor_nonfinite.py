"""Non-finite preprocessing regressions on tiny in-memory visit sequences."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models import VisitTransformer
from pyhealth.processors import MultiClassLabelProcessor, TensorProcessor


def _training_samples() -> list[dict]:
    """Returns three finite synthetic training patients with five features."""
    return [
        {
            "patient_id": f"train-{i}",
            "visits": [[float(i + t + j) for j in range(5)] for t in range(2)],
            "visits_mask": torch.ones(2, dtype=torch.bool),
            "label": label,
        }
        for i, label in enumerate(("CN", "MCI", "AD"))
    ]


def _dataset(normalization: str | None, indicators: bool = False) -> SampleDataset:
    """Fits preprocessing on finite training patients only."""
    samples = _training_samples()
    processor = TensorProcessor(
        normalization=normalization,
        missing_indicators=indicators,
        impute_strategy="zero",
    )
    processor.fit(samples, "visits")
    return create_sample_dataset(
        samples,
        input_schema={"visits": "tensor"},
        output_schema={
            "label": (MultiClassLabelProcessor, {"labels": ["CN", "MCI", "AD"]})
        },
        input_processors={"visits": processor},
    )


def _held_out(training: SampleDataset, samples: list[dict]) -> SampleDataset:
    """Reuses fitted processors without fitting on held-out measurements."""
    return create_sample_dataset(
        samples,
        input_schema=training.input_schema,
        output_schema=training.output_schema,
        input_processors=training.input_processors,
        output_processors=training.output_processors,
    )


def _model(dataset: SampleDataset, architecture: str) -> VisitTransformer:
    """Returns a seeded, minimal model with the existing architecture."""
    torch.manual_seed(0)
    return VisitTransformer(
        dataset,
        embedding_dim=16,
        num_heads=2,
        num_layers=1,
        num_decoder_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        max_visits=2,
        architecture=architecture,
    )


class TestTensorProcessorNonfinite(unittest.TestCase):
    """Zero imputation must not hide infinity from downstream validation."""

    def test_zero_imputation_preserves_signed_infinity(self) -> None:
        for normalization in (None, "zscore", "minmax"):
            for indicators in (False, True):
                with self.subTest(normalization=normalization, indicators=indicators):
                    processor = _dataset(normalization, indicators).input_processors[
                        "visits"
                    ]
                    raw = torch.tensor(
                        [[float("inf"), -float("inf"), float("nan"), 1, 2]]
                    )
                    output = processor.process(raw)
                    self.assertTrue(torch.isposinf(output[0, 0]))
                    self.assertTrue(torch.isneginf(output[0, 1]))
                    self.assertEqual(output[0, 2].item(), 0.0)
                    self.assertTrue(torch.isnan(raw[0, 2]))
                    if indicators:
                        torch.testing.assert_close(
                            output[0, 5:], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
                        )

    def test_training_infinity_is_rejected_before_normalization(self) -> None:
        for normalization in ("zscore", "minmax"):
            for sign in (float("inf"), -float("inf")):
                samples = _training_samples()
                samples[1]["visits"][0][0] = sign
                processor = TensorProcessor(normalization=normalization)
                with (
                    self.subTest(normalization=normalization, value=sign),
                    self.assertRaisesRegex(ValueError, "infinite"),
                ):
                    processor.fit(samples, "visits")

    def test_overflowed_normalization_statistics_are_rejected(self) -> None:
        processor = TensorProcessor(normalization="zscore", dtype=torch.float64)
        with (
            np.errstate(over="ignore", invalid="ignore"),
            self.assertRaisesRegex(ValueError, "normalization statistics"),
        ):
            processor.fit([{"visits": [[1e308], [1.5e308]]}], "visits")

    def test_single_pass_training_keeps_first_sample_in_statistics(self) -> None:
        samples = _training_samples()
        for normalization in ("zscore", "minmax"):
            with self.subTest(normalization=normalization):
                reference = TensorProcessor(normalization=normalization)
                reference.fit(samples, "visits")
                streamed = TensorProcessor(normalization=normalization)
                streamed.fit(iter(samples), "visits")
                np.testing.assert_allclose(streamed.center, reference.center)
                np.testing.assert_allclose(streamed.scale, reference.scale)

    def test_single_pass_training_rejects_infinite_first_sample(self) -> None:
        for normalization in ("zscore", "minmax"):
            for value in (float("inf"), -float("inf")):
                samples = _training_samples()
                samples[0]["visits"][0][0] = value
                processor = TensorProcessor(normalization=normalization)
                with (
                    self.subTest(normalization=normalization, value=value),
                    self.assertRaisesRegex(ValueError, "infinite"),
                ):
                    processor.fit(iter(samples), "visits")


class TestProcessorLoaderModel(unittest.TestCase):
    """Exercise real processing, PyHealth collation and all model variants."""

    def test_held_out_infinity_reaches_model_validation(self) -> None:
        for normalization in (None, "zscore", "minmax"):
            training = _dataset(normalization)
            for value in (float("inf"), -float("inf")):
                held_out = _held_out(
                    training,
                    [
                        {
                            "patient_id": "held-out",
                            "visits": [[value, 1.0, float("nan"), 2.0, 3.0]],
                            "visits_mask": torch.ones(1, dtype=torch.bool),
                            "label": "CN",
                        }
                    ],
                )
                batch = next(iter(get_dataloader(held_out, batch_size=1)))
                for architecture in ("encoder", "encoder_decoder", "tst"):
                    with (
                        self.subTest(
                            normalization=normalization,
                            value=value,
                            architecture=architecture,
                        ),
                        self.assertRaisesRegex(ValueError, "infinite"),
                    ):
                        _model(training, architecture)(**batch)

    def test_nan_batches_keep_finite_predictions_loss_and_gradients(self) -> None:
        samples = [
            {
                "patient_id": "all-missing",
                "visits": [[float("nan")] * 5],
                "visits_mask": torch.ones(1, dtype=torch.bool),
                "label": "CN",
            },
            {
                "patient_id": "mixed",
                "visits": [[1.0, 2.0, float("nan"), 4.0, 5.0], [2.0] * 5],
                "visits_mask": torch.ones(2, dtype=torch.bool),
                "label": "AD",
            },
        ]
        for normalization in (None, "zscore", "minmax"):
            for indicators in (False, True):
                training = _dataset(normalization, indicators)
                processor = training.input_processors["visits"]
                original = (list(processor.center), list(processor.scale))
                held_out = _held_out(training, samples)
                batch = next(iter(get_dataloader(held_out, batch_size=2)))
                self.assertEqual((processor.center, processor.scale), original)
                self.assertTrue(torch.isfinite(batch["visits"]).all())
                torch.testing.assert_close(
                    batch["visits_mask"],
                    torch.tensor([[True, False], [True, True]]),
                )
                for architecture in ("encoder", "encoder_decoder", "tst"):
                    with self.subTest(
                        normalization=normalization,
                        indicators=indicators,
                        architecture=architecture,
                    ):
                        model = _model(training, architecture)
                        output = model(**batch)
                        self.assertTrue(torch.isfinite(output["y_prob"]).all())
                        self.assertTrue(torch.isfinite(output["loss"]))
                        output["loss"].backward()
                        for name, parameter in model.named_parameters():
                            if name != "_dummy_param":
                                self.assertIsNotNone(parameter.grad, name)
                                self.assertTrue(
                                    torch.isfinite(parameter.grad).all(), name
                                )

    def test_processed_discarded_infinity_does_not_affect_predictions(self) -> None:
        for normalization in (None, "zscore", "minmax"):
            training = _dataset(normalization)
            history = [[float("inf")] * 5, [1.0] * 5, [2.0] * 5, [-float("inf")] * 5]
            samples = [
                {
                    "patient_id": "held-out",
                    "visits": history,
                    "visits_mask": torch.tensor([True, True, True, False]),
                    "label": "CN",
                }
            ]
            reference = [
                {
                    "patient_id": "reference",
                    "visits": history[1:3],
                    "visits_mask": torch.ones(2, dtype=torch.bool),
                    "label": "CN",
                }
            ]
            batch = next(
                iter(get_dataloader(_held_out(training, samples), batch_size=1))
            )
            clean = next(
                iter(get_dataloader(_held_out(training, reference), batch_size=1))
            )
            for architecture in ("encoder", "encoder_decoder", "tst"):
                with self.subTest(
                    normalization=normalization, architecture=architecture
                ):
                    model = _model(training, architecture).eval()
                    with torch.no_grad():
                        output = model(**batch)
                        expected = model(**clean)
                    torch.testing.assert_close(output["logit"], expected["logit"])
                    self.assertTrue(torch.isfinite(output["loss"]))


if __name__ == "__main__":
    unittest.main()
