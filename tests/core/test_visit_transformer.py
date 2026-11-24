"""Unit tests for VisitTransformer, on tiny synthetic tensors.

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
    Unit tests for VisitTransformer on synthetic tensors.

Six hand-made sequences with five features are held in an in-memory sample
dataset, so no files are written. Models are tiny and seeded, so every test
runs in milliseconds and gives the same result on every run.
"""

import copy
import io
import unittest
from unittest import mock

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import VisitTransformer
from pyhealth.models.visit_transformer import _BatchNormEncoderLayer
from pyhealth.processors import MultiClassLabelProcessor
from pyhealth.trainer import Trainer

SMALL = {"embedding_dim": 16, "num_heads": 2, "num_layers": 1, "feedforward_dim": 32}


def _samples() -> list[dict]:
    """Six tiny sequences of 1-4 visits with 5 features and 3 classes."""
    rows = [
        ("p0", 3, "CN"),
        ("p1", 1, "MCI"),
        ("p2", 2, "AD"),
        ("p3", 4, "MCI"),
        ("p4", 2, "CN"),
        ("p5", 1, "AD"),
    ]
    out = []
    for i, (pid, n_visits, label) in enumerate(rows):
        visits = [[0.1 * (i + 1), 1.0, -0.5, float(t), 0.2] for t in range(n_visits)]
        out.append({"patient_id": pid, "visits": visits, "label": label})
    return out


class TestVisitTransformer(unittest.TestCase):
    """Instantiation, forward pass, output shapes and gradients."""

    @classmethod
    def setUpClass(cls) -> None:
        """Builds an in-memory sample dataset and one padded batch."""
        cls.dataset = create_sample_dataset(
            samples=_samples(),
            input_schema={"visits": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="synthetic",
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=6)))

    def _model(self, **kwargs: object) -> VisitTransformer:
        """A small, seeded model; ``kwargs`` override the small defaults."""
        torch.manual_seed(0)
        return VisitTransformer(self.dataset, **{**SMALL, **kwargs})

    def test_instantiation(self) -> None:
        """The model reads the input width and the classes from the dataset."""
        model = self._model()
        self.assertEqual(model.input_dim, 5)
        self.assertEqual(model.num_classes, 3)
        self.assertEqual(model.mode, "multiclass")

    def test_forward_shapes(self) -> None:
        """A batch gives logits, probabilities, labels and a finite loss."""
        out = self._model()(**self.batch)
        self.assertEqual(tuple(out["logit"].shape), (6, 3))
        self.assertEqual(tuple(out["y_prob"].shape), (6, 3))
        self.assertEqual(tuple(out["y_true"].shape), (6,))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.allclose(out["y_prob"].sum(-1), torch.ones(6), atol=1e-5))

    def test_label_processor_contract(self) -> None:
        """Every architecture uses the processor's output size, activation and loss."""
        processor = self.dataset.output_processors["label"]
        for arch in ("encoder", "encoder_decoder", "tst"):
            with (
                self.subTest(architecture=arch),
                mock.patch.object(
                    processor, "output_size", wraps=processor.output_size
                ) as output_size,
                mock.patch.object(
                    processor, "activation", wraps=processor.activation
                ) as activation,
                mock.patch.object(processor, "loss", wraps=processor.loss) as loss,
            ):
                model = self._model(
                    architecture=arch, num_decoder_layers=1, max_visits=4
                )
                out = model(**self.batch)
                output_size.assert_called_once_with()
                activation.assert_called_once_with(out["logit"])
                loss.assert_called_once_with(out["logit"], out["y_true"], weight=None)
                self.assertEqual(model.classifier.out_features, processor.size())

    def test_prediction_without_labels(self) -> None:
        """All architectures can predict without labels or a loss."""
        inputs = {"visits": self.batch["visits"]}
        for arch in ("encoder", "encoder_decoder", "tst"):
            with self.subTest(architecture=arch), torch.no_grad():
                model = self._model(
                    architecture=arch, num_decoder_layers=1, max_visits=4
                ).eval()
                out = model(**inputs)
                self.assertEqual(set(out), {"logit", "y_prob"})
                self.assertEqual(tuple(out["y_prob"].shape), (6, 3))
                self.assertTrue(torch.isfinite(out["y_prob"]).all())
                self.assertTrue(
                    torch.allclose(out["y_prob"].sum(-1), torch.ones(6), atol=1e-5)
                )

    def test_fixed_vocabulary_includes_unobserved_class(self) -> None:
        """The processor defines outputs even for classes absent from training."""
        dataset = create_sample_dataset(
            samples=[s for s in _samples() if s["label"] != "AD"],
            input_schema={"visits": "tensor"},
            output_schema={
                "label": (MultiClassLabelProcessor, {"labels": ["CN", "MCI", "AD"]})
            },
        )
        model = VisitTransformer(dataset, **SMALL)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        out = model(**batch)
        self.assertEqual(model.num_classes, 3)
        self.assertEqual(tuple(out["y_prob"].shape), (4, 3))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_trainer_integration(self) -> None:
        """The Trainer trains and evaluates every architecture on one tiny batch."""
        for arch in ("encoder", "encoder_decoder", "tst"):
            with self.subTest(architecture=arch):
                model = self._model(
                    architecture=arch, num_decoder_layers=1, max_visits=4
                )
                before = model.classifier.weight.detach().clone()
                trainer = Trainer(
                    model=model,
                    metrics=["accuracy"],
                    device="cpu",
                    enable_logging=False,
                )
                loader = get_dataloader(self.dataset, batch_size=6)
                trainer.train(train_dataloader=loader, epochs=1)
                self.assertFalse(torch.equal(before, model.classifier.weight))
                scores = trainer.evaluate(loader)
                self.assertTrue(torch.isfinite(torch.tensor(scores["loss"])))
                self.assertTrue(0.0 <= scores["accuracy"] <= 1.0)

    def test_backward_reaches_every_parameter(self) -> None:
        """The loss has a gradient for every weight, in every architecture."""
        for arch in ("encoder", "encoder_decoder", "tst"):
            model = self._model(architecture=arch, num_decoder_layers=1, max_visits=4)
            model(**self.batch)["loss"].backward()
            for name, param in model.named_parameters():
                if name == "_dummy_param":  # BaseModel's device marker
                    continue
                self.assertIsNotNone(param.grad, f"{arch}: {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"{arch}: {name}")

    def test_training_step_reduces_loss(self) -> None:
        """A few optimizer steps on one batch lower its loss."""
        model = self._model(dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = model(**self.batch)["loss"].item()
        for _ in range(5):
            optimizer.zero_grad()
            model(**self.batch)["loss"].backward()
            optimizer.step()
        self.assertLess(model(**self.batch)["loss"].item(), first)

    def test_state_dict_round_trip(self) -> None:
        """Saved weights load into a fresh model and give the same output."""
        for arch in ("encoder", "encoder_decoder", "tst"):
            params = {"architecture": arch, "num_decoder_layers": 1, "max_visits": 4}
            model = self._model(**params).eval()
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            torch.manual_seed(1)  # different initial weights
            restored = VisitTransformer(self.dataset, **{**SMALL, **params}).eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            with torch.no_grad():
                self.assertTrue(
                    torch.equal(
                        model(**self.batch)["logit"], restored(**self.batch)["logit"]
                    ),
                    arch,
                )

    def test_padding_does_not_change_output(self) -> None:
        """Padded visits do not change a sequence's encoding."""
        x = torch.randn(1, 2, 5)
        padded = torch.cat([x, torch.zeros(1, 3, 5)], dim=1)
        for arch in ("encoder", "tst"):
            model = self._model(architecture=arch, max_visits=8).eval()
            with torch.no_grad():
                self.assertTrue(
                    torch.allclose(model.encode(x), model.encode(padded), atol=1e-5),
                    arch,
                )

    def test_tst_output(self) -> None:
        """The TST concatenates all positions before the classifier."""
        model = self._model(architecture="tst", max_visits=4)
        out = model(**self.batch, embed=True)
        self.assertEqual(tuple(out["logit"].shape), (6, 3))
        self.assertEqual(tuple(out["embed"].shape), (6, 4 * 16))
        self.assertEqual(model.classifier.in_features, 4 * 16)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_encoder_decoder_output(self) -> None:
        """The encoder-decoder variant gives one prediction per sequence."""
        model = self._model(architecture="encoder_decoder", num_decoder_layers=1)
        out = model(**self.batch)
        self.assertEqual(tuple(out["logit"].shape), (6, 3))

    def test_embed(self) -> None:
        """``embed=True`` also returns the pooled sequence embedding."""
        out = self._model()(**self.batch, embed=True)
        self.assertEqual(tuple(out["embed"].shape), (6, 16))

    def test_tensor_tuple_input_and_boolean_embed_flag(self) -> None:
        """The annotated tuple input and bool flag follow the tensor contract."""
        for arch in ("encoder", "encoder_decoder", "tst"):
            with self.subTest(architecture=arch):
                model = self._model(
                    architecture=arch, num_decoder_layers=1, max_visits=4
                ).eval()
                wrapped = {**self.batch, "visits": (self.batch["visits"],)}
                with torch.no_grad():
                    direct = model(**self.batch, embed=True)
                    actual = model(**wrapped, embed=True)
                    plain = model(**wrapped, embed=False)
                for key in ("logit", "y_prob", "y_true", "loss", "embed"):
                    torch.testing.assert_close(actual[key], direct[key])
                self.assertNotIn("embed", plain)

    def test_losses(self) -> None:
        """Weighted cross-entropy and focal loss run; bad weights are rejected."""
        weighted = self._model(loss="weighted_ce")
        self.assertEqual(tuple(weighted.class_weights.shape), (3,))
        self.assertTrue(torch.allclose(weighted.class_weights, torch.ones(3)))
        for loss in ("weighted_ce", "focal"):
            out = self._model(loss=loss)(**self.batch)
            self.assertTrue(torch.isfinite(out["loss"]))
        with self.assertRaises(ValueError):
            self._model(class_weights=[1.0, 2.0])

    def test_truncation_ignores_other_patients_padding(self) -> None:
        """Each patient retains its own latest real visits in a mixed batch."""
        torch.manual_seed(7)
        short = torch.randn(1, 2, 5)
        long = torch.randn(1, 6, 5)
        batch = torch.cat([torch.nn.functional.pad(short, (0, 0, 0, 4)), long])
        for arch in ("encoder", "encoder_decoder", "tst"):
            for limit in (1, 4):
                with self.subTest(architecture=arch, max_visits=limit):
                    model = self._model(
                        architecture=arch, num_decoder_layers=1, max_visits=limit
                    ).eval()
                    with torch.no_grad():
                        actual = model.encode(batch)
                        expected = torch.cat(
                            [
                                model.encode(short[:, -limit:]),
                                model.encode(long[:, -limit:]),
                            ]
                        )
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_explicit_mask_preserves_zero_visits(self) -> None:
        """Explicit validity retains real zero rows and ignores masked garbage."""
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]]])
        padded = torch.cat([x, torch.full((1, 3, 5), 99.0)], dim=1)
        mask = torch.tensor([[True, True, False, False, False]])
        for arch in ("encoder", "encoder_decoder", "tst"):
            with self.subTest(architecture=arch):
                model = self._model(
                    architecture=arch, num_decoder_layers=1, max_visits=1
                ).eval()
                with torch.no_grad():
                    actual = model(visits=padded, visits_mask=mask)["logit"]
                    expected = model(
                        visits=x[:, -1:], visits_mask=torch.ones(1, 1, dtype=torch.bool)
                    )["logit"]
                torch.testing.assert_close(actual, expected)

    def test_truncation_backpropagates_only_through_retained_visits(self) -> None:
        """Padding and discarded history receive no input gradients."""
        mask = torch.tensor([[True, True, False, False], [True] * 4])
        for arch in ("encoder", "encoder_decoder", "tst"):
            with self.subTest(architecture=arch):
                model = self._model(
                    architecture=arch, num_decoder_layers=1, max_visits=1
                ).eval()
                x = torch.randn(2, 4, 5, requires_grad=True)
                out = model(visits=x, visits_mask=mask, label=torch.tensor([0, 1]))
                out["loss"].backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertEqual(x.grad[0, 0].abs().sum().item(), 0.0)
                self.assertEqual(x.grad[0, 2:].abs().sum().item(), 0.0)
                self.assertEqual(x.grad[1, :3].abs().sum().item(), 0.0)
                self.assertGreater(x.grad[0, 1].abs().sum().item(), 0.0)
                self.assertGreater(x.grad[1, 3].abs().sum().item(), 0.0)

    def test_visit_mask_validation_and_empty_fallback(self) -> None:
        """Reject malformed masks and keep all-padding samples finite."""
        for arch in ("encoder", "encoder_decoder", "tst"):
            with self.subTest(architecture=arch):
                model = self._model(
                    architecture=arch, num_decoder_layers=1, max_visits=4
                ).eval()
                with self.assertRaisesRegex(ValueError, "mask must have shape"):
                    model.encode(torch.zeros(2, 3, 5), mask=torch.ones(2, 2))
                with torch.no_grad():
                    out = model(visits=torch.zeros(2, 3, 5))
                self.assertTrue(torch.isfinite(out["y_prob"]).all())
        with self.assertRaisesRegex(ValueError, "max_visits must be positive"):
            self._model(max_visits=0)

    def test_infinite_retained_visits_are_rejected(self) -> None:
        """Both public entry points reject infinity before attention or BN."""
        for arch in ("encoder", "encoder_decoder", "tst"):
            model = self._model(architecture=arch, num_decoder_layers=1, max_visits=2)
            for value in (float("inf"), -float("inf")):
                x = torch.ones(1, 2, 5)
                x[0, 1, 0] = value
                for entry in ("forward", "encode"):
                    with (
                        self.subTest(architecture=arch, value=value, entry=entry),
                        self.assertRaisesRegex(ValueError, "infinite"),
                    ):
                        if entry == "forward":
                            model(visits=x, label=torch.tensor([0]))
                        else:
                            model.encode(x)

    def test_infinity_in_masked_or_discarded_visits_is_ignored(self) -> None:
        """Validation follows masking and history truncation for all models."""
        x = torch.tensor(
            [[[float("inf")] * 5, [0.1] * 5, [0.2] * 5, [-float("inf")] * 5]]
        )
        mask = torch.tensor([[True, True, True, False]])
        for arch in ("encoder", "encoder_decoder", "tst"):
            model = self._model(
                architecture=arch, num_decoder_layers=1, max_visits=2
            ).eval()
            with self.subTest(architecture=arch), torch.no_grad():
                actual = model(visits=x, visits_mask=mask)["logit"]
                expected = model(visits=x[:, 1:3])["logit"]
                torch.testing.assert_close(actual, expected)
                self.assertTrue(torch.isfinite(actual).all())

    def test_nan_features_match_zero_imputation_and_finite_gradients(self) -> None:
        """NaN remains supported through encode and forward, with explicit masks."""
        missing = torch.tensor(
            [[[float("nan")] * 5, [0.1, float("nan"), 0.3, 0.4, 0.5]]]
        )
        observed = torch.where(torch.isnan(missing), 0.0, missing)
        mask = torch.ones(1, 2, dtype=torch.bool)
        for arch in ("encoder", "encoder_decoder", "tst"):
            model = self._model(
                architecture=arch, num_decoder_layers=1, max_visits=2
            ).eval()
            with self.subTest(architecture=arch):
                x = missing.clone().requires_grad_()
                expected = model.encode(observed, mask)
                actual = model.encode(x, mask)
                torch.testing.assert_close(actual, expected)
                result = model(
                    visits=x, visits_mask=mask, label=torch.tensor([0]), embed=True
                )
                torch.testing.assert_close(result["embed"], actual)
                self.assertTrue(torch.isfinite(result["loss"]))
                result["loss"].backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertEqual(x.grad[torch.isnan(missing)].abs().sum().item(), 0.0)

    def test_long_sequences_are_truncated(self) -> None:
        """Inputs longer than ``max_visits`` keep the most recent visits."""
        model = self._model(max_visits=2).eval()
        with torch.no_grad():
            z = model.encode(torch.randn(2, 5, 5))
        self.assertEqual(tuple(z.shape), (2, 16))

    def test_invalid_arguments(self) -> None:
        """Invalid heads, architectures and losses are rejected."""
        with self.assertRaises(ValueError):
            self._model(num_heads=3)
        with self.assertRaises(ValueError):
            self._model(architecture="lstm")
        with self.assertRaises(ValueError):
            self._model(loss="hinge")

    def test_positive_integer_dimensions(self) -> None:
        """Invalid dimensions fail at construction with a descriptive error."""
        for name in (
            "embedding_dim",
            "num_heads",
            "num_layers",
            "num_decoder_layers",
            "feedforward_dim",
            "max_visits",
        ):
            for value in (0, -1, 1.5, True):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    self._model(architecture="encoder_decoder", **{name: value})

    def test_finite_dropout_and_focal_gamma(self) -> None:
        """Dropout and focal gamma are validated before modules are built."""
        invalid = {
            "dropout": (-0.1, 1.1, float("nan"), float("inf"), "bad", True),
            "focal_gamma": (-1.0, float("nan"), float("inf"), "bad", True),
        }
        for name, values in invalid.items():
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    self._model(**{name: value})

    def test_class_weights_require_finite_positive_vector(self) -> None:
        """Bad weights never reach cross-entropy or focal loss."""
        invalid = (
            1.0,
            [[1.0, 1.0, 1.0]],
            [1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [float("nan"), 1.0, 1.0],
            [float("inf"), 1.0, 1.0],
        )
        for loss in ("ce", "weighted_ce", "focal"):
            for weights in invalid:
                with (
                    self.subTest(loss=loss, weights=weights),
                    self.assertRaisesRegex(ValueError, "class_weights"),
                ):
                    self._model(loss=loss, class_weights=weights)

    def test_losses_match_manual_values_and_gradients(self) -> None:
        """Explicit weights use weighted CE or the batch mean of focal terms."""
        targets = torch.tensor([0, 1, 2, 1])
        scores = torch.tensor(
            [[2.0, 0.0, -1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
        )
        weights = torch.tensor([1.0, 2.0, 4.0])
        for loss in ("ce", "weighted_ce", "focal"):
            with self.subTest(loss=loss):
                model = self._model(loss=loss, class_weights=weights.tolist())
                logits = scores.clone().requires_grad_()
                reference = scores.clone().requires_grad_()
                actual = model._compute_loss(logits, targets)
                pt = reference.softmax(-1)[torch.arange(4), targets]
                terms = -pt.log() * weights[targets]
                if loss == "focal":
                    expected = ((1 - pt).square() * terms).mean()
                else:
                    expected = terms.sum() / weights[targets].sum()
                torch.testing.assert_close(actual, expected)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(logits.grad, reference.grad)

    def test_automatic_weights_on_imbalanced_labels(self) -> None:
        """Inverse frequencies respect the label vocabulary's class order."""
        samples = _samples()
        labels = ["CN", "CN", "CN", "MCI", "MCI", "AD"]
        for sample, label in zip(samples, labels):
            sample["label"] = label
        dataset = create_sample_dataset(
            samples,
            input_schema={"visits": "tensor"},
            output_schema={"label": "multiclass"},
        )
        model = VisitTransformer(dataset, loss="weighted_ce", **SMALL)
        expected = torch.empty(3)
        for label, index in dataset.output_processors["label"].label_vocab.items():
            expected[index] = 6 / (3 * labels.count(label))
        torch.testing.assert_close(model.class_weights, expected)

    def test_focal_loss_handles_confident_predictions(self) -> None:
        """Zero and fractional gamma keep extreme-score gradients finite."""
        for gamma in (0.0, 0.5, 2.0):
            with self.subTest(gamma=gamma):
                model = self._model(loss="focal", focal_gamma=gamma)
                logits = torch.tensor(
                    [[1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0]],
                    requires_grad=True,
                )
                targets = torch.tensor([0, 2])
                loss = model._compute_loss(logits, targets)
                self.assertTrue(torch.isfinite(loss))
                if gamma == 0:
                    torch.testing.assert_close(
                        loss, torch.nn.functional.cross_entropy(logits, targets)
                    )
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_transformer_layers_have_independent_seeded_initialization(self) -> None:
        """Cloned layers get different weights; a repeated seed reproduces them."""
        for arch in ("encoder", "encoder_decoder"):
            with self.subTest(architecture=arch):
                params = {
                    "architecture": arch,
                    "num_layers": 2,
                    "num_decoder_layers": 2,
                }
                model = self._model(**params)
                repeated = self._model(**params)
                for name, value in model.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, repeated.state_dict()[name]), name
                    )
                stacks = [model.encoder]
                if arch == "encoder_decoder":
                    stacks.append(model.decoder)
                for stack in stacks:
                    first, second = stack.layers
                    for a, b in (
                        (
                            first.self_attn.in_proj_weight,
                            second.self_attn.in_proj_weight,
                        ),
                        (first.linear1.weight, second.linear1.weight),
                    ):
                        self.assertIsNot(a, b)
                        self.assertFalse(torch.equal(a, b))
                for module in model.modules():
                    if isinstance(module, torch.nn.LayerNorm):
                        torch.testing.assert_close(
                            module.weight, torch.ones_like(module.weight)
                        )
                        torch.testing.assert_close(
                            module.bias, torch.zeros_like(module.bias)
                        )

    def test_tst_training_ignores_padding_in_normalization(self) -> None:
        """Masked padding cannot change real outputs, gradients or BN statistics."""
        torch.manual_seed(9)
        layer = _BatchNormEncoderLayer(16, 2, 32, 0.0)
        padded_layer = copy.deepcopy(layer)
        mask = torch.tensor([[False, False, True], [False, False, False]])
        x = torch.randn(2, 3, 16, requires_grad=True)
        padded = torch.cat([x.detach(), torch.full((2, 4, 16), 99.0)], dim=1)
        padded.requires_grad_()
        padded_mask = torch.nn.functional.pad(mask, (0, 4), value=True)
        actual = layer(x, mask)
        extended = padded_layer(padded, padded_mask)
        torch.testing.assert_close(actual[~mask], extended[~padded_mask])
        self.assertEqual(extended[padded_mask].abs().sum().item(), 0.0)
        for a, b in (
            (layer.norm1, padded_layer.norm1),
            (layer.norm2, padded_layer.norm2),
        ):
            torch.testing.assert_close(a.running_mean, b.running_mean)
            torch.testing.assert_close(a.running_var, b.running_var)
            self.assertEqual(a.num_batches_tracked.item(), 1)
            self.assertEqual(b.num_batches_tracked.item(), 1)
        target = torch.randn_like(actual[~mask])
        (actual[~mask] * target).sum().backward()
        (extended[~padded_mask] * target).sum().backward()
        torch.testing.assert_close(x.grad[~mask], padded.grad[~padded_mask])
        self.assertEqual(x.grad[mask].abs().sum().item(), 0.0)
        self.assertEqual(padded.grad[padded_mask].abs().sum().item(), 0.0)
        for a, b in zip(layer.parameters(), padded_layer.parameters()):
            self.assertTrue(torch.isfinite(a.grad).all())
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-5)

    def test_tst_single_valid_visit_trains_without_updating_statistics(self) -> None:
        """Singleton training uses stored BN statistics and retains gradients."""
        for loss in ("ce", "weighted_ce", "focal"):
            with self.subTest(loss=loss):
                model = self._model(architecture="tst", max_visits=1, loss=loss)
                with torch.no_grad():
                    model(**self.batch)
                norms = (model.tst_layers[0].norm1, model.tst_layers[0].norm2)
                before = [copy.deepcopy(norm.state_dict()) for norm in norms]
                x = torch.randn(1, 1, 5, requires_grad=True)
                result = model(visits=x, label=torch.tensor([0]))
                self.assertTrue(torch.isfinite(result["loss"]))
                result["loss"].backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, param in model.named_parameters():
                    if name != "_dummy_param":
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                for norm, state in zip(norms, before):
                    for key, value in norm.state_dict().items():
                        torch.testing.assert_close(value, state[key])

    def test_tst_all_padding_singleton_has_finite_training_loss(self) -> None:
        """The empty-history fallback also works with one training sample."""
        model = self._model(architecture="tst", max_visits=1)
        result = model(
            visits=torch.zeros(1, 2, 5),
            visits_mask=torch.zeros(1, 2, dtype=torch.bool),
            label=torch.tensor([0]),
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()


if __name__ == "__main__":
    unittest.main()
