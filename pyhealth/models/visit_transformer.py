"""Visit-level Transformer for next-visit diagnosis prediction.

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
    Visit-level Transformer for multiclass next-visit diagnosis prediction.

Implements the model of Moghaddami, Schubring and Siadat, "Transformer Model
for Alzheimer's Disease Progression Prediction Using Longitudinal Visit
Sequences" (CHIL 2025, https://proceedings.mlr.press/v287/moghaddami25a.html).

Each clinical visit is one token: a dense vector of measurements, the current
diagnosis and a time feature. Self-attention lets every visit attend to every
other visit, so a trend across visits is not filtered through a recurrent
state.

PyHealth's existing :class:`~pyhealth.models.Transformer` also accepts dense
``tensor`` inputs, but on visit sequences it has no position encoding (the
order of visits is invisible to it), infers the padding mask after a biased
linear projection (so padded visits take part in attention) and represents
a sequence by its first, i.e. oldest, visit. This model masks padded visits
out of attention and pooling, adds learned position embeddings, and
summarizes a sequence by masked mean pooling or, as in the paper, by a
decoder query. It also offers the class-weighted and focal losses used in
our ablations, and the Time Series Transformer (TST) of Zerveas et al.
(KDD 2021), the paper's second Transformer baseline.

Shapes use the row-vector convention, ``Y = X @ W``:
``[B, T, F] @ [F, d] -> [B, T, d]``, then ``[B, d] @ [d, C] -> [B, C]``.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel

ARCHITECTURES = ("encoder", "encoder_decoder", "tst")
LOSSES = ("ce", "weighted_ce", "focal")


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    """Fixed sine/cosine position encodings ``[length, dim]``."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * freq)
    table[:, 1::2] = torch.cos(position * freq[: dim // 2])
    return table


class _BatchNormEncoderLayer(nn.Module):
    """TST encoder layer: post-norm residual blocks with batch normalization.

    Follows ``TransformerBatchNormEncoderLayer`` of Zerveas et al. (2021):
    self-attention and a GELU feed-forward block, each followed by a residual
    connection and ``BatchNorm1d`` over the embedding channels.

    Args:
        d_model: Embedding width ``d``.
        nhead: Attention heads.
        dim_feedforward: Hidden width of the feed-forward block.
        dropout: Dropout rate.
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Encodes ``x`` ``[B, T, d]``; ``padding_mask`` ``[B, T]`` marks padding."""
        attended, _ = self.self_attn(
            x, x, x, key_padding_mask=padding_mask, need_weights=False
        )
        x = x + self.dropout1(attended)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x)))))
        return self.norm2(x.transpose(1, 2)).transpose(1, 2)


class VisitTransformer(BaseModel):
    """Transformer over visit tokens for multiclass next-visit prediction.

    The input feature must be a ``tensor`` of shape ``[T, F]`` per sample,
    padded with all-zero rows by the dataloader. Padded visits are masked out
    of attention and pooling.

    Three architectures are available:

    * ``"encoder"`` (default, the configuration reported in the first
      author's thesis): a Transformer encoder followed by masked mean pooling.
    * ``"encoder_decoder"`` (the configuration reported in the paper): an
      encoder followed by a decoder in which one learned query token attends
      to the encoded visits; the query's output is classified.
    * ``"tst"``: the Time Series Transformer of Zerveas et al. (KDD 2021),
      the paper's other Transformer baseline. Inputs are scaled by
      ``sqrt(d)`` and given fixed sine/cosine positions, the encoder layers
      use batch normalization and GELU, and the outputs of all
      ``max_visits`` positions (padding set to zero) are concatenated,
      ``[B, max_visits * d] @ [max_visits * d, C] -> [B, C]``. Its usual
      settings are ``embedding_dim=64, num_heads=8, num_layers=3,
      feedforward_dim=256, dropout=0.1`` with a small ``max_visits``.

    Args:
        dataset: The :class:`~pyhealth.datasets.SampleDataset` to train on.
            It must have one ``tensor`` input and one ``multiclass`` label.
        embedding_dim: Width ``d`` of the visit embeddings. Default 128.
        num_heads: Attention heads; must divide ``embedding_dim``. Default 8.
        num_layers: Encoder layers. Default 4.
        num_decoder_layers: Decoder layers for ``"encoder_decoder"``.
            Default 8.
        feedforward_dim: Hidden width of each feed-forward block. Default 512.
        dropout: Dropout rate. Default 0.3.
        max_visits: Maximum number of visits; longer inputs keep the most
            recent ``max_visits`` visits. With ``"tst"`` every input is
            padded to this length. Default 64.
        architecture: ``"encoder"``, ``"encoder_decoder"`` or ``"tst"``.
        loss: ``"ce"`` (cross-entropy), ``"weighted_ce"`` (cross-entropy with
            class weights) or ``"focal"`` (focal loss).
        class_weights: Weights per class index for ``"weighted_ce"`` and
            ``"focal"``. ``None`` with ``"weighted_ce"`` computes inverse
            class frequencies from ``dataset``.
        focal_gamma: Focusing parameter of the focal loss. Default 2.0.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import VisitTransformer
        >>> samples = [
        ...     {"patient_id": "p0", "visits": [[0.1, 1.0], [0.3, 1.0]],
        ...      "label": "MCI"},
        ...     {"patient_id": "p1", "visits": [[0.9, 0.0]], "label": "AD"},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"visits": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ... )
        >>> model = VisitTransformer(dataset, embedding_dim=16, num_heads=2)
        >>> model.input_dim
        2
        >>> from pyhealth.datasets import get_dataloader
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> output = model(**batch)
        >>> tuple(output["y_prob"].shape)
        (2, 2)
        >>> sorted(output)
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        num_decoder_layers: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.3,
        max_visits: int = 64,
        architecture: str = "encoder",
        loss: str = "ce",
        class_weights: Sequence[float] | None = None,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError(
                f"VisitTransformer expects one input feature, got {self.feature_keys}"
            )
        if len(self.label_keys) != 1:
            raise ValueError(
                f"VisitTransformer expects one label, got {self.label_keys}"
            )
        if self.mode != "multiclass":
            raise ValueError(f"VisitTransformer supports multiclass, got {self.mode}")
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.embedding_dim = embedding_dim
        self.max_visits = max_visits
        self.architecture = architecture
        self.loss_name = loss
        self.focal_gamma = focal_gamma
        self.input_dim = self._infer_input_dim(dataset)
        self.num_classes = self.get_output_size()

        self.input_projection = nn.Linear(self.input_dim, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        if architecture == "tst":
            self.register_buffer(
                "fixed_positions", _sinusoidal_positions(max_visits, embedding_dim)
            )
            self.tst_layers = nn.ModuleList(
                _BatchNormEncoderLayer(
                    embedding_dim, num_heads, feedforward_dim, dropout
                )
                for _ in range(num_layers)
            )
            self.output_dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(max_visits * embedding_dim, self.num_classes)
        else:
            self.position_embedding = nn.Embedding(max_visits, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )
            self.encoder_norm = nn.LayerNorm(embedding_dim)
            if architecture == "encoder_decoder":
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.decoder = nn.TransformerDecoder(
                    decoder_layer, num_layers=num_decoder_layers
                )
                self.query = nn.Parameter(torch.zeros(1, 1, embedding_dim))
                nn.init.normal_(self.query, std=0.02)
                self.decoder_norm = nn.LayerNorm(embedding_dim)
            self.classifier = nn.Linear(embedding_dim, self.num_classes)

        weights = self._resolve_class_weights(dataset, loss, class_weights)
        self.register_buffer(
            "class_weights",
            weights if weights is not None else torch.ones(self.num_classes),
        )
        self._use_weights = weights is not None

    def _infer_input_dim(self, dataset: SampleDataset) -> int:
        """Reads the number of columns per visit from the dataset."""
        processor = dataset.input_processors.get(self.feature_key)
        size = processor.size() if processor is not None else None
        if isinstance(size, int) and size > 0:
            return size
        first = dataset[0][self.feature_key]
        if isinstance(first, (tuple, list)):
            first = first[0]
        return int(torch.as_tensor(first).shape[-1])

    def _resolve_class_weights(
        self,
        dataset: SampleDataset,
        loss: str,
        class_weights: Sequence[float] | None,
    ) -> torch.Tensor | None:
        """Returns per-class loss weights, or ``None`` for unweighted losses."""
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.numel() != self.num_classes:
                raise ValueError(
                    f"class_weights needs {self.num_classes} values, "
                    f"got {weights.numel()}"
                )
            return weights
        if loss != "weighted_ce":
            return None
        counts = torch.zeros(self.num_classes)
        for sample in dataset:
            counts[int(torch.as_tensor(sample[self.label_key]).item())] += 1
        counts = counts.clamp(min=1)
        return counts.sum() / (self.num_classes * counts)

    def _compute_loss(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Cross-entropy (optionally class-weighted) or focal loss.

        Args:
            logits: Class scores ``[B, C]``.
            y_true: Class indices ``[B]``.

        Returns:
            The mean loss over the batch (a scalar).
        """
        weight = self.class_weights if self._use_weights else None
        if self.loss_name == "focal":
            log_prob = F.log_softmax(logits, dim=-1)
            log_pt = log_prob.gather(1, y_true.unsqueeze(1)).squeeze(1)
            focal = -((1 - log_pt.exp()) ** self.focal_gamma) * log_pt
            if weight is not None:
                focal = focal * weight[y_true]
            return focal.mean()
        return F.cross_entropy(logits, y_true, weight=weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of visit sequences into one vector per sequence.

        Args:
            x: Visit features ``[B, T, F]``; all-zero rows are padding.

        Returns:
            Sequence embeddings ``[B, d]``, or ``[B, max_visits * d]`` for
            ``"tst"``.
        """
        if x.size(1) > self.max_visits:
            x = x[:, -self.max_visits :]
        if self.architecture == "tst":
            return self._encode_tst(x)
        valid = x.abs().sum(dim=-1) != 0
        # A sequence with no visits would make attention return NaN.
        valid[:, 0] = valid[:, 0] | ~valid.any(dim=1)
        positions = torch.arange(x.size(1), device=x.device)
        h = self.input_projection(x) + self.position_embedding(positions)
        h = self.input_dropout(h)
        h = self.encoder(h, src_key_padding_mask=~valid)
        h = self.encoder_norm(h)
        if self.architecture == "encoder_decoder":
            query = self.query.expand(x.size(0), -1, -1)
            out = self.decoder(query, h, memory_key_padding_mask=~valid)
            return self.decoder_norm(out[:, 0])
        weights = valid.unsqueeze(-1).to(h.dtype)
        return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def _encode_tst(self, x: torch.Tensor) -> torch.Tensor:
        """TST encoding: pads to ``max_visits`` and concatenates all positions."""
        if x.size(1) < self.max_visits:
            x = F.pad(x, (0, 0, 0, self.max_visits - x.size(1)))
        valid = x.abs().sum(dim=-1) != 0
        valid[:, 0] = valid[:, 0] | ~valid.any(dim=1)
        h = self.input_projection(x) * math.sqrt(self.embedding_dim)
        h = self.input_dropout(h + self.fixed_positions)
        for layer in self.tst_layers:
            h = layer(h, ~valid)
        h = self.output_dropout(F.gelu(h)) * valid.unsqueeze(-1).to(h.dtype)
        return h.reshape(h.size(0), -1)

    def forward(self, **kwargs: torch.Tensor | list) -> dict[str, torch.Tensor]:
        """Runs the model on a collated batch.

        Args:
            **kwargs: The batch from a PyHealth dataloader. Must contain the
                input feature ``[B, T, F]``; the label ``[B]`` is used for the
                loss when present. ``embed=True`` also returns the sequence
                embeddings.

        Returns:
            A dict with ``logit`` ``[B, C]`` and ``y_prob`` ``[B, C]``; with
            labels, also ``loss`` (scalar) and ``y_true`` ``[B]``; with
            ``embed=True``, also ``embed`` (see :meth:`encode`).
        """
        x = kwargs[self.feature_key]
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = torch.as_tensor(x).to(self.device, dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0)
        z = self.encode(x)
        logits = self.classifier(z)
        results: dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": self.prepare_y_prob(logits),
        }
        if self.label_key in kwargs:
            y_true = torch.as_tensor(kwargs[self.label_key]).to(self.device).long()
            results["loss"] = self._compute_loss(logits, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = z
        return results
