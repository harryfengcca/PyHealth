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
our ablations.

Shapes use the row-vector convention, ``Y = X @ W``:
``[B, T, F] @ [F, d] -> [B, T, d]``, then ``[B, d] @ [d, C] -> [B, C]``.
"""

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset
from .base_model import BaseModel

ARCHITECTURES = ("encoder", "encoder_decoder")
LOSSES = ("ce", "weighted_ce", "focal")


class VisitTransformer(BaseModel):
    """Transformer over visit tokens for multiclass next-visit prediction.

    The input feature must be a ``tensor`` of shape ``[T, F]`` per sample,
    padded with all-zero rows by the dataloader. Padded visits are masked out
    of attention and pooling.

    Two architectures are available:

    * ``"encoder"`` (default, the configuration reported in the first
      author's thesis): a Transformer encoder followed by masked mean pooling.
    * ``"encoder_decoder"`` (the configuration reported in the paper): an
      encoder followed by a decoder in which one learned query token attends
      to the encoded visits; the query's output is classified.

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
            recent ``max_visits`` visits. Default 64.
        architecture: ``"encoder"`` or ``"encoder_decoder"``.
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
        class_weights: Optional[Sequence[float]] = None,
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
        self.position_embedding = nn.Embedding(max_visits, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
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
        class_weights: Optional[Sequence[float]],
    ) -> Optional[torch.Tensor]:
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
            Sequence embeddings ``[B, d]``.
        """
        if x.size(1) > self.max_visits:
            x = x[:, -self.max_visits :]
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

    def forward(self, **kwargs: Union[torch.Tensor, List]) -> Dict[str, torch.Tensor]:
        """Runs the model on a collated batch.

        Args:
            **kwargs: The batch from a PyHealth dataloader. Must contain the
                input feature ``[B, T, F]``; the label ``[B]`` is used for the
                loss when present. ``embed=True`` also returns the sequence
                embeddings.

        Returns:
            A dict with ``logit`` ``[B, C]`` and ``y_prob`` ``[B, C]``; with
            labels, also ``loss`` (scalar) and ``y_true`` ``[B]``; with
            ``embed=True``, also ``embed`` ``[B, d]``.
        """
        x = kwargs[self.feature_key]
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = torch.as_tensor(x).to(self.device, dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0)
        z = self.encode(x)
        logits = self.classifier(z)
        results: Dict[str, torch.Tensor] = {
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
