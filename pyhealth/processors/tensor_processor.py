import math
from collections.abc import Iterable
from itertools import chain
from typing import Any, Literal

import torch

from . import register_processor
from .base_processor import FeatureProcessor
from .utils import NORMALIZATIONS, Normalization, fit_normalization


@register_processor("tensor")
class TensorProcessor(FeatureProcessor):
    """
    Feature processor for converting numerical lists to tensors.

    Input:
        - List of numbers (int/float) or nested lists of numbers

    Processing:
        - Convert input directly to torch.Tensor using torch.tensor()
        - Optionally, along the last dimension: normalization with statistics
          learned in ``fit()`` from the observed (non-NaN) values,
          missing-value indicators (1 where the value is NaN) appended, and
          NaN filled with 0, in that order. After ``"zscore"``
          normalization, 0 is the training mean.
        - Infinite training measurements cannot define normalization statistics
          and raise ValueError. Zero imputation preserves signed infinity so
          downstream models can reject invalid retained observations.

    Output:
        - torch.Tensor with appropriate shape and dtype

    Examples:
        >>> from pyhealth.processors import TensorProcessor
        >>> processor = TensorProcessor()
        >>> processor.fit([{"x": [1.0, 2.0, 3.0]}], "x")
        >>> processor.size()
        3
        >>> processor.process([4.0, 5.0, 6.0])
        tensor([4., 5., 6.])
    """

    stores_tensor = True
    # Defaults for processors pickled before these options existed.
    normalization: Normalization = None
    missing_indicators: bool = False
    impute_strategy: Literal["zero"] | None = None

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        spatial_dims: tuple[bool, ...] | None = None,
        normalization: Normalization = None,
        missing_indicators: bool = False,
        impute_strategy: Literal["zero"] | None = None,
    ) -> None:
        """
        Initialize the TensorProcessor.

        Args:
            dtype: The desired torch data type for the output tensor.
                  Default is torch.float32.
            spatial_dims: Tuple of booleans indicating which dimensions are spatial.
                  If None, defaults to all False. Default is None.
            normalization: ``None``, ``"zscore"`` (zero mean, unit variance)
                  or ``"minmax"`` (scaled to [0, 1] on the fitted range),
                  per entry of the last dimension. Default is None.
            missing_indicators: Whether to append a missing-value indicator
                  per entry of the last dimension. Default is False.
            impute_strategy: ``None`` (keep NaN) or ``"zero"``. Default is None.
        """
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
        if impute_strategy not in (None, "zero"):
            raise ValueError("impute_strategy must be None or 'zero'")
        self.dtype = dtype
        self._n_dim = None
        self._feature_dim = None
        self._spatial_dims = spatial_dims
        self.normalization = normalization
        self.missing_indicators = missing_indicators
        self.impute_strategy = impute_strategy
        self.learns_statistics = normalization is not None
        self.center: list[float] = []
        self.scale: list[float] = []

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Infer n_dim and the feature width from the first valid sample.

        Args:
            samples: Iterable of sample dictionaries.
            field: The field name to extract from samples.

        Raises:
            ValueError: If normalization training values contain infinity,
                or the fitted center or scale is non-finite.
        """
        sample_iterator = iter(samples)
        first_sample = None
        for sample in sample_iterator:
            if field in sample and sample[field] is not None:
                value = sample[field]
                tensor = (
                    value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else torch.tensor(value, dtype=self.dtype)
                )
                self._n_dim = tensor.dim()
                self._feature_dim = tensor.shape[-1] if tensor.dim() > 0 else 1
                first_sample = sample
                break

        if self.normalization is not None:
            # Include the sample used for shape inference in one-pass streams.
            fit_samples = chain(
                () if first_sample is None else (first_sample,), sample_iterator
            )
            self.center, self.scale = fit_normalization(
                (
                    self._normalization_input(sample[field]).numpy()
                    for sample in fit_samples
                    if sample.get(field) is not None
                ),
                self.normalization,
            )
            if not all(math.isfinite(value) for value in self.center + self.scale):
                raise ValueError(
                    "TensorProcessor fitted non-finite normalization statistics"
                )

    def _normalization_input(self, value: Any) -> torch.Tensor:
        """Checks a training value before it contributes to fitted statistics.

        Args:
            value: One raw numeric sample, with NaN for missing entries.

        Returns:
            A float64 tensor of shape ``[N, F]`` for normalization fitting.

        Raises:
            ValueError: If a training measurement is infinite.
        """
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if torch.isinf(tensor).any():
            raise ValueError(
                "TensorProcessor cannot fit normalization on infinite values; "
                "use finite measurements or NaN for missing values."
            )
        return tensor.reshape(-1, self._feature_dim)

    def process(self, value: Any) -> torch.Tensor:
        """
        Process a numerical value or list into a torch.Tensor.

        Args:
            value: Input value (list of numbers or nested lists)

        Returns:
            Processed tensor. With zero imputation, only NaN is replaced;
            signed infinity is preserved for downstream input validation.
        """
        # Prefer to avoid constructing a new tensor from an existing tensor
        # which can trigger a UserWarning. If value is already a tensor,
        # return a detached clone cast to the requested dtype.
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(dtype=self.dtype)
        else:
            tensor = torch.tensor(value, dtype=self.dtype)

        if self.normalization is not None:
            # Shaped like the last dimension, so a scalar input stays a scalar.
            shape = tensor.shape[-1:]
            center = torch.tensor(self.center, dtype=self.dtype).reshape(shape)
            scale = torch.tensor(self.scale, dtype=self.dtype).reshape(shape)
            tensor = (tensor - center) / scale
        if self.missing_indicators:
            if tensor.dim() == 0:
                tensor = tensor.reshape(1)
            indicators = torch.isnan(tensor).to(self.dtype)
        if self.impute_strategy == "zero":
            tensor = tensor.masked_fill(torch.isnan(tensor), 0.0)
        if self.missing_indicators:
            tensor = torch.cat([tensor, indicators], dim=-1)
        return tensor

    def size(self) -> int | None:
        """
        Get the feature size of the processor.

        Returns:
            The width of the last dimension (1 for scalars) seen in ``fit()``,
            doubled with missing-value indicators, or None before fitting.
        """
        feature_dim = getattr(self, "_feature_dim", None)
        if feature_dim is None or not self.missing_indicators:
            return feature_dim
        return feature_dim * 2

    def is_token(self) -> bool:
        """Whether the tensor represents discrete tokens, inferred from dtype.

        Returns:
            True for integer dtype (discrete tokens), False for floating-point
            dtype (continuous values).
        """
        return not self.dtype.is_floating_point

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Number of dimensions for the output tensor.

        Returns:
            (n_dim,). With missing-value indicators, a scalar input becomes a
            vector of 2, so n_dim is at least 1.

        Raises:
            NotImplementedError: If n_dim was not provided and fit() was not called.
        """
        if self._n_dim is None:
            raise NotImplementedError(
                "TensorProcessor cannot determine n_dim automatically. "
                "Call fit() first."
            )
        return (self._output_n_dim(),)

    def _output_n_dim(self) -> int:
        if self.missing_indicators:
            return max(self._n_dim, 1)
        return self._n_dim

    def spatial(self) -> tuple[bool, ...]:
        """Whether each dimension of the output tensor is spatial.

        If spatial_dims was provided at init, returns that. Otherwise defaults
        to all False based on n_dim.
        """
        if self._spatial_dims is not None:
            return self._spatial_dims
        if self._n_dim is None:
            raise NotImplementedError(
                "TensorProcessor cannot determine spatial dims. Call fit() first."
            )
        return tuple(False for _ in range(self._output_n_dim()))

    def __repr__(self) -> str:
        """
        String representation of the processor.

        Returns:
            str: String representation
        """
        return f"TensorProcessor(dtype={self.dtype})"
