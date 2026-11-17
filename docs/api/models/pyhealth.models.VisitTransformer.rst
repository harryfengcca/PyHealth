pyhealth.models.VisitTransformer
===================================

A Transformer over visit tokens for next-visit diagnosis prediction, from `Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_. Each visit is a dense vector of measurements; the model supports the paper's encoder-decoder configuration, an encoder with masked mean pooling, and the Time Series Transformer (TST) of Zerveas et al. (KDD 2021) used as a baseline in the paper, with cross-entropy, class-weighted cross-entropy or focal loss.

The optional ``<feature_key>_mask`` batch field marks real visits, including
visits that are all zero after normalization. Each patient's most recent
``max_visits`` valid visits are selected before padding. Without a mask,
all-zero rows are treated as padding.

.. autoclass:: pyhealth.models.VisitTransformer
    :members:
    :undoc-members:
    :show-inheritance:
