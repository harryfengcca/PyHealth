pyhealth.models.VisitTransformer
===================================

A Transformer over visit tokens for next-visit diagnosis prediction, from `Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_. Each visit is a dense vector of measurements; the model supports the paper's encoder-decoder configuration, an encoder with masked mean pooling, and the Time Series Transformer (TST) of Zerveas et al. (KDD 2021) used as a baseline in the paper, with cross-entropy, class-weighted cross-entropy or focal loss.

.. autoclass:: pyhealth.models.VisitTransformer
    :members:
    :undoc-members:
    :show-inheritance:
