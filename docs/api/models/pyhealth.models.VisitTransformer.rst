pyhealth.models.VisitTransformer
===================================

A Transformer over visit tokens for next-visit diagnosis prediction, from `Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_. Each visit is a dense vector of measurements; the model supports the paper's encoder-decoder configuration and an encoder with masked mean pooling, and cross-entropy, class-weighted cross-entropy or focal loss.

.. autoclass:: pyhealth.models.VisitTransformer
    :members:
    :undoc-members:
    :show-inheritance:
