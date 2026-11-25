pyhealth.models.VisitTransformer
===================================

A Transformer over visit tokens for next-visit diagnosis prediction, from
`Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_.
Each visit is a dense vector of measurements. Padded visits are masked out of
attention and pooling, either from an optional ``<feature_key>_mask`` batch
field or, without it, by treating all-zero rows as padding. Each patient keeps
its most recent ``max_visits`` visits.

Three architectures are available:

* ``encoder`` (default): a Transformer encoder with masked mean pooling.
* ``encoder_decoder``: the paper's encoder-decoder; one learned query attends
  to the encoded visits.
* ``tst``: the Time Series Transformer of Zerveas et al. (KDD 2021), the
  paper's second Transformer baseline.

Losses are cross-entropy, class-weighted cross-entropy and focal loss.

Differences from the paper
--------------------------

The paper gives the encoder-decoder's depth (4 encoder and 8 decoder layers),
width 256, 4 heads and 6,603,267 parameters, but not every detail. The
following are our choices:

* the decoder reads one learned query, and its output is classified;
* positions are learned embeddings, and blocks are pre-norm;
* the feed-forward width is 512, so the model has 8,460,035 parameters with
  the default 26 input columns and 3 classes;
* encoder and decoder layers are initialized independently with Xavier
  uniform weights;
* the TST's batch normalization uses real visits only, so padding does not
  change its statistics.

The default ``encoder`` architecture is a smaller variant (0.81M parameters);
the example's model ablation compares it with the encoder-decoder.

.. autoclass:: pyhealth.models.VisitTransformer
    :members:
    :undoc-members:
    :show-inheritance:
