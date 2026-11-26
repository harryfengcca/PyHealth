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

Compared with ``pyhealth.models.Transformer``
---------------------------------------------

:class:`~pyhealth.models.Transformer` also accepts dense tensors, but it is
built for code sequences. On visit vectors it has no position encoding, so
visit order is invisible to it; it infers padding after a biased projection,
so padded visits are never masked; and it summarizes a sequence by its first,
i.e. oldest, visit. ``VisitTransformer`` adds position embeddings, masks
padded visits out of attention, pooling and normalization, and summarizes a
sequence by masked mean pooling, a learned decoder query or (TST) all visit
states. No other PyHealth prediction model offers the paper's encoder-decoder
or the TST. A separate class keeps the existing ``Transformer`` unchanged for
the tasks that use it.

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
