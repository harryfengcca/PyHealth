pyhealth.models.VisitTransformer
===================================

A Transformer over visit tokens for next-visit diagnosis prediction, implementing
a reproduction attempt of `Moghaddami et al. (CHIL 2025)
<https://proceedings.mlr.press/v287/moghaddami25a.html>`_. Each visit is a dense
vector of measurements. The model supports an encoder with masked mean pooling,
an approximation of the paper's encoder-decoder architecture, and an adaptation
of the Time Series Transformer (TST) of Zerveas et al. (KDD 2021).

The optional ``<feature_key>_mask`` batch field marks real visits, including
visits that are all zero after normalization. Each patient's most recent
``max_visits`` valid visits are selected before padding. Without a mask,
all-zero rows are treated as padding.

Reproduction assumptions and differences
----------------------------------------

The paper's Section 2.4 and Figure 4 describe visits as tokens, an encoder-decoder
Transformer, 4 encoder layers, 8 decoder layers, 4 attention heads, width 256 and
6,603,267 parameters. The published description does not establish the remaining
implementation details below. Matching those dimensions is a partial
reconstruction, and does not establish equivalence to the authors' implementation.

.. list-table:: Model configurations and team choices
   :header-rows: 1
   :widths: 25 75

   * - Setting
     - Current implementation
   * - Default architecture
     - ``encoder`` with width 128, 8 heads, 4 encoder layers and masked mean
       pooling. This is a team variant, separate from the reported paper model.
   * - Paper-oriented architecture
     - ``encoder_decoder`` can use width 256, 4 heads and 4+8 layers. One learned
       query attends to encoder memory and its output is classified. The query
       and readout are team assumptions, not verified author implementation.
   * - Positions and normalization
     - Learned absolute positions, pre-norm residual blocks and separate final
       layer norms for the encoder and decoder are team choices.
   * - Feed-forward width and dropout
     - Defaults are 512 and 0.3. These are team settings; they have not been
       verified against the paper's implementation.
   * - Initialization
     - Encoder and decoder matrices are independently initialized with Xavier
       uniform weights, zero biases and unit layer-normalization scales. This
       avoids identical initial weights in cloned layers; it is not a verified
       paper initialization scheme. The query uses a normal distribution with
       standard deviation 0.02; input projection, position embeddings and the
       classifier retain their PyTorch initializers.
   * - Input representation
     - The task encodes 22 numeric measurements, diagnosis as 3 one-hot columns,
       and one optional time column: 26 columns by default. The paper lists 23
       clinical features including diagnosis and adds a temporal feature; it
       does not establish this exact tensor encoding.
   * - Loss variants
     - Weighted cross-entropy and focal loss are team ablations, not established
       choices of the paper's Transformer.

For the task's 26 input columns and 3 classes, the paper-oriented configuration
with width 256, 4 heads, 4+8 layers, feed-forward width 512 and ``max_visits=64`` has
**8,460,035 parameters**, compared with **6,603,267** reported in the paper. This
unresolved difference must accompany comparisons; the implementation is not
claimed to reproduce the paper's exact architecture or numerical results.

The team retains these architectures as its chosen implementation. A total
parameter count alone cannot determine decoder inputs, readout, normalization
or feed-forward width. The reported layer counts and the remaining team choices
are therefore recorded separately; exact author-source equivalence is unverified.

TST normalization and singleton batches
---------------------------------------

The TST option uses fixed sinusoidal positions, input scaling by ``sqrt(d)``,
GELU, batch normalization and concatenation of all ``max_visits`` output positions.
Its configuration is a team adaptation of the referenced TST baseline; the
Alzheimer's paper does not fully specify its TST settings.

Batch normalization gathers only valid visits across the batch and time axes.
Padded positions do not affect outputs for real visits, running moments or their
gradients, and padded outputs are zero. Training outputs can still depend on
other patients' real visits, as expected for batch normalization. Padding
invariance is tested with dropout disabled because dropout remains stochastic.

When a training batch contains just one valid visit, normalization uses the
stored running mean and variance without updating either statistic or the batch
counter. The initial stored values are zero mean and unit variance. Gradients
still reach the visit features and affine parameters. Masked normalization and
this singleton fallback are team adaptations, not verified choices of the
authors. The existing all-padding fallback supplies a single zero-valued token
for an empty history so predictions remain finite; it does not infer a real
clinical visit from missing data.

Configuration and loss validation
---------------------------------

Both ``forward`` and ``encode`` replace NaN measurements with zero before
inferring validity. Explicit masks still retain real all-zero visits. Infinite
measurements in retained visits raise ``ValueError`` before projection, attention
or normalization. Masked visits and history removed by ``max_visits`` are discarded
before this check, so their values cannot change a prediction. Infinity is not
replaced with an extreme finite value. This behavior is covered for all three
architectures, including finite gradients through NaN-imputed input batches.

Dimensions and layer counts must be positive integers, and attention heads must
divide the embedding width. Dropout must be finite and in ``[0, 1]``; focal gamma
must be finite and nonnegative. Explicit class weights must be a one-dimensional,
finite, strictly positive vector in label-processor class order. They apply to
all loss choices, including ``ce``. Automatic inverse-frequency weights for
``weighted_ce`` use only the dataset passed to the model; pass the training split.

Weighted cross-entropy divides by the sum of target weights in the batch. Focal
loss averages its optionally weighted terms over the batch size; its weight
scaling therefore differs from weighted cross-entropy. Zero gamma matches
ordinary unweighted cross-entropy when no class weights are supplied.

These fixes change initialization and TST training behavior. Use fresh experiment
outputs for the updated implementation. Existing historical tables without run
artifacts remain unverified; passing synthetic tests establishes software
behavior, not real-data predictive performance.

.. autoclass:: pyhealth.models.VisitTransformer
    :members:
    :undoc-members:
    :show-inheritance:
