pyhealth.tasks.ADProgressionPrediction
=======================================

Next-visit Alzheimer's disease stage prediction (CN, MCI or AD) from a subject's earlier visits, with the stable and converter sequences of `Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_. Use with :class:`~pyhealth.datasets.TADPOLEDataset`.

Samples include a boolean ``visits_mask`` alongside ``visits``. The dataloader
pads this mask with False; ``VisitTransformer`` and PyHealth's ``RNN`` use it
to distinguish real visits from padding even when normalized features are zero.
Existing task caches are rebuilt for this sample layout.

Sequence policy version 2 checks both explicit transition strings and observed
stage decreases for reversions before truncating at the first conversion.
Visit preparation is shared by the task, experiment audit, and monthly baselines.

Duplicate handling is a conservative team decision, not a procedure specified
by the paper. Identical selected visit fields collapse to one record. Matching
timestamps and nonempty visit codes allow complementary records to merge when
all overlapping selected values agree. Conflicts or distinct visit codes at the
same timestamp exclude the subject pending private source review. Unknown visit
codes allow only exact duplicates to collapse. Original source events are kept
unchanged, and the task cache key includes the new sequence policy version.

.. autoclass:: pyhealth.tasks.ADProgressionPrediction
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.ad_progression.build_sequence

.. autofunction:: pyhealth.tasks.ad_progression.map_diagnosis

.. autofunction:: pyhealth.tasks.ad_progression.parse_numeric

.. autofunction:: pyhealth.tasks.ad_progression.forward_fill

.. autofunction:: pyhealth.tasks.ad_progression.prepare_patient_visits
