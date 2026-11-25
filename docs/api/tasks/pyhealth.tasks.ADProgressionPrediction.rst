pyhealth.tasks.ADProgressionPrediction
=======================================

Next-visit Alzheimer's disease stage prediction (CN, MCI or AD) from a subject's earlier visits, with the stable and converter sequences of `Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_. Use with :class:`~pyhealth.datasets.TADPOLEDataset`.

Each sample holds the input visits (``visits``, one row per visit), a
``visits_mask`` that marks real visits after batch padding, and the
diagnosis at the next visit (``label``, always encoded as CN=0, MCI=1, AD=2).
Features are z-scored with statistics of the training split.

The paper does not say how repeated records of one exam date are handled.
:func:`~pyhealth.tasks.ad_progression.prepare_patient_visits` merges records
of the same date and visit code when they agree, and drops the subject when
they conflict or carry different visit codes. A transition string such as
``"MCI to NL"`` counts as a reversion.

.. autoclass:: pyhealth.tasks.ADProgressionPrediction
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.ad_progression.build_sequence

.. autofunction:: pyhealth.tasks.ad_progression.map_diagnosis

.. autofunction:: pyhealth.tasks.ad_progression.parse_numeric

.. autofunction:: pyhealth.tasks.ad_progression.forward_fill

.. autofunction:: pyhealth.tasks.ad_progression.prepare_patient_visits
