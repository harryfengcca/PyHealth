pyhealth.tasks.ADProgressionPrediction
=======================================

Next-visit Alzheimer's disease stage prediction (CN, MCI or AD) from a subject's earlier visits, with the stable and converter sequences of `Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_. Use with :class:`~pyhealth.datasets.TADPOLEDataset`.

.. autoclass:: pyhealth.tasks.ADProgressionPrediction
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.ad_progression.build_sequence

.. autofunction:: pyhealth.tasks.ad_progression.map_diagnosis

.. autofunction:: pyhealth.tasks.ad_progression.parse_numeric

.. autofunction:: pyhealth.tasks.ad_progression.forward_fill
