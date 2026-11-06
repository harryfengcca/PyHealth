pyhealth.datasets.TADPOLEDataset
===================================

The TADPOLE challenge dataset (Marinescu et al., 2018), built from the Alzheimer's Disease Neuroimaging Initiative (ADNI). ``TADPOLE_D1_D2.csv`` holds one row per clinical visit with the diagnosis and cognitive, MRI, PET and CSF measurements. Access requires an ADNI account through the `LONI Image and Data Archive <https://ida.loni.usc.edu>`_; see the `TADPOLE challenge <https://tadpole.grand-challenge.org/>`_ for details. Used to reproduce `Moghaddami et al. (CHIL 2025) <https://proceedings.mlr.press/v287/moghaddami25a.html>`_.

.. autoclass:: pyhealth.datasets.TADPOLEDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.tadpole.write_tadpole_fixture

.. autofunction:: pyhealth.datasets.tadpole.generate_synthetic_tadpole
