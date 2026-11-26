"""Full-pipeline replication of Moghaddami et al. (CHIL 2025) in PyHealth.

Contributors:
    Michael Mendes (NetID: mmendes3), Yiqun Ma (NetID: yiqunma3),
    Zhihao Feng (NetID: zhihaof2)

Paper:
    Mahdi Moghaddami, Clayton Schubring, Mohammad-Reza Siadat.
    "Transformer Model for Alzheimer's Disease Progression Prediction Using
    Longitudinal Visit Sequences." Proceedings of the Sixth Conference on
    Health, Inference, and Learning (CHIL 2025), PMLR 287:804-816.
    https://proceedings.mlr.press/v287/moghaddami25a.html

Description:
    Entry point for the full-pipeline replication (extra credit).

Runs data loading (TADPOLEDataset) -> task processing
(ADProgressionPrediction) -> model training (VisitTransformer, PyHealth's
LSTM and GRU, and the paper's TST and month-by-month RNN baselines) ->
evaluation, for the reproduction, the paper's setup and every ablation
(``--experiment all``).

Synthetic data, no ADNI access needed: every experiment runs with 2 folds
and 2 training epochs, 78 runs in about two minutes on a CPU. The numbers
only show that the pipeline runs end to end::

    python examples/my_replication.py

Real TADPOLE data on a GPU, with the full settings (10 folds, 5 for the
ablations, up to 200 epochs)::

    python examples/my_replication.py --root /path/to/tadpole --device cuda

Any argument of the pipeline script can be passed and overrides these
defaults, for example ``--experiment quick``.

See ``tadpole_ad_progression_prediction_visit_transformer.py`` for details.
"""

import importlib.util
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).with_name(
        "tadpole_ad_progression_prediction_visit_transformer.py"
    )
    spec = importlib.util.spec_from_file_location("tadpole_pipeline", script)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    argv = sys.argv[1:]

    def given(flag: str) -> bool:
        """Whether ``flag`` is on the command line, as ``--x v`` or ``--x=v``."""
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    defaults = {"--experiment": "all"}
    if not given("--root"):
        # Synthetic data: small settings, so every experiment runs in minutes.
        defaults.update(
            {
                "--folds": "2",
                "--ablation-folds": "2",
                "--epochs": "2",
                "--monthly-epochs": "2",
            }
        )
    for flag, value in defaults.items():
        if not given(flag):
            argv += [flag, value]
    pipeline.main(argv)
