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
evaluation, for the reproduction, the paper's setup and every ablation.

Synthetic data, no ADNI access needed::

    python examples/my_replication.py

Real TADPOLE data on a GPU (all arguments are passed through)::

    python examples/my_replication.py --root /path/to/tadpole --device cuda

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
    if "--experiment" not in argv:
        argv += ["--experiment", "all" if "--root" in argv else "quick"]
    pipeline.main(argv)
