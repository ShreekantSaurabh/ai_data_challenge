import os
import sys

# Ensure src package is importable when running the test directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.model import CampaignModel

def test_model_training() -> None:
    """Validate that model training completes and returns predictions."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 5))
    y = np.array([0] * 7 + [1] * 7 + [2] * 6)
    rng.shuffle(y)
    model = CampaignModel("rf")
    model.train(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
