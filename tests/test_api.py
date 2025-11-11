import os
import sys

# Ensure src package is importable when running the test directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_endpoint() -> None:
    """Ensure the `/predict` endpoint responds with a prediction payload."""
    features = {f"g1_{i}":0.1 for i in range(1,21)}
    features.update({f"g2_{i}":0.2 for i in range(1,21)})
    features.update({f"c_{i}":0.05 for i in range(1,28)})
    r = client.post("/predict", json=features)
    assert r.status_code == 200
    assert "predicted_target" in r.json()
