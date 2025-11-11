from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.preprocessing import Preprocessor
from src.utils.logger import get_logger

app = FastAPI(title="Customer Group Profitability Predictor")

logger = get_logger(__name__)

model = joblib.load("models/trained_model.pkl")
pre = Preprocessor.load("models/preprocessor.pkl")
EXPECTED_FEATURES: List[str] = list(getattr(pre, "feature_names_", []) or [])


@app.get("/features")
def list_features() -> Dict[str, List[str]]:
    """Return the ordered feature list expected by the model."""
    if not EXPECTED_FEATURES:
        raise HTTPException(status_code=500, detail="Model metadata missing expected feature list.")
    return {"features": EXPECTED_FEATURES}

@app.post("/predict")
def predict(features: Dict[str, float]) -> Dict[str, str]:
    """Predict the most profitable customer group.

    :param features: Mapping of feature names to values supplied by the caller.
    :returns: JSON-serializable payload containing the predicted label.
    """
    if not EXPECTED_FEATURES:
        raise HTTPException(status_code=500, detail="Model metadata missing expected feature list.")

    payload = {}
    missing: List[str] = []
    extra: List[str] = []
    for col in EXPECTED_FEATURES:
        if col in features:
            payload[col] = features[col]
        else:
            payload[col] = 0.0
            missing.append(col)

    for key in features:
        if key not in EXPECTED_FEATURES:
            extra.append(key)

    if missing or extra:
        detail: Dict[str, List[str]] = {}
        if missing:
            detail["missing"] = missing
        if extra:
            detail["ignored"] = extra
        detail["message"] = "Payload adjusted: missing features filled with 0.0 and unsupported fields ignored."
        logger.warning("/predict adjusted payload", extra={"payload_adjustments": detail})

    X = pd.DataFrame([payload], columns=EXPECTED_FEATURES)

    try:
        X_prep = pre.transform(X)
        pred = model.predict(X_prep)[0]
        mapping = {0: "none profitable", 1: "group 1", 2: "group 2"}
        return {"predicted_target": mapping[int(pred)]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc