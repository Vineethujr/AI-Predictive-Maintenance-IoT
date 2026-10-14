"""Load the trained model and score a single virtual sensor reading."""
from pathlib import Path
import joblib
import pandas as pd
from src.train import add_features, MODEL_FEATURES
from src.simulator import FEATURES

MODEL_PATH = Path("models/predictive_maintenance.joblib")

def predict_machine(reading: dict, history: pd.DataFrame | None = None) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python main.py` or train from the Streamlit app first.")
    missing = set(FEATURES) - set(reading)
    if missing:
        raise ValueError(f"Missing sensor values: {sorted(missing)}")

    bundle = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{key: float(reading[key]) for key in FEATURES}])
    if history is not None and not history.empty:
        combined = pd.concat([history[FEATURES].tail(4), row], ignore_index=True)
        engineered = add_features(combined).tail(1)
    else:
        engineered = add_features(row)
    X = engineered[MODEL_FEATURES]
    probability = float(bundle["model"].predict_proba(X)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    prediction = int(probability >= threshold)
    alert = "HIGH RISK" if probability >= 0.70 else ("WATCH" if probability >= 0.30 else "NORMAL")
    return {
        "predicted_failure": prediction,
        "failure_probability": probability,
        "alert": alert,
        "model_name": bundle.get("model_name", "Trained model"),
        "threshold": threshold,
    }
