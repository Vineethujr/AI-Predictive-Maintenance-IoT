"""Train and evaluate baseline models for the virtual maintenance demo."""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.simulator import FEATURES, TARGET, save_sensor_data

DATA_PATH = Path("data/raw/virtual_sensor_data.csv")
MODEL_PATH = Path("models/predictive_maintenance.joblib")

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add compact rolling features using only preceding/current readings."""
    result = frame.copy()
    for col in ["temperature", "vibration", "current"]:
        result[f"{col}_rolling_mean"] = result[col].rolling(5, min_periods=1).mean()
        result[f"{col}_rolling_std"] = result[col].rolling(5, min_periods=1).std().fillna(0)
    return result

MODEL_FEATURES = FEATURES + [
    "temperature_rolling_mean", "temperature_rolling_std",
    "vibration_rolling_mean", "vibration_rolling_std",
    "current_rolling_mean", "current_rolling_std",
]

def train_and_evaluate(data: pd.DataFrame | None = None) -> dict:
    if data is None:
        if not DATA_PATH.exists():
            data = save_sensor_data()
        else:
            data = pd.read_csv(DATA_PATH)

    required = set(FEATURES + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    clean = data.dropna(subset=FEATURES + [TARGET]).copy()
    clean[TARGET] = clean[TARGET].astype(int)
    if clean[TARGET].nunique() < 2:
        raise ValueError("Training data needs both failure classes (0 and 1). Generate a larger dataset.")

    featured = add_features(clean)
    X = featured[MODEL_FEATURES]
    y = featured[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }
    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        try:
            auc = float(roc_auc_score(y_test, prob))
        except ValueError:
            auc = None
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": auc,
        }
        fitted[name] = model

    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "features": MODEL_FEATURES,
        "base_features": FEATURES,
        "threshold": 0.50,
    }, MODEL_PATH)

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/model_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "models": results}, f, indent=2)
    predictions = X_test.copy()
    predictions["actual_failure"] = y_test.to_numpy()
    predictions["predicted_failure"] = best_pred
    predictions["failure_probability"] = best_model.predict_proba(X_test)[:, 1]
    predictions.to_csv("outputs/predictions/test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig("outputs/figures/confusion_matrix.png", dpi=160)
    plt.close(fig)

    metric_frame = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]]
    ax = metric_frame.plot(kind="bar", figsize=(8, 4), ylim=(0, 1), rot=0)
    ax.set_title("Baseline Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/figures/model_comparison.png", dpi=160)
    plt.close()

    return {
        "best_model": best_name,
        "metrics": results,
        "classification_report": classification_report(y_test, best_pred, zero_division=0),
        "test_rows": int(len(X_test)),
        "model_path": str(MODEL_PATH),
    }

if __name__ == "__main__":
    result = train_and_evaluate()
    print(f"Best model: {result['best_model']}")
    print(f"Test rows: {result['test_rows']}")
    for name, metrics in result["metrics"].items():
        print(f"\n{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value if value is None else f'{value:.3f}'}")
    print(f"\nSaved model to {result['model_path']}")
    print("Note: scores reflect synthetic rule-generated labels, not real-world performance.")
