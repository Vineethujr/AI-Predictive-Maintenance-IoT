"""Generate reproducible virtual IoT sensor data for a learning demo."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "temperature", "vibration", "current",
    "rotational_speed", "torque", "tool_wear"
]
TARGET = "failure"

def generate_sensor_data(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Return simulated sensor readings and a rule-generated failure label.

    Important: this is synthetic data. Labels are created by transparent
    thresholds and do not represent validated real-world failure outcomes.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100.")
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2025-01-01", periods=n_samples, freq="h")

    temperature = np.clip(rng.normal(62, 10, n_samples), 25, 110)
    vibration = np.clip(rng.gamma(2.2, 1.0, n_samples), 0.1, 12)
    current = np.clip(rng.normal(8.0, 1.8, n_samples), 1, 18)
    rotational_speed = np.clip(rng.normal(1500, 180, n_samples), 500, 2300)
    torque = np.clip(rng.normal(40, 8, n_samples), 5, 80)
    tool_wear = np.clip(np.cumsum(rng.uniform(0.0, 0.08, n_samples)), 0, 300)

    # Inject occasional abnormal operating periods to make the demo useful.
    abnormal_count = max(1, int(n_samples * 0.10))
    abnormal_idx = rng.choice(n_samples, size=abnormal_count, replace=False)
    temperature[abnormal_idx] += rng.uniform(12, 25, abnormal_count)
    vibration[abnormal_idx] += rng.uniform(1.5, 4.0, abnormal_count)
    current[abnormal_idx] += rng.uniform(1.5, 4.0, abnormal_count)
    tool_wear[abnormal_idx] += rng.uniform(15, 70, abnormal_count)

    temperature = np.clip(temperature, 25, 120)
    vibration = np.clip(vibration, 0.1, 15)
    current = np.clip(current, 1, 22)
    tool_wear = np.clip(tool_wear, 0, 360)

    # A transparent synthetic target rule—not a real failure label.
    score = (
        (temperature > 82).astype(int)
        + (vibration > 5.2).astype(int)
        + (current > 12.0).astype(int)
        + ((tool_wear > 180) & (torque > 45)).astype(int)
    )
    failure = (score >= 2).astype(int)

    return pd.DataFrame({
        "timestamp": timestamp,
        "temperature": temperature.round(2),
        "vibration": vibration.round(3),
        "current": current.round(2),
        "rotational_speed": rotational_speed.round(1),
        "torque": torque.round(2),
        "tool_wear": tool_wear.round(2),
        "failure": failure,
    })

def save_sensor_data(output_path: str = "data/raw/virtual_sensor_data.csv",
                     n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    data = generate_sensor_data(n_samples=n_samples, seed=seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data

if __name__ == "__main__":
    data = save_sensor_data()
    print(f"Generated {len(data):,} virtual sensor rows.")
    print(f"Failure label counts: {data['failure'].value_counts().to_dict()}")
    print("Saved to data/raw/virtual_sensor_data.csv")
