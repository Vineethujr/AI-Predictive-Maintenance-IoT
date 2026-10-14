from pathlib import Path
import sys

# Make the repository root importable when pytest is launched from different environments.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator import generate_sensor_data, FEATURES


def test_generated_data_has_expected_columns_and_values():
    data = generate_sensor_data(n_samples=1000, seed=7)
    assert set(FEATURES + ["failure", "timestamp"]).issubset(data.columns)
    assert len(data) == 1000
    assert data.isna().sum().sum() == 0
    assert set(data["failure"].unique()).issubset({0, 1})


def test_generator_is_reproducible():
    first = generate_sensor_data(n_samples=500, seed=21)
    second = generate_sensor_data(n_samples=500, seed=21)
    assert first.equals(second)
